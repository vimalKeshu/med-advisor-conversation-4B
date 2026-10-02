from __future__ import annotations

import json
import os
import math
import queue as queue_mod
import statistics as st
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM

from common import completion_logprobs, load_config
from wsync import WeightSender, weight_metas

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# batch state
# ---------------------------------------------------------------------------

@dataclass
class Batch:

    planned: int
    received: int = 0
    usable: int = 0
    loss: float = 0.0
    advantages: list[float] = field(default_factory=list)

    tokens: int = 0
    abs_ratio_err: float = 0.0
    signed_dlogp: float = 0.0
    kl: float = 0.0
    by_turn: dict = field(default_factory=lambda: defaultdict(lambda: [0.0, 0]))
    max_ratio_err: float = 0.0
    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    neg_logp_sum: float = 0.0    

    @property
    def full(self) -> bool:
        return self.received >= self.planned

    @property
    def scale(self) -> float:
        """DDP's uneven-rank correction. Weights were divided by `planned`, but
        only `usable` groups contributed, so the accumulated gradient is short by
        exactly this factor."""
        return self.planned / self.usable if self.usable else 0.0

    def weight(self, n_tokens: int, n_turns: int, n_trajs: int) -> float:
        """The four-level mean, as one scalar per token: tokens in a turn, turns
        in a trajectory, trajectories in a group, groups in a batch. Uses
        `planned` because the usable count is unknowable until the batch ends;
        `scale` corrects it before the step."""
        return 1.0 / (n_tokens * n_turns * n_trajs * self.planned)

    def group_token_weight(self, group_tokens: int) -> float:
        return 1.0 / (group_tokens * self.planned)

# ---------------------------------------------------------------------------
# advantages -- the ordering is mandatory
# ---------------------------------------------------------------------------

def prepare_group(group: dict, cfg: dict) -> tuple[list[tuple], str | None]:
    eligible = []

    for trajectory in group["trajectories"]:
        reward_info = trajectory["reward"]

        if not reward_info["valid"]:
            continue

        reward = reward_info["trajectory_reward"]
        if reward is None:
            continue

        turns = [
            (i, turn)
            for i, turn in enumerate(trajectory.get("turns") or [])
            if turn.get("completion_ids")
        ]

        if not turns:
            continue

        eligible.append((trajectory, reward, turns))

    if len(eligible) < cfg["min_eligible"]:
        return [], "too_few_eligible"

    rewards = [reward for _, reward, _ in eligible]

    mean = st.fmean(rewards)
    std = st.pstdev(rewards)

    if std <= cfg["std_epsilon"]:
        return [], "zero_reward_variance"

    members = []

    for trajectory, reward, turns in eligible:
        # Dr-GRPO
        advantage = reward - mean  

        # Vanilla GRPO 
        # advantage = (reward - mean) / (std + cfg["std_epsilon"])          

        members.append((advantage, reward, turns))

    return members, None

# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def turn_loss(
    model: Any,
    turn: dict,
    advantage: float,
    weight: float,
    cfg: dict,
    batch: Batch,
) -> torch.Tensor:

    prompt_ids = turn["prompt_ids"]
    completion_ids = turn["completion_ids"]
    device = next(model.parameters()).device

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logp = completion_logprobs(
            model,
            prompt_ids + completion_ids,
            len(prompt_ids),
            cfg["logprob_temperature"],
            chunk=cfg["logit_chunk"],
            grad=True,
        )
    # Do policy-gradient math in fp32.
    logp = logp.float()
    assert logp.numel() == len(completion_ids)
    # ---------------------------------------------------------
    # On-policy policy gradient
    #
    # Rollout policy == trainer policy, and each rollout is
    # consumed exactly once before optimizer.step().
    #
    # grad[-A * log pi] = -A * grad[log pi]
    # ---------------------------------------------------------
    policy = -logp * advantage
    # ---------------------------------------------------------
    # Optional reference-policy KL
    # ---------------------------------------------------------
    beta = cfg["kl_beta"]
    if beta > 0.0:
        logp_ref = turn.get("logp_ref")

        if logp_ref is None:
            raise RuntimeError(
                "kl_beta > 0 but logp_ref is missing"
            )      

        ref = torch.tensor(
            logp_ref,
            dtype=torch.float32,
            device=device,
        )
        assert ref.numel() == logp.numel()
        d = ref - logp
        # k3 estimator:
        # exp(log pi_ref - log pi)
        #   - (log pi_ref - log pi)
        #   - 1
        #
        # E_pi[k3] = KL(pi || pi_ref)
        kl = d.exp() - d - 1.0
        policy = policy + beta * kl
        # Diagnostic only -- raw token sum.
        batch.kl += float(kl.detach().sum())

    # ---------------------------------------------------------
    # Rollout/trainer synchronization diagnostics ONLY
    # ---------------------------------------------------------
    with torch.no_grad():
        logp_old = torch.tensor(
            turn["logp_old"],
            dtype=torch.float32,
            device=device,
        )
        assert logp_old.numel() == logp.numel()
        delta = logp.detach() - logp_old
        # Keep ratio only as telemetry.
        ratio = delta.exp()
        ratio_err = (ratio - 1.0).abs()
        batch.tokens += delta.numel()
        batch.abs_ratio_err += float(ratio_err.sum())
        batch.signed_dlogp += float(delta.sum())
        batch.max_ratio_err = max(
            batch.max_ratio_err,
            float(ratio_err.max()),
        )
        batch.neg_logp_sum += float(-logp.detach().sum())        
        seen = batch.by_turn[turn["_index"]]
        seen[0] += float(ratio_err.sum())
        seen[1] += delta.numel()

    # weight = 1 / (group_tokens * planned_groups)
    return policy.sum() * weight

# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Owns the policy, the optimizer and the weight-sync channel."""

    def __init__(self, full: dict, ctrl_q: Any = None) -> None:
        self.full, self.cfg, self.ctrl_q = full, full["train"], ctrl_q
        self.version = self.steps = 0
        self.skips: Counter = Counter()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # fp32 masters, bf16 autocast -- the phase-1 recipe. Pure-bf16 parameters
        # round Adam updates to zero, and at 1e-6 there is even less to round.
        self.model = AutoModelForCausalLM.from_pretrained(
            full["model"]["policy"], torch_dtype=torch.float32,
            attn_implementation="sdpa",
        ).to("cuda")
        self.model.config.use_cache = False
        if self.cfg["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.train()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg["learning_rate"],
            betas=tuple(self.cfg["betas"]), eps=self.cfg["epsilon"],
            weight_decay=self.cfg["weight_decay"], fused=True,
        )
        self.optimizer.zero_grad(set_to_none=True)

        if self.cfg.get("min_lr_ratio", None) is None:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda s: min(1.0, (s + 1) / max(1, self.cfg["warmup_steps"])),
            )
        else:
            total_steps = int(self.cfg["max_steps"])
            warmup_steps = int(self.cfg["warmup_steps"])
            min_lr_ratio = float(self.cfg.get("min_lr_ratio", 0.0))            
            def lr_lambda(step: int) -> float:
                update = step + 1  # Actual optimizer update number: 1, 2, ...

                if warmup_steps > 0 and update <= warmup_steps:
                    return update / warmup_steps

                # With no warmup, the first update starts at peak LR.
                peak_update = max(1, warmup_steps)
                progress = (update - peak_update) / max(1, total_steps - peak_update)
                progress = min(1.0, max(0.0, progress))

                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)        

        self.sender = self._connect()
        self.out_dir = HERE / full["run"]["out_dir"]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = (self.out_dir / "train_metrics.jsonl").open("a", encoding="utf-8")
        self.cursor = 0
        self.epoch = 0
        resumed = self._restore() if self.cfg.get("save_resume_checkpoint", True) else False
        if self.ctrl_q is not None:
            if resumed and self.sender is not None:
                self.ctrl_q.put(("sync",))
                self.sender.send()
            self.ctrl_q.put(("start", self.cursor, self.version))
    
    def _connect(self) -> Any:
        """NCCL handshake. Order is load-bearing: the actor must be told to join
        before this side blocks in the rendezvous, and metas must exist first
        because the actor sizes its receive buffers from them."""
        if self.ctrl_q is None:
            return None
        host, port = self.full["sync"]["host"], int(self.full["sync"]["port"])
        self.ctrl_q.put(("init_nccl", host, port, weight_metas(self.model)))
        sender = WeightSender(self.model, host, port)
        print("NCCL weight sync established", flush=True)
        return sender

    # -------------------------------------------------------------- one group

    def accumulate(self, group: dict, batch: Batch) -> None:
        """Advantages for one group, then backward -- while the actor is still
        generating the next one."""
        batch.received += 1
        self.cursor = max(self.cursor, int(group.get("cursor", -1)) + 1)
        self.epoch = max(self.epoch, int(group.get("epoch", 0)))
        members, cause = prepare_group(group, self.cfg)

        if cause:
            self.skips[cause] += 1
            return

        if not members:
            return

        batch.usable += 1

        group_tokens = sum(
            len(turn["completion_ids"])
            for _, _, turns in members
            for _, turn in turns
        )     
          
        for advantage, reward, turns in members:
            batch.advantages.append(advantage)        
            batch.rewards.append(reward)   
            batch.lengths.append(sum(len(t["completion_ids"]) for _, t in turns))
            for index, turn in turns:
                turn["_index"] = index
                weight = batch.group_token_weight(group_tokens)
                loss = turn_loss(self.model, turn, advantage, weight, self.cfg, batch)
                loss.backward()
                batch.loss += float(loss.detach())

    # --------------------------------------------------------- batch boundary

    def optimize(self, batch: Batch) -> bool:
        """Rescale, clip, step, release the actor, log. False when max_steps hit."""
        self.steps += 1
        updated = batch.usable > 0

        lr = float(self.optimizer.param_groups[0]["lr"])
        grad_norm = self._apply(batch)
        if updated:
            self.version += 1
        self.optimizer.zero_grad(set_to_none=True)

        max_steps = self.cfg.get("max_steps")
        reached_max_steps = (
            max_steps is not None
            and self.steps >= int(max_steps)
        )
        save_resume = self.cfg.get("save_resume_checkpoint", True)
        export_final = self.cfg.get("export_final_model", True)
        export_interval = int(self.cfg.get("export_interval", 0))

        if reached_max_steps:
            # The model changed, but we intentionally do not synchronize/release the
            # actor because there will be no subsequent rollout batch.
            self._log(batch, lr, grad_norm)

            if save_resume:
                print(
                    f"saving terminal resume checkpoint at step {self.steps}...",
                    flush=True,
                )
                self._save(force=True)
                print("terminal resume checkpoint saved", flush=True)

            if export_final:
                print(
                    f"exporting final Hugging Face model at step {self.steps}...",
                    flush=True,
                )
                self._export_final()
                print("final Hugging Face model saved", flush=True)

            # run() will send "stop" in its finally block. Until then the actor
            # remains blocked at actor.synced.wait(), so it cannot start another batch.
            return False

        self._release(batch)
        self._log(batch, lr, grad_norm)
        if save_resume:
            self._save()
        if export_final and (export_interval > 0 and  self.steps % export_interval == 0):
            self._export_final()        
        return not reached_max_steps

    def _apply(self, batch: Batch) -> float:
        if not batch.usable:
            self.skips["empty_batch"] += 1
            return 0.0
        if batch.usable != batch.planned:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.mul_(batch.scale)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg["max_grad_norm"],
            error_if_nonfinite=True))
        self.optimizer.step()
        self.scheduler.step()
        return grad_norm

    def _release(self, batch: Batch) -> None:
        """The actor barriers on every batch, so it must be released on every
        batch -- including ones that produced no update."""
        if self.ctrl_q is None:
            return
        if batch.usable and self.sender is not None:
            self.ctrl_q.put(("sync",))      # actor drains, then joins the broadcast
            self.sender.send()              # blocks until the worker arrives
        else:
            self.ctrl_q.put(("resume",))

    def _log(self, batch: Batch, lr: float, grad_norm: float) -> None:
        n = max(1, batch.tokens)
        row = {
            "step": self.steps, "version": self.version, "loss": batch.loss,
            "lr": lr, "grad_norm": grad_norm, "tokens": batch.tokens,
            "groups": batch.usable, "trajectories": len(batch.advantages),
            "mean_abs_ratio_err": batch.abs_ratio_err / n,
            "signed_mean_dlogp": batch.signed_dlogp / n,
            "kl": batch.kl / n,
            "mean_kl": batch.kl / n,
            "beta_kl": self.cfg["kl_beta"] * (batch.kl / n),
            "advantage_mean": st.fmean(batch.advantages) if batch.advantages else None,
            "advantage_absmean": (st.fmean(map(abs, batch.advantages))
                                  if batch.advantages else None),
            "by_turn_ratio_err": {i: round(s / c, 4)
                                  for i, (s, c) in sorted(batch.by_turn.items()) if c},
            "max_abs_ratio_err": batch.max_ratio_err,
            "reward_mean": st.fmean(batch.rewards) if batch.rewards else None,
            "len_mean": st.fmean(batch.lengths) if batch.lengths else None,
            "len_max": max(batch.lengths, default=0),
            "entropy_est": batch.neg_logp_sum / n,                                  
            "skips": dict(self.skips),
            "cursor": self.cursor, 
            "epoch": self.epoch,
        }
        self.metrics.write(json.dumps(row) + "\n")
        self.metrics.flush()
        print(json.dumps(row), flush=True)

    def _restore(self) -> bool:
        """Load the current checkpoint, or the previous one if it is missing.
        Counters live inside the payload, so they cannot
        disagree with the weights they were saved beside."""
        d = self.out_dir / "resume"
        path = next((p for p in (d / "state.pt", d / "state.prev.pt")
                     if p.exists()), None)
        if path is None:
            return False
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.steps, self.version = state["steps"], state["version"]
        self.cursor = state["cursor"]
        self.epoch = state["epoch"]
        print(f"resumed from {path.name}: step {self.steps} version {self.version} "
              f"cursor {self.cursor} epoch {self.epoch}", flush=True)
        return True

    def _save(self, force: bool = False) -> None:
        """Save state for resuming it later."""
        every = self.cfg["checkpoint_every"]
        if not force and not (every and self.steps % every == 0):
            return
        d = self.out_dir / "resume"
        d.mkdir(parents=True, exist_ok=True)
        tmp, cur, prev = d / "state.tmp", d / "state.pt", d / "state.prev.pt"

        with tmp.open("wb") as handle:
            torch.save({"model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "steps": self.steps, "version": self.version,
                        "cursor": self.cursor, "epoch": self.epoch}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if cur.exists():
            cur.replace(prev)
        tmp.replace(cur)

        print(f"resume checkpoint committed: {cur}", flush=True)

    def _export_final(self) -> None:
        """Export inference-ready BF16 weights after reaching max_steps."""
        from transformers import AutoTokenizer

        path = self.out_dir / "final" / f"policy_step{self.steps}"

        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing final model: {path}"
            )

        # Training requires use_cache=False, but inference should use the KV cache.
        previous_use_cache = self.model.config.use_cache
        self.model.config.use_cache = True

        state = {
            name: parameter.detach().to("cpu", torch.bfloat16)
            for name, parameter in self.model.state_dict().items()
        }

        try:
            self.model.save_pretrained(
                path,
                state_dict=state,
                safe_serialization=True,
            )

            # Make the final directory independently loadable by vLLM/Hugging Face.
            tokenizer = AutoTokenizer.from_pretrained(
                self.full["model"]["tokenizer"]
            )
            tokenizer.save_pretrained(path)
            
        finally:
            self.model.config.use_cache = previous_use_cache
            del state

        print(f"exported final model: {path}", flush=True)

    # ------------------------------------------------------------------- loop

    def run(self, source: Iterator[dict]) -> int:
        batch = Batch(int(self.cfg["groups_per_step"]))
        print(f"trainer: batch={batch.planned} groups, "
              f"sync={'on' if self.sender else 'off'}, "
              f"resuming at step {self.steps} v{self.version}", flush=True)
        try:
            for group in source:
                self.accumulate(group, batch)
                if batch.full:
                    keep = self.optimize(batch)
                    batch = Batch(batch.planned)
                    if not keep:
                        break
            else:
                # Shutdown with a part-filled batch: step on what arrived
                # rather than discarding the backward work already done.
                if batch.received:
                    self.optimize(batch)
        finally:
            # Always, including on exception: otherwise the actor blocks forever
            # and the pod holds two idle A100s.
            if self.ctrl_q is not None:
                self.ctrl_q.put(("stop",))
            self.metrics.close()

        print(json.dumps({"status": "complete", "steps": self.steps,
                          "version": self.version, "skips": dict(self.skips)}))
        return 0

# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def queue_groups(traj_q: Any, timeout: float = 1800.0) -> Iterator[dict]:
    """Queue mode. The actor barriers each batch, so the gap between groups is one
    group's generation time -- minutes, not seconds."""
    while True:
        try:
            item = traj_q.get(timeout=timeout)
        except queue_mod.Empty:
            raise RuntimeError(f"no group for {timeout:.0f}s -- actor stalled?")
        if item is None:
            return
        yield item

def trainer_main(full: dict, traj_q: Any, ctrl_q: Any) -> int:
    """run.py's child. CUDA_VISIBLE_DEVICES is already GPU 1."""
    return Trainer(full, ctrl_q).run(queue_groups(traj_q))