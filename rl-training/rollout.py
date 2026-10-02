import asyncio
import os
import statistics
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any

from common import (
    JsonlWriter,
    Renderer,
    Scenario,
    completion_logprobs,
    load_scenarios,
    split_think,
)
from reward import Judge

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# reference model -- built only when kl_beta > 0
# ---------------------------------------------------------------------------

class Reference:

    def __init__(self, cfg: dict) -> None:
        import torch
        from concurrent.futures import ThreadPoolExecutor
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["reference"] or cfg["model"]["policy"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda").eval()
        self.model.config.use_cache = False
        self.temperature = cfg["train"]["logprob_temperature"]
        self.chunk = cfg["train"]["logit_chunk"]
        self.pool = ThreadPoolExecutor(max_workers=1)

    def _score(self, trajectories: list[dict]) -> None:
        for trajectory in trajectories:
            for turn in trajectory["turns"]:
                if not turn["completion_ids"]:
                    continue

                ids = turn["prompt_ids"] + turn["completion_ids"]

                logp = completion_logprobs(
                    self.model,
                    ids,
                    len(turn["prompt_ids"]),
                    self.temperature,
                    chunk=self.chunk,
                )

                turn["logp_ref"] = logp.tolist()

    async def score(self, trajectories: list[dict]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.pool, self._score, trajectories)

# ---------------------------------------------------------------------------
# simulator
# ---------------------------------------------------------------------------

class Persona:
    """One conversation's simulated user.

    Holds its own history, so the role inversion is explicit: the advisor's answer
    arrives as `user`, the persona's own line is stored as `assistant` -- including
    the opening, or it would be replying to an answer to a question it has no
    memory of asking.
    """

    def __init__(self, client: Any, cfg: dict, system: str, opening: str) -> None:
        self.client, self.cfg = client, cfg
        self.messages = [
            {"role": "system", "content": system},
            {"role": "assistant", "content": opening},
        ]

    async def reply(self, advisor_answer: str) -> str:
        self.messages.append({"role": "user", "content": advisor_answer})
        response = await self.client.chat.completions.create(
            model=self.cfg["model"],
            messages=self.messages,
            temperature=self.cfg["temperature"],
            max_tokens=self.cfg["max_tokens"],
        )
        text = (response.choices[0].message.content or "").strip()
        self.messages.append({"role": "assistant", "content": text})
        return text

# ---------------------------------------------------------------------------
# actor -- the engine, the control protocol, and everything a group needs
# ---------------------------------------------------------------------------

class Actor:
    """The rollout process's world.

    policy_version lives here because this is the process that applies weights --
    it knows a swap happened because it just performed one. The trainer counts the
    syncs it initiated; the two agree without exchanging the number, because every
    sync is a blocking NCCL rendezvous.
    """

    def __init__(self, cfg: dict) -> None:
        from openai import AsyncOpenAI
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from vllm.v1.engine.async_llm import AsyncLLM

        self.cfg, self.r, v = cfg, cfg["rollout"], cfg["vllm"]
        self.run_id = time.strftime("%Y%m%dT%H%M%S")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer"])
        self.renderer = Renderer(self.tokenizer, cfg["system_message"])
        self.scenarios = load_scenarios(cfg, HERE)
        self.judge = Judge(cfg["judge"], cfg.get("reward"))

        self.sim = cfg["simulator"]
        self.client = AsyncOpenAI(
            base_url=self.sim["base_url"],
            api_key=os.environ[self.sim["api_key_env"]],
            timeout=float(self.sim["timeout"]),
            max_retries=int(self.sim["max_retries"]),
        )

        # Reference BEFORE the engine: vLLM profiles free memory at init and sizes
        # its KV budget from it. Loading 8 GB afterwards is how you OOM at turn 40.
        self.reference = Reference(cfg) if cfg["train"]["kl_beta"] > 0 else None

        self._sampling_params = SamplingParams
        self._output_kind = RequestOutputKind.FINAL_ONLY
        self.stop_ids = [self.tokenizer.convert_tokens_to_ids("<|im_end|>")]
        self.sampling = {
            **{k: self.r[k] for k in ("temperature", "top_p", "top_k", "min_p",
                                      "presence_penalty", "max_new_tokens")},
            "logprobs_mode": v["logprobs_mode"],   # what logp_old actually means
            "model": cfg["model"]["policy"],
        }
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=cfg["model"]["policy"],
            tokenizer=cfg["model"]["tokenizer"],
            dtype=v["dtype"],
            max_model_len=v["max_model_len"],
            gpu_memory_utilization=v["gpu_memory_utilization"],
            enforce_eager=v["enforce_eager"],
            enable_prefix_caching=v["enable_prefix_caching"],
            logprobs_mode=v["logprobs_mode"],
            worker_extension_cls="wsync.WeightSync",
            tensor_parallel_size=1,
            trust_remote_code=False,
        ))

        # protocol state
        self.policy_version = 0
        self.cursor = 0
        self.inflight = 0
        self._metas: list | None = None
        self.gate = asyncio.Event()
        self.gate.set()
        self.synced = asyncio.Event()     # released when a batch may generate again
        self.started = asyncio.Event()    # released once the trainer sends the resume point

    # ------------------------------------------------------------- generation

    def persona(self, scenario: Scenario) -> Persona:
        return Persona(self.client, self.sim, scenario.sim_prompt(), scenario.opening)

    def seed_for(self, epoch: int, k: int) -> int:
        """Either counter advancing must re-roll exploration: policy_version when
        weights are swapped, epoch when the scenario file is replayed."""
        return int(self.r["seed0"]) + (self.policy_version + epoch) * 10_000 + k

    async def generate(self, prompt_ids: list[int], seed: int, request_id: str) -> dict:
        from vllm.inputs import TokensPrompt

        await self.gate.wait()            # open except during a weight swap
        self.inflight += 1
        try:
            params = self._sampling_params(
                temperature=self.r["temperature"],
                top_p=self.r["top_p"],
                top_k=self.r["top_k"],
                min_p=self.r["min_p"],
                presence_penalty=self.r["presence_penalty"],
                max_tokens=self.r["max_new_tokens"],
                seed=seed,
                logprobs=0,          # the sampled token only -- pi_old, never recomputed
                stop_token_ids=self.stop_ids,
                output_kind=self._output_kind,
            )
            final = None
            async for out in self.engine.generate(
                TokensPrompt(prompt_token_ids=prompt_ids), params, request_id
            ):
                final = out
        finally:
            self.inflight -= 1

        completion = final.outputs[0]
        ids, logprobs = list(completion.token_ids), completion.logprobs
        # zip would silently truncate to the shorter of the two and misalign every
        # ratio from that point onward.
        if logprobs is None or len(logprobs) != len(ids):
            raise RuntimeError(
                f"logprob mismatch: {len(ids)} tokens, "
                f"{None if logprobs is None else len(logprobs)} logprobs"
            )
        return {
            "completion_ids": ids,
            "logp_old": [lp[t].logprob for t, lp in zip(ids, logprobs)],
            "finish_reason": completion.finish_reason,
        }

    # --------------------------------------------------------------- protocol

    async def handle(self, message: tuple) -> bool:
        """One control message from the trainer. False on stop."""
        kind = message[0]

        if kind == "stop":
            return False

        if kind == "start":
            # Sent last, so this also means "startup is complete": NCCL is up, and
            # on a resume the trained weights are already in the engine.
            _, self.cursor, self.policy_version = message
            self.started.set()

        elif kind == "init_nccl":
            _, host, port, self._metas = message
            await self.engine.collective_rpc("init_weight_sync", args=(host, port, 1, 2))

        elif kind == "resume":
            # No update this batch (all groups skipped, or enabled: false). Release
            # the barrier without a weight transfer.
            self.synced.set()

        elif kind == "sync":
            self.gate.clear()               # stop issuing new generations
            while self.inflight:            # let running decodes finish: swapping
                await asyncio.sleep(0.05)   # mid-decode extends old KV with new weights
            await self.engine.collective_rpc("recv_weights", args=(self._metas,))
            # Not optional: vLLM hashes the token prefix, not the weights, so blocks
            # cached under the previous policy would be reused silently.
            await self.engine.reset_prefix_cache()
            self.policy_version += 1
            self.gate.set()
            self.synced.set()

        return True

    async def control(self, ctrl_q: Any, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        while True:
            message = await loop.run_in_executor(None, ctrl_q.get)
            if not await self.handle(message):
                break
        stop.set()
        # Release both barriers: a shutdown may arrive while the loop waits for a
        # sync, or before the very first start, and neither will ever come now.
        self.synced.set()
        self.started.set()

# ---------------------------------------------------------------------------
# one trajectory
# ---------------------------------------------------------------------------

async def converse(actor: Actor, scenario: Scenario, persona: Persona,
                   trajectory_id: str, seed: int) -> dict:
    """One conversation to completion.

    `chat` is the visible transcript: what the next prompt is rendered from and
    what the judge grades. `turns` is the training view -- one entry per assistant
    turn, carrying the exact ids and the sampler's logprobs.
    """
    thinking = scenario.thinking
    chat: list[dict[str, str]] = [{"role": "user", "content": scenario.opening}]
    turns: list[dict] = []
    started = time.monotonic()
    end_reason = "max_turns"

    for turn_index in range(scenario.max_turns):
        prompt_ids = actor.renderer.prompt_ids(chat, thinking)
        sampled = await actor.generate(
            prompt_ids, seed, f"{trajectory_id}_t{turn_index}")
        text = actor.tokenizer.decode(
            sampled["completion_ids"], skip_special_tokens=True)
        answer, reasoning = split_think(text)
        turns.append({
            "prompt_ids": prompt_ids,
            **sampled,
            "policy_version": actor.policy_version,  # a sync can land mid-conversation
            "logp_ref": None,                        # filled by Reference when kl_beta > 0
            "reasoning": reasoning,
        })

        if sampled["finish_reason"] == "length":
            end_reason = "advisor_truncated"         # from vLLM, not inferred
            break
        if not answer:
            end_reason = "advisor_no_answer"
            break
        chat.append({"role": "assistant", "content": answer})

        if turn_index == scenario.max_turns - 1:
            break                                    # do not ask what we will not answer

        reply = await persona.reply(answer)
        if not reply:
            end_reason = "persona_no_response"
            break
        if "[[DONE]]" in reply:
            end_reason = "persona_satisfied"
            break
        chat.append({"role": "user", "content": reply})

    return {
        "trajectory_id": trajectory_id,
        "seed": seed,
        "end_reason": end_reason,
        "advisor_turns": len(turns),
        "completion_tokens": sum(len(t["completion_ids"]) for t in turns),
        "seconds": round(time.monotonic() - started, 2),
        "chat": chat,
        "turns": turns,
    }

# ---------------------------------------------------------------------------
# one group
# ---------------------------------------------------------------------------

async def run_group(actor: Actor, scenario: Scenario, epoch: int, cursor: int) -> dict:
    """K trajectories for one scenario, then one judge call each.

    All K share identical sampling; only the seed differs. That is what makes them
    comparable rather than a mixed-settings sweep.
    """
    group_id = f"{actor.run_id}_e{epoch}_{scenario.id}"
    k_range = range(int(actor.r["group_size"]))
    tag = f"e{epoch} #{cursor}"
    began = time.monotonic()
    print(f"  [start ] {tag} v{actor.policy_version} K={len(k_range)} "
          f"thinking={scenario.thinking} {scenario.id[:44]}", flush=True)

    results = await asyncio.gather(
        *(
            converse(actor, scenario, actor.persona(scenario),
                     f"{group_id}_k{k}", actor.seed_for(epoch, k))
            for k in k_range
        ),
        return_exceptions=True,
    )
    trajectories = [
        result
        if not isinstance(result, BaseException)
        else {
            "trajectory_id": f"{group_id}_k{k}",
            "seed": actor.seed_for(epoch, k),
            "end_reason": f"error: {type(result).__name__}: {result}",
            "advisor_turns": 0,
            "completion_tokens": 0,
            "seconds": 0.0,
            "chat": [],
            "turns": [],
        }
        for k, result in zip(k_range, results)
    ]

    print(f"  [talked] {tag} {time.monotonic() - began:6.1f}s "
          f"turns={sum(t['advisor_turns'] for t in trajectories)} "
          f"tokens={sum(t['completion_tokens'] for t in trajectories)} "
          f"{dict(Counter(t['end_reason'].split(':')[0] for t in trajectories))}",
          flush=True)

    # KL scoring overlaps judging for free: one is GPU-bound on card 0, the other
    # is a network call.
    scoring = (asyncio.create_task(actor.reference.score(trajectories))
               if actor.reference else None)
    graded = await asyncio.gather(*(actor.judge(scenario, t) for t in trajectories))
    if scoring:
        await scoring
    for trajectory, (judgment, reward) in zip(trajectories, graded):
        trajectory["judge"], trajectory["reward"] = judgment, reward
        if not reward["valid"]:
            print(
                f"[judge-invalid] {trajectory['trajectory_id']} "
                f"{judgment.get('evaluation_status')}: "
                f"{reward.get('exclusion_reason')}",
                flush=True,
            )        

    rewards = [t["reward"]["trajectory_reward"] for t in trajectories
               if t["reward"]["valid"]]
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    usable = len(rewards) > 1 and std > 1e-6
    summary = (f"mean={statistics.fmean(rewards):.3f} std={std:.3f} "
               f"{'USABLE' if usable else 'SKIP'}") if rewards else "SKIP"
    print(f"  [judged] {tag} {time.monotonic() - began:6.1f}s "
          f"eligible={len(rewards)}/{len(k_range)} {summary}", flush=True)

    versions = [t["policy_version"] for tr in trajectories for t in tr["turns"]]
    return {
        "run_id": actor.run_id,
        "group_id": group_id,
        "scenario_id": scenario.id,
        "domain": scenario.domain,
        "thinking": scenario.thinking,
        "sampling": actor.sampling,
        # min, not the version at group start: the oldest policy that contributed
        # is what the trainer has to reason about.
        "policy_version": min(versions) if versions else actor.policy_version,
        "cursor": cursor,
        "epoch": epoch,
        "trajectories": trajectories,
    }

# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

async def actor_run(cfg: dict, traj_q: Any, ctrl_q: Any) -> int:
    actor = Actor(cfg)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    work: asyncio.Queue = asyncio.Queue()
    writer = JsonlWriter(HERE / cfg["run"]["out_dir"] / f"groups_{actor.run_id}.jsonl")

    async def worker() -> None:
        while True:
            cursor, epoch, scenario = await work.get()
            try:
                if stop.is_set():
                    continue
                group = await run_group(actor, scenario, epoch, cursor)
                writer.write(group)                 # durable, before the handoff
                # Blocking put: at most groups_per_step are ever outstanding, so it
                # cannot fill. Through an executor because it would otherwise freeze
                # the event loop.
                waited = time.monotonic()
                await loop.run_in_executor(None, traj_q.put, group)
                print(f"  [posted] e{epoch} #{cursor} v{group['policy_version']} "
                      f"queue_wait={time.monotonic() - waited:.1f}s", flush=True)
            except Exception as error:              # noqa: BLE001 -- one bad group
                print(f"  group #{cursor} failed: "
                      f"{type(error).__name__}: {error}", flush=True)
            finally:
                work.task_done()

    tasks = [asyncio.create_task(worker())
             for _ in range(int(cfg["rollout"]["concurrent_groups"]))]
    tasks.append(asyncio.create_task(actor.control(ctrl_q, stop)))

    def items(start: int):
        """(cursor, epoch, scenario) from a fixed infinite sequence.

        The shuffle is seeded, so cursor -> scenario is stable across processes and
        restarts. Resuming means starting the same sequence at the cursor the
        trainer last checkpointed.
        """
        n = len(actor.scenarios)
        cursor = start
        while True:
            yield cursor, cursor // n, actor.scenarios[cursor % n]
            cursor += 1

    await actor.started.wait()                      # resume point from the trainer
    stream = items(actor.cursor)
    batch_size = int(cfg["train"]["groups_per_step"])
    print(f"actor {actor.run_id}: {len(actor.scenarios)} scenarios, "
          f"K={cfg['rollout']['group_size']}, batch={batch_size}, "
          f"from cursor {actor.cursor} at v{actor.policy_version}", flush=True)

    try:
        while not stop.is_set():
            # Arm BEFORE producing: the trainer can finish the batch and sync while
            # we are still inside work.join(), and clearing afterwards would discard
            # that signal and deadlock.
            actor.synced.clear()
            for item in islice(stream, batch_size):
                work.put_nowait(item)
            await work.join()
            await actor.synced.wait()               # next batch runs on the new policy
    finally:
        for task in tasks:
            task.cancel()
        writer.close()
    return 0


def actor_main(full: dict, traj_q: Any, ctrl_q: Any) -> int:
    """run.py's actor child. CUDA_VISIBLE_DEVICES is already GPU 0."""
    return asyncio.run(actor_run(full, traj_q, ctrl_q))
