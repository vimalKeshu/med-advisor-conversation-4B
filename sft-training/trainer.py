"""Single-GPU full-parameter SFT. Run: python trainer.py config.yaml."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import time
from itertools import islice
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
import yaml
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import device_batches, digest, loader, prepare, require


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(identity):
    return hashlib.sha256(json.dumps({k: identity[k] for k in ("config", "code")}, sort_keys=True).encode()).hexdigest()


def loss_sums(model, batch, chunk_size):
    """Exactly the causal completion loss; only project supervised positions into the vocabulary."""
    hidden = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                         use_cache=False).last_hidden_state[:, :-1]
    kinds = batch["kinds"][:, 1:]
    selected = kinds > 0
    hidden, labels, kinds = hidden[selected], batch["input_ids"][:, 1:][selected], kinds[selected]

    def chunk(h, y, group):
        nll = F.cross_entropy(model.lm_head(h).float(), y, reduction="none")
        return nll.new_zeros(3).scatter_add(0, group - 1, nll)

    sums = hidden.new_zeros(3, dtype=torch.float32)
    for start in range(0, len(labels), chunk_size):
        args = hidden[start:start + chunk_size], labels[start:start + chunk_size], kinds[start:start + chunk_size]
        # Recompute each small vocabulary projection in backward instead of retaining all logits.
        sums = sums + (checkpoint(chunk, *args, use_reentrant=False) if torch.is_grad_enabled() else chunk(*args))
    return sums, torch.bincount(kinds, minlength=4)[1:]


def metrics(sums, counts):
    groups = {"overall": (0, 1, 2), "thinking": (0, 1), "non_thinking": (2,),
              "reasoning": (0,), "answer": (1, 2)}
    result = {}
    for name, indices in groups.items():
        n = sum(counts[i] for i in indices)
        result[name] = {"loss": sum(sums[i] for i in indices) / n if n else None, "tokens": n}
    return result


def evaluate(model, batches, cfg, device):
    model.eval()
    sums = torch.zeros(3, dtype=torch.float64, device=device)
    counts = torch.zeros(3, dtype=torch.long, device=device)
    started = reported = time.monotonic()
    completed, total = 0, len(batches.dataset)
    print(f"Validation starting: {total} targets, batch size {batches.batch_size}; waiting for the first batch.", flush=True)
    with torch.no_grad():
        for cpu, batch in device_batches(batches, device, cfg["training"]["cuda_prefetch"]):
            if completed == 0:
                print(f"Validation first batch loaded: {batch['input_ids'].shape[1]} tokens; running forward.", flush=True)
            with torch.autocast(device.type, dtype=getattr(torch, cfg["model"]["compute_dtype"]),
                                enabled=cfg["model"]["compute_dtype"] != "float32"):
                losses, tokens = loss_sums(model, batch, cfg["training"]["loss_chunk_tokens"])
            sums += losses.double()
            counts += tokens
            first = completed == 0
            completed += len(cpu["target_ids"])
            if first or completed == total or time.monotonic() - reported >= cfg["run"]["progress_every_seconds"]:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                reported = time.monotonic()
                elapsed = reported - started
                print(json.dumps({"event": "validation_progress", "completed": completed, "total": total,
                                  "seconds": round(elapsed, 1),
                                  "eta_seconds": round(elapsed * (total - completed) / completed, 1)}), flush=True)
    require(bool(torch.isfinite(sums).all()), "Non-finite validation loss")
    model.train()
    return metrics(sums.tolist(), counts.tolist())


def export(model, tokenizer, directory, step, cfg):
    destination = directory / f"step-{step:06d}"
    if destination.exists():
        destination = directory / f"step-{step:06d}-{time.time_ns()}"
    temporary = destination.with_suffix(".tmp")
    temporary.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, cfg["model"]["export_dtype"])
    # Never cast live FP32 training parameters down and then back up.
    weights, converted = {}, {}
    for key, value in model.state_dict().items():
        identity = (value.data_ptr(), tuple(value.shape))
        if identity not in converted:
            converted[identity] = value.detach().to(device="cpu", dtype=dtype)
        weights[key] = converted[identity]  # Preserve Qwen's tied embedding/output weights.
    model.save_pretrained(temporary, state_dict=weights, safe_serialization=True,
                          max_shard_size=cfg["run"]["max_shard_size"])
    tokenizer.save_pretrained(temporary)
    saved = json.loads((temporary / "config.json").read_text())
    saved["torch_dtype"] = cfg["model"]["export_dtype"]
    saved["use_cache"] = True
    if "dtype" in saved:
        saved["dtype"] = cfg["model"]["export_dtype"]
    write_json(temporary / "config.json", saved)
    temporary.rename(destination)
    return str(destination.resolve())


def save_resume(path, model, optimizer, state):
    state = dict(state, model=model.state_dict(), optimizer=optimizer.state_dict(),
                 torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)  # The previous complete checkpoint survives until this succeeds.


def learning_rate(step, total, cfg):
    warmup = max(1, math.ceil(total * cfg["warmup_ratio"]))
    scale = (step + 1) / warmup if step < warmup else (
        0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))
    return cfg["learning_rate"] * scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(Path(__file__).with_name("config.yaml")))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    opt, run, mc = cfg["training"], cfg["run"], cfg["model"]
    resuming = run["resume"]
    validation_enabled = run.get("validation_enabled", False)
    require(type(validation_enabled) is bool, "run.validation_enabled must be true or false")
    for key in ("epochs", "microbatch_size", "accumulation_steps", "loss_chunk_tokens"):
        require(type(opt[key]) is int and opt[key] > 0, f"Invalid training.{key}")
    for key in ("evaluate_every_steps", "checkpoint_every_steps"):
        require(type(run[key]) is int and run[key] > 0, f"Invalid run.{key}")
    validation_batch_size = run.get("validation_batch_size", opt["microbatch_size"])
    require(type(validation_batch_size) is int and validation_batch_size > 0, "Invalid validation batch size")
    require(type(run["first_checkpoint_step"]) is int and run["first_checkpoint_step"] >= 0, "Invalid first checkpoint step")
    require(run["progress_every_seconds"] > 0, "Invalid progress interval")
    require(type(opt["num_workers"]) is int and opt["num_workers"] >= 0, "Invalid worker count")
    require(type(opt["prefetch_factor"]) is int and opt["prefetch_factor"] > 0, "Invalid prefetch factor")
    require(not opt["cuda_prefetch"] or opt["pin_memory"], "CUDA prefetch requires pinned CPU memory")
    require(0 <= opt["warmup_ratio"] < 1 and opt["learning_rate"] > 0, "Invalid learning rate/schedule")
    require(mc["parameter_dtype"] == "float32", "Full-parameter training requires FP32 master weights here")
    root = config_path.parent
    out = root / run["directory"]
    resume_path = out / "resume.pt"
    if resuming:
        require(resume_path.exists(), f"No checkpoint at {resume_path}. Restore the saved run folder here, "
                "or use run.resume: false with an empty run directory to start new training.")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not resuming:
        require(not any(p.name != ".lock" for p in out.iterdir()), "Run directory is not empty; choose a new name or resume")
    device = torch.device(run["device"])
    if device.type == "cuda":
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "A BF16-capable CUDA GPU is required")
        torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(mc["repository"], revision=mc["tokenizer_revision"],
                                              trust_remote_code=mc["trust_remote_code"], local_files_only=mc["local_files_only"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    datasets, audit = prepare(cfg, root, tokenizer)
    print(json.dumps({"data_audit": audit}), flush=True)
    signature_config = dict(cfg, run={k: v for k, v in run.items() if k not in ("resume", "allow_performance_tuning")})
    identity = {"config": signature_config, "code": {name: digest(Path(__file__).with_name(name)) for name in ("trainer.py", "data.py")}}
    signature = fingerprint(identity)
    packages = ("torch", "transformers", "safetensors", "PyYAML", "tokenizers", "huggingface-hub")
    if mc["attention"] == "flash_attention_2":
        packages += ("flash-attn",)
    versions = {name: importlib.metadata.version(name) for name in packages}
    if resuming:
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        original = json.loads((out / "run.json").read_text())
        if saved["signature"] != original["signature"]:
            original = json.loads((out / f"run-{saved['signature']}.json").read_text())
        require(original["signature"] == saved["signature"] == fingerprint(original), "Checkpoint metadata does not match")
        print(f"Loading checkpoint at step {saved['step']}, epoch {saved['epoch'] + 1}, position {saved['position']}.", flush=True)

    torch.manual_seed(opt["seed"])
    torch.backends.cuda.matmul.allow_tf32 = opt["tf32"]
    torch.backends.cudnn.allow_tf32 = opt["tf32"]
    model = AutoModelForCausalLM.from_pretrained(mc["repository"], revision=mc["revision"],
        dtype=getattr(torch, mc["parameter_dtype"]), attn_implementation=mc["attention"],
        trust_remote_code=mc["trust_remote_code"], local_files_only=mc["local_files_only"]).to(device)
    require(model.config.model_type == "qwen3", "This compact trainer targets Qwen3")
    model.config.use_cache = False
    if opt["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["learning_rate"], betas=tuple(opt["betas"]),
                                  eps=opt["epsilon"], weight_decay=opt["weight_decay"], fused=device.type == "cuda")
    total = math.ceil(math.ceil(len(datasets["train"]) / opt["microbatch_size"]) / opt["accumulation_steps"]) * opt["epochs"]
    state = {"signature": signature, "step": 0, "epoch": 0, "position": 0, "best": None, "baseline": None}
    if resuming:
        model.load_state_dict(saved.pop("model"))
        optimizer.load_state_dict(saved.pop("optimizer"))
        torch.set_rng_state(saved.pop("torch_rng"))
        cuda_rng = saved.pop("cuda_rng")
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda_rng)
        state = dict(saved, signature=signature)
    manifest = out / (f"run-{signature}.json" if resuming else "run.json")
    if not manifest.exists():
        write_json(manifest, dict(identity, signature=signature, data=audit, versions=versions,
                                 cuda=torch.version.cuda, device=str(device)))
    log_path = out / "metrics.jsonl"
    if resuming and log_path.exists():
        # Discard events after the saved position; they will be reproduced after resume.
        lines = []
        raw = log_path.read_text().splitlines()
        for index, line in enumerate(raw):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                require(index == len(raw) - 1, "Corrupted metrics before the final line")
                break  # A killed process may leave its last log entry incomplete.
            if event["step"] <= state["step"]:
                lines.append(line)
        log_path.write_text("\n".join(lines) + "\n")

    def log(event):
        event = dict(event, step=state["step"])
        with log_path.open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps(event, allow_nan=False), flush=True)

    if resuming:
        log({"event": "resume", "position": state["position"], "previous_signature": original["signature"],
             "signature": signature, "loss_chunk_tokens": opt["loss_chunk_tokens"], "prefetch_factor": opt["prefetch_factor"],
             "validation_batch_size": validation_batch_size, "validation_enabled": validation_enabled})
    validation_loader = (loader(datasets["validation"], dict(opt, microbatch_size=validation_batch_size), tokenizer.pad_token_id)
                         if validation_enabled else None)
    training_loader = loader(datasets["train"], opt, tokenizer.pad_token_id)
    if validation_enabled and state["step"] == 0 and run["evaluate_at_start"]:
        state["baseline"] = evaluate(model, validation_loader, cfg, device)
        log({"event": "baseline", "validation": state["baseline"]})
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(state["epoch"], opt["epochs"]):
        start = state["position"] if epoch == state["epoch"] else 0
        training_loader.sampler.epoch, training_loader.sampler.start = epoch, start
        started = time.monotonic()
        batches = iter(training_loader)
        while True:
            group = list(islice(batches, opt["accumulation_steps"]))
            if not group:
                break
            data_wait = time.monotonic() - started
            tokens = sum(int((b["kinds"][:, 1:] > 0).sum()) for b in group)
            sums = torch.zeros(3, device=device)
            counts = torch.zeros(3, dtype=torch.long, device=device)
            for cpu, batch in device_batches(group, device, opt["cuda_prefetch"]):
                with torch.autocast(device.type, dtype=getattr(torch, mc["compute_dtype"]), enabled=mc["compute_dtype"] != "float32"):
                    losses, n = loss_sums(model, batch, opt["loss_chunk_tokens"])
                require(bool(torch.isfinite(losses).all()), f"Non-finite loss: {cpu['target_ids']}")
                (losses.sum() / tokens).backward()
                sums += losses.detach()
                counts += n
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), opt["max_grad_norm"], error_if_nonfinite=True))
            lr = learning_rate(state["step"], total, opt)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1
            state["epoch"] = epoch
            state["position"] += sum(len(b["target_ids"]) for b in group)
            train_metrics = metrics(sums.tolist(), counts.tolist())
            elapsed = time.monotonic() - started
            log({"event": "train", "epoch": epoch + 1, "position": state["position"], "lr": lr,
                 "grad_norm": norm, "train": train_metrics, "seconds": elapsed, "data_wait_seconds": data_wait,
                 "tokens_per_second": tokens / elapsed,
                 "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                 "target_ids": [tid for b in group for tid in b["target_ids"]] if norm > opt["log_outlier_grad_norm"] else []})
            final = state["step"] == total
            evaluated = validation_enabled and (final or state["step"] % run["evaluate_every_steps"] == 0)
            if evaluated:
                result = evaluate(model, validation_loader, cfg, device)
                improved = state["best"] is None or result["overall"]["loss"] < state["best"]["loss"]
                path = export(model, tokenizer, out, state["step"], cfg) if improved or final or run["export_each_evaluation"] else None
                if improved:
                    state["best"] = {"step": state["step"], "loss": result["overall"]["loss"], "model": path}
                log({"event": "validation", "validation": result, "model": path, "best_validation": state["best"]})
            elif final:
                path = export(model, tokenizer, out, state["step"], cfg)
                log({"event": "export", "model": path})
            if final:
                state["final_model"] = path
            early_save = run["first_checkpoint_step"] > 0 and state["step"] == run["first_checkpoint_step"]
            if final or evaluated or early_save or state["step"] % run["checkpoint_every_steps"] == 0:
                checkpoint_started = time.monotonic()
                save_resume(resume_path, model, optimizer, state)
                log({"event": "checkpoint", "seconds": time.monotonic() - checkpoint_started})
            started = time.monotonic()
        state["epoch"], state["position"] = epoch + 1, 0
    write_json(out / "training_report.json", {"status": "complete", "steps": state["step"], "data": audit,
        "validation_enabled": validation_enabled, "final_model": state.get("final_model"),
        "baseline_validation": state["baseline"], "best_validation": state["best"],
        "note": ("Lowest validation loss is a candidate for generated-answer evaluation, not an accuracy or RL promotion decision."
                 if validation_enabled else "Validation disabled for this continuation; any restored validation metrics are historical.")})
    print(f"Complete. Report: {out / 'training_report.json'}", flush=True)


if __name__ == "__main__":
    main()

