"""Validate an exported SFT model. Run: python -u validation.py [config.yaml] MODEL_DIR."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import loader, prepare, require
from trainer import evaluate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("model", help="Model directory; relative paths resolve beside the config")
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    root, mc, opt, run = cfg_path.parent, cfg["model"], cfg["training"], cfg["run"]
    directory = root / run["directory"]
    model_path = (root / args.model).resolve()
    output, batch_size = directory / "validation.json", run["validation_batch_size"]
    require(model_path.is_dir(), f"No exported model at {model_path}")
    require(type(batch_size) is int and batch_size > 0, "run.validation_batch_size must be positive")

    device = torch.device(run["device"])
    if device.type == "cuda":
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "A BF16-capable CUDA GPU is required")
        torch.cuda.set_device(device)
    torch.manual_seed(opt["seed"])
    torch.backends.cuda.matmul.allow_tf32 = opt["tf32"]
    torch.backends.cudnn.allow_tf32 = opt["tf32"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=mc["trust_remote_code"],
                                              local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    datasets, audit = prepare(cfg, root, tokenizer, ("validation",))
    print(json.dumps({"data_audit": audit["validation"]}), flush=True)
    batches = loader(datasets["validation"], dict(opt, microbatch_size=batch_size), tokenizer.pad_token_id)
    del datasets

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, mc["export_dtype"]),
        attn_implementation=mc["attention"], trust_remote_code=mc["trust_remote_code"],
        local_files_only=True).to(device)
    require(model.config.model_type == "qwen3", "This validator targets Qwen3")
    model.config.use_cache = False
    started = time.monotonic()
    result = evaluate(model, batches, cfg, device)
    report = {"model": str(model_path.resolve()),
              "weights_dtype": mc["export_dtype"],
              "compute_dtype": mc["compute_dtype"], "data": audit["validation"], "batch_size": batch_size,
              "validation": result, "seconds": time.monotonic() - started,
              "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps({"event": "validation", **report}, allow_nan=False), flush=True)
    print(f"Complete. Report: {output}", flush=True)


if __name__ == "__main__":
    main()
