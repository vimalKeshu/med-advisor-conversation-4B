"""V5 validation, native Qwen rendering, token cache, and batching."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def render(record, tokenizer, config):
    history = [{"role": "system", "content": config["system_message"]}]
    thinking = record["thinking"]
    require(type(thinking) is bool, f"{record['record_id']}: thinking must be boolean")
    conversation = record["conversation"]
    require(len(conversation) >= 2 and len(conversation) % 2 == 0, "Incomplete conversation")
    for index, message in enumerate(conversation):
        target_id = f"{record['record_id']}_t{index:03d}"
        role = "assistant" if index % 2 else "user"
        require(message["role"] == role and message["turn_index"] == index, target_id)
        require(isinstance(message["content"], str) and message["content"].strip(), target_id)
        if role == "assistant":
            mode = "thinking" if thinking else "non_thinking"
            reasoning = message["reasoning_content"]
            require(message["reasoning_mode"] == mode, f"{target_id}: mode mismatch")
            require(isinstance(reasoning, str), f"{target_id}: reasoning must be text")
            require(bool(reasoning.strip()) if thinking else reasoning == "", target_id)
            prompt = tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            current = {key: message[key] for key in ("role", "content", "reasoning_content")}
            complete = tokenizer.apply_chat_template(
                history + [current], tokenize=False, add_generation_prompt=False)
            suffix = "<|im_start|>assistant\n" + ("" if thinking else "<think>\n\n</think>\n\n")
            segment = complete.rsplit("<|im_start|>assistant\n", 1)[-1]
            require(prompt.endswith(suffix) and complete.startswith(prompt), f"{target_id}: template drift")
            require(segment.count("<think>") == segment.count("</think>") == 1, target_id)
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            ids = tokenizer.encode(complete, add_special_tokens=False)
            p = len(prefix)
            require(ids[:p] == prefix and 0 < p < len(ids), f"{target_id}: token prefix mismatch")
            require(len(ids) <= config["max_length"], f"{target_id}: {len(ids)} > max_length")
            q = ids.index(tokenizer.convert_tokens_to_ids("</think>"), p) + 1 if thinking else p
            yield {"ids": torch.tensor(ids, dtype=torch.int32), "prompt": p, "reasoning_end": q,
                   "thinking": thinking, "target_id": target_id, "domain": record["domain"],
                   "turn": index // 2 + 1}
        # Previous reasoning and all scenario/persona metadata stay out of model input.
        history.append({"role": role, "content": message["content"]})


def prepare(config, root, tokenizer, splits=("train", "validation")):
    spec = config["data"]
    require(splits and len(splits) == len(set(splits)) and set(splits) <= {"train", "validation"},
            "Invalid data split selection")
    cache = root / spec["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    datasets, reports, split_keys = {}, {}, {}
    for split in splits:
        expected = spec[split]
        path = root / expected["file"]
        require(digest(path) == expected["sha256"], f"{split}: dataset hash changed")
        with path.open() as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        require(len(records) == expected["records"], f"{split}: record count changed")
        keys, ids = set(), set()
        for record in records:
            require(record["schema_version"] == spec["schema_version"], "Unexpected data schema")
            require(record["training_split"] == split, f"{split}: wrong split label")
            require(not record.get("evaluation_only") and not record.get("training_use_forbidden"), "Protected data")
            rid = record["record_id"]
            require(rid not in ids, f"Duplicate record: {rid}")
            ids.add(rid)
            keys.update((key, record[key]) for key in ("record_id", "scenario_id", "split_group_id"))
            keys.update((key, value) for key in ("source_record_id", "seed_id")
                        if (value := record.get("provenance", {}).get(key)))
        split_keys[split] = keys
        signature = {"data": spec, "model": config["model"], "split": split,
                     "template": tokenizer.chat_template, "code": digest(__file__)}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
        cached = cache / f"{split}-{key}.pt"
        if cached.exists():
            targets = torch.load(cached, map_location="cpu", weights_only=True)
        else:
            targets = []
            for n, record in enumerate(records, 1):
                targets.extend(render(record, tokenizer, spec))
                if n % spec["progress_every_records"] == 0:
                    print(f"Tokenized {split}: {n}/{len(records)} conversations", flush=True)
            temporary = cached.with_suffix(".tmp")
            torch.save(targets, temporary)
            temporary.replace(cached)
        require(len(targets) == expected["targets"], f"{split}: target count changed")
        require(len({t['target_id'] for t in targets}) == len(targets), "Duplicate targets")
        supervised = sum(len(t["ids"]) - t["prompt"] for t in targets)
        require(supervised == expected["supervised_tokens"], f"{split}: supervised token count changed")
        reports[split] = {"records": len(records), "targets": len(targets), "supervised_tokens": supervised,
                          "max_length": max(len(t["ids"]) for t in targets), "sha256": expected["sha256"]}
        datasets[split] = targets
    if "train" in split_keys and "validation" in split_keys:
        require(not split_keys["train"] & split_keys["validation"], "Train/validation family or source overlap")
    return datasets, reports


def collate(targets, pad_token_id):
    width = max(len(t["ids"]) for t in targets)
    ids = torch.full((len(targets), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    kinds = torch.zeros_like(ids)
    for row, target in enumerate(targets):
        n, p, q = len(target["ids"]), target["prompt"], target["reasoning_end"]
        ids[row, :n], mask[row, :n] = target["ids"], 1
        # 1: serialized thinking span; 2: thinking answer; 3: non-thinking answer.
        if target["thinking"]:
            kinds[row, p:q], kinds[row, q:n] = 1, 2
        else:
            kinds[row, p:n] = 3
    return {"input_ids": ids, "attention_mask": mask, "kinds": kinds,
            "target_ids": [t["target_id"] for t in targets]}


class SampleOrder:
    """Rebuild the epoch order from the committed cursor, independent of worker prefetch."""
    def __init__(self, size, seed, epoch, start):
        self.size, self.seed, self.epoch, self.start = size, seed, epoch, start

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + (self.epoch or 0))
        order = torch.randperm(self.size, generator=generator).tolist() if self.epoch is not None else range(self.size)
        return iter(order[self.start:])

    def __len__(self):
        return self.size - self.start


def loader(targets, config, pad_token_id, epoch=None, start=0):
    from functools import partial
    workers = config["num_workers"]
    options = {"prefetch_factor": config["prefetch_factor"], "persistent_workers": config["persistent_workers"]} if workers else {}
    return DataLoader(targets, sampler=SampleOrder(len(targets), config["seed"], epoch, start),
                      batch_size=config["microbatch_size"],
                      collate_fn=partial(collate, pad_token_id=pad_token_id),
                      num_workers=workers, pin_memory=config["pin_memory"],
                      generator=torch.Generator().manual_seed(config["seed"]), **options)


def device_batches(batches, device, prefetch=False):
    """Yield (CPU metadata, device tensors), optionally copying one batch ahead."""
    def move(cpu):
        return {k: v.to(device, non_blocking=True) for k, v in cpu.items() if torch.is_tensor(v)}

    if not prefetch or device.type != "cuda":
        for cpu in batches:
            yield cpu, move(cpu)
        return
    stream = torch.cuda.Stream(device=device)
    batches = iter(batches)

    def stage():
        cpu = next(batches, None)
        if cpu is not None:
            with torch.cuda.stream(stream):
                batch = move(cpu)
                ready = stream.record_event()
            return cpu, batch, ready

    pending = None
    try:
        pending = stage()
        while pending is not None:
            cpu, batch, ready = pending
            compute = torch.cuda.current_stream(device)
            compute.wait_event(ready)
            for tensor in batch.values():
                tensor.record_stream(compute)
            pending = stage()
            yield cpu, batch
            ready.synchronize()  # Retain the current CPU buffers until their copy has completed.
    finally:
        stream.synchronize()  # Also protect a pending copy when iteration stops early.
