# SFT

Supervised fine-tuning (SFT) of the **Qwen3-4B** model on a single 80 GB A100 GPU.

- `trainer.py` — training, checkpoints, and model export
- `validation.py` — standalone evaluation of exported models
- `data.py` — dataset validation, tokenization/cache, and batching
- `config.yaml` — run configuration; relative paths resolve from this file

## Training configuration

| Setting | Default |
| --- | --- |
| Model | Qwen3-4B (original hybrid) |
| Epochs / optimizer updates | 1 / 1,585 |
| Peak learning rate | `1e-5` |
| Microbatch / gradient accumulation | 1 / 32 |
| Precision | FP32 weights; BF16 compute and export |
| Maximum sequence length | 10,240 tokens (no silent truncation) |
| Loss chunk size | 1,024 tokens |
| Data loader | 2 workers; prefetch factor 32 |
| CUDA prefetch | Disabled |
| In-training validation | Disabled |
| Resume checkpoints | Every 400 updates and at completion |

The loss is computed at supervised token positions, with token-weighted gradient accumulation. Sequences are dynamically sized rather than padded to the maximum length.

## Run training

Use Linux with CUDA, Python 3.11 or 3.12, and an 80 GB A100. Budget **approximately 130 GB of free output space** for checkpoints and exports; additional exports require more.

```bash
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
python -u trainer.py config.yaml
```

For a new run, set `run.resume: false` and choose an unused `run.directory`. The trainer validates the dataset and prepares/reuses the token cache before training.

## Validation

In-training validation is off by default. Set `run.validation_enabled: true` to evaluate at steps 400, 800, 1,200, and completion. Set `evaluate_at_start: true` for an additional baseline. Validation batch size is configured separately with `run.validation_batch_size` (default 16).

Evaluate an exported model independently:

```bash
python -u validation.py config.yaml outputs/v5-sft-01/step-001585
```

This writes `validation.json` under `run.directory` without modifying training checkpoints. It reports supervised-token-weighted losses for overall, thinking, non-thinking, reasoning, and answer tokens.

## Outputs

The default output directory is `outputs/v5-sft-01/`.

| File | Contents |
| --- | --- |
| `run.json`, `run-<signature>.json` | Run configuration and metadata |
| `metrics.jsonl` | Training/validation metrics and performance timings |
| `resume.pt` | Model, optimizer, RNG state, and training position for resume |
| `step-XXXXXX/` | BF16 Hugging Face model exports |
| `training_report.json` | Final export path and available validation summary |

The final BF16 export is produced even when in-training validation is disabled.

## Resume training

Set `run.resume: true` and keep the existing run directory, including `resume.pt` and its metadata. Then rerun:

```bash
python -u trainer.py config.yaml
```

Resume restores the saved model, optimizer, RNG state, data position, and learning-rate progress. Training resumes from the **last completed checkpoint**, so updates after that checkpoint may be repeated. No automatic checkpoint is created on Ctrl+C.

To permit limited performance-setting changes on resume, set `run.allow_performance_tuning: true`. Model, dataset, effective batch size, optimizer, schedule, and package versions must remain compatible. For a separate experiment, use a new `run.directory` with `run.resume: false`.

**Note:** With `first_checkpoint_step: 0`, the first resumable checkpoint is saved at update 400. Set it to `1` if an early checkpoint is needed.

