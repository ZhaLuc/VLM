# Reproducibility foundation

This document records how experiment runs are identified and what is (and is
not) guaranteed about determinism.

## What every initialized run records

`initialize_experiment` creates `runs/<run_id>/` and writes:

| File | Purpose |
|------|---------|
| `config.yaml` / `config.json` | Full serializable experiment config |
| `metadata.json` | Identity + environment + determinism + code state |
| `environment.json` | Python/platform/torch/CUDA/git snapshot |
| `device.json` | Resolved device (`cpu` or `cuda:N`) |
| `determinism.json` | Seed settings + honesty level |
| `result_changing_parameters.json` | Knobs that can change scientific results |
| `run_manifest.json` | Compact legacy-compatible summary |
| `run.log` | File + stderr logging |

Identity fields always include: model, checkpoint kind/path, dataset version,
split, task, training method, reward function, generation config, seed,
hardware/runtime (via environment/device), code/config hashes, output directory.

## Zero-shot baseline vs post-trained

A run is a **zero-shot baseline** when:

- `training_method: none`
- `checkpoint.kind` is `base` or `stub`

These runs should keep `baseline_immutable: true`.

Post-trained runs must use `training_method` in `{sft,dpo,grpo,ppo}` and/or
`checkpoint.kind` in `{post_trained, adapter}`, and must **not** set
`baseline_immutable: true`.

## Determinism status

| Level | When reported | Meaning |
|-------|---------------|---------|
| `unavailable` | seeds skipped | No seed controls applied |
| `partially_controlled` | default after `set_seed` | RNGs seeded; bitwise equality **not** claimed |
| `guaranteed` | **never** by current code | Would require a verified harness we do not provide |

Factors that can still change results under the same seed:

- CPU vs CUDA builds, driver/CUDA versions
- Hugging Face / model generate kernels
- Unset `PYTHONHASHSEED` (noted in `determinism.json`)
- Dataset/media content changes with the same manifest path
- Any change listed in `result_changing_parameters.json`

## Device selection

`device.preference` is `auto | cpu | cuda`. Optional `cuda_index` selects
`cuda:N` only when explicitly configured. Machine-specific GPU UUIDs are not
hard-coded.

## Initialize without loading a VLM

```bash
magic-vlm-init --config configs/baseline_stub.yaml
# or
python scripts/init_experiment.py --config configs/baseline_stub.yaml
```
