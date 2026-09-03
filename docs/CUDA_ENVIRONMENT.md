# CUDA / GPU environment notes

Verified on 2026-09-03 for the first real Qwen2.5-VL smoke:

| Item | Value |
|------|-------|
| OS | Windows 10 (AMD64) |
| GPU | NVIDIA GeForce RTX 3060 (12 GiB) |
| Driver | 610.88 (CUDA UMD 13.3) |
| Python | 3.11.9 (`C:\Users\Lucas\AppData\Local\Programs\Python\Python311\python.exe`) |
| Prior torch | `2.13.0+cpu` (`torch.cuda.is_available() == False`) |
| Installed torch | `2.13.0+cu130` from `https://download.pytorch.org/whl/cu130` |
| Transformers | 5.16.1 (unchanged) |
| Accelerate | 1.14.0 (unchanged) |
| PEFT | 0.20.0 (unchanged) |
| TRL | 1.12.0 (unchanged) |
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` in HF hub cache (not committed) |

No project `.venv` existed; the CUDA wheel replaced the CPU wheel in the existing
Python 3.11 environment used by this repository.

## Install command used

```bash
python -m pip uninstall -y torch
python -m pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
```

## Smoke command used

```bash
python scripts/run_baseline.py --config configs/baseline_qwen25vl_3b.yaml --run-id REAL_ZERO_SHOT_BASELINE_SMOKE_TEST --load-frames --allow-download
```

Evidence (no weights): `reports/real_zero_shot_baseline_smoke/`.
Full run dir (gitignored): `runs/REAL_ZERO_SHOT_BASELINE_SMOKE_TEST/`.

## Formal next baseline command

```bash
magic-vlm-baseline --config configs/baseline_qwen25vl_3b.yaml --run-id baseline-real-v1 --load-frames
```

Cached hub weights load without `--allow-download`. Do not commit weights or videos.

Formal baseline evidence (committed, distinct from smoke):
`reports/real_zero_shot_baseline/` with label `REAL_ZERO_SHOT_BASELINE`.
Full run dir (gitignored): `runs/baseline-real-v1/`.

n=1 exact-match accuracy must not be over-interpreted scientifically.
