# Visual Reasoning from Magic and Mentalism Demonstrations

**Open research prototype** for evaluating vision-language models on **hidden-state** reasoning in short magic and mentalism videos - with a reproducible evaluation spine and unfinished preference / reward post-training scaffolds (DPO, GRPO).

| | |
|---|---|
| **Status** | Paused - zero-shot prototype complete |
| **Post-training** | Not completed |
| **Primary model** | [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| **Verified result** | One human-approved example (n = 1), exact-match correct |
| **License** | MIT |

---

## Overview

Magic and mentalism clips are a useful stress test for multimodal models: the true object state can be hidden, misdirection can contradict the final frame, and the correct answer often depends on event order rather than a single still image.

This repository provides:

- dataset schemas, validation, and train / held-out leakage checks
- deterministic video frame sampling (with optional temporal shuffle utilities)
- VLM loading and inference with preserved raw model text
- exact-match evaluation and experiment artifact logging
- scaffolds for preferences, Bradley-Terry reward models, DPO, and GRPO
- project-health auditing

It is documented honestly as a **paused prototype**, not as a completed post-training study.

```text
video + question  ->  VLM  ->  answer  ->  score
                         \
                          ->  (planned) DPO / GRPO  ->  held-out evaluation
```

---

## Key result

On one human-approved Mac King no-reveal clip (**S6**):

| Field | Value |
|-------|--------|
| Question | Which hand contains the coin after the apparent transfer? |
| Model output | `right` |
| Ground truth | `right` |
| n | 1 |
| Accuracy | 1.0 |
| Evidence | [`reports/real_zero_shot_baseline/`](reports/real_zero_shot_baseline/) |

**Interpretation:** the end-to-end zero-shot pipeline works on real video and a real open VLM. A single correct answer does **not** establish strong reasoning, temporal or causal understanding, generalization, or post-training improvement.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/OVERVIEW.md`](docs/OVERVIEW.md) | Visual overview with short bullet cues |
| [`docs/TECHNICAL_GUIDE.md`](docs/TECHNICAL_GUIDE.md) | Full technical explanation: concepts, pipeline, methods, limitations |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Generated project-health status |
| [`docs/OPEN_ITEMS.md`](docs/OPEN_ITEMS.md) | Remaining research / data items |

---

## Repository layout

```text
configs/           Experiment YAML
data/examples/     Manifests and reviews (raw videos are not committed)
docs/              Public documentation and diagrams
reports/           Committed baseline and health evidence
scripts/           CLI entrypoints
src/magic_vlm/     Core Python package
tests/             Automated tests
```

---

## Quick start

Requires **Python 3.11+**.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Optional stacks:

```bash
python -m pip install -e ".[dev,video,models]"
```

Project health:

```bash
python scripts/project_health.py
# or: magic-vlm-project-health
```

Formal baseline configuration (already recorded under `reports/`):

```bash
python scripts/run_baseline.py \
  --config configs/baseline_qwen25vl_3b.yaml \
  --run-id baseline-real-v1 \
  --load-frames
```

Notes:

- Model weights are downloaded or cached via Hugging Face; they are **not** stored in this repository.
- Local MP4 / OGV media under `data/videos/` are gitignored. Reproduce experiments with your own licensed copies of the cited source videos.

---

## What is implemented vs incomplete

| Completed | Not completed |
|-----------|----------------|
| Research architecture and tests | 5-clip / 15-25 clip benchmarks |
| Dataset validation and leakage tooling | Human preference dataset |
| Video preprocessing | Bradley-Terry RM on real preferences |
| CUDA Qwen2.5-VL-3B load and inference | Real DPO / GRPO / PPO training |
| One approved hidden-state gold example (S6) | Temporal / causal post-training study |
| Formal zero-shot baseline (n = 1) | Evidence of post-training improvement |

Wikimedia / PeerJ cups-and-balls pilots remain **controls** (visible state / reveal leakage). They are not hidden-state gold.

---

## Data and ethics

- Mac King supplementary stimuli are associated with Cui et al., 2011 (*Front. Hum. Neurosci.*): [PMC3202226](https://pmc.ncbi.nlm.nih.gov/articles/PMC3202226/).
- Provenance metadata is recorded under `data/provenance/` and example manifests.
- Media files are intentionally excluded from git. Confirm reuse rights before redistribution or training.

---

## Citation

If you use this repository, please cite the upstream stimulus sources above and acknowledge this codebase as an open research prototype for VLM hidden-state evaluation.

---

## Contributing

Issues and pull requests that improve documentation clarity, reproducibility, or evaluation integrity are welcome. Please do not commit model weights, private credentials, or raw video binaries.
