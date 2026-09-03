# Visual Reasoning from Magic and Mentalism Demonstrations

Paused VLM research prototype. Real zero-shot pipeline works on **one** human-approved hidden-state clip. Post-training (DPO / GRPO) was **not** completed.

## Talk to Professor Xu with this file

**[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)** - images + short bullet cues for the meeting.

## Study this before the meeting

**[docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md)** - deep explanation of everything: concepts, preprocessing, inference, results, DPO/GRPO, and Q&A drill.

Live status: [PROJECT_STATUS.md](PROJECT_STATUS.md) (`PAUSED - ZERO-SHOT PROTOTYPE COMPLETE`).

## What worked

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Clip: Mac King S6 (`Movie6.MP4`)
- Answer: `right` (matches ground truth)
- **n = 1**, accuracy **1.0**
- Evidence: `reports/real_zero_shot_baseline/`

This does **not** prove strong reasoning or that training would help.

## Repo layout

```
src/magic_vlm/     code (data, video, model, eval, training scaffolds)
configs/           experiment YAML
scripts/           CLI wrappers
data/examples/     manifests / reviews (videos are local, not in git)
reports/           committed baseline + health evidence
docs/WALKTHROUGH.md
docs/STUDY_GUIDE.md
tests/
```

## Setup

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/project_health.py
```

Real baseline (already recorded):

```bash
python scripts/run_baseline.py --config configs/baseline_qwen25vl_3b.yaml --run-id baseline-real-v1 --load-frames
```

## License

MIT (see `pyproject.toml`).
