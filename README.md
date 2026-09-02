# Visual Reasoning from Magic and Mentalism Demonstrations

Research code for post-training and evaluating vision-language models on short
magic/mentalism demonstration clips.

**Working title:** Visual Reasoning and Explanation from Magic and Mentalism
Demonstrations: A Post-Training Study of Vision-Language Models

## Status

This repository currently provides a **minimal research architecture** only:

- dataset schemas + JSONL loading with split-boundary checks
- deterministic video frame-index preprocessing (optional OpenCV decode)
- model loading separated from training algorithms
- inference artifacts that preserve raw model text
- exact-match evaluation and standalone rewards
- experiment configuration / run manifests
- preference-pair schema I/O

**Not implemented yet:** Dataset B collection, real Qwen baseline runs, DPO,
GRPO, PPO, reward-model training, or temporal-shuffle experiments beyond the
preprocessing flag.

See `docs/ARCHITECTURE.md` and `magic-vlm-research-plan-v2.md`.

## Layout

```
configs/               # experiment YAML
data/examples/         # toy manifests (no raw videos committed)
docs/                  # architecture / integrity notes
scripts/               # thin wrappers
src/magic_vlm/         # package
tests/                 # unit + interface smoke tests
```

## Setup

Requires Python 3.11+.

```bash
python -m pip install -e ".[dev]"
```

Optional extras (later stages):

```bash
python -m pip install -e ".[dev,video,models]"
```

## Smoke test (no model download)

```bash
python -m pytest
python -m magic_vlm.cli --config configs/baseline_stub.yaml
```

## Recommended first real experiment

1. Build Dataset B manifests with frozen `held_out` membership.
2. Install `models` / `video` extras and obtain Qwen2.5-VL-7B-Instruct locally
   (or deliberately enable download).
3. Copy `configs/baseline_qwen25vl_7b.yaml`, point it at the real manifest, and
   run a zero-shot baseline that writes raw `predictions.jsonl`.

## License

MIT (see `pyproject.toml`). Research plan documents remain project notes.
