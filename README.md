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

**Not implemented yet:** Dataset B collection, real Qwen baseline runs, GRPO,
PPO. DPO, a Bradley-Terry text reward model, and a temporal-order diagnostic
(ordered vs shuffled presentation of the same sampled frames) are implemented
as separate modules.

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

## Reproducibility init (no model load)

```bash
python -m pytest
magic-vlm-init --config configs/baseline_stub.yaml
# or: python scripts/init_experiment.py --config configs/baseline_stub.yaml
```

## Zero-shot baseline (immutable reference)

```bash
magic-vlm-baseline --config configs/baseline_stub.yaml --run-id baseline-stub-heldout
```

Default split is `held_out`. See `docs/BASELINE.md`.

```bash
magic-vlm-analyze-baseline --run-dir runs/baseline-stub-heldout-v1
```

See `docs/BASELINE_ANALYSIS.md` for failure diagnostics (per-trick, answer
distribution, inspectable error exports).

```bash
magic-vlm-infer --config configs/baseline_stub.yaml
```

See `docs/INFERENCE.md`.

```bash
magic-vlm-sample-frames --video path/to/clip.mp4 --max-frames 8 --also-shuffled --json-out runs/sample.json
```

See `docs/VIDEO_PREPROCESSING.md`.

```bash
magic-vlm-validate --manifest data/examples/toy_manifest.jsonl --json-out runs/validation.json
# media files may be missing for the toy manifest; use --no-media-check while scaffolding
```

See `docs/DATASET_VALIDATION.md`.

```bash
magic-vlm-validate-preferences --prefs data/examples/toy_preferences.jsonl
```

See `docs/PREFERENCE_SCHEMA.md`.

```bash
magic-vlm-annotate --annotator lucas \
  --queue data/examples/toy_annotation_queue.jsonl \
  --out data/annotations/preferences.jsonl \
  --rubric configs/annotation_rubric.yaml
```

See `docs/ANNOTATION_WORKFLOW.md`.

```bash
magic-vlm-analyze-preferences --prefs data/examples/toy_annotated_preferences.jsonl
```

See `docs/PREFERENCE_QUALITY.md`.

```bash
magic-vlm-train-reward --config configs/reward_model_bt_synthetic.yaml
magic-vlm-score-reward --prefs tests/fixtures/reward_model/synthetic_bt_prefs.jsonl \
  --checkpoint runs/reward_model/bt_rm_synthetic_smoke/checkpoint_best.pt \
  --out runs/reward_model/scored.jsonl
```

See `docs/REWARD_MODEL.md`. Preference agreement is **not** reasoning accuracy.

```bash
magic-vlm-train-dpo --probe-only
magic-vlm-train-dpo --config configs/dpo_smoke_text.yaml --smoke-local-lm
```

See `docs/DPO.md`. DPO loss reduction is **not** reasoning improvement.

Objective rewards for a future GRPO trainer (not implemented): see
`docs/OBJECTIVE_REWARDS.md`, including `temporal_iou` /
`docs/TEMPORAL_CAUSAL_REWARD.md` (independent of hidden-state exact match; no
hybrid weighting).

```bash
magic-vlm-compare-objective \
  --manifest data/examples/toy_temporal_causal.jsonl \
  --predictions path/to/predictions.jsonl \
  --out runs/compare_objective.jsonl
```

```bash
magic-vlm-temporal-shuffle --config configs/temporal_shuffle_stub.yaml --run-id temporal-stub
```

See `docs/TEMPORAL_SHUFFLE.md`. This is a temporal-order diagnostic, **not**
proof of causal reasoning, and must not be used for training.

```bash
magic-vlm-run --config configs/baseline_stub.yaml --run-id baseline-dispatch-1
magic-vlm-run --list-types
```

See `docs/EXPERIMENT_RUNNER.md` for the common config-driven dispatcher.

```bash
magic-vlm-smoke --config configs/baseline_stub.yaml
```

See `docs/REPRODUCIBILITY.md` for metadata fields and determinism policy.

## Recommended first real experiment

1. Build Dataset B manifests with frozen `held_out` membership.
2. Install `models` / `video` extras and obtain Qwen2.5-VL-7B-Instruct locally
   (or deliberately enable download).
3. Copy `configs/baseline_qwen25vl_7b.yaml`, point it at the real manifest, and
   run a zero-shot baseline that writes raw `predictions.jsonl`.

## License

MIT (see `pyproject.toml`). Research plan documents remain project notes.
