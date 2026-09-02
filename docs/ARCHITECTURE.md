# Architecture notes

This document records consequential research-integrity decisions for the
minimal code architecture. It is not a redesign of the research plan in
`magic-vlm-research-plan-v2.md`.

## Test-set boundaries

- Every `ExampleRecord` carries an explicit `split` in `{train, val, held_out}`.
- `held_out` is reserved for final comparative evaluation.
- `ExperimentConfig.allow_held_out_in_training` defaults to `false` and raises
  if set `true`.
- `magic_vlm.dataset.iter_for_stage` and `magic_vlm.training.validate_training_split`
  refuse held-out leakage into training-like stages.

Freeze held-out membership when the real Dataset B manifest is created. Do not
move examples into/out of `held_out` after baseline numbers are published.

## Raw output preservation

- `InferenceArtifact.raw_text` stores the untouched decoder string.
- `parsed_answer` is derived and never a substitute when writing artifacts.
- `preserve_raw_outputs=true` in experiment configs writes `predictions.jsonl`.

## Experiment metadata

- Each smoke/run writes `run_manifest.json` with `run_id`, UTC timestamp,
  full config, `config_hash`, optional `git_commit`, stage, and
  `baseline_immutable`.

## Baseline immutability

- Zero-shot baseline runs should set `baseline_immutable: true`.
- Treat those metrics as a frozen reference; later training comparisons must
  point at a baseline `run_id` rather than silently re-running and overwriting.

## Future temporal shuffling

- `VideoPreprocessConfig.temporal_shuffle` exists as a preprocessing flag.
- Default is `false`. Enabling shuffle changes frame **order only** and must
  not rewrite source media.
- Shuffle diagnostics belong to a later experiment stage; this architecture
  only guarantees a deterministic, seed-stable hook.

## Framework compatibility

- Real Qwen2.5-VL loading goes through Hugging Face Transformers when the
  optional `models` extra is installed.
- Transformers/TRL/PEFT/vLLM versions are **not** pinned yet; compatibility
  must be verified at the baseline-inference stage before claiming support.
- Architecture tests use `stub/` model ids and never download weights.
