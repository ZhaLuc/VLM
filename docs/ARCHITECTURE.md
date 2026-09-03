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

## Ground-truth storage

- Dataset `ground_truth` is authoritative and must not be silently normalized.
- See `docs/DATASET_SCHEMA.md`.

## Experiment metadata

- Prefer `initialize_experiment`, which writes `metadata.json`,
  `environment.json`, `determinism.json`, serialized config, and
  `run_manifest.json`. See `docs/REPRODUCIBILITY.md`.
- Common dispatcher: `magic-vlm-run` / `docs/EXPERIMENT_RUNNER.md` (baseline,
  temporal_shuffle, dpo, reward_model only; no overwrite; failures recorded).

## Baseline immutability

- Zero-shot baseline runs should set `baseline_immutable: true`.
- Treat those metrics as a frozen reference; later training comparisons must
  point at a baseline `run_id` rather than silently re-running and overwriting.

## Temporal-order diagnostic

- Sampling is independent of shuffle; see `docs/VIDEO_PREPROCESSING.md`.
- `ordered_and_shuffled_pair` builds both views from one sample plan.
- The paired experiment runner is `magic_vlm.temporal` (`docs/TEMPORAL_SHUFFLE.md`).
- Temporal-shuffle sensitivity is a diagnostic of order dependence, not proof
  of causal reasoning. Shuffle outputs must not be used for training.

## Preference records

- Pairwise judgments use `PreferencePair` with nested `generation_meta` and
  `annotation` objects; raw `response_a` / `response_b` are never normalized.
- See `docs/PREFERENCE_SCHEMA.md`.
- A small Bradley-Terry **text** reward model can be fit on preferences
  (`docs/REWARD_MODEL.md`). Preference agreement ≠ reasoning accuracy.
- DPO for explanation preferences: `docs/DPO.md` (TRL + PEFT). Probe the stack
  before claiming VLM readiness. Never overwrite immutable baselines.
- GRPO on modular objective rewards: `docs/GRPO.md` (TRL + PEFT; default
  `hidden_state_exact_match`). Reward gains ≠ reasoning improvement.
- Cross-method comparative evaluation: `docs/COMPARISON.md` (locked held-out
  protocol; accuracy / generalization / temporal / reward kept separate).
- Reward-hacking diagnostics: `docs/REWARD_HACKING.md` (reward vs independent
  accuracy / RM / human; observational `possible_*` tags only).
- Objective rewards: `docs/OBJECTIVE_REWARDS.md`
  (`hidden_state_exact_match` and `temporal_iou` v1.0.0). Not reasoning metrics;
  no hybrid weighting. Causal annotation status is exposed in reports
  (`docs/TEMPORAL_CAUSAL_REWARD.md`).

## Framework compatibility

- This project defaults to **project-sampled frames** for Qwen2.5-VL, not
  processor-internal `fps` resampling. See `docs/INFERENCE.md`.
- Transformers/TRL/PEFT/vLLM versions are **not** pinned yet; compatibility
  must be verified at the baseline-inference stage before claiming support.
- Architecture tests use `stub/` model ids and never download weights.
