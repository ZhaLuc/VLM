# GRPO on objective reward (hidden-state exact match)

First GRPO post-training path. Compares an **untouched base checkpoint** to a
**GRPO-adapted** checkpoint on the same held-out benchmark.

Reward logic stays in `magic_vlm.rewards`. The trainer only adapts datasets and
invokes `ObjectiveReward.evaluate`. Reward gain is **not** reasoning improvement.

## Stack probe

```bash
magic-vlm-train-grpo --probe-only
```

| Component | Role |
|-----------|------|
| TRL `GRPOTrainer` | Group relative policy optimization |
| PEFT / LoRA | Parameter-efficient adaptation |
| CUDA | Required for practical Qwen2.5-VL GRPO |
| vLLM | Optional generation acceleration (not required) |

- **Text GRPO** (`ready_for_text_grpo`): torch + TRL GRPOTrainer + transformers + datasets.
- **VLM GRPO** (`ready_for_vlm_grpo`): also needs CUDA, PEFT, and TRL source mentions of
  multimodal / vision paths (`mm_token_type_ids`). Do not invent workarounds when the
  probe reports not ready.

## Integrity rules

- Do not train on `held_out` examples or modify held-out membership.
- Do not select checkpoints using final held-out / test scores (`last_train_step` only).
- Failed / malformed rollouts score **0.0** (never silently dropped, never `None`).
- Do not hard-code task reward inside the trainer; configure `reward_id`.
- Do not combine rewards without explicit configuration (not supported in this stage).
- Do not overwrite immutable baseline run directories.

## Commands

```bash
# Text smoke (tiny local LM, no Hub download)
magic-vlm-train-grpo --config configs/grpo_smoke_text.yaml --smoke-local-lm

# Via common runner
magic-vlm-run --config configs/grpo_smoke_text.yaml

# Qwen2.5-VL template (blocked until probe ready_for_vlm_grpo)
# magic-vlm-train-grpo --config configs/grpo_qwen25vl_3b.yaml
```

## Artifacts

Under `runs/grpo/<run_id>/`:

- `checkpoint/` — GRPO (LoRA) weights; isolated from baseline
- `train_metadata.json` — base checkpoint, reward version, group size, generation,
  optimizer/LR/batch, seed, PEFT, hardware, steps, checkpoint selection
- `raw_completions.jsonl` — preserved rollout text + reward fields
- `reward_stats.json` — mean reward / parse-fail counts (not a reasoning claim)
- `held_out_eval.json` — independent eval after train (not used for selection)
- `DISCLAIMER.json`, `comparison_plan.json`, `baseline_protection.json`

## Scientific reading

1. Report GRPO training reward statistics separately from held-out exact-match accuracy.
2. Compare GRPO checkpoint vs untouched baseline on the **same** held-out set.
3. Do not interpret higher train reward as better causal / temporal reasoning.
