# Open items

Remaining research and data items for this paused prototype.

## Current blockers for a full post-training study

- Only **1 of 5** intended pilot gold clips is approved (`mac_king_s006`).
- `mac_king_s007` remains **PENDING** human review.
- No real human preference dataset has been collected.
- Real DPO / GRPO training on the VLM has not been run.

## Guidelines

- Do not gold-label Wikimedia clips.
- Do not splice reveal counterparts (S1 / S2) onto no-reveal clips (S6 / S7).
- Treat n = 1 accuracy as pipeline evidence only - not generalization or post-training benefit.
- Clip still pending: `mac_king_s007` / Movie7.MP4

## Formal baseline already recorded

- Run: `baseline-real-v1`
- Label: `REAL_ZERO_SHOT_BASELINE`
- Distinct from: `REAL_ZERO_SHOT_BASELINE_SMOKE_TEST`
- Evidence: `reports/real_zero_shot_baseline/`
