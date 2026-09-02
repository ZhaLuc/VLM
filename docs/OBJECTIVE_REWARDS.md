# Objective rewards (GRPO-ready interface)

Modular rewards for a **future** GRPO trainer. GRPO itself is **not** implemented
here. Preference reward-model training is separate (`docs/REWARD_MODEL.md`).

## API

```python
from magic_vlm.rewards import build_reward, RewardConfig

reward = build_reward("hidden_state_exact_match")
# or: RewardConfig.from_yaml("configs/reward_hidden_state_exact_match.yaml").build()

result = reward.evaluate(artifact, example)  # RewardResult
scalar = reward.score(artifact, example)     # float
```

`RewardResult` fields: `value`, `reward_id`, `version`, `parse_failed`,
`prediction`, `gold`, `matched`, `extras`.

## Separated concerns

1. **Parse** — `extract_prediction` / `parse_answer` (does not mutate raw text)
2. **Canonicalize** — `canonicalize_label` (compare-time only)
3. **Score** — `compute_hidden_state_exact_match_value` → `{0.0, 1.0}`

## Initial reward

| Field | Value |
|-------|--------|
| id | `hidden_state_exact_match` |
| version | `1.0.0` |
| correct | `1.0` |
| incorrect / malformed | `0.0` |

**Not a reasoning metric.**

## Extension points (stubs)

Reserved ids: `temporal_localization_correctness`, `temporal_iou`,
`explanation_reward`, `hybrid_reward` (hybrid **not** combined in this stage).

## Shortcut risks

- Answer-frequency exploitation
- Parser exploitation
- Camera / identity leakage

## Config

`configs/reward_hidden_state_exact_match.yaml`
