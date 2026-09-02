# Objective rewards (GRPO-ready interface)

Modular rewards for a **future** GRPO trainer. GRPO itself is **not** implemented
here. Preference reward-model training is separate (`docs/REWARD_MODEL.md`).

## API

```python
from magic_vlm.rewards import build_reward, RewardConfig, compare_hidden_state_and_temporal

reward = build_reward("hidden_state_exact_match")
# or: RewardConfig.from_yaml("configs/reward_hidden_state_exact_match.yaml").build()

result = reward.evaluate(artifact, example)  # RewardResult
scalar = reward.score(artifact, example)     # float

# Independent temporal/causal comparison (no weighting):
compare_hidden_state_and_temporal(artifact, example)
```

`RewardResult` fields: `value`, `reward_id`, `version`, `parse_failed`,
`prediction`, `gold`, `matched`, `extras`.

## Separated concerns

1. **Parse** — `extract_prediction` / `parse_answer` / `parse_interval_text`
2. **Canonicalize** — `canonicalize_label` (labels) or interval validation
3. **Score** — exact-match `{0,1}` or temporal IoU (binary / partial)

## Implemented rewards

| id | version | signal |
|----|---------|--------|
| `hidden_state_exact_match` | `1.0.0` | binary short-label correctness |
| `temporal_iou` | `1.0.0` | IoU vs defensible `causal` span (`binary` or `partial`) |
| `temporal_localization_correctness` | alias | same as `temporal_iou` (binary default) |

**Neither is a reasoning-quality metric.** Hybrid weighting is reserved / forbidden.

See `docs/TEMPORAL_CAUSAL_REWARD.md` for causal annotation status rules.

## Shortcut risks

Hidden-state: answer-frequency, parser, camera leakage.

Temporal: salient-motion exploitation, interval-parser exploitation, ambiguous-cause collapse.

## Config

- `configs/reward_hidden_state_exact_match.yaml`
- `configs/reward_temporal_iou.yaml`
- `configs/reward_temporal_iou_partial.yaml`
