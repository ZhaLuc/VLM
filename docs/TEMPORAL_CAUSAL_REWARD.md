# Temporal / causal objective reward

Independently callable IoU-based reward for Dataset-C-style localization.
Comparable with `hidden_state_exact_match`. **Not** a hybrid and **not** GRPO.

## Eligibility

Scores only when `example.causal` has:

- `status` in `{known, researcher_annotated, ambiguous}`
- its own `provenance`
- a usable `causal_moment` interval (seconds and/or frames)

| Status | Scored as gold? |
|--------|-----------------|
| `known` (objectively established) | yes |
| `researcher_annotated` | yes (judgment, not unique-cause proof) |
| `ambiguous` | **no** (exposed in reports; value 0) |

Clip-level `example.temporal` is **never** used as causal gold.

`unique_cause=false` is allowed and reported; the reward does **not** assume every
trick has one causal action. A salient action is not a proven causal action.

## Modes

- `binary` — `1.0` if IoU ≥ `iou_threshold` else `0.0`
- `partial` — reward value = IoU in `[0, 1]`

## API

```python
from magic_vlm.rewards import build_reward, compare_hidden_state_and_temporal

reward = build_reward("temporal_iou", mode="binary", iou_threshold=0.5)
result = reward.evaluate(artifact, example)

# Independent comparison (no weighting):
compare_hidden_state_and_temporal(artifact, example)
```

Configs: `configs/reward_temporal_iou.yaml`, `configs/reward_temporal_iou_partial.yaml`.

Fixture with authored causal spans: `data/examples/toy_temporal_causal.jsonl`.

## CLI

```bash
magic-vlm-compare-objective \
  --manifest data/examples/toy_temporal_causal.jsonl \
  --predictions path/to/predictions.jsonl \
  --out runs/compare_objective.jsonl
```

## Integrity

Ambiguous labels are retained and surfaced (`annotation_status` in extras).
Missing causal annotations score 0 without inventing intervals.
GRPO and reward weighting are out of scope for this stage.
