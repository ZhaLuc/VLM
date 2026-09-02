# Bradley-Terry preference reward model

Small text reward model for explanation preferences:

```
L = -log(sigmoid(r(x, y_w) - r(x, y_l)))
```

## What `x` is

For this dataset size, `r(x, y)` encodes:

`[TASK] … [CLIP] … [INSTRUCTION] … [RESPONSE] …`

Clip identity + instruction are the **task/video representation**. Pixel/video
encoders are intentionally **not** used here (too large for ~tens of labels).

## Integrity

**Preference-model agreement is not ground-truth reasoning accuracy.**  
Do not use this RM as a standalone factual evaluator. Do not train on
`held_out` preference rows. This stage does **not** run GRPO.

## Commands

```bash
# Synthetic smoke (CPU)
magic-vlm-train-reward --config configs/reward_model_bt_synthetic.yaml

# Score pairs with a checkpoint (inference only)
magic-vlm-score-reward \
  --prefs tests/fixtures/reward_model/synthetic_bt_prefs.jsonl \
  --checkpoint runs/reward_model/bt_rm_synthetic_smoke/checkpoint_best.pt \
  --out runs/reward_model/bt_rm_synthetic_smoke/scored.jsonl
```

## Outputs

Under `runs/reward_model/<run_id>/`:

| File | Contents |
|------|----------|
| `checkpoint_best.pt` / `checkpoint_last.pt` | Model + tokenizer + config |
| `metrics.jsonl` | Per-epoch train/val loss + preference accuracy |
| `train_scores.jsonl` / `val_scores.jsonl` | Example-level rewards |
| `train_metadata.json` | Dataset split, architecture, optimizer, seed, hardware |
| `DISCLAIMER.json` | Preference ≠ reasoning correctness |

## Validation metric

Primary: **held-out preference accuracy** on the preference `val` split  
(fraction of pairs with `r(chosen) > r(rejected)`). Training loss alone is
insufficient.
