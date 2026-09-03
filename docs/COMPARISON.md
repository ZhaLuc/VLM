# Comparative evaluation under a locked held-out protocol.

Compares implemented methods on the **same** held-out set, task, metric, and
(when declared) generation policy. Does **not** invent a reasoning score.

## Dimensions (kept separate)

| Dimension | Meaning |
|-----------|---------|
| Task accuracy | Exact-match on locked held-out examples |
| Generalization slices | Unseen trick / performer / camera / prop / wording; known-trick variations |
| Temporal sensitivity | Ordered vs shuffled diagnostic (attached when available) |
| Reward change | Held-out mean reward and optional training `reward_stats.json` |
| Human quality | Not computed here |

None of these is automatically labeled as **reasoning improvement**.

## Inputs

Each method arm provides either:

- `run_dir` containing `predictions.jsonl` (baseline) or `held_out_eval_rows.jsonl` (GRPO), or
- explicit `predictions_path`

Optional: `temporal_summary_path` / `reward_stats_path` / `generation_policy`.

## Command

```bash
magic-vlm-compare-methods --config configs/compare_methods_toy.yaml
# or
python scripts/compare_methods.py --config configs/compare_methods_toy.yaml
```

## Outputs

Under `runs/comparison/<run_id>/`:

- `aligned_examples.jsonl` — one row per locked example; every method cell present (missing flagged)
- `comparison_metrics.json` — aggregates, per-group, slices, temporal, reward, deltas
- `comparison_report.md` — human-readable tables
- `DISCLAIMER.json` / `comparison_config.json` / `result.json`

## Integrity

- Same locked held-out membership for all arms
- Missing predictions are visible (`missing: true`); full coverage required by default
- Incompatible `generation_policy` fingerprints raise unless `allow_incompatible_protocols: true`
- SFT / PPO arms are rejected until implemented
- Do not use this report for checkpoint selection on final test performance
