# Reward-hacking / reward–quality divergence diagnostics

Detects **possible** cases where an optimized reward moves without matching
independent task performance, RM scores, or human labels.

A single example is **never** proof of reward hacking. Tags are `possible_*`
only. Training reward is never the sole independent evaluator.

## Compared axes (kept separate)

| Axis | Source |
|------|--------|
| Programmatic / RM reward | `quadrant_reward: rm \| objective` + optional `rm_scores_path` |
| Ground-truth accuracy | Exact-match on locked held-out predictions |
| Preference-model score | Optional `rm_score` column |
| Human evaluation | Optional `human_labels_path` with provenance |

## Cohorts / heuristics

- High-reward / low-accuracy and low-reward / high-accuracy exports
- Length, confidence-marker, keyword-stuffing, markdownish style heuristics
- Hidden-state shortcuts: answer-frequency, parser exploitation, camera leakage
- Training `reward_stats` vs held-out accuracy gap (when files present)

## Command

```bash
magic-vlm-analyze-reward-hacking --config configs/reward_hacking_toy.yaml
```

## Outputs

Under `runs/reward_hacking/<run_id>/`:

- `reward_hacking_metrics.json` / `reward_hacking_report.md`
- `examples_inspectable.jsonl`
- `high_reward_low_accuracy.jsonl` / `low_reward_high_accuracy.jsonl`
- `heuristic_flagged.jsonl` / `findings.jsonl`
- `DISCLAIMER.json` (`single_example_is_not_proof: true`)

## Integrity

- Do not hide negative findings
- Do not call one example proof of hacking
- State when human evaluation is unavailable
- Objective exact-match reward is tied to accuracy — use an RM column for
  RM-style quadrants
