# Research-quality experiment reporting

Aggregates stored run artifacts into deterministic markdown + JSON summaries
for lab meetings and later paper prep. **Read-only** — does not re-run models
or invent missing numbers.

## Required sections

1. Experiment summary  
2. Configuration  
3. Dataset / split  
4. Model / checkpoint  
5. Training method  
6. Reward  
7. Evaluation  
8. Generalization  
9. Temporal shuffle  
10. Reward-hacking analysis  
11. Representative successes  
12. Representative failures  
13. Unresolved issues  

Missing sections are marked `unavailable` and listed under unresolved issues.

## Example selection rule

Printed in every report:

> Pool examples from `examples_inspectable.jsonl`, else `aligned_examples.jsonl`,
> else `predictions.jsonl` / `held_out_eval_rows.jsonl`. Success = `correct is
> True`; failure = `correct is False`. Sort by `(trick_id, example_id)`.
> Round-robin by `trick_id` until caps. Never pad. Truncate `raw_text` for
> display; full rows remain in linked JSONL.

## Commands

```bash
magic-vlm-report --config configs/experiment_report_toy.yaml
magic-vlm-report --run-dir tests/fixtures/comparison/zero_shot --run-id report_from_baseline
```

## Outputs

Under `runs/reports/<run_id>/`:

- `experiment_report.md`
- `experiment_report.json` (sorted keys)
- `experiment_report_config.json`
- `DISCLAIMER.json`
- `result.json`

## Integrity

- Do not invent scientific conclusions
- Do not hide failed / inconclusive runs
- Do not call a method better without a defined comparison
- Do not cherry-pick examples outside the documented rule
- Never auto-translate metric changes into "reasoning improved"
