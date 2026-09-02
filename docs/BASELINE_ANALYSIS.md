# Baseline evaluation analysis

Post-hoc diagnostics for a frozen zero-shot baseline run. Accuracy is reported,
but the goal is **inspectable failures**, group variation, and answer-frequency
structure — not a claim about reasoning ability.

## What is computed

| Analysis | Notes |
|----------|--------|
| Overall exact-match accuracy | Same denominator as baseline (`n` predictions) |
| Per-trick / per-performer / per-camera | When metadata is available via manifest join |
| Gold vs predicted answer distributions | Includes majority-class baseline accuracy |
| Parse failures | Always visible; never dropped from exports |
| Correct / incorrect example lists | Full exports |
| Observational tags | See below |

## Observational tags (not causal claims)

| Tag | Meaning |
|-----|---------|
| `correct` | Exact-match success |
| `parse_failure` | No usable parsed label |
| `inference_error` | Runtime error recorded on the row |
| `empty_raw_response` | Empty model text |
| `wrong_label` | Parsed a label that does not match gold |
| `visual_input_missing` | No pixels provided (`indices_only…`) |
| `incorrect_with_pixels_present` | Incorrect while pixels were present |
| `possible_answer_frequency_shortcut` | Wrong prediction equals majority gold label |
| `ambiguity_or_task_design_note` | Notes/metadata hint at ambiguity |

**Never** infer “reasoning failure” from a tag alone.

## Outputs

Written under the run directory (or `--out-dir`):

- `analysis_metrics.json` — machine-readable aggregates + all diagnoses
- `analysis_report.md` — human-readable summary
- `errors.jsonl` — every incorrect example
- `successes.jsonl` — every correct example
- `examples_inspectable.jsonl` — all examples with tags

Baseline runs also emit these automatically after scoring.

## Command

```bash
magic-vlm-analyze-baseline --run-dir runs/baseline-stub-heldout-v1
# or
python scripts/analyze_baseline.py --run-dir runs/baseline-stub-heldout-v1 \
  --manifest data/examples/toy_manifest.jsonl
```

## Integrity

- Do not retune prompts or protocol against `held_out` after reading the report.
- Do not suppress ambiguous or uninteresting failures.
- Compare later post-training runs against the same baseline `run_id`.
