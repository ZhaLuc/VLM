# Preference data quality analysis

QC tooling for pairwise preference JSONL **before** DPO or Bradley-Terry reward
modeling. Non-destructive: nothing is deleted or silently filtered.

## Checks

| Check | Severity class |
|-------|----------------|
| Malformed JSON / schema failures | hard error |
| Identical A/B within a pair | hard error |
| Duplicate `judgment_id` / non-canonical `pair_id` | hard error |
| A/B winner imbalance | warning / info |
| Duplicate content `pair_id` groups | warning |
| Contradictory labels on same pair | warning (not auto-labeled as annotator error) |
| Exact response strings reused across pairs | warning |
| Repeated-judgment agreement | warning/info; IRR only if ≥2 annotators |
| Preferred-longer length association | possible bias (correlation ≠ reward hacking) |
| Confidence-marker / markdownish correlations | possible bias |
| Clip / task / trick / annotator distributions | descriptive |

## Command

```bash
magic-vlm-analyze-preferences --prefs data/examples/toy_annotated_preferences.jsonl \
  --out-dir runs/pref_quality_toy
```

Outputs:

- `preference_quality.json`
- `preference_quality_report.md`
- `preference_quality_findings.jsonl`
- `malformed_records.jsonl`

## Integrity rules

- Do not silently filter examples.
- Do not treat all disagreement as annotation error.
- Do not assume longer explanations are better.
- Do not claim inter-rater reliability with only one annotator.
- Preserve uncertainty: correlations are observational.
