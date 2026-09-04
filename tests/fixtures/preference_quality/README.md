# Preference quality fixtures

Synthetic JSONL used by `tests/test_preference_quality.py`.

- `synthetic_quality.jsonl` - known counts for imbalance, duplicates,
  contradictions, length/confidence associations, identical A/B.
- `synthetic_quality_with_malformed.jsonl` - same plus two malformed lines.

Do not treat these as real human labels.
