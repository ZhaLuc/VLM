# Dataset validation

Command:

```bash
magic-vlm-validate --manifest data/examples/toy_manifest.jsonl --json-out runs/validation.json
# or
python scripts/validate_dataset.py --manifest data/examples/toy_manifest.jsonl --no-media-check
```

The validator **never** repairs, moves, deletes, or re-labels data. It only reports.

## Severity levels

| Severity | Meaning | Default process exit |
|----------|---------|----------------------|
| `error` | Hard integrity / quality failure | fail |
| `leakage` | Scientifically critical train/val ↔ `held_out` contamination | fail |
| `review` | Heuristic / incomplete metadata needing human judgment | pass (reported) |

`--allow-leakage` keeps leakage findings in the report but does not fail the process.
`--no-media-check` skips file existence/decode checks (emits a review finding).

## Rule catalog (scientific meaning)

### Hard errors
- `malformed_metadata` / `manifest_integrity` / `duplicate_example_id`
- `missing_ground_truth` / `invalid_ground_truth_whitespace` / `invalid_answer_not_in_vocab`
- `inconsistent_clip_metadata` / `inconsistent_clip_path_or_split` / `inconsistent_clip_ground_truth`
- `duplicate_video_path` (two `clip_id`s share one file)
- `missing_video` / `unreadable_video` / `unreadable_video_empty`
- `invalid_temporal_interval` / `unexpected_fps`

### Scientific leakage (non-held_out vs `held_out`)
- `leakage_clip_id` — same clip in development and held-out
- `leakage_video_path` — same media path across the boundary
- `leakage_trick_id` — trick identity overlap defeats “unseen trick” claims
- `leakage_performer_id` — performer overlap allows style shortcuts

These are **not** soft nits: on a 15–25 clip dataset they can change conclusions.

### Manual review (not definitive leakage)
- `missing_fps` / `fps_deviates_from_expected` / `fps_duration_inconsistency`
- `train_val_*_overlap` — independence of val only
- `near_duplicate_content_hash` / `duplicate_question_phrasing` — heuristics
- `repeated_trick_performer_combo` — coverage concentration
- `no_held_out_split` / media-check availability notices

## Machine-readable output

`--json-out` writes a `ValidationReport` JSON with `passed`, counts, and findings
(including `scientific_meaning` per finding). Compatible with the existing
`magic_vlm.utils.write_json` artifact style.
