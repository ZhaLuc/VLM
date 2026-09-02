# Dataset schema (hidden-state first)

## Record model

Each JSONL row is one :class:`magic_vlm.schemas.ExampleRecord`.

| Field | Required | Notes |
|-------|----------|-------|
| `example_id` | yes | Unique across the manifest |
| `clip_id` | yes | Shared by question variants of the same filmed moment |
| `trick_id` | yes | Trick family / identity |
| `performer_id` | yes | |
| `camera_id` | yes | |
| `video.path` | yes | Plus optional hash/fps/duration/num_frames |
| `task` | yes | `hidden_state` (now) or `explanation` (later) |
| `question` | yes | |
| `ground_truth` | yes for `hidden_state` | Canonical stored label; never silently rewritten |
| `justification` | no | Optional human note, not required for scoring |
| `temporal` | no | `start_s`/`end_s` and/or frame indices |
| `causal` | no | Reserved for later temporal/causal labels |
| `split` | yes | `train` \| `val` \| `held_out` |
| `provenance` | yes | At least `source` |
| `notes` | no | |
| `question_variant` | no | Label for paraphrases (`canonical`, `paraphrase_1`, …) |

Multiple questions for one clip: same `clip_id` + video path + split, different
`example_id` / `question` / `question_variant`.

## Ground truth vs prediction normalization

- **Storage:** `ground_truth` is written and loaded exactly as authored.
- **Evaluation:** `magic_vlm.evaluation.normalize_label` may case-fold / collapse
  whitespace **only when comparing** a prediction to a copy of the gold label.
- Metrics must not write normalized strings back into manifests.

Legacy manifests that used `"answer"` are accepted on load and mapped to
`ground_truth` without changing the string value. New writes use `ground_truth`
only.

## Split integrity

- Split is explicit on every row.
- All variants of a `clip_id` must share one split.
- Loaders return all rows; use `load_split` / `filter_split` for a partition.
- Training-like stages refuse `held_out` via `iter_for_stage`.

## Extension path

- `task: explanation` can omit short-label `ground_truth` (validated separately
  when that stage is implemented).
- Optional `causal` annotations can be added later without changing existing
  hidden-state rows.
