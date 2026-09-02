# Video preprocessing

Minimal CPU pipeline for baseline inference and temporal-shuffle diagnostics.

## Guarantees

- Sampling (`uniform` / `first_n`) is deterministic for a fixed `max_frames` and
  source frame count.
- Temporal shuffle **does not resample**. Ordered and shuffled conditions share
  the same `ordered_indices` (same source frames); only presentation order changes.
- Source videos are never overwritten.
- Prompt construction is out of scope for this module.

## Recorded metadata (`SampledClip.to_dict`)

- sampling policy / `max_frames`
- `ordered_indices` and presentation `frame_indices`
- ordering (`temporal` vs `shuffled`) and `shuffle_seed`
- optional resize policy
- source path / fps / content hash when provided
- whether frames were loaded

## Integrity note

Temporal-shuffle sensitivity is evidence that predictions depend on frame
**order**. It is **not** sufficient to claim genuine causal reasoning.

## CLI

```bash
magic-vlm-sample-frames --video path/to/clip.mp4 --max-frames 8 \
  --also-shuffled --shuffle-seed 0 --json-out runs/sample.json
```

Use `--load-frames` only when OpenCV decode is needed; default writes indices
and metadata only.
