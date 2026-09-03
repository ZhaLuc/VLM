# Wikimedia clip preparation (first real baseline)

Prepare downloaded Wikimedia Commons OGV files as **pilot/control** footage.
They fail the hidden-state rubric (transparent cups, late reveal). Ground-truth
and questions were **not** invented. Do not gold-label these rows as
`hidden_state`. See `docs/HIDDEN_STATE_ELIGIBILITY.md`.

Pipeline completed here:

```text
Wikimedia source OGV -> local MP4 -> schema-shaped pilot rows -> decode smoke
```

Blocked for a real Qwen2.5-VL run: authored labels, CUDA GPU, local Qwen weights.

## Clips found

Five source files under `data/videos/` (no other `.ogv` / `.mp4` besides these
conversions and `.gitkeep`). Local SHA-1 and byte size match Wikimedia Commons
`imageinfo` for each file.

| clip_id | source file | format | duration (s) | resolution | fps | Commons SHA-1 match | MP4 converted | pipeline |
| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| `peerj_01_19_s003` | `...-s003.ogv` | Ogg Theora | 11.345 | 720×480 | 29.97 | yes | yes | OGV and MP4 decode |
| `peerj_01_19_s004` | `...-s004.ogv` | Ogg Theora | 10.777 | 720×480 | 29.97 | yes | yes | OGV and MP4 decode |
| `peerj_01_19_s005` | `...-s005.ogv` | Ogg Theora | 11.111 | 720×480 | 29.97 | yes | yes | OGV and MP4 decode |
| `peerj_01_19_s006` | `...-s006.ogv` | Ogg Theora | 11.245 | 720×480 | 29.97 | yes | yes | OGV and MP4 decode |
| `peerj_01_19_s007` | `...-s007.ogv` | Ogg Theora | 10.978 | 720×480 | 29.97 | yes | yes | OGV and MP4 decode |

Provenance type: Wikimedia Commons open-access PeerJ supplement
(`10.7717/peerj.19`, CC BY 3.0). File pages and conversion hashes:
`data/provenance/wikimedia_peerj_01_19.json`.

Commons *page captions* (not research labels): s003 Standard / S1; s004 No ball /
S2; s005 Lift / S3; s006 Table / S4; s007 Drop / S5.

FFmpeg printed `Broken file, keyframe not correctly marked` on each OGV. OpenCV
still probed nonzero frame counts and decoded sampled frames.

## Clips converted

All five OGV files converted beside the originals (originals kept):

```text
ffmpeg -hide_banner -y -i INPUT.ogv -an -c:v libx264 -pix_fmt yuv420p
       -preset medium -crf 18 -movflags +faststart OUTPUT.mp4
```

Helper: `python scripts/convert_ogv_to_mp4.py`

This host had no `ffmpeg` on PATH. Conversion used the FFmpeg 7.1 essentials
binary bundled with `imageio-ffmpeg==0.6.0`. Videos are gitignored.

## Clips readable / not readable

Readable: all five MP4s and all five OGVs (`probe_video`, frame decode,
`preprocess_video` with `max_frames=8` uniform sampling).

Not readable: none of the downloaded clips.

## Metadata found / missing

Found (verified locally and/or via Commons API): original URL, filename, license
(CC BY 3.0), duration, resolution, fps, frame count, SHA-1 match, conversion
command, local MP4 path/hash.

Missing (must not be inferred from filename): `trick_id`, `performer_id`,
`camera_id`, research `question`, `ground_truth`. Split `held_out` and task
`hidden_state` are first-baseline pipeline conventions only; confirm or change.

## Pilot manifest

`data/examples/wikimedia_pilot_manifest.template.jsonl` — five `ExampleRecord`
rows. Schema-required research fields are the sentinel `HUMAN_FILL_REQUIRED`
(not labels). Copy, fill, then validate the filled file before any baseline
scoring.

## Validation results

- `magic-vlm-validate --manifest data/examples/wikimedia_pilot_manifest.template.jsonl`
  → PASSED (0 errors, 0 leakage). Media checked on the real MP4s. Review-only:
  `only_held_out_present`, `repeated_trick_performer_combo` (placeholder IDs).
- One-clip preprocess: `magic-vlm-sample-frames --load-frames --also-shuffled` on
  s003 → `frames_loaded=true`, 8 uniform indices from 340 source frames.
- pytest: 212 passed (includes video/validate/dataset + convert helper + template).
- `python scripts/project_health.py`: overall BLOCKED; first_baseline_ready
  PARTIALLY (5 real MP4s; still no CUDA / Qwen). Health does not treat
  `HUMAN_FILL_REQUIRED` as missing gold.

Media/schema can pass while research labels remain unauthored. Do not treat a
pass on the template as a gold hidden-state set.

## First-baseline blockers

| Item | Status |
| --- | --- |
| Real video | yes (local MP4) |
| Working video decode / frame sampling | yes |
| Manifest matching `ExampleRecord` | template only; labels unfilled |
| Ground truth | **missing** |
| Qwen2.5-VL weights | **missing** on this host |
| Working GPU / CUDA torch | **missing** (`torch` is CPU-only) |

## Git policy

`.gitignore` now excludes `data/videos/*.{ogv,mp4,webm,mkv}`. Do not commit
media, weights, or credentials.
