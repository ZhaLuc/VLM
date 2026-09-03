# Hidden-state candidate inventory

Inspected 2026-09-03. Frames for the five local Wikimedia MP4s were previously
sampled into `reports/wikimedia_clip_review/stills/` (transparent cups, late
reveal). No other real magic videos are on disk. No `QUALIFIES` row.

Machine-readable copy: `data/examples/hidden_state_candidate_inventory.json`.
Visual page: `reports/hidden_state_candidates/index.html`.
Rubric: `docs/HIDDEN_STATE_ELIGIBILITY.md`.

| Candidate | Source | Duration | Trick/condition | Genuine occlusion? | Temporal dependence? | Ground truth? | Leakage risk | Status |
| --------- | ------ | -------: | --------------- | ------------------ | -------------------- | ------------- | ------------ | ------ |
| peerj_01_19_s003 | Wikimedia Commons / PeerJ 10.7717/peerj.19 S1 | 11.34 s | Standard cups-and-balls, transparent cups | No | Plausible for the third-cup sequence; proposed question is a visible palm transfer | Visible-event only (not hidden-state) | High: transparent cups + late lift | `NOT_SUITABLE` |
| peerj_01_19_s004 | Commons / PeerJ S2 | 10.78 s | No ball | No | Proposed start-state question is readable at t=0 | Opening absence is visible | High: transparent cups + late lift | `NOT_SUITABLE` |
| peerj_01_19_s005 | Commons / PeerJ S3 | 11.11 s | Lift | No | Plausible for the lift action; lift itself is visible | Lift height visible; pocket destination not established on video | High: transparent cups + late lift | `NOT_SUITABLE` |
| peerj_01_19_s006 | Commons / PeerJ S4 | 11.24 s | Table | No | Plausible for third-cup order; extra ball is on the table in view | On-table ball visible | High: transparent cups + visible fourth ball + late lift | `NOT_SUITABLE` |
| peerj_01_19_s007 | Commons / PeerJ S5 | 10.98 s | Drop | No | Weak; landing not clear | Ambiguous (paper: floor; video: off-table blur) | High: transparent cups + late lift | `NOT_SUITABLE` |
| peerj_01_19_s008 | Commons / PeerJ S6 (not downloaded) | ~11 s (Commons) | Stuck | Unknown (not inspected) | Unknown | Not established locally | Unknown; sibling clips were transparent | `MISSING_FROM_REPOSITORY` |
| peerj_01_19_opaque_cups | Paper conditions; no Commons files found besides s003–s008 | unknown | Opaque-cup experimental cells | Unknown | Unknown | Unknown | Unknown | `MISSING_FROM_REPOSITORY` |
| toy_cups_* | Synthetic schema fixtures | n/a | Toy cups questions | Not real footage | n/a | Fixture labels only | n/a | `NOT_SUITABLE` |

Control roles for the five local clips (keep on disk; not gold):

| Clip | Hidden-state | Control roles | Keep in repo? |
| ---- | ------------ | ------------- | ------------- |
| s003 Standard | `NOT_SUITABLE_FOR_HIDDEN_STATE` | `VISIBLE_EVENT_CONTROL`, `TEMPORAL_CONTROL`, `PIPELINE_SMOKE` | yes |
| s004 No ball | `NOT_SUITABLE_FOR_HIDDEN_STATE` | `VISIBLE_EVENT_CONTROL`, `PIPELINE_SMOKE` | yes |
| s005 Lift | `NOT_SUITABLE_FOR_HIDDEN_STATE` | `VISIBLE_EVENT_CONTROL`, `TEMPORAL_CONTROL`, `PIPELINE_SMOKE` | yes |
| s006 Table | `NOT_SUITABLE_FOR_HIDDEN_STATE` | `VISIBLE_EVENT_CONTROL`, `TEMPORAL_CONTROL`, `PIPELINE_SMOKE` | yes (strongest visible-event / smoke clip) |
| s007 Drop | `NOT_SUITABLE_FOR_HIDDEN_STATE` | `PIPELINE_SMOKE` | yes |

s008 is not classified as transparent without local frames. Sibling supplements
s003–s007 were transparent-cup stage clips; do not download s008 expecting
hidden-state gold.

## Recommended first composition

| Milestone | Size | Why |
| --------- | ---: | --- |
| Next (this week) | **5** qualifying clips | Smallest held-out-only set worth scoring; one question each; still tiny |
| Absolute minimum smoke | 3 | Only if 5 cannot be filmed/sourced immediately |
| Later Dataset B | 15–25 | Research-plan target; **not** justified by currently available footage |

Do not count the Wikimedia five toward those numbers.
