# Wikimedia pilot annotation review

HUMAN APPROVAL: PENDING

Do not treat this file as gold. Commons captions name the experimental
condition; they are not questions or answers. Final `question` /
`ground_truth` stay `HUMAN_FILL_REQUIRED` until you approve.

Shared facts from the local MP4s (all five):

- Transparent inverted cups (load is visible through the cup).
- After the third-cup action, a ball is visible under each of the three cups.
- Each clip includes a late cup-lift reveal.
- These files are therefore **not occluded hidden-state items** as the
  repository defines that task.

Paper: Rieiro, Martinez-Conde, Macknik (2013), PeerJ 1:e19, CC BY 3.0.

| Clip | Proposed Question | Proposed Answer | Evidence | Confidence | Human Decision |
| ---- | ----------------- | --------------- | -------- | ---------- | -------------- |
| peerj_01_19_s003 Standard | Where is the ball that started on top of the right-hand cup immediately after it leaves the cup? | in the magician's hand | Video ~5.51s (ball in open palm) + paper Standard | HIGH for that visible event; not hidden-state | PENDING |
| peerj_01_19_s004 No ball | At the start of the clip, is there a ball on top of the right-hand cup? | no | Opening frame empty right cup + paper No ball | HIGH for start state; not hidden-state | PENDING |
| peerj_01_19_s005 Lift | During the third-cup action, does the magician lift the former top ball to about head/shoulder height? | yes | Video ~5.9–6.1s + paper Lift | HIGH for the lift; pocket destination not verified | PENDING |
| peerj_01_19_s006 Table | After the magician handles the right-hand cup, is there a ball sitting on the table beside the cups (not only under them)? | yes | Video ~6.2s and ~6.7s (fourth ball on table) + paper Table | HIGH for on-table ball; not hidden-state | PENDING |
| peerj_01_19_s007 Drop | During the third-cup action, does the former top ball remain on the table next to the cups? | AMBIGUOUS | Paper: dropped off-screen/floor. Video: downward/off-table blur at ~6.0s; landing not clear | LOW–MEDIUM | PENDING |

All five: `NOT_SUITABLE_FOR_HIDDEN_STATE_TASK` unless you explicitly accept a
visible-event proxy or obtain opaque-cup media.

Strongest first-baseline *visible* candidate: **peerj_01_19_s006** (Table).
Review that clip first. Visual review page:
`reports/wikimedia_clip_review/index.html`.

Full proposals: `data/examples/wikimedia_pilot_annotation_proposals.json`.
Review-ready manifest (labels still unfilled):
`data/examples/wikimedia_pilot_review.jsonl`.
Original template left in place:
`data/examples/wikimedia_pilot_manifest.template.jsonl`.
