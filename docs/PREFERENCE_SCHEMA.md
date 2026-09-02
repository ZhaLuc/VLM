# Preference data representation

Pairwise human preference judgments over free-text mechanism explanations
(Dataset D / Task A). Designed for later **DPO** and **Bradley-Terry** reward
modeling — this stage only defines storage, identity, and validation.

## Record layout

Each JSONL row is one **judgment** over a content **pair**:

| Field group | Contents |
|-------------|----------|
| Identity | `pair_id` (content hash), `judgment_id` (annotator decision) |
| Clip / task | `clip_id`, optional `example_id` / `video`, `task`, `instruction` |
| Candidates | `response_a`, `response_b` — **raw, never normalized** |
| `generation_meta` | Source model(s), checkpoint fields, per-side generation configs |
| `annotation` | `annotator_id`, `timestamp`, `winner`, optional `rationale`, `allow_ties` |
| `provenance` | Required collection provenance |
| `split` | train / val / held_out |

Generation metadata and annotation metadata are **separate objects** in
serialization so training code can consume labels without conflating sampling
knobs.

## Identity

- `pair_id = pref_` + SHA256(clip_id, instruction, response_a, response_b, task)[:24]
- `judgment_id = judg_` + SHA256(pair_id, annotator_id, timestamp)[:24]

Same content pair + different annotators → same `pair_id`, different
`judgment_id`.

## Ties

Default protocol is forced choice (`winner` ∈ `{a,b}`). `winner=tie` requires
`allow_ties=True` on the record **and** validation `allow_ties=True`.

## Validation

```bash
python -c "from magic_vlm.preferences import load_preference_pairs, validate_preference_pairs; \
r=validate_preference_pairs(load_preference_pairs('data/examples/toy_preferences.jsonl')); print(r.format_human())"
```

Or:

```bash
magic-vlm-validate-preferences --prefs data/examples/toy_preferences.jsonl
```

Checks include: identical responses, invalid/missing winner (on parse), missing
provenance (on parse), duplicate `judgment_id`, duplicate `pair_id` (optional
strict mode), multi-annotator disagreement (review).

## Downstream projections (no training)

- `dpo_training_rows` → `{instruction, chosen, rejected, ...}` (skips ties)
- `bradley_terry_rows` → `{response_a, response_b, label∈{+1,-1,0}, ...}`

## Integrity

- Do not collect AI-generated preference labels in this schema stage.
- Do not rewrite candidate text on load/save.
- Do not use `held_out` preferences for reward-model / DPO fitting without an
  explicit research decision.
