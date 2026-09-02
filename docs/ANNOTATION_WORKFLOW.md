# Minimal preference annotation workflow

Research tooling for collecting ~60–100+ human pairwise judgments over
mechanism explanations. Not a commercial annotation platform.

## Interface

CLI session (`magic-vlm-annotate`):

1. Load candidate queue JSONL (clip, video path, instruction, raw A/B, generation meta)
2. Skip items this annotator already judged (resume-safe)
3. Display rubric + task + raw Response A/B (never rewritten)
4. Expose / optionally open the local video file
5. Collect forced choice `a`/`b` and optional rationale
6. Append an immutable `PreferencePair` to the judgment store

Scripted (tests / smoke):

```bash
magic-vlm-annotate --annotator ann_smoke --queue data/examples/toy_annotation_queue.jsonl \
  --out data/annotations/smoke_preferences.jsonl --no-open-video --limit 1 \
  --winner a --rationale "Palm-and-load cites a concrete mechanism."
```

Interactive:

```bash
magic-vlm-annotate --annotator lucas \
  --queue data/examples/toy_annotation_queue.jsonl \
  --out data/annotations/preferences.jsonl \
  --rubric configs/annotation_rubric.yaml
```

Prompts: `a` / `b` / `s` (skip) / `q` (quit), then optional rationale.

## Rubric

`configs/annotation_rubric.yaml` (`explanation_pref_v1`):

- Prefer: factual correctness, evidence from the demonstration, mechanism specificity
- Do **not** prefer: verbosity, confidence, fluent unsupported invention

Documented rules beyond the research plan are listed in that YAML `notes` field.

## Storage

- Queue: `data/examples/toy_annotation_queue.jsonl` (candidates only)
- Judgments: append-only JSONL (default `data/annotations/preferences.jsonl`)
- Session summary: sibling `*.session.json`
- Duplicate `judgment_id` and same `(annotator_id, pair_id)` are refused
- Candidate `response_a` / `response_b` are copied verbatim into the preference record

## Integrity

- No AI ranking
- No user accounts / cloud sync / databases
- Do not edit candidate text during annotation
- Validate later with `magic-vlm-validate-preferences --prefs ...`
