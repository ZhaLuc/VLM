# Zero-shot baseline experiment

The baseline evaluates an **untouched** VLM on the **fixed** `held_out` split.

## Guarantees

- `training_method: none`, checkpoint `base` or `stub` only
- Deterministic decoding (`do_sample: false`)
- Ordered frames only (`temporal_shuffle: false`)
- Raw responses preserved; parsing is separate and never overwrites raw text
- Every selected example appears in `predictions.jsonl` (no silent drops)
- Held-out IDs are locked in `split_lock.json` from the manifest

## Command

```bash
magic-vlm-baseline --config configs/baseline_stub.yaml --run-id baseline-stub-heldout
# Real model (when CUDA + weights available):
magic-vlm-baseline --config configs/baseline_qwen25vl_7b.yaml --load-frames --allow-download
```

## Outputs (`runs/<run_id>/`)

| File | Contents |
|------|----------|
| `predictions.jsonl` | Per-example raw/parsed/correctness/latency |
| `raw_responses.jsonl` | Raw text artifacts |
| `metrics.json` / `baseline_summary.json` | Overall + per-trick accuracy, parse failures |
| `split_lock.json` | Frozen example/clip/trick IDs for the split |
| `prompt_template.txt` | Prompt template used |
| `BASELINE_IMMUTABLE.json` | Marker that this run is the reference |
| `metadata.json` / `environment.json` | Reproducibility bundle |
| `analysis_metrics.json` / `analysis_report.md` | Failure analysis (see `docs/BASELINE_ANALYSIS.md`) |
| `errors.jsonl` / `successes.jsonl` / `examples_inspectable.jsonl` | Example-level diagnostics |

## Integrity

Do not tune prompts against held_out. Do not overwrite an immutable baseline run;
start a new `run_id` for any change to code, prompt, preprocessing, or model.
