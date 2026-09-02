# Temporal-order diagnostic

Paired evaluation of **the same sampled frames** in temporal vs shuffled order.

This is a **temporal-order diagnostic**. It is **not** proof of causal reasoning
and **must not** be used as training data.

## Guarantees

- Sampling happens once (`ordered_and_shuffled_pair`); shuffle does not resample
- Identical question, prompt (`hidden_state_v1`), model, generation settings, and
  `hidden_state_exact_match` scorer
- Reproducible `shuffle_seed`

## Shuffle method

`magic_vlm.video._shuffle_indices`: Fisher–Yates permutation of the **already
sampled** indices, driven by a 32-bit linear congruential generator
(`state = (1664525 * state + 1013904223) & 0xFFFFFFFF`, `seed & 0xFFFFFFFF`).
Recorded as `lcg_fisher_yates_permutation_of_sampled_indices`.

Some seeds (including `0` on several 4-frame sample sets) yield an identity
permutation. The runner refuses those cases when more than one frame is sampled.
Default configs use `shuffle_seed: 7`.

## Metrics

- Ordered / shuffled accuracy
- Difference (ordered − shuffled)
- Paired counts: both correct, both incorrect, ordered-only, shuffled-only

## Command

```bash
magic-vlm-temporal-shuffle --config configs/temporal_shuffle_stub.yaml --run-id temporal-stub
```

Index-only smoke (no OpenCV / no videos on disk):

```bash
python -m pytest tests/test_temporal_shuffle.py
```

Real clips (decode pixels): add `--load-frames` and a local VLM checkpoint.

Outputs under `runs/<run_id>/`:

- `temporal_shuffle_pairs.jsonl` — per-example ordered/shuffled raw + correctness
- `temporal_shuffle_summary.json` / `metrics.json`
- `temporal_shuffle_metadata.json` — seed, method, sampled indices, model, generation

## Interpretation limits

A drop (or rise) under shuffle shows that **this model, on this split, with
these sampled frames**, is sensitive to presentation order. That does **not**
establish causal mechanism understanding, rule out shortcuts, or justify
training on shuffled clips.
