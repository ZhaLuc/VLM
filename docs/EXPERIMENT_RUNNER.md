# Common experiment runner

Config-driven dispatch over **existing** experiment implementations.

## Supported types

| `experiment_type` | Implementation |
|-------------------|----------------|
| `baseline` | `run_zero_shot_baseline` |
| `temporal_shuffle` | `run_temporal_shuffle_experiment` |
| `dpo` | `train_dpo` |
| `grpo` | `train_grpo` |
| `reward_model` | `train_bradley_terry_reward_model` |
| `comparison` | `run_comparison` (locked held-out multi-method report) |

Unsupported (not implemented): `sft`, `ppo`, plugins, distributed jobs.

## What every run records

- Unique run directory (overwrite refused)
- Serialized config + `config_hash`
- `dispatch_config.json` / `dispatch_result.json` / `status.json`
- Existing runner artifacts (predictions, metrics, checkpoints)
- On failure: `failure.json` + `failure_traceback.txt` (no silent retry)

## Sample command

```bash
magic-vlm-run --config configs/baseline_stub.yaml --run-id baseline-dispatch-1
magic-vlm-run --config configs/compare_methods_toy.yaml --run-id compare-dispatch-1
magic-vlm-run --list-types
```

```bash
python scripts/run_experiment.py --config configs/temporal_shuffle_stub.yaml --run-id temporal-1
```

## Integrity

- Does not choose checkpoints from final held-out/test performance
- Does not overwrite prior runs
- Does not invent missing scientific parameters
- Failed runs are retained so numbers remain auditable
