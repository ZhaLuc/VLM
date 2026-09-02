"""Tests for the common experiment dispatcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from magic_vlm.experiment import load_experiment_config
from magic_vlm.runner import (
    ExperimentDispatchError,
    list_supported_experiments,
    load_raw_config,
    resolve_experiment_type,
    run_experiment,
    validate_dispatch_config,
)
from magic_vlm.utils import RunDirectoryError, allocate_run_directory


def test_supported_types() -> None:
    types = list_supported_experiments()
    assert set(types) == {"baseline", "temporal_shuffle", "dpo", "grpo", "reward_model"}


def test_valid_config_load_and_resolve() -> None:
    raw = load_raw_config("configs/baseline_stub.yaml")
    assert resolve_experiment_type(raw) == "baseline"
    validate_dispatch_config(raw, experiment_type="baseline")
    cfg = load_experiment_config("configs/baseline_stub.yaml")
    assert cfg.experiment_type == "baseline"


def test_invalid_config_missing_fields() -> None:
    with pytest.raises(ExperimentDispatchError, match="missing"):
        validate_dispatch_config(
            {"experiment_type": "baseline", "name": "x"},
            experiment_type="baseline",
        )


def test_invalid_experiment_type() -> None:
    with pytest.raises(ExperimentDispatchError, match="Unsupported"):
        resolve_experiment_type({"experiment_type": "ppo"})


def test_grpo_config_resolves() -> None:
    raw = load_raw_config("configs/grpo_smoke_text.yaml")
    assert resolve_experiment_type(raw) == "grpo"
    validate_dispatch_config(raw, experiment_type="grpo")


def test_dpo_rejects_held_out_checkpoint_selection() -> None:
    raw = load_raw_config("configs/dpo_smoke_text.yaml")
    raw["checkpoint_selection"] = "held_out_best"
    with pytest.raises(ExperimentDispatchError, match="held-out"):
        validate_dispatch_config(raw, experiment_type="dpo")


def test_grpo_rejects_held_out_checkpoint_selection() -> None:
    raw = load_raw_config("configs/grpo_smoke_text.yaml")
    raw["checkpoint_selection"] = "held_out_best"
    with pytest.raises(ExperimentDispatchError, match="held-out"):
        validate_dispatch_config(raw, experiment_type="grpo")


def test_unique_run_directory(tmp_path: Path) -> None:
    allocate_run_directory(tmp_path, "r1")
    with pytest.raises(RunDirectoryError, match="already exists"):
        allocate_run_directory(tmp_path, "r1")


def test_dispatch_baseline_smoke(tmp_path: Path) -> None:
    cfg_text = Path("configs/baseline_stub.yaml").read_text(encoding="utf-8")
    cfg_text = cfg_text.replace("output_dir: runs", f"output_dir: {tmp_path.as_posix()}/runs")
    cfg_path = tmp_path / "baseline.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    result = run_experiment(cfg_path, run_id="dispatch-baseline")
    assert result.status == "ok"
    assert result.experiment_type == "baseline"
    run_dir = Path(result.run_dir)
    assert (run_dir / "status.json").exists()
    assert (run_dir / "dispatch_config.json").exists()
    assert (run_dir / "dispatch_result.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.jsonl").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "config.yaml").exists()
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "ok"
    assert status["config_hash"]
    assert "n_examples" in result.metrics or "overall_accuracy" in result.metrics

    # Refuse overwrite of the same run id.
    with pytest.raises(RunDirectoryError):
        run_experiment(cfg_path, run_id="dispatch-baseline")


def test_dispatch_temporal_shuffle(tmp_path: Path) -> None:
    cfg_text = Path("configs/temporal_shuffle_stub.yaml").read_text(encoding="utf-8")
    cfg_text = cfg_text.replace("output_dir: runs", f"output_dir: {tmp_path.as_posix()}/runs")
    cfg_path = tmp_path / "temporal.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    result = run_experiment(cfg_path, run_id="dispatch-temporal")
    assert result.status == "ok"
    assert result.experiment_type == "temporal_shuffle"
    assert (Path(result.run_dir) / "temporal_shuffle_pairs.jsonl").exists()


def test_failure_propagation_and_record(tmp_path: Path) -> None:
    bad = {
        "experiment_type": "baseline",
        "name": "bad_baseline",
        "training_method": "none",
        "baseline_immutable": True,
        "output_dir": str(tmp_path / "runs"),
        "model": {"model_id": "stub/echo"},
        "checkpoint": {"kind": "stub"},
        "dataset": {"manifest": str(tmp_path / "missing_manifest.jsonl"), "split": "held_out"},
        "generation": {"max_new_tokens": 8, "do_sample": False},
        "device": {"preference": "cpu"},
        "seed_config": {"seed": 0},
    }
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(bad), encoding="utf-8")

    with pytest.raises(Exception):
        run_experiment(cfg_path, run_id="fail-run", propagate_errors=True)

    # Soft mode still records failure without raising.
    result = run_experiment(cfg_path, run_id="fail-run-soft", propagate_errors=False)
    assert result.status == "failed"
    assert result.error_type
    run_dir = Path(result.run_dir)
    assert run_dir.exists()
    assert (run_dir / "failure.json").exists()
    assert (run_dir / "failure_traceback.txt").exists()
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["note"].startswith("Failed run recorded")


def test_repeatable_config_loading() -> None:
    a = load_raw_config("configs/baseline_stub.yaml")
    b = load_raw_config("configs/baseline_stub.yaml")
    assert a == b
    assert resolve_experiment_type(a) == resolve_experiment_type(b)


def test_cli_run_baseline(tmp_path: Path) -> None:
    from magic_vlm.cli import run_main

    cfg_text = Path("configs/baseline_stub.yaml").read_text(encoding="utf-8")
    cfg_text = cfg_text.replace("output_dir: runs", f"output_dir: {tmp_path.as_posix()}/runs")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    code = run_main(["--config", str(cfg_path), "--run-id", "cli-dispatch"])
    assert code == 0
    assert (tmp_path / "runs" / "cli-dispatch" / "dispatch_result.json").exists()


def test_cli_list_types() -> None:
    from magic_vlm.cli import run_main

    assert run_main(["--list-types"]) == 0
