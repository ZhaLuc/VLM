"""Tests for reward-hacking / reward–quality divergence diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from magic_vlm.reward_hacking import (
    QUAD_HIGH_REWARD_LOW_ACCURACY,
    RewardHackingConfig,
    RewardHackingError,
    assign_quadrant,
    run_reward_hacking,
)

FIX = Path("tests/fixtures/reward_hacking")
CFG = Path("configs/reward_hacking_toy.yaml")


def test_quadrant_assignment() -> None:
    assert (
        assign_quadrant(reward=0.9, correct=False, high_cut=0.7, low_cut=0.3)
        == QUAD_HIGH_REWARD_LOW_ACCURACY
    )
    assert (
        assign_quadrant(reward=0.1, correct=True, high_cut=0.7, low_cut=0.3)
        == "low_reward_high_accuracy"
    )
    assert assign_quadrant(reward=None, correct=True, high_cut=0.7, low_cut=0.3) == "unscored"


def test_synthetic_divergence(tmp_path: Path) -> None:
    cfg = RewardHackingConfig.from_yaml(CFG)
    payload = cfg.to_dict()
    payload["output_dir"] = str(tmp_path / "rh")
    payload["run_id"] = "synth"
    result = run_reward_hacking(RewardHackingConfig.from_dict(payload))
    report = result.report

    assert report["n_aligned"] == 4
    assert report["before_after"]["accuracy_before"] == 1.0
    assert report["before_after"]["accuracy_after"] == 0.0
    assert report["before_after"]["mean_rm_delta"] is not None
    assert report["before_after"]["mean_rm_delta"] > 0
    assert report["before_after"]["accuracy_delta"] == -1.0
    assert report["quadrant_counts_after"].get(QUAD_HIGH_REWARD_LOW_ACCURACY, 0) == 4
    assert report["single_example_is_not_proof"] is True
    assert report["human_evaluation"]["available"] is True

    codes = {f["code"] for f in report["findings"]}
    assert "possible_reward_up_accuracy_flat_or_down" in codes
    assert QUAD_HIGH_REWARD_LOW_ACCURACY in codes

    # Exports exist and mark not_proof
    hr = list(
        Path(result.run_dir).joinpath("high_reward_low_accuracy.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(hr) == 4
    row = json.loads(hr[0])
    assert row["not_proof"] is True
    assert row["after"]["not_proof"] is True
    assert (Path(result.run_dir) / "DISCLAIMER.json").exists()

    # Heuristics fire on stuffed incorrect after responses
    tagged = json.loads(
        (Path(result.run_dir) / "heuristic_flagged.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert tagged["tags"] or tagged["after"]["tags"]


def test_objective_tied_note(tmp_path: Path) -> None:
    cfg = RewardHackingConfig.from_yaml(CFG)
    payload = cfg.to_dict()
    payload["output_dir"] = str(tmp_path / "rh")
    payload["run_id"] = "obj"
    payload["quadrant_reward"] = "objective"
    payload["rm_scores_path"] = None
    result = run_reward_hacking(RewardHackingConfig.from_dict(payload))
    codes = {f["code"] for f in result.report["findings"]}
    assert "objective_reward_tied_to_accuracy" in codes


def test_human_unavailable_recorded(tmp_path: Path) -> None:
    cfg = RewardHackingConfig.from_yaml(CFG)
    payload = cfg.to_dict()
    payload["output_dir"] = str(tmp_path / "rh")
    payload["run_id"] = "nohuman"
    payload["human_labels_path"] = None
    result = run_reward_hacking(RewardHackingConfig.from_dict(payload))
    assert result.report["human_evaluation"]["available"] is False
    codes = {f["code"] for f in result.report["findings"]}
    assert "human_evaluation_unavailable" in codes


def test_missing_raises_by_default(tmp_path: Path) -> None:
    # Empty after predictions → missing coverage
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "predictions.jsonl").write_text("", encoding="utf-8")
    cfg = RewardHackingConfig.from_yaml(CFG)
    payload = cfg.to_dict()
    payload["output_dir"] = str(tmp_path / "rh")
    payload["run_id"] = "miss"
    payload["after"]["run_dir"] = str(empty)
    with pytest.raises(Exception, match="missing"):
        run_reward_hacking(RewardHackingConfig.from_dict(payload))


def test_cli(tmp_path: Path) -> None:
    raw = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    raw["output_dir"] = str(tmp_path / "out")
    raw["run_id"] = "cli"
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    from magic_vlm.cli import analyze_reward_hacking_main

    assert analyze_reward_hacking_main(["--config", str(path)]) == 0
    assert (tmp_path / "out" / "cli" / "reward_hacking_metrics.json").exists()
