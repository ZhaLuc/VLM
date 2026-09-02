from __future__ import annotations

import json
from pathlib import Path

import pytest

from magic_vlm.baseline import (
    BaselineConfigError,
    assert_zero_shot_baseline_config,
    resolve_eval_split,
    run_zero_shot_baseline,
)
from magic_vlm.evaluation import exact_match, is_parse_failure
from magic_vlm.experiment import load_experiment_config
from magic_vlm.inference import parse_answer
from magic_vlm.schemas import Split


def test_parse_answer_heuristics() -> None:
    assert parse_answer("Because of the pass.\nAnswer: left") == "left"
    assert parse_answer("Final answer: Right cup") == "Right cup"
    assert parse_answer("line1\nline2\ncenter") == "center"
    raw = "Keep me\nAnswer: left"
    assert parse_answer(raw) == "left"
    assert raw.startswith("Keep me")


def test_parse_failure_detection() -> None:
    assert is_parse_failure("something", None)
    assert is_parse_failure("something", "")
    assert is_parse_failure("something", "   ")
    assert not is_parse_failure("something", "left")
    assert not is_parse_failure("", "")


def test_resolve_default_split_is_held_out() -> None:
    cfg = load_experiment_config("configs/baseline_stub.yaml")
    assert resolve_eval_split(cfg) is Split.HELD_OUT
    assert cfg.dataset.split == "held_out"


def test_refuses_sampling_and_training() -> None:
    cfg = load_experiment_config("configs/baseline_stub.yaml")
    assert_zero_shot_baseline_config(cfg)
    from magic_vlm.experiment import experiment_config_from_dict

    payload = cfg.to_dict()
    payload["generation"]["do_sample"] = True
    sampled = experiment_config_from_dict(payload)
    with pytest.raises(BaselineConfigError, match="do_sample"):
        assert_zero_shot_baseline_config(sampled)

    payload2 = cfg.to_dict()
    payload2["training_method"] = "dpo"
    payload2["baseline_immutable"] = False
    payload2["checkpoint"]["kind"] = "post_trained"
    trained = experiment_config_from_dict(payload2)
    with pytest.raises(BaselineConfigError):
        assert_zero_shot_baseline_config(trained)


def test_smoke_baseline_held_out(tmp_path: Path) -> None:
    cfg = load_experiment_config("configs/baseline_stub.yaml")
    payload = cfg.to_dict()
    payload["output_dir"] = str(tmp_path / "runs")
    from magic_vlm.experiment import experiment_config_from_dict

    cfg = experiment_config_from_dict(payload)
    result = run_zero_shot_baseline(cfg, run_id="stub-heldout", load_frames=False)
    assert result.split == "held_out"
    assert result.immutable is True
    assert result.summary.n_examples == 1
    assert len(result.predictions) == result.summary.n_examples
    assert len(result.artifacts) == result.summary.n_examples
    # Stub does not match gold; still counted (not excluded).
    assert result.summary.overall_accuracy == 0.0
    assert result.summary.n_correct == 0
    assert "cups_ball_v2" in result.summary.per_trick_accuracy
    assert result.summary.per_trick_counts["cups_ball_v2"] == 1

    run_dir = Path(result.run_dir)
    preds = list(run_dir.glob("predictions.jsonl"))
    assert preds
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["raw_text"]
    assert "parsed_answer" in rows[0]
    assert (run_dir / "BASELINE_IMMUTABLE.json").exists()
    lock = json.loads((run_dir / "split_lock.json").read_text(encoding="utf-8"))
    assert lock["example_ids"] == ["toy_held_out_001_q1"]
    assert lock["split"] == "held_out"

    # Metric consistency: accuracy == n_correct / n
    assert result.summary.overall_accuracy == result.summary.n_correct / result.summary.n_examples


def test_cli_baseline(tmp_path: Path) -> None:
    from magic_vlm.cli import baseline_main

    cfg_text = Path("configs/baseline_stub.yaml").read_text(encoding="utf-8")
    cfg_text = cfg_text.replace("output_dir: runs", f"output_dir: {tmp_path.as_posix()}/runs")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    code = baseline_main(["--config", str(cfg_path), "--run-id", "cli-base"])
    assert code == 0
    assert (tmp_path / "runs" / "cli-base" / "predictions.jsonl").exists()


def test_exact_match_still_works() -> None:
    assert exact_match(" Left ", "left")
    assert not exact_match("", "left")
