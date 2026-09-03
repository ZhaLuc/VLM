"""Tests for research-quality experiment reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from magic_vlm.reporting import (
    EXAMPLE_SELECTION_RULE,
    ReportConfig,
    UNAVAILABLE,
    build_experiment_report,
    detect_run_kind,
    generate_experiment_report,
    render_experiment_report_markdown,
    select_representative_examples,
    ExampleSelectionConfig,
)
from magic_vlm.reward_hacking import RewardHackingConfig, run_reward_hacking

CFG = Path("configs/experiment_report_toy.yaml")
ZERO = Path("tests/fixtures/comparison/zero_shot")
GRPO = Path("tests/fixtures/comparison/grpo")
RH_CFG = Path("configs/reward_hacking_toy.yaml")


def test_detect_run_kinds() -> None:
    assert detect_run_kind(ZERO) == "baseline"
    assert detect_run_kind(GRPO) == "grpo"
    assert detect_run_kind(Path("tests/fixtures/comparison/temporal")) == "temporal"


def test_example_selection_deterministic() -> None:
    rows = [
        {"example_id": "b", "trick_id": "t1", "correct": False, "raw_text": "x"},
        {"example_id": "a", "trick_id": "t1", "correct": True, "raw_text": "y"},
        {"example_id": "c", "trick_id": "t2", "correct": False, "raw_text": "z"},
    ]
    sel = ExampleSelectionConfig(max_successes=2, max_failures=2)
    first = select_representative_examples(rows, sel)
    second = select_representative_examples(rows, sel)
    assert first == second
    assert first["successes"][0]["example_id"] == "a"
    assert {r["example_id"] for r in first["failures"]} == {"b", "c"}


def test_build_report_marks_missing(tmp_path: Path) -> None:
    cfg = ReportConfig.from_yaml(CFG)
    payload = cfg.to_dict()
    payload["output_dir"] = str(tmp_path / "reports")
    report = build_experiment_report(ReportConfig.from_dict(payload))
    assert report["evaluation"]["status"] == "available"
    assert report["generalization"]["status"] == UNAVAILABLE
    assert report["reward_hacking_analysis"]["status"] == UNAVAILABLE
    assert report["training_method"]["status"] == "available"
    assert report["training_method"]["method"] == "grpo"
    assert report["temporal_shuffle"]["status"] == "available"
    assert any("comparison" in u or "generalization" in u for u in report["unresolved_issues"])
    assert report["dimension_legend"]["reasoning_improvement"].startswith("NOT inferred")
    assert "reasoning improved" not in json.dumps(report).lower()


def test_deterministic_write(tmp_path: Path) -> None:
    raw = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    raw["output_dir"] = str(tmp_path / "r1")
    raw["run_id"] = "det_a"
    raw["generated_at"] = "2026-09-02T00:00:00+00:00"
    a = generate_experiment_report(ReportConfig.from_dict(raw))
    raw["output_dir"] = str(tmp_path / "r2")
    raw["run_id"] = "det_b"
    b = generate_experiment_report(ReportConfig.from_dict(raw))
    md_a = Path(a.markdown_path).read_text(encoding="utf-8")
    md_b = Path(b.markdown_path).read_text(encoding="utf-8")
    # Same content aside from paths inside artifact index — compare JSON core sections
    ja = json.loads(Path(a.json_path).read_text(encoding="utf-8"))
    jb = json.loads(Path(b.json_path).read_text(encoding="utf-8"))
    for key in (
        "evaluation",
        "training_method",
        "reward",
        "temporal_shuffle",
        "representative_failures",
        "dimension_legend",
    ):
        assert ja[key] == jb[key]
    # Fingerprint includes run_id/output_dir by design; core scientific content matches.
    assert ja["config_fingerprint"] != jb["config_fingerprint"] or raw.get("run_id")
    assert EXAMPLE_SELECTION_RULE in md_a
    assert "NOT inferred" in md_a
    assert (Path(a.run_dir) / "DISCLAIMER.json").exists()


def test_from_run_dir(tmp_path: Path) -> None:
    cfg = ReportConfig.from_run_dir(
        ZERO,
        output_dir=str(tmp_path / "out"),
        run_id="from_run",
        generated_at="2026-09-02T00:00:00+00:00",
    )
    result = generate_experiment_report(cfg)
    report = result.report
    assert report["representative_failures"]
    assert report["representative_failures"][0]["example_id"] == "toy_held_out_001_q1"
    assert report["representative_successes"] == []


def test_with_reward_hacking_attached(tmp_path: Path) -> None:
    rh_payload = RewardHackingConfig.from_yaml(RH_CFG).to_dict()
    rh_payload["output_dir"] = str(tmp_path / "rh")
    rh_payload["run_id"] = "rh_for_report"
    rh = run_reward_hacking(RewardHackingConfig.from_dict(rh_payload))

    raw = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    raw["output_dir"] = str(tmp_path / "reports")
    raw["run_id"] = "with_rh"
    raw["artifacts"]["reward_hacking"] = rh.run_dir
    result = generate_experiment_report(ReportConfig.from_dict(raw))
    section = result.report["reward_hacking_analysis"]
    assert section["status"] == "available"
    assert section["single_example_is_not_proof"] is True
    md = render_experiment_report_markdown(result.report).lower()
    assert "reasoning improved" not in md
    assert "this proves reward hacking" not in md
    assert "no single example proves" in md


def test_cli(tmp_path: Path) -> None:
    raw = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    raw["output_dir"] = str(tmp_path / "out")
    raw["run_id"] = "cli"
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    from magic_vlm.cli import report_main

    assert report_main(["--config", str(path)]) == 0
    assert (tmp_path / "out" / "cli" / "experiment_report.md").exists()
