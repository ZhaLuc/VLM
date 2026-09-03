"""Tests for cross-method comparative evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from magic_vlm.comparison import (
    ComparisonConfig,
    ComparisonError,
    MethodSpec,
    ProtocolSpec,
    align_methods,
    check_protocol_compatibility,
    compute_deltas,
    compute_seen_sets,
    example_axis_flags,
    load_method_predictions,
    locked_held_out_examples,
    run_comparison,
)
from magic_vlm.dataset import load_manifest
from magic_vlm.schemas import Split

MANIFEST = Path("data/examples/toy_manifest.jsonl")
ZERO = Path("tests/fixtures/comparison/zero_shot")
GRPO = Path("tests/fixtures/comparison/grpo")
TEMPORAL = Path("tests/fixtures/comparison/temporal/temporal_shuffle_summary.json")


def test_locked_held_out_and_unseen_axes() -> None:
    examples = load_manifest(MANIFEST)
    locked = locked_held_out_examples(
        examples, split=Split.HELD_OUT.value, task="hidden_state"
    )
    assert len(locked) == 1
    seen = compute_seen_sets(examples)
    flags = example_axis_flags(locked[0], seen)
    assert flags["unseen_trick"] is True
    assert flags["unseen_performer"] is True
    assert flags["unseen_camera"] is True
    assert flags["known_trick_variation"] is False


def test_load_baseline_and_grpo_formats() -> None:
    z = load_method_predictions(
        MethodSpec(method_id="z", kind="zero_shot", run_dir=str(ZERO))
    )
    g = load_method_predictions(MethodSpec(method_id="g", kind="grpo", run_dir=str(GRPO)))
    assert z["toy_held_out_001_q1"].correct is False
    assert g["toy_held_out_001_q1"].correct is True
    assert g["toy_held_out_001_q1"].reward == 1.0


def test_alignment_and_missing_visible(tmp_path: Path) -> None:
    examples = load_manifest(MANIFEST)
    locked = locked_held_out_examples(
        examples, split="held_out", task="hidden_state"
    )
    z = load_method_predictions(
        MethodSpec(method_id="zero_shot", kind="zero_shot", run_dir=str(ZERO))
    )
    aligned, coverage = align_methods(
        locked, {"zero_shot": z, "incomplete": {}}, require_full_coverage=False
    )
    assert aligned[0]["methods"]["incomplete"]["missing"] is True
    assert coverage["methods"]["incomplete"]["n_missing"] == 1
    with pytest.raises(ComparisonError, match="missing"):
        align_methods(locked, {"incomplete": {}}, require_full_coverage=True)


def test_incompatible_protocols_labeled() -> None:
    methods = (
        MethodSpec(
            method_id="a",
            kind="zero_shot",
            generation_policy={"temperature": 0.0},
        ),
        MethodSpec(
            method_id="b",
            kind="grpo",
            generation_policy={"temperature": 0.7},
        ),
    )
    with pytest.raises(ComparisonError, match="Incompatible"):
        check_protocol_compatibility(methods, allow_incompatible=False)
    report = check_protocol_compatibility(methods, allow_incompatible=True)
    assert report["compatible"] is False
    assert report["n_distinct_policies"] == 2


def test_deltas_known() -> None:
    aggregates = {
        "zero_shot": {"accuracy": 0.0, "mean_reward": 0.0},
        "grpo": {"accuracy": 1.0, "mean_reward": 1.0},
    }
    deltas = compute_deltas(aggregates, reference_method_id="zero_shot")
    assert deltas["deltas"]["grpo"]["accuracy_delta_vs_reference"] == 1.0
    assert deltas["deltas"]["grpo"]["mean_reward_delta_vs_reference"] == 1.0


def test_run_comparison_synthetic(tmp_path: Path) -> None:
    cfg = ComparisonConfig(
        protocol=ProtocolSpec(
            manifest=str(MANIFEST),
            require_full_coverage=True,
            allow_incompatible_protocols=False,
        ),
        methods=(
            MethodSpec(
                method_id="zero_shot",
                kind="zero_shot",
                run_dir=str(ZERO),
                generation_policy={"temperature": 0.0, "max_new_tokens": 32},
            ),
            MethodSpec(
                method_id="grpo",
                kind="grpo",
                run_dir=str(GRPO),
                temporal_summary_path=str(TEMPORAL),
                generation_policy={"temperature": 0.0, "max_new_tokens": 32},
            ),
        ),
        output_dir=str(tmp_path / "comparison"),
        run_id="synth",
        reference_method_id="zero_shot",
    )
    result = run_comparison(cfg)
    report = result.report
    assert report["aggregates"]["zero_shot"]["accuracy"] == 0.0
    assert report["aggregates"]["grpo"]["accuracy"] == 1.0
    assert report["deltas"]["deltas"]["grpo"]["accuracy_delta_vs_reference"] == 1.0
    assert report["protocol_compatibility"]["compatible"] is True
    assert report["temporal"]["grpo"]["ordered_accuracy"] == 1.0
    assert report["reward"]["grpo"]["training_reward_stats"]["mean_reward"] == 0.42
    # Dimension legend refuses a reasoning collapse
    assert "NOT inferred" in report["dimension_legend"]["reasoning_improvement"]

    aligned = Path(result.aligned_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(aligned) == 1
    row = json.loads(aligned[0])
    assert row["methods"]["zero_shot"]["correct"] is False
    assert row["methods"]["grpo"]["correct"] is True
    assert row["axes"]["unseen_trick"] is True
    assert (Path(result.run_dir) / "comparison_report.md").exists()
    assert (Path(result.run_dir) / "DISCLAIMER.json").exists()


def test_rejects_unimplemented_sft(tmp_path: Path) -> None:
    cfg = ComparisonConfig(
        protocol=ProtocolSpec(manifest=str(MANIFEST), require_full_coverage=False),
        methods=(
            MethodSpec(
                method_id="sft_arm",
                kind="sft",
                predictions_path=str(ZERO / "predictions.jsonl"),
            ),
        ),
        output_dir=str(tmp_path / "cmp"),
        run_id="sft",
    )
    with pytest.raises(ComparisonError, match="not implemented"):
        run_comparison(cfg)


def test_yaml_config_and_cli(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/compare_methods_toy.yaml").read_text(encoding="utf-8"))
    raw["output_dir"] = str(tmp_path / "out")
    raw["run_id"] = "cli_toy"
    cfg_path = tmp_path / "cmp.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    from magic_vlm.cli import compare_methods_main

    assert compare_methods_main(["--config", str(cfg_path)]) == 0
    metrics = json.loads(
        (tmp_path / "out" / "cli_toy" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["aggregates"]["grpo"]["n_correct"] == 1
