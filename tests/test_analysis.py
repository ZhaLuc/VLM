"""Hand-calculated fixtures for baseline analysis integrity."""

from __future__ import annotations

import json
from pathlib import Path

from magic_vlm.analysis import (
    TAG_PARSE_FAILURE,
    TAG_POSSIBLE_FREQUENCY_SHORTCUT,
    TAG_VISUAL_INPUT_MISSING,
    TAG_WRONG_LABEL,
    analyze_baseline_run,
    analyze_predictions,
    assign_diagnostic_tags,
    format_analysis_report,
    write_analysis_outputs,
)
from magic_vlm.dataset import load_manifest
from magic_vlm.schemas import (
    ExampleRecord,
    Provenance,
    Split,
    TaskType,
    VideoRef,
)


def _ex(
    eid: str,
    *,
    trick: str,
    performer: str,
    camera: str,
    gold: str,
    notes: str | None = None,
) -> ExampleRecord:
    return ExampleRecord(
        example_id=eid,
        clip_id=f"clip_{eid}",
        trick_id=trick,
        performer_id=performer,
        camera_id=camera,
        video=VideoRef(path=f"data/videos/{eid}.mp4", num_frames=8),
        task=TaskType.HIDDEN_STATE,
        question="Which cup contains the ball?",
        ground_truth=gold,
        split=Split.HELD_OUT,
        provenance=Provenance(source="fixture", created_by="test_analysis"),
        notes=notes,
    )


def _pred(
    eid: str,
    *,
    trick: str,
    gold: str,
    parsed: str | None,
    raw: str,
    correct: bool,
    parse_failed: bool = False,
    video_mode: str = "indices_only_no_pixels",
    error: str | None = None,
) -> dict:
    return {
        "example_id": eid,
        "clip_id": f"clip_{eid}",
        "trick_id": trick,
        "split": "held_out",
        "question": "Which cup contains the ball?",
        "ground_truth": gold,
        "raw_text": raw,
        "parsed_answer": parsed,
        "parse_failed": parse_failed,
        "correct": correct,
        "latency_s": 0.01,
        "model_id": "stub/echo",
        "preprocessing": {"video_input_mode": video_mode},
        "error": error,
    }


def test_hand_calculated_fixture_metrics() -> None:
    """Five examples with known aggregates (computed by hand).

    Gold labels: left, left, right, center, left  → majority=left (3/5=0.6)
    Predictions:
      e1 left  → correct
      e2 right → wrong (wrong_label); not majority
      e3 left  → wrong; matches majority gold → possible shortcut tag
      e4 None  → parse failure
      e5 left  → correct
    Hand totals: n=5, correct=2, accuracy=0.4, parse_failures=1
    Per trick cups_a: e1,e2,e3 → 1/3; cups_b: e4,e5 → 1/2
    Per performer A: e1,e2 → 1/2; B: e3,e4,e5 → 1/3
    """
    examples = [
        _ex("e1", trick="cups_a", performer="A", camera="front", gold="left"),
        _ex("e2", trick="cups_a", performer="A", camera="front", gold="left"),
        _ex("e3", trick="cups_a", performer="B", camera="side", gold="right"),
        _ex(
            "e4",
            trick="cups_b",
            performer="B",
            camera="side",
            gold="center",
            notes="Label is ambiguous per annotators.",
        ),
        _ex("e5", trick="cups_b", performer="B", camera="side", gold="left"),
    ]
    predictions = [
        _pred("e1", trick="cups_a", gold="left", parsed="left", raw="Answer: left", correct=True),
        _pred(
            "e2",
            trick="cups_a",
            gold="left",
            parsed="right",
            raw="Answer: right",
            correct=False,
        ),
        _pred(
            "e3",
            trick="cups_a",
            gold="right",
            parsed="left",
            raw="Answer: left",
            correct=False,
        ),
        _pred(
            "e4",
            trick="cups_b",
            gold="center",
            parsed=None,
            raw="I am unsure about the cups.",
            correct=False,
            parse_failed=True,
        ),
        _pred(
            "e5",
            trick="cups_b",
            gold="left",
            parsed="left",
            raw="Answer: left",
            correct=True,
            video_mode="decoded_pixels",
        ),
    ]

    analysis = analyze_predictions(predictions, examples=examples, run_id="fix", split="held_out")

    assert analysis.n_examples == 5
    assert analysis.n_correct == 2
    assert analysis.n_incorrect == 3
    assert analysis.overall_accuracy == 0.4
    assert analysis.n_parse_failures == 1
    assert analysis.parse_failure_rate == 0.2
    assert analysis.integrity["per_trick_n_matches_total"] is True
    assert analysis.integrity["n_correct_plus_incorrect"] is True
    assert analysis.integrity["raw_text_present_for_all"] is True

    by_trick = {g.key: g for g in analysis.per_trick}
    assert by_trick["cups_a"].n == 3
    assert by_trick["cups_a"].n_correct == 1
    assert by_trick["cups_a"].accuracy == 1 / 3
    assert by_trick["cups_b"].n == 2
    assert by_trick["cups_b"].n_correct == 1
    assert by_trick["cups_b"].accuracy == 0.5

    by_perf = {g.key: g for g in analysis.per_performer}
    assert by_perf["A"].accuracy == 0.5
    assert by_perf["B"].n == 3
    assert by_perf["B"].n_correct == 1
    assert by_perf["B"].accuracy == 1 / 3

    by_cam = {g.key: g for g in analysis.per_camera}
    assert by_cam["front"].n_correct == 1
    assert by_cam["side"].n == 3

    dist = analysis.answer_distribution
    assert dist.majority_gold_label == "left"
    assert dist.majority_gold_count == 3
    assert dist.majority_gold_fraction == 0.6
    assert dist.majority_class_baseline_accuracy == 0.6
    assert dist.gold_counts["left"] == 3
    assert dist.gold_counts["right"] == 1
    assert dist.gold_counts["center"] == 1

    assert set(analysis.example_ids_correct) == {"e1", "e5"}
    assert set(analysis.example_ids_incorrect) == {"e2", "e3", "e4"}
    assert analysis.example_ids_parse_failed == ("e4",)

    by_id = {d.example_id: d for d in analysis.diagnoses}
    assert TAG_WRONG_LABEL in by_id["e2"].tags
    assert TAG_VISUAL_INPUT_MISSING in by_id["e2"].tags
    assert TAG_POSSIBLE_FREQUENCY_SHORTCUT in by_id["e3"].tags
    assert TAG_PARSE_FAILURE in by_id["e4"].tags
    assert "ambiguity_or_task_design_note" in by_id["e4"].tags
    # Do not claim reasoning failure tags exist.
    assert all("reasoning" not in t for d in analysis.diagnoses for t in d.tags)


def test_exports_include_every_incorrect(tmp_path: Path) -> None:
    predictions = [
        _pred("a", trick="t", gold="left", parsed="right", raw="Answer: right", correct=False),
        _pred("b", trick="t", gold="left", parsed="left", raw="Answer: left", correct=True),
        _pred(
            "c",
            trick="t",
            gold="right",
            parsed=None,
            raw="???",
            correct=False,
            parse_failed=True,
        ),
    ]
    examples = [
        _ex("a", trick="t", performer="p", camera="c", gold="left"),
        _ex("b", trick="t", performer="p", camera="c", gold="left"),
        _ex("c", trick="t", performer="p", camera="c", gold="right"),
    ]
    analysis = analyze_predictions(predictions, examples=examples, run_id="x")
    paths = write_analysis_outputs(analysis, tmp_path)
    errors = [
        json.loads(line)
        for line in paths["errors"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    successes = [
        json.loads(line)
        for line in paths["successes"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    inspectable = [
        json.loads(line)
        for line in paths["inspectable"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {e["example_id"] for e in errors} == {"a", "c"}
    assert {s["example_id"] for s in successes} == {"b"}
    assert len(inspectable) == 3
    assert all("raw_text" in row for row in inspectable)
    report = paths["report"].read_text(encoding="utf-8")
    assert "Incorrect examples" in report
    assert "`a`" in report and "`c`" in report
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["n_examples"] == 3
    assert metrics["overall_accuracy"] == 1 / 3


def test_assign_tags_do_not_overclaim() -> None:
    tags = assign_diagnostic_tags(
        correct=False,
        parse_failed=False,
        raw_text="Answer: left",
        parsed_answer="left",
        error=None,
        video_input_mode="indices_only_no_pixels",
        majority_gold_label="left",
    )
    assert TAG_POSSIBLE_FREQUENCY_SHORTCUT in tags
    assert TAG_VISUAL_INPUT_MISSING in tags
    assert "reasoning_failure" not in tags


def test_analyze_real_stub_run(tmp_path: Path) -> None:
    from magic_vlm.baseline import run_zero_shot_baseline
    from magic_vlm.experiment import experiment_config_from_dict, load_experiment_config

    cfg = load_experiment_config("configs/baseline_stub.yaml")
    payload = cfg.to_dict()
    payload["output_dir"] = str(tmp_path / "runs")
    cfg = experiment_config_from_dict(payload)
    result = run_zero_shot_baseline(cfg, run_id="analysis-stub", load_frames=False)
    run_dir = Path(result.run_dir)
    assert (run_dir / "analysis_metrics.json").exists()
    assert (run_dir / "analysis_report.md").exists()
    assert (run_dir / "errors.jsonl").exists()
    assert (run_dir / "examples_inspectable.jsonl").exists()

    # Re-run CLI path
    analysis = analyze_baseline_run(
        run_dir,
        manifest="data/examples/toy_manifest.jsonl",
        write=True,
        out_dir=tmp_path / "reanalyze",
    )
    assert analysis.n_examples == 1
    assert analysis.overall_accuracy == 0.0
    assert analysis.n_incorrect == 1
    assert analysis.per_trick[0].key == "cups_ball_v2"
    assert analysis.per_performer[0].key == "performer_b"
    assert analysis.example_ids_incorrect == ("toy_held_out_001_q1",)
    report = format_analysis_report(analysis)
    assert "reasoning failure" not in report.lower() or "not" in report.lower()
    # Stronger: report must include caveats
    assert "does not measure reasoning" in report.lower()


def test_cli_analyze(tmp_path: Path) -> None:
    from magic_vlm.baseline import run_zero_shot_baseline
    from magic_vlm.cli import analyze_main
    from magic_vlm.experiment import experiment_config_from_dict, load_experiment_config

    cfg = load_experiment_config("configs/baseline_stub.yaml")
    payload = cfg.to_dict()
    payload["output_dir"] = str(tmp_path / "runs")
    result = run_zero_shot_baseline(
        experiment_config_from_dict(payload),
        run_id="cli-analyze",
        load_frames=False,
    )
    code = analyze_main(["--run-dir", result.run_dir, "--out-dir", str(tmp_path / "out")])
    assert code == 0
    assert (tmp_path / "out" / "analysis_metrics.json").exists()


def test_manifest_join_uses_toy_metadata() -> None:
    examples = load_manifest("data/examples/toy_manifest.jsonl")
    held = [ex for ex in examples if ex.split is Split.HELD_OUT]
    predictions = [
        _pred(
            "toy_held_out_001_q1",
            trick="cups_ball_v2",
            gold="center",
            parsed="nope",
            raw="Answer: nope",
            correct=False,
        )
    ]
    analysis = analyze_predictions(predictions, examples=held)
    assert analysis.diagnoses[0].performer_id == "performer_b"
    assert analysis.diagnoses[0].camera_id == "cam_side"
