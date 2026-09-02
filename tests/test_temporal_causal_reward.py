"""Tests for temporal/causal IoU reward and interval parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from magic_vlm.dataset import load_manifest
from magic_vlm.rewards import (
    TEMPORAL_CAUSAL_VERSION,
    TEMPORAL_IOU_ID,
    RewardConfig,
    TemporalCausalReward,
    build_reward,
    compare_hidden_state_and_temporal,
    compute_temporal_causal_value,
)
from magic_vlm.schemas import (
    AnnotationStatus,
    CausalAnnotation,
    ExampleRecord,
    InferenceArtifact,
    Provenance,
    SchemaError,
    Split,
    TaskType,
    TemporalSpan,
    VideoRef,
)
from magic_vlm.temporal_parse import (
    Interval,
    gold_causal_interval,
    interval_iou,
    parse_interval_text,
)


def _prov(**kwargs) -> Provenance:
    return Provenance(source="unit_test", **kwargs)


def _ex_with_causal(
    *,
    status: AnnotationStatus = AnnotationStatus.KNOWN,
    start_s: float = 1.0,
    end_s: float = 1.5,
    unique_cause: bool | None = True,
    example_id: str = "e1",
    gold_label: str = "left",
) -> ExampleRecord:
    return ExampleRecord(
        example_id=example_id,
        clip_id="clip_e1",
        trick_id="t",
        performer_id="p",
        camera_id="cam",
        video=VideoRef(path="e1.mp4", num_frames=24, fps=8.0),
        task=TaskType.HIDDEN_STATE,
        question="Which cup contains the ball?",
        ground_truth=gold_label,
        temporal=TemporalSpan(start_s=0.0, end_s=3.0),
        causal=CausalAnnotation(
            status=status,
            provenance=_prov(collection_notes="test causal"),
            causal_moment=TemporalSpan(start_s=start_s, end_s=end_s, label="load"),
            unique_cause=unique_cause,
            annotator_notes="fixture",
        ),
        split=Split.VAL,
        provenance=_prov(),
    )


def _art(raw: str, example_id: str = "e1") -> InferenceArtifact:
    return InferenceArtifact(
        example_id=example_id,
        model_id="stub",
        prompt="p",
        raw_text=raw,
        parsed_answer=None,
    )


def test_interval_iou_exact_partial_none() -> None:
    a = Interval(1.0, 2.0, "seconds")
    assert interval_iou(a, Interval(1.0, 2.0, "seconds")) == 1.0
    assert abs(interval_iou(a, Interval(1.5, 2.5, "seconds")) - (0.5 / 1.5)) < 1e-9
    assert interval_iou(a, Interval(3.0, 4.0, "seconds")) == 0.0


def test_invalid_intervals() -> None:
    assert Interval(2.0, 1.0, "seconds").is_valid is False
    assert Interval(-1.0, 1.0, "seconds").is_valid is False
    with pytest.raises(ValueError):
        interval_iou(Interval(2.0, 1.0, "seconds"), Interval(0.0, 1.0, "seconds"))
    pred, failed, reason = parse_interval_text("start_s=2.0 end_s=1.0")
    assert failed is True
    assert pred is None
    assert reason == "invalid_predicted_interval"


def test_parse_seconds_and_frames() -> None:
    iv, failed, _ = parse_interval_text("Answer: start_s=1.0 end_s=1.5")
    assert failed is False
    assert iv == Interval(1.0, 1.5, "seconds")
    iv2, failed2, _ = parse_interval_text("frames 4-7")
    assert failed2 is False
    assert iv2 == Interval(4.0, 7.0, "frames")
    _, failed3, reason = parse_interval_text("I am unsure where it happens.")
    assert failed3 is True
    assert reason == "no_interval_parsed"


def test_clip_temporal_never_used_as_gold() -> None:
    ex = ExampleRecord(
        example_id="no_causal",
        clip_id="c",
        trick_id="t",
        performer_id="p",
        camera_id="cam",
        video=VideoRef(path="x.mp4"),
        task=TaskType.HIDDEN_STATE,
        question="q",
        ground_truth="left",
        temporal=TemporalSpan(start_s=0.0, end_s=3.0),
        split=Split.TRAIN,
        provenance=_prov(),
    )
    meta = gold_causal_interval(ex)
    assert meta["eligible"] is False
    assert meta["used_clip_temporal_as_gold"] is False
    assert meta["reason"] == "no_causal_annotation"
    result = TemporalCausalReward().evaluate(_art("start_s=0.0 end_s=3.0", "no_causal"), ex)
    assert result.value == 0.0
    assert result.extras["eligible"] is False


def test_exact_overlap_binary_and_partial() -> None:
    ex = _ex_with_causal()
    art = _art("start_s=1.0 end_s=1.5")
    binary = TemporalCausalReward(mode="binary", iou_threshold=0.5)
    partial = TemporalCausalReward(mode="partial", iou_threshold=0.5)
    b = binary.evaluate(art, ex)
    p = partial.evaluate(art, ex)
    assert b.value == 1.0
    assert b.matched is True
    assert b.extras["iou"] == 1.0
    assert p.value == 1.0
    assert p.extras["annotation_status"] == "known"
    assert p.extras["status_label"] == "objectively_established"
    assert p.extras["causal_provenance"]["source"] == "unit_test"
    assert p.extras["used_clip_temporal_as_gold"] is False
    assert p.extras["salient_action_is_not_causal_proof"] is True


def test_partial_overlap_reward() -> None:
    ex = _ex_with_causal(start_s=1.0, end_s=2.0)
    art = _art("start_s=1.5 end_s=2.5")
    iou = 0.5 / 1.5
    partial = TemporalCausalReward(mode="partial").evaluate(art, ex)
    binary = TemporalCausalReward(mode="binary", iou_threshold=0.5).evaluate(art, ex)
    assert abs(partial.value - iou) < 1e-9
    assert binary.value == 0.0
    assert binary.matched is False
    assert abs(
        compute_temporal_causal_value(
            iou=iou, mode="partial", iou_threshold=0.5, eligible=True, parse_failed=False
        )
        - iou
    ) < 1e-9


def test_no_overlap() -> None:
    ex = _ex_with_causal()
    result = TemporalCausalReward(mode="partial").evaluate(
        _art("start_s=2.5 end_s=3.0"), ex
    )
    assert result.value == 0.0
    assert result.extras["iou"] == 0.0
    assert result.notes == "no_overlap"


def test_ambiguous_not_scored_as_gold() -> None:
    ex = _ex_with_causal(status=AnnotationStatus.AMBIGUOUS)
    result = TemporalCausalReward().evaluate(_art("start_s=1.0 end_s=1.5"), ex)
    assert result.value == 0.0
    assert result.extras["eligible"] is False
    assert result.extras["annotation_status"] == "ambiguous"
    assert result.notes == "ambiguous_annotation"
    assert result.extras["gold_interval"] is not None


def test_researcher_annotated_is_eligible() -> None:
    ex = _ex_with_causal(status=AnnotationStatus.RESEARCHER_ANNOTATED, unique_cause=False)
    result = TemporalCausalReward().evaluate(_art("start_s=1.0 end_s=1.5"), ex)
    assert result.extras["eligible"] is True
    assert result.extras["annotation_status"] == "researcher_annotated"
    assert result.extras["unique_cause"] is False
    assert result.value == 1.0


def test_independent_comparison_with_hidden_state() -> None:
    ex = _ex_with_causal()
    art = InferenceArtifact(
        example_id="e1",
        model_id="stub",
        prompt="p",
        raw_text="Answer: left\nstart_s=1.0 end_s=1.5",
        parsed_answer="left",
    )
    report = compare_hidden_state_and_temporal(art, ex)
    assert report["combined"] is False
    assert report["weighted"] is False
    assert report["hidden_state"]["value"] == 1.0
    assert report["temporal_iou"]["value"] == 1.0
    assert report["hidden_state"]["reward_id"] == "hidden_state_exact_match"
    assert report["temporal_iou"]["reward_id"] == TEMPORAL_IOU_ID


def test_config_build_modes() -> None:
    binary = RewardConfig.from_yaml("configs/reward_temporal_iou.yaml").build()
    partial = RewardConfig.from_yaml("configs/reward_temporal_iou_partial.yaml").build()
    assert isinstance(binary, TemporalCausalReward)
    assert binary.mode == "binary"
    assert binary.version == TEMPORAL_CAUSAL_VERSION
    assert isinstance(partial, TemporalCausalReward)
    assert partial.mode == "partial"
    alias = build_reward("temporal_localization_correctness")
    assert isinstance(alias, TemporalCausalReward)


def test_fixture_manifest_genuine_annotations() -> None:
    examples = load_manifest("data/examples/toy_temporal_causal.jsonl")
    assert len(examples) == 4
    by_id = {ex.example_id: ex for ex in examples}
    known = by_id["tc_known_001"]
    assert known.causal is not None
    assert known.causal.status is AnnotationStatus.KNOWN
    assert known.causal.provenance.source == "synthetic_fixture"
    assert known.temporal is not None
    assert known.causal.causal_moment is not None
    assert (known.temporal.start_s, known.temporal.end_s) != (
        known.causal.causal_moment.start_s,
        known.causal.causal_moment.end_s,
    )

    reward = TemporalCausalReward(mode="binary")
    exact = reward.evaluate(
        InferenceArtifact(
            example_id="tc_known_001",
            model_id="stub",
            prompt="p",
            raw_text="start_s=1.0 end_s=1.5",
        ),
        known,
    )
    assert exact.value == 1.0
    assert exact.extras["status_label"] == "objectively_established"

    ambiguous = by_id["tc_ambiguous_001"]
    amb = reward.evaluate(
        InferenceArtifact(
            example_id="tc_ambiguous_001",
            model_id="stub",
            prompt="p",
            raw_text="start_s=0.9 end_s=1.4",
        ),
        ambiguous,
    )
    assert amb.value == 0.0
    assert amb.extras["annotation_status"] == "ambiguous"

    frames = by_id["tc_known_frames_001"]
    fr = reward.evaluate(
        InferenceArtifact(
            example_id="tc_known_frames_001",
            model_id="stub",
            prompt="p",
            raw_text="frames 4-7",
        ),
        frames,
    )
    assert fr.value == 1.0
    assert fr.extras["gold_interval"]["unit"] == "frames"


def test_causal_schema_requires_status_and_provenance() -> None:
    with pytest.raises(SchemaError, match="status"):
        CausalAnnotation.from_dict({"causal_moment": {"start_s": 0.0, "end_s": 1.0}})
    with pytest.raises(SchemaError, match="provenance"):
        CausalAnnotation.from_dict(
            {
                "status": "known",
                "causal_moment": {"start_s": 0.0, "end_s": 1.0},
            }
        )


def test_cli_compare_objective(tmp_path: Path) -> None:
    from magic_vlm.cli import compare_objective_main

    preds = tmp_path / "preds.jsonl"
    rows = [
        {
            "example_id": "tc_known_001",
            "model_id": "stub",
            "prompt": "p",
            "raw_text": "Answer: left\nstart_s=1.0 end_s=1.5",
            "parsed_answer": "left",
        },
        {
            "example_id": "tc_ambiguous_001",
            "model_id": "stub",
            "prompt": "p",
            "raw_text": "Answer: right\nstart_s=0.9 end_s=1.4",
            "parsed_answer": "right",
        },
    ]
    preds.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "compare.jsonl"
    code = compare_objective_main(
        [
            "--manifest",
            "data/examples/toy_temporal_causal.jsonl",
            "--predictions",
            str(preds),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    lines = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    assert lines[0]["hidden_state"]["value"] == 1.0
    assert lines[0]["temporal_iou"]["value"] == 1.0
    assert lines[1]["temporal_iou"]["extras"]["annotation_status"] == "ambiguous"
    assert lines[0]["combined"] is False
