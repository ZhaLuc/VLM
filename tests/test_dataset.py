from __future__ import annotations

import json
from pathlib import Path

import pytest

from magic_vlm.dataset import (
    SplitBoundaryError,
    assert_no_held_out,
    examples_for_clip,
    filter_split,
    iter_for_stage,
    load_manifest,
    load_split,
    write_manifest,
)
from magic_vlm.schemas import (
    ExampleRecord,
    Provenance,
    SchemaError,
    Split,
    TaskType,
    TemporalSpan,
    VideoRef,
    validate_manifest_records,
)


def _prov(**kwargs) -> Provenance:
    base = {"source": "unit_test"}
    base.update(kwargs)
    return Provenance(**base)


def _example(
    example_id: str,
    split: Split,
    *,
    clip_id: str | None = None,
    question: str = "Which cup?",
    ground_truth: str = "left",
    camera_id: str = "cam_front",
    video_path: str | None = None,
) -> ExampleRecord:
    cid = clip_id or example_id
    return ExampleRecord(
        example_id=example_id,
        clip_id=cid,
        trick_id="cups",
        performer_id="a",
        camera_id=camera_id,
        video=VideoRef(path=video_path or f"{cid}.mp4", num_frames=8),
        task=TaskType.HIDDEN_STATE,
        question=question,
        ground_truth=ground_truth,
        split=split,
        provenance=_prov(),
    )


def test_roundtrip_manifest(tmp_path: Path) -> None:
    records = [_example("e1", Split.TRAIN), _example("e2", Split.HELD_OUT)]
    path = tmp_path / "m.jsonl"
    write_manifest(path, records)
    loaded = load_manifest(path)
    assert [r.example_id for r in loaded] == ["e1", "e2"]
    assert loaded[1].split is Split.HELD_OUT
    assert loaded[0].ground_truth == "left"
    assert "answer" not in loaded[0].to_dict()


def test_filter_and_boundary() -> None:
    records = [_example("t", Split.TRAIN), _example("h", Split.HELD_OUT)]
    assert len(filter_split(records, Split.TRAIN)) == 1
    with pytest.raises(SplitBoundaryError):
        list(iter_for_stage(records, stage="training"))
    allowed = list(iter_for_stage(records, stage="baseline", allow_held_out=True))
    assert {r.example_id for r in allowed} == {"t", "h"}


def test_assert_no_held_out() -> None:
    with pytest.raises(SplitBoundaryError):
        assert_no_held_out([_example("h", Split.HELD_OUT)], context="unit")


def test_from_dict_accepts_string_split() -> None:
    raw = {
        "example_id": "x",
        "clip_id": "clip_x",
        "split": "val",
        "video": {"path": "x.mp4"},
        "question": "q",
        "trick_id": "t",
        "performer_id": "p",
        "camera_id": "cam",
        "ground_truth": "Left Cup",
        "task": "hidden_state",
        "provenance": {"source": "unit"},
    }
    record = ExampleRecord.from_dict(raw)
    assert record.split is Split.VAL
    assert record.ground_truth == "Left Cup"  # not normalized
    assert json.loads(json.dumps(record.to_dict()))["split"] == "val"


def test_legacy_answer_alias_preserves_value() -> None:
    raw = {
        "example_id": "x",
        "clip_id": "c",
        "split": "train",
        "video": {"path": "x.mp4"},
        "question": "q",
        "trick_id": "t",
        "performer_id": "p",
        "camera_id": "cam",
        "answer": "  Right  ",
        "provenance": {"source": "legacy"},
    }
    record = ExampleRecord.from_dict(raw)
    assert record.ground_truth == "  Right  "
    assert record.answer == "  Right  "


def test_toy_manifest_loads() -> None:
    records = load_manifest("data/examples/toy_manifest.jsonl")
    assert len(records) == 4
    train = load_split("data/examples/toy_manifest.jsonl", Split.TRAIN)
    assert len(train) == 2
    variants = examples_for_clip(records, "toy_cups_train")
    assert len(variants) == 2
    assert {v.question_variant for v in variants} == {"canonical", "paraphrase_1"}
    assert variants[0].ground_truth == variants[1].ground_truth == "left"


def test_missing_required_fields() -> None:
    with pytest.raises(SchemaError, match="missing required field"):
        ExampleRecord.from_dict(
            {
                "example_id": "x",
                "split": "train",
                "video": {"path": "x.mp4"},
                "question": "q",
                "provenance": {"source": "u"},
            }
        )


def test_invalid_task_and_split() -> None:
    base = {
        "example_id": "x",
        "clip_id": "c",
        "trick_id": "t",
        "performer_id": "p",
        "camera_id": "cam",
        "video": {"path": "x.mp4"},
        "question": "q",
        "ground_truth": "left",
        "provenance": {"source": "u"},
    }
    with pytest.raises(SchemaError, match="invalid task"):
        ExampleRecord.from_dict({**base, "split": "train", "task": "telepathy"})
    with pytest.raises(SchemaError, match="invalid split"):
        ExampleRecord.from_dict({**base, "split": "test", "task": "hidden_state"})


def test_malformed_temporal_ranges() -> None:
    with pytest.raises(SchemaError, match="start_s"):
        TemporalSpan(start_s=3.0, end_s=1.0)
    with pytest.raises(SchemaError, match="start_frame"):
        TemporalSpan(start_frame=10, end_frame=2)


def test_duplicate_example_ids() -> None:
    records = [
        _example("dup", Split.TRAIN, clip_id="c1"),
        _example("dup", Split.TRAIN, clip_id="c2", video_path="c2.mp4"),
    ]
    with pytest.raises(SchemaError, match="duplicate example_id"):
        validate_manifest_records(records)


def test_clip_variants_must_share_split_and_video() -> None:
    a = _example("a", Split.TRAIN, clip_id="same", video_path="a.mp4")
    b = _example("b", Split.VAL, clip_id="same", video_path="a.mp4")
    with pytest.raises(SchemaError, match="multiple splits"):
        validate_manifest_records([a, b])
    c = _example("c", Split.TRAIN, clip_id="same", video_path="other.mp4")
    with pytest.raises(SchemaError, match="multiple video paths"):
        validate_manifest_records([a, c])


def test_hidden_state_requires_ground_truth() -> None:
    with pytest.raises(SchemaError, match="ground_truth"):
        ExampleRecord(
            example_id="x",
            clip_id="c",
            trick_id="t",
            performer_id="p",
            camera_id="cam",
            video=VideoRef(path="x.mp4"),
            task=TaskType.HIDDEN_STATE,
            question="q",
            ground_truth=None,
            split=Split.TRAIN,
            provenance=_prov(),
        )


def test_explanation_task_allows_missing_ground_truth() -> None:
    record = ExampleRecord(
        example_id="exp1",
        clip_id="c",
        trick_id="t",
        performer_id="p",
        camera_id="cam",
        video=VideoRef(path="x.mp4"),
        task=TaskType.EXPLANATION,
        question="How was this done?",
        ground_truth=None,
        split=Split.TRAIN,
        provenance=_prov(),
        notes="future preference/explanation path",
    )
    assert record.ground_truth is None
    assert record.task is TaskType.EXPLANATION


def test_optional_fields_and_canonical_answer_preserved(tmp_path: Path) -> None:
    record = ExampleRecord(
        example_id="opt1",
        clip_id="clip_opt",
        trick_id="t",
        performer_id="p",
        camera_id="cam_side",
        video=VideoRef(path="opt.mp4", content_hash="abc"),
        task=TaskType.HIDDEN_STATE,
        question="Where is the card?",
        ground_truth="Top Pocket",
        justification="Filmed with known placement.",
        temporal=TemporalSpan(start_s=0.5, end_s=1.5, start_frame=4, end_frame=12),
        notes="keep casing",
        question_variant="canonical",
        split=Split.TRAIN,
        provenance=_prov(created_by="annotator"),
    )
    path = tmp_path / "opt.jsonl"
    write_manifest(path, [record])
    loaded = load_manifest(path)[0]
    assert loaded.ground_truth == "Top Pocket"
    assert loaded.justification.startswith("Filmed")
    assert loaded.temporal is not None
    assert loaded.temporal.start_frame == 4
