from __future__ import annotations

import json
from pathlib import Path

import pytest

from magic_vlm.dataset import (
    SplitBoundaryError,
    assert_no_held_out,
    filter_split,
    iter_for_stage,
    load_manifest,
    write_manifest,
)
from magic_vlm.schemas import ExampleRecord, Split, VideoRef


def _example(example_id: str, split: Split) -> ExampleRecord:
    return ExampleRecord(
        example_id=example_id,
        split=split,
        video=VideoRef(path=f"{example_id}.mp4", num_frames=8),
        question="Which cup?",
        answer="left",
        trick_id="cups",
        performer_id="a",
    )


def test_roundtrip_manifest(tmp_path: Path) -> None:
    records = [_example("e1", Split.TRAIN), _example("e2", Split.HELD_OUT)]
    path = tmp_path / "m.jsonl"
    write_manifest(path, records)
    loaded = load_manifest(path)
    assert [r.example_id for r in loaded] == ["e1", "e2"]
    assert loaded[1].split is Split.HELD_OUT


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
        "split": "val",
        "video": {"path": "x.mp4"},
        "question": "q",
        "trick_id": "t",
        "performer_id": "p",
    }
    record = ExampleRecord.from_dict(raw)
    assert record.split is Split.VAL
    assert json.loads(json.dumps(record.to_dict()))["split"] == "val"
