from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from magic_vlm.validate import ValidatorConfig, validate_dataset


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return path


def base_row(
    *,
    example_id: str,
    clip_id: str,
    split: str,
    trick_id: str = "trick_a",
    performer_id: str = "performer_a",
    camera_id: str = "cam_front",
    video_path: str = "videos/a.mp4",
    question: str = "Which cup contains the ball?",
    ground_truth: str = "left",
    fps: float | None = 30.0,
    num_frames: int | None = 60,
    duration_s: float | None = 2.0,
    temporal: dict[str, Any] | None = None,
    content_hash: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    video: dict[str, Any] = {"path": video_path}
    if fps is not None:
        video["fps"] = fps
    if num_frames is not None:
        video["num_frames"] = num_frames
    if duration_s is not None:
        video["duration_s"] = duration_s
    if content_hash is not None:
        video["content_hash"] = content_hash
    row: dict[str, Any] = {
        "example_id": example_id,
        "clip_id": clip_id,
        "trick_id": trick_id,
        "performer_id": performer_id,
        "camera_id": camera_id,
        "video": video,
        "task": "hidden_state",
        "question": question,
        "ground_truth": ground_truth,
        "split": split,
        "provenance": {"source": "validation_fixture"},
    }
    if temporal is not None:
        row["temporal"] = temporal
    row.update(extra)
    return row


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    videos = tmp_path / "videos"
    videos.mkdir()
    # Non-empty placeholder files (decode checked only when OpenCV can open them).
    for name in ("train.mp4", "val.mp4", "held.mp4", "other.mp4"):
        (videos / name).write_bytes(b"not-a-real-video-but-nonempty")
    (videos / "empty.mp4").write_bytes(b"")
    return tmp_path


def _valid_rows(media_root: Path) -> list[dict]:
    return [
        base_row(
            example_id="tr1",
            clip_id="clip_train",
            split="train",
            trick_id="trick_train",
            performer_id="perf_train",
            video_path="videos/train.mp4",
            ground_truth="left",
        ),
        base_row(
            example_id="tr1b",
            clip_id="clip_train",
            split="train",
            trick_id="trick_train",
            performer_id="perf_train",
            video_path="videos/train.mp4",
            question="Where is the ball now?",
            question_variant="paraphrase_1",
            ground_truth="left",
        ),
        base_row(
            example_id="va1",
            clip_id="clip_val",
            split="val",
            trick_id="trick_val",
            performer_id="perf_val",
            video_path="videos/val.mp4",
            ground_truth="right",
        ),
        base_row(
            example_id="ho1",
            clip_id="clip_held",
            split="held_out",
            trick_id="trick_held",
            performer_id="perf_held",
            camera_id="cam_side",
            video_path="videos/held.mp4",
            ground_truth="center",
        ),
    ]


def test_valid_dataset_passes_without_opencv_decode(media_root: Path) -> None:
    manifest = write_jsonl(media_root / "valid.jsonl", _valid_rows(media_root))
    # OpenCV will likely mark placeholder bytes unreadable; disable decode path
    # for the "structurally valid + files exist" case by patching: we use
    # check_media True but accept unreadable as failure — so for valid pass we
    # need either real videos or no media check.
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(root=media_root, check_media=False, fail_on_leakage=True),
    )
    assert report.passed
    assert not report.errors
    assert not report.leakages
    assert any(f.code == "media_check_disabled" for f in report.reviews)


def test_missing_video_is_error(media_root: Path) -> None:
    rows = _valid_rows(media_root)
    rows[0]["video"]["path"] = "videos/does_not_exist.mp4"
    rows[1]["video"]["path"] = "videos/does_not_exist.mp4"
    manifest = write_jsonl(media_root / "missing.jsonl", rows)
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(root=media_root, check_media=True),
    )
    assert not report.passed
    assert any(f.code == "missing_video" for f in report.errors)


def test_empty_video_is_unreadable_error(media_root: Path) -> None:
    rows = [
        base_row(
            example_id="e1",
            clip_id="c1",
            split="train",
            trick_id="t1",
            performer_id="p1",
            video_path="videos/empty.mp4",
        ),
        base_row(
            example_id="e2",
            clip_id="c2",
            split="held_out",
            trick_id="t2",
            performer_id="p2",
            video_path="videos/held.mp4",
            ground_truth="right",
        ),
    ]
    manifest = write_jsonl(media_root / "empty.jsonl", rows)
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(root=media_root, check_media=True),
    )
    assert not report.passed
    assert any(f.code == "unreadable_video_empty" for f in report.errors)


def test_duplicate_example_ids(media_root: Path) -> None:
    rows = _valid_rows(media_root)
    rows.append(dict(rows[0]))
    rows[-1]["example_id"] = "tr1"  # duplicate
    manifest = write_jsonl(media_root / "dup_id.jsonl", rows)
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(root=media_root, check_media=False),
    )
    assert not report.passed
    assert any(f.code in {"duplicate_example_id", "manifest_integrity"} for f in report.errors)


def test_trick_leakage(media_root: Path) -> None:
    rows = _valid_rows(media_root)
    rows[-1]["trick_id"] = "trick_train"  # held_out shares train trick
    manifest = write_jsonl(media_root / "trick_leak.jsonl", rows)
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(root=media_root, check_media=False),
    )
    assert not report.passed
    assert any(f.code == "leakage_trick_id" for f in report.leakages)


def test_performer_leakage(media_root: Path) -> None:
    rows = _valid_rows(media_root)
    rows[-1]["performer_id"] = "perf_train"
    manifest = write_jsonl(media_root / "perf_leak.jsonl", rows)
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(root=media_root, check_media=False),
    )
    assert not report.passed
    assert any(f.code == "leakage_performer_id" for f in report.leakages)


def test_clip_leakage(media_root: Path) -> None:
    rows = _valid_rows(media_root)
    # Force same clip_id into held_out with matching path would also fail split
    # consistency; use same clip_id different path to hit leakage + integrity.
    rows[-1]["clip_id"] = "clip_train"
    rows[-1]["video"]["path"] = "videos/held.mp4"
    manifest = write_jsonl(media_root / "clip_leak.jsonl", rows)
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(root=media_root, check_media=False),
    )
    assert not report.passed
    codes = {f.code for f in report.findings}
    assert "leakage_clip_id" in codes or "inconsistent_clip_path_or_split" in codes or (
        "manifest_integrity" in codes
    )


def test_malformed_temporal_metadata(media_root: Path) -> None:
    rows = _valid_rows(media_root)
    rows[0]["temporal"] = {"start_s": 5.0, "end_s": 1.0}
    manifest = write_jsonl(media_root / "bad_temporal.jsonl", rows)
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(root=media_root, check_media=False),
    )
    assert not report.passed
    assert any(f.code == "malformed_metadata" for f in report.errors)


def test_invalid_answers_vocab(media_root: Path) -> None:
    rows = _valid_rows(media_root)
    manifest = write_jsonl(media_root / "vocab.jsonl", rows)
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(
            root=media_root,
            check_media=False,
            allowed_answers=frozenset({"left", "right"}),  # missing center
        ),
    )
    assert not report.passed
    assert any(f.code == "invalid_answer_not_in_vocab" for f in report.errors)


def test_missing_ground_truth_line(media_root: Path) -> None:
    # Raw JSONL bypassing schema constructor defaults
    row = base_row(example_id="bad", clip_id="c", split="train", video_path="videos/train.mp4")
    del row["ground_truth"]
    held = base_row(
        example_id="ho",
        clip_id="ch",
        split="held_out",
        trick_id="th",
        performer_id="ph",
        video_path="videos/held.mp4",
        ground_truth="center",
    )
    manifest = write_jsonl(media_root / "no_gt.jsonl", [row, held])
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(root=media_root, check_media=False),
    )
    assert not report.passed
    assert any(f.code == "malformed_metadata" for f in report.errors)


def test_human_and_machine_readable(media_root: Path, tmp_path: Path) -> None:
    manifest = write_jsonl(media_root / "valid2.jsonl", _valid_rows(media_root))
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(root=media_root, check_media=False),
    )
    text = report.format_human()
    assert "PASSED" in text
    assert "HARD ERRORS" in text
    payload = report.to_dict()
    assert payload["passed"] is True
    assert "findings" in payload
    out = tmp_path / "report.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    assert out.exists()


def test_allow_leakage_flag(media_root: Path) -> None:
    rows = _valid_rows(media_root)
    rows[-1]["trick_id"] = "trick_train"
    manifest = write_jsonl(media_root / "allow_leak.jsonl", rows)
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(root=media_root, check_media=False, fail_on_leakage=False),
    )
    assert report.leakages
    assert report.passed  # leakage present but not failing


def test_cli_validate(media_root: Path) -> None:
    from magic_vlm.cli import validate_main

    rows = _valid_rows(media_root)
    rows[-1]["performer_id"] = "perf_train"
    manifest = write_jsonl(media_root / "cli_leak.jsonl", rows)
    json_out = media_root / "out.json"
    code = validate_main(
        [
            "--manifest",
            str(manifest),
            "--root",
            str(media_root),
            "--no-media-check",
            "--json-out",
            str(json_out),
        ]
    )
    assert code == 1
    assert json_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["counts"]["leakage"] >= 1


def test_unreadable_media_with_opencv_if_available(media_root: Path) -> None:
    pytest.importorskip("cv2")
    rows = [
        base_row(
            example_id="e1",
            clip_id="c1",
            split="train",
            trick_id="t1",
            performer_id="p1",
            video_path="videos/train.mp4",  # nonempty garbage
        ),
        base_row(
            example_id="e2",
            clip_id="c2",
            split="held_out",
            trick_id="t2",
            performer_id="p2",
            video_path="videos/held.mp4",
            ground_truth="right",
        ),
    ]
    manifest = write_jsonl(media_root / "corrupt.jsonl", rows)
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(root=media_root, check_media=True),
    )
    assert not report.passed
    assert any(f.code == "unreadable_video" for f in report.errors)
