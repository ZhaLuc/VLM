"""Shared helpers for dataset validation fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
