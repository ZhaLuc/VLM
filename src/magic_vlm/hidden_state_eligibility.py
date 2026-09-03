"""Benchmark eligibility checks for approved hidden-state gold rows.

Does not invent labels. Human approval is required to close leakage/occlusion
warnings that were flagged as PARTIAL at candidate time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from magic_vlm.dataset import load_manifest
from magic_vlm.schemas import ExampleRecord
from magic_vlm.video import VideoPreprocessConfig, preprocess_video, probe_video

FILL = "HUMAN_FILL_REQUIRED"
APPROVED_VALUES = {"APPROVE", "APPROVED"}
WIKIMEDIA_PREFIX = "peerj_"


def _approval(record: ExampleRecord) -> str:
    meta = record.metadata or {}
    proposal = meta.get("annotation_proposal") or {}
    return str(
        meta.get("human_approval")
        or meta.get("human_decision")
        or proposal.get("human_decision")
        or ""
    )


def evaluate_gold_record(record: ExampleRecord, root: Path) -> dict[str, Any]:
    errors: list[str] = []
    meta = record.metadata or {}
    proposal = meta.get("annotation_proposal") or {}

    if record.clip_id.startswith(WIKIMEDIA_PREFIX):
        errors.append("wikimedia control must not enter hidden-state gold")
    if record.question.strip() == "" or record.question == FILL:
        errors.append("question missing")
    if record.ground_truth is None or record.ground_truth.strip() == "" or record.ground_truth == FILL:
        errors.append("ground_truth missing")
    if not record.provenance.source.strip():
        errors.append("provenance.source missing")
    if _approval(record) not in APPROVED_VALUES:
        errors.append("human approval missing")
    if meta.get("unresolved_leakage_warning") is True:
        errors.append("unresolved leakage warning")
    leakage_ok = bool(meta.get("leakage_resolved_by_human")) or str(
        meta.get("answer_leakage_status") or proposal.get("answer_leakage_status") or ""
    ) in {"PASS", "RESOLVED_BY_HUMAN", "NONE"}
    if not leakage_ok:
        errors.append("answer leakage not resolved")
    reveal = str(meta.get("reveal_status") or proposal.get("reveal_status") or "")
    if reveal == "REVEAL_PRESENT":
        errors.append("evaluation clip includes a reveal")

    video_path = Path(record.video.path)
    if not video_path.is_absolute():
        video_path = Path(root) / video_path
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        errors.append(f"video missing: {record.video.path}")
    else:
        try:
            info = probe_video(video_path)
            if int(info.get("num_frames") or 0) < 1:
                errors.append("video probe found no frames")
            sampled = preprocess_video(
                video_path,
                config=VideoPreprocessConfig(max_frames=8, sample_strategy="uniform"),
                load_frames=True,
            )
            if sampled.frames is None or len(sampled.frames) < 1:
                errors.append("preprocessing failed")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"video decode/preprocess failed: {type(exc).__name__}: {exc}")

    return {"clip_id": record.clip_id, "passed": not errors, "errors": errors}


def evaluate_gold_manifest(path: Path, root: Path) -> dict[str, Any]:
    records = load_manifest(path)
    rows = [evaluate_gold_record(record, root) for record in records]
    return {
        "manifest": str(path),
        "n": len(records),
        "passed": all(row["passed"] for row in rows) and len(rows) >= 1,
        "rows": rows,
    }
