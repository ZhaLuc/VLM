"""Dataset quality and train/held-out leakage validation.

This module **never** repairs, rewrites, deletes, or re-splits data. It only
reports findings so a researcher can decide whether the dataset is safe for
model experiments.

Severity model
--------------
- ``error``: hard data-quality / integrity failures (unusable or unsafe as-is)
- ``leakage``: scientifically critical train↔held_out contamination signals
- ``review``: heuristics / incompleteness that need human judgment (not definitive)

Default pass/fail: any ``error`` or ``leakage`` fails validation. ``review``
alone does not fail. Use ``fail_on_leakage=False`` only for exploratory reports.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from magic_vlm.schemas import ExampleRecord, SchemaError, Split, TaskType, validate_manifest_records

Severity = Literal["error", "leakage", "review"]

# Prototype clips are short; extreme FPS values usually mean metadata mistakes.
_FPS_MIN = 1.0
_FPS_MAX = 120.0
_FPS_DURATION_TOLERANCE = 0.35  # relative mismatch before review


@dataclass(frozen=True)
class Finding:
    severity: Severity
    code: str
    message: str
    scientific_meaning: str
    example_ids: tuple[str, ...] = ()
    clip_ids: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["example_ids"] = list(self.example_ids)
        payload["clip_ids"] = list(self.clip_ids)
        return payload


@dataclass(frozen=True)
class ValidationReport:
    manifest_path: str
    n_records: int
    n_clips: int
    findings: tuple[Finding, ...]
    passed: bool
    fail_on_leakage: bool
    media_checked: bool

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def leakages(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "leakage")

    @property
    def reviews(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "review")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "n_records": self.n_records,
            "n_clips": self.n_clips,
            "passed": self.passed,
            "fail_on_leakage": self.fail_on_leakage,
            "media_checked": self.media_checked,
            "counts": {
                "error": len(self.errors),
                "leakage": len(self.leakages),
                "review": len(self.reviews),
            },
            "findings": [f.to_dict() for f in self.findings],
        }

    def format_human(self) -> str:
        lines = [
            f"Dataset validation: {'PASSED' if self.passed else 'FAILED'}",
            f"  manifest: {self.manifest_path}",
            f"  records: {self.n_records}  clips: {self.n_clips}",
            f"  errors: {len(self.errors)}  leakage: {len(self.leakages)}  "
            f"review: {len(self.reviews)}",
            f"  media_checked: {self.media_checked}  fail_on_leakage: {self.fail_on_leakage}",
            "",
        ]
        for label, group in (
            ("HARD ERRORS", self.errors),
            ("SCIENTIFIC LEAKAGE", self.leakages),
            ("MANUAL REVIEW", self.reviews),
        ):
            lines.append(f"== {label} ({len(group)}) ==")
            if not group:
                lines.append("  (none)")
            for finding in group:
                ids = ",".join(finding.example_ids) if finding.example_ids else "-"
                lines.append(f"  [{finding.code}] {finding.message}")
                lines.append(f"      examples: {ids}")
                lines.append(f"      meaning: {finding.scientific_meaning}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


@dataclass(frozen=True)
class ValidatorConfig:
    """Options for :func:`validate_dataset`.

    ``allowed_answers``, when provided, is an exact-string allow-list compared
    to stored ``ground_truth`` **without** normalizing labels.
    """

    root: Path | None = None
    check_media: bool = True
    fail_on_leakage: bool = True
    allowed_answers: frozenset[str] | None = None
    expected_fps: float | None = None
    fps_tolerance: float = 0.51


def validate_dataset(
    manifest_path: str | Path,
    *,
    config: ValidatorConfig | None = None,
) -> ValidationReport:
    """Validate a JSONL manifest for quality and train/held-out leakage."""
    cfg = config or ValidatorConfig()
    path = Path(manifest_path)
    root = cfg.root if cfg.root is not None else path.parent.parent if path.parent.name else Path.cwd()
    # Prefer repo root when manifests live under data/examples/
    if cfg.root is None:
        candidate = path.resolve().parent
        for parent in [candidate, *candidate.parents]:
            if (parent / "pyproject.toml").exists() or (parent / "src" / "magic_vlm").exists():
                root = parent
                break
        else:
            root = Path.cwd()

    findings: list[Finding] = []
    records, parse_findings = _load_records_collecting_errors(path)
    findings.extend(parse_findings)

    if records:
        findings.extend(_check_manifest_integrity(records))
        findings.extend(_check_ground_truth(records, cfg.allowed_answers))
        findings.extend(_check_clip_consistency(records))
        findings.extend(_check_duplicate_video_paths(records))
        findings.extend(_check_temporal_and_fps(records, cfg))
        findings.extend(_check_media(records, root=root, check_media=cfg.check_media))
        findings.extend(_check_split_leakage(records))
        findings.extend(_check_suspicious_patterns(records))

    media_checked = bool(cfg.check_media)
    n_clips = len({r.clip_id for r in records})
    has_error = any(f.severity == "error" for f in findings)
    has_leakage = any(f.severity == "leakage" for f in findings)
    passed = (not has_error) and (not has_leakage if cfg.fail_on_leakage else True)

    # Stable ordering for reproducible reports
    order = {"error": 0, "leakage": 1, "review": 2}
    findings_sorted = tuple(
        sorted(findings, key=lambda f: (order[f.severity], f.code, f.message))
    )
    return ValidationReport(
        manifest_path=str(path),
        n_records=len(records),
        n_clips=n_clips,
        findings=findings_sorted,
        passed=passed,
        fail_on_leakage=cfg.fail_on_leakage,
        media_checked=media_checked,
    )


def _load_records_collecting_errors(
    path: Path,
) -> tuple[list[ExampleRecord], list[Finding]]:
    findings: list[Finding] = []
    records: list[ExampleRecord] = []
    if not path.exists():
        findings.append(
            Finding(
                severity="error",
                code="manifest_missing",
                message=f"Manifest file not found: {path}",
                scientific_meaning="Without a manifest, no experiment can be defined.",
            )
        )
        return [], findings

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
                records.append(ExampleRecord.from_dict(payload))
            except (json.JSONDecodeError, SchemaError, TypeError, ValueError) as exc:
                findings.append(
                    Finding(
                        severity="error",
                        code="malformed_metadata",
                        message=f"Line {line_no}: {exc}",
                        scientific_meaning=(
                            "Malformed rows cannot be trusted as labeled research examples."
                        ),
                        details={"line": line_no},
                    )
                )
    return records, findings


def _check_manifest_integrity(records: Sequence[ExampleRecord]) -> list[Finding]:
    findings: list[Finding] = []
    try:
        validate_manifest_records(list(records))
    except SchemaError as exc:
        findings.append(
            Finding(
                severity="error",
                code="manifest_integrity",
                message=str(exc),
                scientific_meaning=(
                    "Duplicate IDs or clip/split inconsistencies break unique example "
                    "identity and can mix evaluation conditions."
                ),
            )
        )

    # Explicit duplicate example_id report with IDs (validate_manifest_records already errors)
    counts: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        counts[record.example_id].append(idx)
    for example_id, idxs in counts.items():
        if len(idxs) > 1:
            findings.append(
                Finding(
                    severity="error",
                    code="duplicate_example_id",
                    message=f"Duplicate example_id {example_id!r} appears {len(idxs)} times",
                    scientific_meaning="Duplicate example IDs make metrics and joins ambiguous.",
                    example_ids=(example_id,),
                )
            )
    return findings


def _check_ground_truth(
    records: Sequence[ExampleRecord],
    allowed_answers: frozenset[str] | None,
) -> list[Finding]:
    findings: list[Finding] = []
    for record in records:
        if record.task is TaskType.HIDDEN_STATE:
            if record.ground_truth is None or record.ground_truth == "":
                findings.append(
                    Finding(
                        severity="error",
                        code="missing_ground_truth",
                        message=f"Missing ground_truth for example {record.example_id!r}",
                        scientific_meaning=(
                            "Hidden-state scoring requires an authored gold label; "
                            "the validator will not infer one."
                        ),
                        example_ids=(record.example_id,),
                        clip_ids=(record.clip_id,),
                    )
                )
            elif record.ground_truth.strip() == "":
                findings.append(
                    Finding(
                        severity="error",
                        code="invalid_ground_truth_whitespace",
                        message=(
                            f"ground_truth for {record.example_id!r} is whitespace-only"
                        ),
                        scientific_meaning=(
                            "A whitespace-only label is not a usable canonical answer."
                        ),
                        example_ids=(record.example_id,),
                    )
                )
            elif allowed_answers is not None and record.ground_truth not in allowed_answers:
                findings.append(
                    Finding(
                        severity="error",
                        code="invalid_answer_not_in_vocab",
                        message=(
                            f"ground_truth {record.ground_truth!r} for {record.example_id!r} "
                            "is not in the provided answer vocabulary"
                        ),
                        scientific_meaning=(
                            "Out-of-vocabulary gold labels break closed-set evaluation "
                            "contracts. Compared exactly as stored (no normalization)."
                        ),
                        example_ids=(record.example_id,),
                        details={"ground_truth": record.ground_truth},
                    )
                )
            elif record.ground_truth != record.ground_truth.strip():
                findings.append(
                    Finding(
                        severity="review",
                        code="ground_truth_peripheral_whitespace",
                        message=(
                            f"ground_truth for {record.example_id!r} has leading/trailing "
                            "whitespace (left unchanged)"
                        ),
                        scientific_meaning=(
                            "Not an automatic failure; confirm the authored label is intentional. "
                            "Validators never strip stored ground truth."
                        ),
                        example_ids=(record.example_id,),
                    )
                )
    return findings


def _check_clip_consistency(records: Sequence[ExampleRecord]) -> list[Finding]:
    findings: list[Finding] = []
    by_clip: dict[str, list[ExampleRecord]] = defaultdict(list)
    for record in records:
        by_clip[record.clip_id].append(record)

    for clip_id, group in by_clip.items():
        tricks = {r.trick_id for r in group}
        performers = {r.performer_id for r in group}
        cameras = {r.camera_id for r in group}
        paths = {r.video.path for r in group}
        splits = {r.split for r in group}
        gts = {r.ground_truth for r in group if r.task is TaskType.HIDDEN_STATE}
        if len(tricks) > 1 or len(performers) > 1 or len(cameras) > 1:
            findings.append(
                Finding(
                    severity="error",
                    code="inconsistent_clip_metadata",
                    message=(
                        f"clip_id {clip_id!r} has inconsistent trick/performer/camera "
                        f"across question variants"
                    ),
                    scientific_meaning=(
                        "Question variants of one clip must describe the same filmed moment."
                    ),
                    example_ids=tuple(r.example_id for r in group),
                    clip_ids=(clip_id,),
                    details={
                        "trick_ids": sorted(tricks),
                        "performer_ids": sorted(performers),
                        "camera_ids": sorted(cameras),
                    },
                )
            )
        if len(paths) > 1 or len(splits) > 1:
            # Also covered by validate_manifest_records; keep explicit.
            findings.append(
                Finding(
                    severity="error",
                    code="inconsistent_clip_path_or_split",
                    message=f"clip_id {clip_id!r} maps to multiple paths or splits",
                    scientific_meaning=(
                        "A clip cannot belong to two evaluation partitions or two media files."
                    ),
                    example_ids=tuple(r.example_id for r in group),
                    clip_ids=(clip_id,),
                )
            )
        if len(gts) > 1:
            findings.append(
                Finding(
                    severity="error",
                    code="inconsistent_clip_ground_truth",
                    message=(
                        f"clip_id {clip_id!r} has conflicting ground_truth values across variants"
                    ),
                    scientific_meaning=(
                        "Paraphrased questions for one hidden state must share one gold label."
                    ),
                    example_ids=tuple(r.example_id for r in group),
                    clip_ids=(clip_id,),
                    details={"ground_truths": sorted(str(g) for g in gts)},
                )
            )
    return findings


def _check_duplicate_video_paths(records: Sequence[ExampleRecord]) -> list[Finding]:
    findings: list[Finding] = []
    path_to_clips: dict[str, set[str]] = defaultdict(set)
    path_to_examples: dict[str, list[str]] = defaultdict(list)
    for record in records:
        path_to_clips[record.video.path].add(record.clip_id)
        path_to_examples[record.video.path].append(record.example_id)
    for path, clips in path_to_clips.items():
        if len(clips) > 1:
            findings.append(
                Finding(
                    severity="error",
                    code="duplicate_video_path",
                    message=(
                        f"Video path {path!r} is used by multiple clip_ids: {sorted(clips)}"
                    ),
                    scientific_meaning=(
                        "Distinct clip IDs sharing one file collapse identity and can "
                        "silently duplicate training signal."
                    ),
                    example_ids=tuple(path_to_examples[path]),
                    clip_ids=tuple(sorted(clips)),
                )
            )
    return findings


def _check_temporal_and_fps(
    records: Sequence[ExampleRecord],
    cfg: ValidatorConfig,
) -> list[Finding]:
    findings: list[Finding] = []
    for record in records:
        temporal = record.temporal
        if temporal is not None:
            # Schema already rejects start>end; re-state for validator reports if somehow present
            if (
                temporal.start_s is not None
                and temporal.end_s is not None
                and temporal.start_s > temporal.end_s
            ):
                findings.append(
                    Finding(
                        severity="error",
                        code="invalid_temporal_interval",
                        message=(
                            f"Invalid temporal interval for {record.example_id!r}: "
                            f"{temporal.start_s} > {temporal.end_s}"
                        ),
                        scientific_meaning=(
                            "Invalid intervals make temporal/causal analyses undefined."
                        ),
                        example_ids=(record.example_id,),
                    )
                )

        fps = record.video.fps
        causal = record.causal
        if causal is not None:
            findings.append(
                Finding(
                    severity="review",
                    code="causal_annotation_present",
                    message=(
                        f"Causal annotation on {record.example_id!r}: "
                        f"status={causal.status.value}, unique_cause={causal.unique_cause}"
                    ),
                    scientific_meaning=(
                        "Causal status must be exposed in reports. Ambiguous labels must "
                        "not be treated as gold. Clip-level temporal is not causal gold."
                    ),
                    example_ids=(record.example_id,),
                    details={
                        "annotation_status": causal.status.value,
                        "status_label": (
                            "objectively_established"
                            if causal.status.value == "known"
                            else causal.status.value
                        ),
                        "unique_cause": causal.unique_cause,
                        "provenance": causal.provenance.to_dict(),
                        "eligible_as_gold": causal.is_eligible_gold
                        and causal.causal_moment is not None,
                    },
                )
            )
            if causal.status.value == "ambiguous":
                findings.append(
                    Finding(
                        severity="review",
                        code="ambiguous_causal_annotation",
                        message=(
                            f"Ambiguous causal annotation on {record.example_id!r} "
                            "(retained; not scorable as gold)"
                        ),
                        scientific_meaning=(
                            "Simultaneous actions can make a single causal moment "
                            "undefinable. Do not hide or auto-promote ambiguous labels."
                        ),
                        example_ids=(record.example_id,),
                    )
                )
            if causal.causal_moment is None and causal.is_eligible_gold:
                findings.append(
                    Finding(
                        severity="error",
                        code="causal_missing_moment",
                        message=(
                            f"Non-ambiguous causal annotation on {record.example_id!r} "
                            "lacks causal_moment"
                        ),
                        scientific_meaning=(
                            "Eligible causal statuses require an interval to score IoU."
                        ),
                        example_ids=(record.example_id,),
                    )
                )

        if fps is None:
            findings.append(
                Finding(
                    severity="review",
                    code="missing_fps",
                    message=f"No fps recorded for {record.example_id!r}",
                    scientific_meaning=(
                        "Missing FPS is not leakage, but frame sampling reproducibility "
                        "is weaker without it."
                    ),
                    example_ids=(record.example_id,),
                    clip_ids=(record.clip_id,),
                )
            )
        else:
            if fps < _FPS_MIN or fps > _FPS_MAX:
                findings.append(
                    Finding(
                        severity="error",
                        code="unexpected_fps",
                        message=f"Unexpected fps={fps} for {record.example_id!r}",
                        scientific_meaning=(
                            "Extreme FPS values usually indicate metadata corruption and "
                            "can distort frame-index / duration math."
                        ),
                        example_ids=(record.example_id,),
                        details={"fps": fps},
                    )
                )
            if cfg.expected_fps is not None and abs(fps - cfg.expected_fps) > cfg.fps_tolerance:
                findings.append(
                    Finding(
                        severity="review",
                        code="fps_deviates_from_expected",
                        message=(
                            f"fps={fps} for {record.example_id!r} differs from expected "
                            f"{cfg.expected_fps}"
                        ),
                        scientific_meaning=(
                            "Not definitive error; confirm whether mixed capture rates are intended."
                        ),
                        example_ids=(record.example_id,),
                    )
                )

        if (
            record.video.fps
            and record.video.num_frames
            and record.video.duration_s
            and record.video.duration_s > 0
        ):
            implied = record.video.num_frames / record.video.fps
            rel = abs(implied - record.video.duration_s) / record.video.duration_s
            if rel > _FPS_DURATION_TOLERANCE:
                findings.append(
                    Finding(
                        severity="review",
                        code="fps_duration_inconsistency",
                        message=(
                            f"num_frames/fps≈{implied:.3f}s vs duration_s="
                            f"{record.video.duration_s} for {record.example_id!r}"
                        ),
                        scientific_meaning=(
                            "Heuristic inconsistency only; may be OK if duration is approximate."
                        ),
                        example_ids=(record.example_id,),
                    )
                )
    return findings


def _resolve_video_path(video_path: str, *, root: Path, manifest_path_hint: Path | None = None) -> Path:
    path = Path(video_path)
    if path.is_absolute():
        return path
    candidates = [root / path, Path.cwd() / path]
    if manifest_path_hint is not None:
        candidates.insert(0, manifest_path_hint.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return root / path


def _check_media(
    records: Sequence[ExampleRecord],
    *,
    root: Path,
    check_media: bool,
) -> list[Finding]:
    findings: list[Finding] = []
    if not check_media:
        findings.append(
            Finding(
                severity="review",
                code="media_check_disabled",
                message="Video existence/readability checks were disabled",
                scientific_meaning=(
                    "Skipping media checks can hide missing or corrupt clips until inference."
                ),
            )
        )
        return findings

    try:
        import cv2  # type: ignore

        has_cv2 = True
    except ImportError:
        has_cv2 = False
        findings.append(
            Finding(
                severity="review",
                code="media_read_check_unavailable",
                message=(
                    "OpenCV not installed; corrupt/unreadable checks limited to existence/size"
                ),
                scientific_meaning=(
                    "Install magic-vlm[video] for stronger media validation. "
                    "Missing files are still reported as hard errors."
                ),
            )
        )

    seen_paths: set[str] = set()
    for record in records:
        if record.video.path in seen_paths:
            continue
        seen_paths.add(record.video.path)
        resolved = _resolve_video_path(record.video.path, root=root)
        if not resolved.exists():
            findings.append(
                Finding(
                    severity="error",
                    code="missing_video",
                    message=f"Missing video file for clip {record.clip_id!r}: {resolved}",
                    scientific_meaning=(
                        "Examples without media cannot be used; the validator will not skip them."
                    ),
                    example_ids=(record.example_id,),
                    clip_ids=(record.clip_id,),
                    details={"path": str(resolved)},
                )
            )
            continue
        if resolved.stat().st_size == 0:
            findings.append(
                Finding(
                    severity="error",
                    code="unreadable_video_empty",
                    message=f"Video file is empty (0 bytes): {resolved}",
                    scientific_meaning="Empty media files are corrupt for research use.",
                    example_ids=(record.example_id,),
                    clip_ids=(record.clip_id,),
                )
            )
            continue
        if has_cv2:
            capture = cv2.VideoCapture(str(resolved))
            opened = bool(capture.isOpened())
            frame_ok = False
            if opened:
                frame_ok, _ = capture.read()
            capture.release()
            if not opened or not frame_ok:
                findings.append(
                    Finding(
                        severity="error",
                        code="unreadable_video",
                        message=f"Video could not be opened/decoded: {resolved}",
                        scientific_meaning=(
                            "Corrupt media silently dropped at train time would bias results; "
                            "validation fails instead of skipping."
                        ),
                        example_ids=(record.example_id,),
                        clip_ids=(record.clip_id,),
                    )
                )
    return findings


def _split_groups(records: Sequence[ExampleRecord]) -> dict[Split, list[ExampleRecord]]:
    groups: dict[Split, list[ExampleRecord]] = defaultdict(list)
    for record in records:
        groups[record.split].append(record)
    return groups


def _check_split_leakage(records: Sequence[ExampleRecord]) -> list[Finding]:
    """Train/val vs held_out overlaps on clip, video, trick, and performer.

    These are classified as ``leakage`` (scientifically critical), not mere review.
    """
    findings: list[Finding] = []
    groups = _split_groups(records)
    held = groups.get(Split.HELD_OUT, [])
    if not held:
        findings.append(
            Finding(
                severity="review",
                code="no_held_out_split",
                message="No held_out examples present in manifest",
                scientific_meaning=(
                    "A frozen held-out split is required before claiming generalization."
                ),
            )
        )
        return findings

    non_held = [r for r in records if r.split is not Split.HELD_OUT]
    if not non_held:
        findings.append(
            Finding(
                severity="review",
                code="only_held_out_present",
                message="Manifest contains only held_out examples",
                scientific_meaning="No train/val partition exists to compare against held_out.",
            )
        )
        return findings

    def _overlap(
        code: str,
        attr: str,
        meaning: str,
    ) -> None:
        left = {getattr(r, attr) for r in non_held}
        right = {getattr(r, attr) for r in held}
        shared = sorted(left & right)
        if shared:
            ex = tuple(
                r.example_id
                for r in records
                if getattr(r, attr) in shared
            )
            findings.append(
                Finding(
                    severity="leakage",
                    code=code,
                    message=f"Overlap on {attr} between non-held_out and held_out: {shared}",
                    scientific_meaning=meaning,
                    example_ids=ex,
                    details={"shared": shared, "attribute": attr},
                )
            )

    _overlap(
        "leakage_clip_id",
        "clip_id",
        "The same filmed clip must not appear in both development and held-out partitions.",
    )
    # Video path overlap (even if clip_ids differ)
    non_held_paths = {r.video.path for r in non_held}
    held_paths = {r.video.path for r in held}
    shared_paths = sorted(non_held_paths & held_paths)
    if shared_paths:
        findings.append(
            Finding(
                severity="leakage",
                code="leakage_video_path",
                message=f"Shared video paths across held_out boundary: {shared_paths}",
                scientific_meaning=(
                    "Identical media in train/val and held_out is direct contamination."
                ),
                example_ids=tuple(
                    r.example_id for r in records if r.video.path in shared_paths
                ),
                details={"shared": shared_paths},
            )
        )

    _overlap(
        "leakage_trick_id",
        "trick_id",
        "Held-out evaluation is intended to stress unseen tricks; trick overlap "
        "inflates apparent generalization.",
    )
    _overlap(
        "leakage_performer_id",
        "performer_id",
        "Held-out evaluation is intended to stress unseen performers; performer "
        "overlap allows identity/style shortcuts.",
    )

    # Also flag train↔val trick/performer overlap as review (not held-out critical)
    train = groups.get(Split.TRAIN, [])
    val = groups.get(Split.VAL, [])
    if train and val:
        for attr, code in (
            ("trick_id", "train_val_trick_overlap"),
            ("performer_id", "train_val_performer_overlap"),
            ("clip_id", "train_val_clip_overlap"),
        ):
            shared = sorted({getattr(r, attr) for r in train} & {getattr(r, attr) for r in val})
            if shared:
                findings.append(
                    Finding(
                        severity="review",
                        code=code,
                        message=f"train/val overlap on {attr}: {shared}",
                        scientific_meaning=(
                            "Not held-out leakage. May be acceptable for tuning, but "
                            "reduces the independence of val as an early-stopping signal."
                        ),
                        details={"shared": shared},
                    )
                )
    return findings


def _normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _check_suspicious_patterns(records: Sequence[ExampleRecord]) -> list[Finding]:
    """Heuristics only - never reported as definitive leakage."""
    findings: list[Finding] = []

    # Same content_hash under different clip_ids
    hash_to_clips: dict[str, set[str]] = defaultdict(set)
    hash_examples: dict[str, list[str]] = defaultdict(list)
    for record in records:
        digest = record.video.content_hash
        if digest:
            hash_to_clips[digest].add(record.clip_id)
            hash_examples[digest].append(record.example_id)
    for digest, clips in hash_to_clips.items():
        if len(clips) > 1:
            findings.append(
                Finding(
                    severity="review",
                    code="near_duplicate_content_hash",
                    message=(
                        f"content_hash {digest!r} shared by clip_ids {sorted(clips)}"
                    ),
                    scientific_meaning=(
                        "Heuristic near-duplicate signal only. Confirm whether files are "
                        "true duplicates before treating as leakage."
                    ),
                    example_ids=tuple(hash_examples[digest]),
                    clip_ids=tuple(sorted(clips)),
                )
            )

    # Duplicate normalized questions within a clip
    by_clip: dict[str, list[ExampleRecord]] = defaultdict(list)
    for record in records:
        by_clip[record.clip_id].append(record)
    for clip_id, group in by_clip.items():
        seen: dict[str, list[str]] = defaultdict(list)
        for record in group:
            seen[_normalize_question(record.question)].append(record.example_id)
        for norm_q, ids in seen.items():
            if len(ids) > 1:
                findings.append(
                    Finding(
                        severity="review",
                        code="duplicate_question_phrasing",
                        message=(
                            f"Near-duplicate questions on clip {clip_id!r}: {ids}"
                        ),
                        scientific_meaning=(
                            "Heuristic only (case/whitespace normalized). May be intentional "
                            "paraphrase collapse; not counted as split leakage."
                        ),
                        example_ids=tuple(ids),
                        clip_ids=(clip_id,),
                        details={"normalized_question": norm_q},
                    )
                )

    # Repeated (trick, performer) pairs across many clips - annotation density review
    pair_clips: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        pair_clips[(record.trick_id, record.performer_id)].add(record.clip_id)
    for (trick, performer), clips in pair_clips.items():
        if len(clips) >= 4:
            findings.append(
                Finding(
                    severity="review",
                    code="repeated_trick_performer_combo",
                    message=(
                        f"trick={trick!r} performer={performer!r} appears in {len(clips)} clips"
                    ),
                    scientific_meaning=(
                        "Not leakage by itself; check whether the small dataset is "
                        "over-concentrated on one combo and under-covers others."
                    ),
                    clip_ids=tuple(sorted(clips)),
                )
            )
    return findings


def write_report(report: ValidationReport, json_path: str | Path | None = None) -> None:
    if json_path is None:
        return
    from magic_vlm.utils import write_json

    write_json(json_path, report.to_dict())
