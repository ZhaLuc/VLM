"""Preference-pair representation, I/O, and validation.

Supports later DPO and Bradley-Terry reward modeling consumers. Does **not**
train models, collect AI labels, or alter raw candidate responses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from magic_vlm.schemas import (
    PreferenceGenerationMeta,
    PreferencePair,
    Provenance,
    SchemaError,
    Split,
    TaskType,
    VideoRef,
)
from magic_vlm.utils import stable_json, utc_now_iso, write_json


class PreferenceSeverity(str, Enum):
    ERROR = "error"
    REVIEW = "review"


@dataclass(frozen=True)
class PreferenceFinding:
    severity: PreferenceSeverity
    code: str
    message: str
    judgment_id: str | None = None
    pair_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreferenceValidationConfig:
    """Policy for preference-collection validation (not training)."""

    allow_ties: bool = False
    allow_identical_responses: bool = False
    require_rationale: bool = False
    fail_on_review: bool = False
    # When False, more than one judgment sharing the same content pair_id is an error.
    allow_multiple_annotations_per_pair: bool = True


@dataclass(frozen=True)
class PreferenceValidationReport:
    n_records: int
    n_unique_pair_ids: int
    n_unique_judgment_ids: int
    findings: tuple[PreferenceFinding, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "n_unique_pair_ids": self.n_unique_pair_ids,
            "n_unique_judgment_ids": self.n_unique_judgment_ids,
            "passed": self.passed,
            "findings": [f.to_dict() for f in self.findings],
        }

    def format_human(self) -> str:
        lines = [
            f"Preference validation: passed={self.passed} "
            f"n={self.n_records} unique_pairs={self.n_unique_pair_ids} "
            f"unique_judgments={self.n_unique_judgment_ids}",
            "",
        ]
        if not self.findings:
            lines.append("No findings.")
        for finding in self.findings:
            lines.append(
                f"[{finding.severity.value}] {finding.code}: {finding.message}"
            )
        lines.append("")
        return "\n".join(lines)


def compute_content_pair_id(
    *,
    clip_id: str,
    instruction: str,
    response_a: str,
    response_b: str,
    task: str | TaskType = TaskType.EXPLANATION,
) -> str:
    """Stable content identity for duplicate detection.

    Hashes exact clip / instruction / raw responses / task. Does not include
    annotator, winner, or generation knobs - those identify judgments, not pairs.
    """
    task_value = task.value if isinstance(task, TaskType) else str(task)
    payload = {
        "clip_id": clip_id,
        "instruction": instruction,
        "response_a": response_a,
        "response_b": response_b,
        "task": task_value,
    }
    digest = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
    return f"pref_{digest[:24]}"


def compute_judgment_id(
    *,
    pair_id: str,
    annotator_id: str,
    timestamp: str,
) -> str:
    """Stable judgment identity for one annotator's decision on a content pair."""
    payload = {
        "pair_id": pair_id,
        "annotator_id": annotator_id,
        "timestamp": timestamp,
    }
    digest = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
    return f"judg_{digest[:24]}"


def build_preference_pair(
    *,
    clip_id: str,
    instruction: str,
    response_a: str,
    response_b: str,
    winner: str,
    annotator_id: str,
    provenance: Provenance,
    generation_meta: PreferenceGenerationMeta,
    timestamp: str | None = None,
    task: TaskType = TaskType.EXPLANATION,
    example_id: str | None = None,
    video: VideoRef | None = None,
    rationale: str | None = None,
    rubric_version: str | None = None,
    allow_ties: bool = False,
    split: Split = Split.TRAIN,
    metadata: dict[str, Any] | None = None,
    pair_id: str | None = None,
    judgment_id: str | None = None,
) -> PreferencePair:
    """Construct a preference record with stable IDs (responses preserved exactly)."""
    ts = timestamp or utc_now_iso()
    pid = pair_id or compute_content_pair_id(
        clip_id=clip_id,
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
        task=task,
    )
    jid = judgment_id or compute_judgment_id(
        pair_id=pid,
        annotator_id=annotator_id,
        timestamp=ts,
    )
    return PreferencePair(
        pair_id=pid,
        judgment_id=jid,
        clip_id=clip_id,
        example_id=example_id,
        video=video,
        task=task,
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
        winner=winner,
        annotator_id=annotator_id,
        timestamp=ts,
        rationale=rationale,
        rubric_version=rubric_version,
        allow_ties=allow_ties,
        provenance=provenance,
        generation_meta=generation_meta,
        split=split,
        metadata=dict(metadata or {}),
    )


def load_preference_pairs(path: str | Path) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                pairs.append(PreferencePair.from_dict(json.loads(text)))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid preference row at {path}:{line_no}") from exc
    return pairs


def write_preference_pairs(path: str | Path, pairs: Sequence[PreferencePair]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_dict(), ensure_ascii=True) + "\n")


def filter_preference_split(pairs: Sequence[PreferencePair], split: Split) -> list[PreferencePair]:
    return [pair for pair in pairs if pair.split is split]


def group_judgments_by_pair(
    pairs: Sequence[PreferencePair],
) -> dict[str, list[PreferencePair]]:
    """Group multiple annotations that share the same content ``pair_id``."""
    grouped: dict[str, list[PreferencePair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.pair_id, []).append(pair)
    return grouped


def validate_preference_pairs(
    pairs: Sequence[PreferencePair],
    *,
    config: PreferenceValidationConfig | None = None,
) -> PreferenceValidationReport:
    """Validate preference records for DPO / BT readiness (no training)."""
    cfg = config or PreferenceValidationConfig()
    findings: list[PreferenceFinding] = []
    seen_judgment: dict[str, PreferencePair] = {}
    pair_ids: set[str] = set()

    for pair in pairs:
        pair_ids.add(pair.pair_id)

        expected = compute_content_pair_id(
            clip_id=pair.clip_id,
            instruction=pair.instruction,
            response_a=pair.response_a,
            response_b=pair.response_b,
            task=pair.task,
        )
        if pair.pair_id != expected:
            findings.append(
                PreferenceFinding(
                    severity=PreferenceSeverity.REVIEW,
                    code="pair_id_mismatch",
                    message=(
                        f"judgment {pair.judgment_id!r} pair_id does not match "
                        f"canonical content hash {expected!r}"
                    ),
                    judgment_id=pair.judgment_id,
                    pair_id=pair.pair_id,
                    details={"expected_pair_id": expected},
                )
            )

        if pair.judgment_id in seen_judgment:
            findings.append(
                PreferenceFinding(
                    severity=PreferenceSeverity.ERROR,
                    code="duplicate_judgment_id",
                    message=f"duplicate judgment_id {pair.judgment_id!r}",
                    judgment_id=pair.judgment_id,
                    pair_id=pair.pair_id,
                )
            )
        else:
            seen_judgment[pair.judgment_id] = pair

        if pair.winner == "tie" and not (pair.allow_ties and cfg.allow_ties):
            # Record may have allow_ties=True but collection policy forbids ties.
            if not cfg.allow_ties:
                findings.append(
                    PreferenceFinding(
                        severity=PreferenceSeverity.ERROR,
                        code="tie_not_allowed",
                        message="winner='tie' but validation config has allow_ties=False",
                        judgment_id=pair.judgment_id,
                        pair_id=pair.pair_id,
                    )
                )

        if pair.response_a == pair.response_b:
            severity = (
                PreferenceSeverity.REVIEW
                if cfg.allow_identical_responses
                else PreferenceSeverity.ERROR
            )
            findings.append(
                PreferenceFinding(
                    severity=severity,
                    code="identical_responses",
                    message=(
                        "response_a and response_b are byte-identical; "
                        "preference is uninformative for DPO/BT"
                    ),
                    judgment_id=pair.judgment_id,
                    pair_id=pair.pair_id,
                )
            )

        if cfg.require_rationale and (
            pair.rationale is None or not str(pair.rationale).strip()
        ):
            findings.append(
                PreferenceFinding(
                    severity=PreferenceSeverity.ERROR,
                    code="missing_rationale",
                    message="rationale required by validation config",
                    judgment_id=pair.judgment_id,
                    pair_id=pair.pair_id,
                )
            )

        if pair.split is Split.HELD_OUT:
            findings.append(
                PreferenceFinding(
                    severity=PreferenceSeverity.REVIEW,
                    code="held_out_preference",
                    message=(
                        "Preference on held_out split - confirm this is intentional "
                        "and not used for reward-model / DPO training"
                    ),
                    judgment_id=pair.judgment_id,
                    pair_id=pair.pair_id,
                )
            )

    grouped = group_judgments_by_pair(pairs)
    for pid, group in grouped.items():
        if len(group) > 1:
            annotators = sorted({g.annotator_id for g in group})
            if not cfg.allow_multiple_annotations_per_pair:
                findings.append(
                    PreferenceFinding(
                        severity=PreferenceSeverity.ERROR,
                        code="duplicate_pair_id",
                        message=(
                            f"duplicate content pair_id {pid!r} "
                            f"({len(group)} judgments); multi-annotation disabled"
                        ),
                        pair_id=pid,
                        details={
                            "judgment_ids": [g.judgment_id for g in group],
                        },
                    )
                )
            else:
                findings.append(
                    PreferenceFinding(
                        severity=PreferenceSeverity.REVIEW,
                        code="multiple_annotations",
                        message=(
                            f"pair_id {pid!r} has {len(group)} judgments "
                            f"(annotators={annotators})"
                        ),
                        pair_id=pid,
                        details={
                            "n_judgments": len(group),
                            "judgment_ids": [g.judgment_id for g in group],
                            "annotator_ids": annotators,
                            "winners": [g.winner for g in group],
                        },
                    )
                )
                winners = {g.winner for g in group}
                if len(winners) > 1:
                    findings.append(
                        PreferenceFinding(
                            severity=PreferenceSeverity.REVIEW,
                            code="annotator_disagreement",
                            message=(
                                f"pair_id {pid!r} has disagreeing winners {sorted(winners)}"
                            ),
                            pair_id=pid,
                            details={"winners": sorted(winners)},
                        )
                    )

    errors = [f for f in findings if f.severity is PreferenceSeverity.ERROR]
    reviews = [f for f in findings if f.severity is PreferenceSeverity.REVIEW]
    passed = not errors and (not cfg.fail_on_review or not reviews)
    return PreferenceValidationReport(
        n_records=len(pairs),
        n_unique_pair_ids=len(pair_ids),
        n_unique_judgment_ids=len(seen_judgment),
        findings=tuple(findings),
        passed=passed,
    )


def dpo_training_rows(pairs: Sequence[PreferencePair]) -> list[dict[str, Any]]:
    """Project preference judgments into DPO-ready rows (no training).

    Skips ties. Preserves raw chosen/rejected strings exactly.
    """
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        if pair.winner == "tie":
            continue
        chosen, rejected = pair.chosen_rejected()
        rows.append(
            {
                "judgment_id": pair.judgment_id,
                "pair_id": pair.pair_id,
                "clip_id": pair.clip_id,
                "example_id": pair.example_id,
                "instruction": pair.instruction,
                "chosen": chosen,
                "rejected": rejected,
                "task": pair.task.value,
                "split": pair.split.value,
                "generation_meta": pair.generation_meta.to_dict(),
            }
        )
    return rows


def bradley_terry_rows(pairs: Sequence[PreferencePair]) -> list[dict[str, Any]]:
    """Project judgments into Bradley-Terry style comparison rows (no fitting)."""
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        rows.append(
            {
                "judgment_id": pair.judgment_id,
                "pair_id": pair.pair_id,
                "clip_id": pair.clip_id,
                "instruction": pair.instruction,
                "response_a": pair.response_a,
                "response_b": pair.response_b,
                "label": pair.bradley_terry_label(),
                "winner": pair.winner,
                "annotator_id": pair.annotator_id,
                "split": pair.split.value,
            }
        )
    return rows


def write_preference_validation_report(
    report: PreferenceValidationReport,
    path: str | Path,
) -> None:
    write_json(path, report.to_dict())
