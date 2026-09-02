"""Temporal interval parsing and IoU (compare-time only; never mutates datasets).

Does not invent causal ground truth. Clip-level spans are not causal moments.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from magic_vlm.schemas import ExampleRecord, TemporalSpan

IntervalUnit = Literal["seconds", "frames"]


@dataclass(frozen=True)
class Interval:
    """Closed interval in seconds or frame indices."""

    start: float
    end: float
    unit: IntervalUnit

    def __post_init__(self) -> None:
        if self.unit not in {"seconds", "frames"}:
            raise ValueError(f"unsupported interval unit: {self.unit!r}")
        object.__setattr__(self, "start", float(self.start))
        object.__setattr__(self, "end", float(self.end))

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def is_valid(self) -> bool:
        if self.start < 0 or self.end < 0:
            return False
        if self.start > self.end:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def interval_intersection_length(a: Interval, b: Interval) -> float:
    if a.unit != b.unit:
        raise ValueError("interval units must match for intersection")
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def interval_iou(a: Interval, b: Interval) -> float:
    """Intersection-over-union for two intervals in the same unit.

    Degenerate (zero-length) intervals: IoU is 1.0 only when both are the same
    point; otherwise 0.0. Never invents conversions across units.
    """
    if a.unit != b.unit:
        raise ValueError("interval units must match for IoU")
    if not a.is_valid or not b.is_valid:
        raise ValueError("both intervals must be valid for IoU")
    inter = interval_intersection_length(a, b)
    len_a = a.length
    len_b = b.length
    if len_a == 0.0 and len_b == 0.0:
        return 1.0 if a.start == b.start else 0.0
    union = len_a + len_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def span_to_interval(
    span: TemporalSpan,
    *,
    preferred_unit: IntervalUnit | Literal["auto"] = "auto",
) -> tuple[Interval | None, str | None]:
    """Convert a TemporalSpan to an Interval.

    Returns ``(interval, error)``. Prefers seconds when both units are present
    unless ``preferred_unit`` forces frames.
    """
    has_s = span.start_s is not None and span.end_s is not None
    has_f = span.start_frame is not None and span.end_frame is not None
    unit: IntervalUnit
    if preferred_unit == "seconds":
        if not has_s:
            return None, "span_missing_seconds"
        unit = "seconds"
        start, end = float(span.start_s), float(span.end_s)  # type: ignore[arg-type]
    elif preferred_unit == "frames":
        if not has_f:
            return None, "span_missing_frames"
        unit = "frames"
        start, end = float(span.start_frame), float(span.end_frame)  # type: ignore[arg-type]
    else:
        if has_s:
            unit = "seconds"
            start, end = float(span.start_s), float(span.end_s)  # type: ignore[arg-type]
        elif has_f:
            unit = "frames"
            start, end = float(span.start_frame), float(span.end_frame)  # type: ignore[arg-type]
        else:
            return None, "span_missing_both_ends"
    interval = Interval(start=start, end=end, unit=unit)
    if not interval.is_valid:
        return None, "invalid_interval"
    return interval, None


# Structured parsers (prefer explicit tags to reduce parser exploitation).
_SECONDS_PATTERNS = (
    re.compile(
        r"start_s\s*[=:]\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*[,\s]+end_s\s*[=:]\s*([+-]?(?:\d+\.?\d*|\.\d+))",
        re.IGNORECASE,
    ),
    re.compile(
        r"interval\s*[=:]\s*\[?\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*[,–-]\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*\]?\s*(?:s|sec|seconds)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:from\s+)?([+-]?(?:\d+\.?\d*|\.\d+))\s*(?:s|sec|seconds)\s*(?:-|–|to)\s*"
        r"([+-]?(?:\d+\.?\d*|\.\d+))\s*(?:s|sec|seconds)?",
        re.IGNORECASE,
    ),
)

_FRAMES_PATTERNS = (
    re.compile(
        r"start_frame\s*[=:]\s*(\d+)\s*[,\s]+end_frame\s*[=:]\s*(\d+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"frames?\s*[=:]?\s*(\d+)\s*(?:-|–|to)\s*(\d+)",
        re.IGNORECASE,
    ),
)


def parse_interval_text(
    raw_text: str,
    *,
    preferred_unit: IntervalUnit | Literal["auto"] = "auto",
) -> tuple[Interval | None, bool, str | None]:
    """Parse a predicted interval from model text.

    Returns ``(interval, parse_failed, reason)``. Does not mutate ``raw_text``.
    """
    text = (raw_text or "").strip()
    if not text:
        return None, True, "empty_response"

    seconds_hit: Interval | None = None
    frames_hit: Interval | None = None
    for pattern in _SECONDS_PATTERNS:
        match = pattern.search(text)
        if match:
            seconds_hit = Interval(
                start=float(match.group(1)),
                end=float(match.group(2)),
                unit="seconds",
            )
            break
    for pattern in _FRAMES_PATTERNS:
        match = pattern.search(text)
        if match:
            frames_hit = Interval(
                start=float(match.group(1)),
                end=float(match.group(2)),
                unit="frames",
            )
            break

    chosen: Interval | None = None
    if preferred_unit == "seconds":
        chosen = seconds_hit
        if chosen is None and frames_hit is not None:
            return None, True, "predicted_frames_but_seconds_required"
    elif preferred_unit == "frames":
        chosen = frames_hit
        if chosen is None and seconds_hit is not None:
            return None, True, "predicted_seconds_but_frames_required"
    else:
        chosen = seconds_hit or frames_hit

    if chosen is None:
        return None, True, "no_interval_parsed"
    if not chosen.is_valid:
        return None, True, "invalid_predicted_interval"
    return chosen, False, None


def gold_causal_interval(
    example: ExampleRecord,
    *,
    preferred_unit: IntervalUnit | Literal["auto"] = "auto",
) -> dict[str, Any]:
    """Resolve gold causal interval without inventing labels.

    Never falls back to ``example.temporal`` (clip span ≠ causal moment).
    """
    causal = example.causal
    base: dict[str, Any] = {
        "eligible": False,
        "interval": None,
        "annotation_status": None,
        "status_label": None,
        "unique_cause": None,
        "provenance": None,
        "reason": None,
        "used_clip_temporal_as_gold": False,
    }
    if causal is None:
        base["reason"] = "no_causal_annotation"
        return base
    base["annotation_status"] = causal.status.value
    base["status_label"] = (
        "objectively_established"
        if causal.status.value == "known"
        else causal.status.value
    )
    base["unique_cause"] = causal.unique_cause
    base["provenance"] = causal.provenance.to_dict()
    if not causal.is_eligible_gold:
        base["reason"] = "ambiguous_annotation"
        # Still surface the authored span for inspection without treating as gold.
        if causal.causal_moment is not None:
            interval, err = span_to_interval(causal.causal_moment, preferred_unit=preferred_unit)
            base["interval"] = None if interval is None else interval.to_dict()
            if err:
                base["reason"] = f"ambiguous_annotation;{err}"
        return base
    if causal.causal_moment is None:
        base["reason"] = "missing_causal_moment"
        return base
    interval, err = span_to_interval(causal.causal_moment, preferred_unit=preferred_unit)
    if interval is None:
        base["reason"] = err or "invalid_gold_interval"
        return base
    base["eligible"] = True
    base["interval"] = interval.to_dict()
    base["reason"] = None
    return base


def resolve_gold_interval_object(
    example: ExampleRecord,
    *,
    preferred_unit: IntervalUnit | Literal["auto"] = "auto",
) -> tuple[Interval | None, dict[str, Any]]:
    meta = gold_causal_interval(example, preferred_unit=preferred_unit)
    if not meta["eligible"] or meta["interval"] is None:
        return None, meta
    payload = meta["interval"]
    return Interval(start=payload["start"], end=payload["end"], unit=payload["unit"]), meta
