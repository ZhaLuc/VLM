"""Stable data schemas for dataset records and research artifacts.

Ground-truth integrity
----------------------
Stored ``ground_truth`` strings are authoritative research labels. Loaders and
serializers must not silently normalize, case-fold, strip semantics beyond
structural parsing, or overwrite them. Prediction-side normalization used by
metrics (see ``magic_vlm.evaluation.normalize_label``) is compare-time only and
must never be written back into dataset manifests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Split(str, Enum):
    """Dataset partition.

    ``held_out`` is reserved for final comparative evaluation and must not be
    used for training, preference-model fitting, or iterative prompt tuning.
    """

    TRAIN = "train"
    VAL = "val"
    HELD_OUT = "held_out"


class TaskType(str, Enum):
    """Supported task types.

    ``hidden_state`` is the first prototype task. ``explanation`` is reserved
    for the later free-text / preference stage and does not require the same
    short-label ground truth.
    """

    HIDDEN_STATE = "hidden_state"
    EXPLANATION = "explanation"


class SchemaError(ValueError):
    """Raised when a record or manifest violates the dataset schema."""


@dataclass(frozen=True)
class VideoRef:
    """Pointer to a source video clip.

    ``content_hash`` is optional at authoring time and should be filled once
    media is ingested so preprocessing stays reproducible.
    """

    path: str
    content_hash: str | None = None
    duration_s: float | None = None
    fps: float | None = None
    num_frames: int | None = None

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise SchemaError("video.path must be a non-empty string")
        if self.duration_s is not None and self.duration_s < 0:
            raise SchemaError("video.duration_s must be >= 0")
        if self.fps is not None and self.fps <= 0:
            raise SchemaError("video.fps must be > 0 when provided")
        if self.num_frames is not None and self.num_frames < 1:
            raise SchemaError("video.num_frames must be >= 1 when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoRef:
        return cls(**dict(data))


@dataclass(frozen=True)
class TemporalSpan:
    """Optional temporal metadata for a clip or annotated moment.

    Hidden-state examples may omit this entirely. When both ends of a pair are
    present, the start must be <= end.
    """

    start_s: float | None = None
    end_s: float | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.start_s is not None and self.end_s is not None and self.start_s > self.end_s:
            raise SchemaError(
                f"temporal start_s ({self.start_s}) must be <= end_s ({self.end_s})"
            )
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.start_frame > self.end_frame
        ):
            raise SchemaError(
                f"temporal start_frame ({self.start_frame}) must be <= end_frame ({self.end_frame})"
            )
        for name, value in (
            ("start_frame", self.start_frame),
            ("end_frame", self.end_frame),
        ):
            if value is not None and value < 0:
                raise SchemaError(f"temporal.{name} must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TemporalSpan | None:
        if data is None:
            return None
        return cls(**dict(data))


@dataclass(frozen=True)
class CausalAnnotation:
    """Optional future temporal/causal annotation.

    Not required for hidden-state examples. Present so later Dataset C-style
    labels can attach without breaking existing records.
    """

    causal_moment: TemporalSpan | None = None
    cause_description: str | None = None
    effect_description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "causal_moment": None if self.causal_moment is None else self.causal_moment.to_dict(),
            "cause_description": self.cause_description,
            "effect_description": self.effect_description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CausalAnnotation | None:
        if data is None:
            return None
        payload = dict(data)
        moment = payload.get("causal_moment")
        if isinstance(moment, dict):
            payload["causal_moment"] = TemporalSpan.from_dict(moment)
        payload.setdefault("metadata", {})
        return cls(**payload)


@dataclass(frozen=True)
class Provenance:
    """Where an example came from (filming, synthetic fixture, etc.)."""

    source: str
    created_by: str | None = None
    created_at: str | None = None
    license: str | None = None
    collection_notes: str | None = None

    def __post_init__(self) -> None:
        if not str(self.source).strip():
            raise SchemaError("provenance.source must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Provenance:
        if data is None:
            raise SchemaError("provenance is required")
        return cls(**dict(data))


@dataclass(frozen=True)
class ExampleRecord:
    """One QA example over a magic/mentalism clip.

    ``clip_id`` identifies the underlying filmed moment. ``example_id`` uniquely
    identifies this (clip, question phrasing) pair so multiple phrasings of the
    same clip are first-class rows without duplicating media identity.

    ``ground_truth`` is the canonical stored label for ``hidden_state`` tasks.
    It is never silently rewritten by loaders or metrics.
    """

    example_id: str
    clip_id: str
    trick_id: str
    performer_id: str
    camera_id: str
    video: VideoRef
    task: TaskType
    question: str
    split: Split
    provenance: Provenance
    ground_truth: str | None = None
    justification: str | None = None
    temporal: TemporalSpan | None = None
    causal: CausalAnnotation | None = None
    notes: str | None = None
    question_variant: str | None = None
    prop_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("example_id", self.example_id),
            ("clip_id", self.clip_id),
            ("trick_id", self.trick_id),
            ("performer_id", self.performer_id),
            ("camera_id", self.camera_id),
            ("question", self.question),
        ):
            if not str(value).strip():
                raise SchemaError(f"{name} must be a non-empty string")
        if not isinstance(self.task, TaskType):
            raise SchemaError(f"invalid task type: {self.task!r}")
        if not isinstance(self.split, Split):
            raise SchemaError(f"invalid split: {self.split!r}")
        if self.task is TaskType.HIDDEN_STATE:
            if self.ground_truth is None or self.ground_truth == "":
                raise SchemaError(
                    "hidden_state examples require a non-empty ground_truth string "
                    "(stored exactly as authored; not normalized)"
                )
    @property
    def answer(self) -> str | None:
        """Compatibility alias for ``ground_truth`` (read-only)."""
        return self.ground_truth

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "clip_id": self.clip_id,
            "trick_id": self.trick_id,
            "performer_id": self.performer_id,
            "camera_id": self.camera_id,
            "video": self.video.to_dict(),
            "task": self.task.value,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "justification": self.justification,
            "temporal": None if self.temporal is None else self.temporal.to_dict(),
            "causal": None if self.causal is None else self.causal.to_dict(),
            "split": self.split.value,
            "provenance": self.provenance.to_dict(),
            "notes": self.notes,
            "question_variant": self.question_variant,
            "prop_id": self.prop_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExampleRecord:
        """Parse one record.

        Accepts legacy key ``answer`` as an input alias for ``ground_truth`` when
        loading older manifests. The in-memory and rewritten field name is always
        ``ground_truth``; the original string value is preserved byte-for-byte as
        provided (no case-folding or whitespace collapsing).
        """
        if not isinstance(data, dict):
            raise SchemaError("example record must be a JSON object")
        payload = dict(data)

        try:
            task = TaskType(str(payload.get("task", TaskType.HIDDEN_STATE.value)))
        except ValueError as exc:
            raise SchemaError(f"invalid task type: {payload.get('task')!r}") from exc
        try:
            split = Split(str(payload["split"]))
        except KeyError as exc:
            raise SchemaError("missing required field: split") from exc
        except ValueError as exc:
            raise SchemaError(f"invalid split: {payload.get('split')!r}") from exc

        video = payload.get("video")
        if not isinstance(video, dict):
            raise SchemaError("video must be an object with at least path")
        video_ref = VideoRef.from_dict(video)

        if "provenance" not in payload:
            raise SchemaError("missing required field: provenance")
        provenance = Provenance.from_dict(payload.get("provenance"))

        if "ground_truth" in payload:
            ground_truth = payload.get("ground_truth")
        elif "answer" in payload:
            ground_truth = payload.get("answer")
        else:
            ground_truth = None

        temporal = TemporalSpan.from_dict(payload.get("temporal"))
        causal = CausalAnnotation.from_dict(payload.get("causal"))

        required = (
            "example_id",
            "clip_id",
            "trick_id",
            "performer_id",
            "camera_id",
            "question",
        )
        missing = [key for key in required if key not in payload or payload[key] is None]
        if missing:
            raise SchemaError(f"missing required field(s): {', '.join(missing)}")

        return cls(
            example_id=str(payload["example_id"]),
            clip_id=str(payload["clip_id"]),
            trick_id=str(payload["trick_id"]),
            performer_id=str(payload["performer_id"]),
            camera_id=str(payload["camera_id"]),
            video=video_ref,
            task=task,
            question=str(payload["question"]),
            ground_truth=None if ground_truth is None else str(ground_truth),
            justification=(
                None if payload.get("justification") is None else str(payload.get("justification"))
            ),
            temporal=temporal,
            causal=causal,
            split=split,
            provenance=provenance,
            notes=None if payload.get("notes") is None else str(payload.get("notes")),
            question_variant=(
                None
                if payload.get("question_variant") is None
                else str(payload.get("question_variant"))
            ),
            prop_id=None if payload.get("prop_id") is None else str(payload.get("prop_id")),
            metadata=dict(payload.get("metadata") or {}),
        )


def validate_manifest_records(records: list[ExampleRecord]) -> None:
    """Validate cross-record constraints (unique IDs, consistent clip media)."""
    seen_example_ids: set[str] = set()
    clip_videos: dict[str, str] = {}
    clip_splits: dict[str, Split] = {}

    for record in records:
        if record.example_id in seen_example_ids:
            raise SchemaError(f"duplicate example_id: {record.example_id!r}")
        seen_example_ids.add(record.example_id)

        prior_video = clip_videos.get(record.clip_id)
        if prior_video is None:
            clip_videos[record.clip_id] = record.video.path
        elif prior_video != record.video.path:
            raise SchemaError(
                f"clip_id {record.clip_id!r} maps to multiple video paths: "
                f"{prior_video!r} vs {record.video.path!r}"
            )

        prior_split = clip_splits.get(record.clip_id)
        if prior_split is None:
            clip_splits[record.clip_id] = record.split
        elif prior_split is not record.split:
            raise SchemaError(
                f"clip_id {record.clip_id!r} appears in multiple splits: "
                f"{prior_split.value!r} vs {record.split.value!r}. "
                "Question variants of the same clip must share one split."
            )


@dataclass(frozen=True)
class InferenceArtifact:
    """First-class preservation of a single model response.

    ``raw_text`` is the untouched decoder output. Parsed fields are derived and
    must never replace the raw string when writing run artifacts.
    """

    example_id: str
    model_id: str
    prompt: str
    raw_text: str
    parsed_answer: str | None = None
    frame_indices: tuple[int, ...] = ()
    generation: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    clip_id: str | None = None
    task: str | None = None
    question: str | None = None
    model_revision: str | None = None
    checkpoint_kind: str | None = None
    checkpoint_path: str | None = None
    preprocessing: dict[str, Any] = field(default_factory=dict)
    device: str | None = None
    latency_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frame_indices"] = list(self.frame_indices)
        return payload


@dataclass(frozen=True)
class PreferencePair:
    """Pairwise preference over two candidate responses for the same example."""

    pair_id: str
    example_id: str
    response_a: str
    response_b: str
    winner: str  # "a" | "b" | "tie"
    annotator_id: str | None = None
    rubric_version: str | None = None
    split: Split = Split.TRAIN
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["split"] = self.split.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreferencePair:
        payload = dict(data)
        payload["split"] = Split(payload.get("split", Split.TRAIN.value))
        payload.setdefault("metadata", {})
        return cls(**payload)
