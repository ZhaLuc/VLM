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
class PreferenceGenerationMeta:
    """How candidates A/B were produced.

    Kept separate from human annotation fields so DPO / Bradley-Terry training
    can consume responses + judgment without mixing sampling knobs into labels.
    """

    model_id_a: str
    model_id_b: str
    generation_a: dict[str, Any] = field(default_factory=dict)
    generation_b: dict[str, Any] = field(default_factory=dict)
    model_revision_a: str | None = None
    model_revision_b: str | None = None
    checkpoint_kind_a: str | None = None
    checkpoint_kind_b: str | None = None
    checkpoint_path_a: str | None = None
    checkpoint_path_b: str | None = None
    sampling_run_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("model_id_a", self.model_id_a), ("model_id_b", self.model_id_b)):
            if not str(value).strip():
                raise SchemaError(f"generation_meta.{name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id_a": self.model_id_a,
            "model_id_b": self.model_id_b,
            "generation_a": dict(self.generation_a),
            "generation_b": dict(self.generation_b),
            "model_revision_a": self.model_revision_a,
            "model_revision_b": self.model_revision_b,
            "checkpoint_kind_a": self.checkpoint_kind_a,
            "checkpoint_kind_b": self.checkpoint_kind_b,
            "checkpoint_path_a": self.checkpoint_path_a,
            "checkpoint_path_b": self.checkpoint_path_b,
            "sampling_run_id": self.sampling_run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PreferenceGenerationMeta:
        if data is None:
            raise SchemaError("generation_meta is required")
        if not isinstance(data, dict):
            raise SchemaError("generation_meta must be an object")
        payload = dict(data)
        try:
            model_id_a = str(payload["model_id_a"])
            model_id_b = str(payload["model_id_b"])
        except KeyError as exc:
            raise SchemaError(
                "generation_meta requires model_id_a and model_id_b"
            ) from exc
        return cls(
            model_id_a=model_id_a,
            model_id_b=model_id_b,
            generation_a=dict(payload.get("generation_a") or {}),
            generation_b=dict(payload.get("generation_b") or {}),
            model_revision_a=(
                None
                if payload.get("model_revision_a") is None
                else str(payload.get("model_revision_a"))
            ),
            model_revision_b=(
                None
                if payload.get("model_revision_b") is None
                else str(payload.get("model_revision_b"))
            ),
            checkpoint_kind_a=(
                None
                if payload.get("checkpoint_kind_a") is None
                else str(payload.get("checkpoint_kind_a"))
            ),
            checkpoint_kind_b=(
                None
                if payload.get("checkpoint_kind_b") is None
                else str(payload.get("checkpoint_kind_b"))
            ),
            checkpoint_path_a=(
                None
                if payload.get("checkpoint_path_a") is None
                else str(payload.get("checkpoint_path_a"))
            ),
            checkpoint_path_b=(
                None
                if payload.get("checkpoint_path_b") is None
                else str(payload.get("checkpoint_path_b"))
            ),
            sampling_run_id=(
                None
                if payload.get("sampling_run_id") is None
                else str(payload.get("sampling_run_id"))
            ),
        )


@dataclass(frozen=True)
class PreferencePair:
    """One human preference judgment over two raw candidate explanations.

    ``response_a`` / ``response_b`` are stored exactly as produced — never
    normalized, stripped of meaning, or rewritten at I/O time.

    ``pair_id`` is a *content* identity over (clip, instruction, A, B, task) for
    duplicate-pair detection. ``judgment_id`` identifies this annotator's
    judgment so multiple annotations of the same content pair are first-class.

    Ties (``winner="tie"``) are rejected unless ``allow_ties=True``.
    """

    pair_id: str
    judgment_id: str
    clip_id: str
    instruction: str
    response_a: str
    response_b: str
    winner: str  # "a" | "b" | "tie"
    annotator_id: str
    timestamp: str
    provenance: Provenance
    generation_meta: PreferenceGenerationMeta
    task: TaskType = TaskType.EXPLANATION
    example_id: str | None = None
    video: VideoRef | None = None
    rationale: str | None = None
    rubric_version: str | None = None
    allow_ties: bool = False
    split: Split = Split.TRAIN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("pair_id", self.pair_id),
            ("judgment_id", self.judgment_id),
            ("clip_id", self.clip_id),
            ("instruction", self.instruction),
            ("annotator_id", self.annotator_id),
            ("timestamp", self.timestamp),
        ):
            if not str(value).strip():
                raise SchemaError(f"{name} must be a non-empty string")
        # Responses may be empty only as an explicit stored string; still must be str.
        if not isinstance(self.response_a, str) or not isinstance(self.response_b, str):
            raise SchemaError("response_a and response_b must be strings (raw text preserved)")
        if self.winner not in {"a", "b", "tie"}:
            raise SchemaError(
                f"invalid winner {self.winner!r}; expected 'a', 'b', or 'tie' "
                "(ties require allow_ties=True)"
            )
        if self.winner == "tie" and not self.allow_ties:
            raise SchemaError(
                "winner='tie' is only allowed when allow_ties=True "
                "(ties are opt-in; default preference protocol is forced choice)"
            )
        if not isinstance(self.task, TaskType):
            raise SchemaError(f"invalid task type: {self.task!r}")
        if not isinstance(self.split, Split):
            raise SchemaError(f"invalid split: {self.split!r}")
        if not isinstance(self.provenance, Provenance):
            raise SchemaError("provenance is required")
        if not isinstance(self.generation_meta, PreferenceGenerationMeta):
            raise SchemaError("generation_meta is required")

    @property
    def preferred_response(self) -> str | None:
        """Raw preferred text, or None on an allowed tie."""
        if self.winner == "a":
            return self.response_a
        if self.winner == "b":
            return self.response_b
        return None

    @property
    def rejected_response(self) -> str | None:
        """Raw rejected text, or None on an allowed tie."""
        if self.winner == "a":
            return self.response_b
        if self.winner == "b":
            return self.response_a
        return None

    def chosen_rejected(self) -> tuple[str, str]:
        """DPO-oriented (chosen, rejected) raw texts. Raises on ties."""
        if self.winner == "tie":
            raise SchemaError("chosen_rejected() is undefined for ties")
        chosen = self.preferred_response
        rejected = self.rejected_response
        assert chosen is not None and rejected is not None
        return chosen, rejected

    def bradley_terry_label(self) -> int:
        """+1 if A preferred, -1 if B preferred, 0 if tie."""
        if self.winner == "a":
            return 1
        if self.winner == "b":
            return -1
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize with generation_meta and annotation kept in separate objects."""
        return {
            "pair_id": self.pair_id,
            "judgment_id": self.judgment_id,
            "clip_id": self.clip_id,
            "example_id": self.example_id,
            "video": None if self.video is None else self.video.to_dict(),
            "task": self.task.value,
            "instruction": self.instruction,
            "response_a": self.response_a,
            "response_b": self.response_b,
            "split": self.split.value,
            "provenance": self.provenance.to_dict(),
            "generation_meta": self.generation_meta.to_dict(),
            "annotation": {
                "annotator_id": self.annotator_id,
                "timestamp": self.timestamp,
                "winner": self.winner,
                "rationale": self.rationale,
                "rubric_version": self.rubric_version,
                "allow_ties": self.allow_ties,
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreferencePair:
        """Parse one preference judgment.

        Accepts the nested ``generation_meta`` / ``annotation`` layout and a
        limited flat legacy layout (older stubs with only response/winner fields).
        Response strings are preserved exactly as provided.
        """
        if not isinstance(data, dict):
            raise SchemaError("preference record must be a JSON object")
        payload = dict(data)

        annotation = dict(payload.get("annotation") or {})
        # Flat legacy / convenience keys fall through into annotation.
        for key in (
            "annotator_id",
            "timestamp",
            "winner",
            "rationale",
            "rubric_version",
            "allow_ties",
        ):
            if key in payload and key not in annotation:
                annotation[key] = payload[key]
            elif key in payload and key in annotation and annotation[key] is None:
                annotation[key] = payload[key]

        if "winner" not in annotation or annotation.get("winner") is None:
            raise SchemaError("missing required field: winner (annotation.winner)")

        gen_raw = payload.get("generation_meta")
        if gen_raw is None and "model_id" in payload:
            # Minimal legacy: single model_id applied to both sides.
            mid = str(payload.get("model_id"))
            gen_raw = {
                "model_id_a": mid,
                "model_id_b": mid,
                "generation_a": dict(payload.get("generation") or {}),
                "generation_b": dict(payload.get("generation") or {}),
            }
        generation_meta = PreferenceGenerationMeta.from_dict(
            None if gen_raw is None else dict(gen_raw)
        )

        if "provenance" not in payload or payload.get("provenance") is None:
            raise SchemaError("missing required field: provenance")
        provenance = Provenance.from_dict(payload.get("provenance"))

        try:
            task = TaskType(str(payload.get("task", TaskType.EXPLANATION.value)))
        except ValueError as exc:
            raise SchemaError(f"invalid task type: {payload.get('task')!r}") from exc
        try:
            split = Split(str(payload.get("split", Split.TRAIN.value)))
        except ValueError as exc:
            raise SchemaError(f"invalid split: {payload.get('split')!r}") from exc

        video = None
        if payload.get("video") is not None:
            if not isinstance(payload["video"], dict):
                raise SchemaError("video must be an object when provided")
            video = VideoRef.from_dict(payload["video"])

        instruction = payload.get("instruction")
        if instruction is None:
            instruction = payload.get("question")
        if instruction is None:
            raise SchemaError("missing required field: instruction (or question alias)")

        clip_id = payload.get("clip_id")
        if clip_id is None or not str(clip_id).strip():
            raise SchemaError("missing required field: clip_id")

        if "response_a" not in payload or "response_b" not in payload:
            raise SchemaError("missing required field(s): response_a and/or response_b")
        # Preserve exactly — do not strip.
        response_a = payload["response_a"]
        response_b = payload["response_b"]
        if not isinstance(response_a, str) or not isinstance(response_b, str):
            raise SchemaError("response_a and response_b must be JSON strings")

        pair_id = payload.get("pair_id")
        if pair_id is None or not str(pair_id).strip():
            raise SchemaError("missing required field: pair_id")
        judgment_id = payload.get("judgment_id")
        if judgment_id is None or not str(judgment_id).strip():
            raise SchemaError("missing required field: judgment_id")

        annotator_id = annotation.get("annotator_id")
        if annotator_id is None or not str(annotator_id).strip():
            raise SchemaError("missing required field: annotator_id")
        timestamp = annotation.get("timestamp")
        if timestamp is None or not str(timestamp).strip():
            raise SchemaError("missing required field: timestamp")

        example_id = payload.get("example_id")
        if example_id is not None:
            example_id = str(example_id)

        return cls(
            pair_id=str(pair_id),
            judgment_id=str(judgment_id),
            clip_id=str(clip_id),
            example_id=example_id,
            video=video,
            task=task,
            instruction=str(instruction),
            response_a=response_a,
            response_b=response_b,
            winner=str(annotation["winner"]),
            annotator_id=str(annotator_id),
            timestamp=str(timestamp),
            rationale=(
                None if annotation.get("rationale") is None else str(annotation.get("rationale"))
            ),
            rubric_version=(
                None
                if annotation.get("rubric_version") is None
                else str(annotation.get("rubric_version"))
            ),
            allow_ties=bool(annotation.get("allow_ties", False)),
            provenance=provenance,
            generation_meta=generation_meta,
            split=split,
            metadata=dict(payload.get("metadata") or {}),
        )
