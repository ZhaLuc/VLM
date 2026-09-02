"""Stable data schemas for dataset records and research artifacts."""

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


@dataclass(frozen=True)
class ExampleRecord:
    """One Task-B/A style example.

    Split membership is part of the record so loaders can refuse accidental
    train/held-out mixing without consulting a separate sidecar.
    """

    example_id: str
    split: Split
    video: VideoRef
    question: str
    trick_id: str
    performer_id: str
    answer: str | None = None
    camera_id: str | None = None
    prop_id: str | None = None
    task: str = "hidden_state"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["split"] = self.split.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExampleRecord:
        payload = dict(data)
        split = payload.get("split", Split.TRAIN.value)
        payload["split"] = Split(split)
        video = payload.get("video")
        if isinstance(video, dict):
            payload["video"] = VideoRef(**video)
        payload.setdefault("metadata", {})
        return cls(**payload)


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
