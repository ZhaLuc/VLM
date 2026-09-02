"""VLM inference with raw-output preservation.

Parsing is independent of generation: ``parse_answer`` never mutates ``raw_text``.
No weight updates occur in this module.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from magic_vlm.models import VLMModel
from magic_vlm.runtime import DeviceInfo
from magic_vlm.schemas import ExampleRecord, InferenceArtifact
from magic_vlm.video import PreprocessedVideo, SampledClip

ParseFn = Callable[[str], str | None]


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding parameters that can change scientific results.

    Always serialize the full object into run metadata. ``top_p`` / ``top_k``
    are recorded even when ``do_sample`` is False so configs remain explicit.
    """

    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "sampling_mode": "sample" if self.do_sample else "greedy",
        }
        if self.extras:
            payload["extras"] = dict(self.extras)
        return payload

    def to_generate_kwargs(self) -> dict[str, Any]:
        """Hugging Face ``model.generate`` kwargs only (no metadata keys)."""
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            if self.top_k > 0:
                kwargs["top_k"] = self.top_k
        extras = dict(self.extras)
        extras.pop("sampling_mode", None)
        kwargs.update(extras)
        return kwargs

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GenerationConfig:
        raw = dict(data or {})
        raw.pop("sampling_mode", None)
        extras = dict(raw.pop("extras", {}) or {})
        known = {
            "max_new_tokens": int(raw.pop("max_new_tokens", 128)),
            "temperature": float(raw.pop("temperature", 0.0)),
            "top_p": float(raw.pop("top_p", 1.0)),
            "top_k": int(raw.pop("top_k", 0)),
            "do_sample": bool(raw.pop("do_sample", False)),
        }
        extras.update(raw)
        return cls(**known, extras=extras)


def build_prompt(example: ExampleRecord) -> str:
    """Default prompt template for hidden-state / explanation questions."""
    return (
        "You are analyzing a short magic or mentalism demonstration.\n"
        f"Question: {example.question}\n"
        "Answer briefly and concretely."
    )


def parse_answer(raw_text: str) -> str:
    """Lightweight parse that does not discard the raw string elsewhere."""
    text = raw_text.strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def _bgr_frames_to_rgb(frames: Sequence[Any]) -> list[Any]:
    converted: list[Any] = []
    for frame in frames:
        if hasattr(frame, "ndim") and getattr(frame, "ndim", 0) == 3 and frame.shape[-1] == 3:
            converted.append(frame[..., ::-1].copy())
        else:
            converted.append(frame)
    return converted


def _video_payload(preprocessed: PreprocessedVideo | SampledClip | None) -> tuple[list[Any] | None, dict[str, Any]]:
    """Return (videos_for_model, preprocessing_metadata)."""
    if preprocessed is None:
        return None, {}
    meta = {
        "source_path": getattr(preprocessed, "source_path", None),
        "ordered_indices": list(getattr(preprocessed, "ordered_indices", ()) or preprocessed.frame_indices),
        "frame_indices": list(preprocessed.frame_indices),
        "temporal_shuffled": bool(preprocessed.temporal_shuffled),
        "shuffle_seed": getattr(preprocessed, "shuffle_seed", None),
        "sample_strategy": getattr(preprocessed, "sample_strategy", None),
        "max_frames": getattr(preprocessed, "max_frames", None),
        "n_frames": getattr(preprocessed, "n_frames", len(preprocessed.frame_indices)),
        "resize": None
        if getattr(preprocessed, "resize", None) is None
        else preprocessed.resize.to_dict(),  # type: ignore[union-attr]
        "source_fps": getattr(preprocessed, "source_fps", None),
        "source_content_hash": getattr(preprocessed, "source_content_hash", None),
    }
    frames = getattr(preprocessed, "frames", None)
    if frames:
        meta["video_input_mode"] = "project_sampled_frames"
        return [_bgr_frames_to_rgb(list(frames))], meta
    meta["video_input_mode"] = "indices_only_no_pixels"
    return None, meta


def run_inference(
    model: VLMModel,
    example: ExampleRecord,
    *,
    preprocessed: PreprocessedVideo | SampledClip | None = None,
    images: Sequence[Any] | None = None,
    videos: Sequence[Any] | None = None,
    generation: GenerationConfig | None = None,
    prompt: str | None = None,
    parse: ParseFn | None = parse_answer,
    device: DeviceInfo | str | None = None,
    checkpoint_kind: str | None = None,
    checkpoint_path: str | None = None,
) -> InferenceArtifact:
    """Run one inference call and return a preservable artifact.

    ``raw_text`` is always the untouched model string. Parsing is applied
    afterwards and never overwrites the raw field.
    """
    gen = generation or GenerationConfig()
    used_prompt = prompt if prompt is not None else build_prompt(example)
    image_list = list(images) if images is not None else None
    derived_videos, preprocess_meta = _video_payload(preprocessed)
    video_list = list(videos) if videos is not None else derived_videos

    device_str: str | None
    if isinstance(device, DeviceInfo):
        device_str = device.resolved
    elif isinstance(device, str):
        device_str = device
    else:
        device_str = getattr(model, "device", None)

    started = time.perf_counter()
    raw_text = model.generate(
        used_prompt,
        images=image_list,
        videos=video_list,
        **gen.to_generate_kwargs(),
    )
    latency_s = time.perf_counter() - started

    parsed = parse(raw_text) if parse is not None else None
    frame_indices = preprocessed.frame_indices if preprocessed is not None else ()
    return InferenceArtifact(
        example_id=example.example_id,
        clip_id=example.clip_id,
        task=example.task.value if hasattr(example.task, "value") else str(example.task),
        question=example.question,
        model_id=getattr(model, "model_id", "unknown"),
        model_revision=getattr(model, "revision", None),
        checkpoint_kind=checkpoint_kind,
        checkpoint_path=checkpoint_path,
        prompt=used_prompt,
        raw_text=raw_text,
        parsed_answer=parsed,
        frame_indices=frame_indices,
        generation=gen.to_dict(),
        preprocessing=preprocess_meta,
        device=device_str,
        latency_s=latency_s,
        extras={
            "temporal_shuffled": bool(preprocessed.temporal_shuffled) if preprocessed else False,
        },
    )


def run_inference_batch(
    model: VLMModel,
    examples: Sequence[ExampleRecord],
    *,
    preprocessed: Sequence[PreprocessedVideo | SampledClip | None] | None = None,
    generation: GenerationConfig | None = None,
    parse: ParseFn | None = parse_answer,
    device: DeviceInfo | str | None = None,
    checkpoint_kind: str | None = None,
    checkpoint_path: str | None = None,
    batch_size: int = 1,
) -> list[InferenceArtifact]:
    """Run inference over many examples.

    Default ``batch_size=1`` (sequential). True padded video batching is not
    assumed; requesting ``batch_size>1`` still runs sequentially and records
    ``batch_size`` in extras so claims stay honest.
    """
    artifacts: list[InferenceArtifact] = []
    prep_list: Sequence[PreprocessedVideo | SampledClip | None]
    if preprocessed is None:
        prep_list = [None] * len(examples)
    else:
        if len(preprocessed) != len(examples):
            raise ValueError("preprocessed length must match examples")
        prep_list = preprocessed
    for example, prep in zip(examples, prep_list):
        artifact = run_inference(
            model,
            example,
            preprocessed=prep,
            generation=generation,
            parse=parse,
            device=device,
            checkpoint_kind=checkpoint_kind,
            checkpoint_path=checkpoint_path,
        )
        extras = dict(artifact.extras)
        extras["requested_batch_size"] = batch_size
        extras["batching"] = "sequential"
        artifacts.append(replace(artifact, extras=extras))
    return artifacts
