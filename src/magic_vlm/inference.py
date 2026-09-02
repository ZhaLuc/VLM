"""VLM inference with raw-output preservation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from magic_vlm.models import VLMModel
from magic_vlm.schemas import ExampleRecord, InferenceArtifact
from magic_vlm.video import PreprocessedVideo


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            **self.extras,
        }


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
    # Prefer the last non-empty line for short label-style answers.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def run_inference(
    model: VLMModel,
    example: ExampleRecord,
    *,
    preprocessed: PreprocessedVideo | None = None,
    images: Sequence[Any] | None = None,
    generation: GenerationConfig | None = None,
    prompt: str | None = None,
) -> InferenceArtifact:
    """Run one inference call and return a preservable artifact.

    ``raw_text`` is always the untouched model string. Callers must serialize
    the full :class:`InferenceArtifact`, not only ``parsed_answer``.
    """
    gen = generation or GenerationConfig()
    used_prompt = prompt if prompt is not None else build_prompt(example)
    image_list = list(images) if images is not None else None
    raw_text = model.generate(
        used_prompt,
        images=image_list,
        **gen.to_dict(),
    )
    frame_indices = preprocessed.frame_indices if preprocessed is not None else ()
    return InferenceArtifact(
        example_id=example.example_id,
        model_id=getattr(model, "model_id", "unknown"),
        prompt=used_prompt,
        raw_text=raw_text,
        parsed_answer=parse_answer(raw_text),
        frame_indices=frame_indices,
        generation=gen.to_dict(),
        extras={
            "temporal_shuffled": bool(preprocessed.temporal_shuffled) if preprocessed else False,
        },
    )
