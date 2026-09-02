"""Model loading independent of training algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelSpec:
    """Identifier and runtime options for a VLM checkpoint."""

    model_id: str
    revision: str | None = None
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True


@runtime_checkable
class VLMModel(Protocol):
    """Minimal inference-facing model surface.

    Training loops must not be embedded here; adapters/trainers belong in
    ``training`` once those stages are implemented.
    """

    model_id: str

    def generate(self, prompt: str, images: list[Any] | None = None, **kwargs: Any) -> str:
        """Return raw text output for a multimodal prompt."""


@dataclass
class EchoStubVLM:
    """Deterministic stub used for architecture tests without weight downloads."""

    model_id: str = "stub/echo"

    def generate(self, prompt: str, images: list[Any] | None = None, **kwargs: Any) -> str:
        n_images = 0 if images is None else len(images)
        return f"STUB_RESPONSE images={n_images} :: {prompt.strip()}"


def load_vlm(spec: ModelSpec, *, allow_download: bool = False) -> VLMModel:
    """Load a VLM for inference.

    Architecture validation uses ``EchoStubVLM`` when ``model_id`` starts with
    ``stub/``. Real checkpoints require ``magic-vlm[models]`` and explicit
    ``allow_download=True`` (or a local path already present).
    """
    if spec.model_id.startswith("stub/"):
        return EchoStubVLM(model_id=spec.model_id)

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for real model loading. Install magic-vlm[models]."
        ) from exc

    if not allow_download and not _looks_like_local_path(spec.model_id):
        raise RuntimeError(
            "Refusing to download weights during load_vlm(allow_download=False). "
            "Pass allow_download=True once intentionally ready, or point model_id "
            "at a local directory."
        )

    processor = AutoProcessor.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        torch_dtype=spec.torch_dtype,
        device_map=spec.device_map,
        trust_remote_code=spec.trust_remote_code,
    )
    return TransformersVLM(model_id=spec.model_id, model=model, processor=processor)


def _looks_like_local_path(model_id: str) -> bool:
    from pathlib import Path

    path = Path(model_id)
    return path.exists()


@dataclass
class TransformersVLM:
    """Thin wrapper keeping load/generate free of training concerns."""

    model_id: str
    model: Any
    processor: Any

    def generate(self, prompt: str, images: list[Any] | None = None, **kwargs: Any) -> str:
        # Real multimodal chat templating is deferred to the baseline-inference stage.
        # This method exists so model loading stays usable without baking in trainers.
        max_new_tokens = int(kwargs.get("max_new_tokens", 128))
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text
