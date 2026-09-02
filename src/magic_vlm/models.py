"""Model loading independent of training algorithms.

Verified against Hugging Face Transformers 5.16.1 (local install):
- ``Qwen2_5_VLForConditionalGeneration`` exists and is the documented Qwen2.5-VL class
- ``AutoProcessor`` / ``Qwen2_5_VLProcessor.apply_chat_template`` exists
- Official video messages use ``{"type": "video", "path": ...}`` plus optional ``fps``
- This project prefers passing **already sampled frames** so later temporal-shuffle
  experiments reuse the same sample set (processor-side resampling is a distinct mode)

Weights are never modified here. No LoRA / DPO / GRPO / PPO.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from magic_vlm.runtime import DeviceConfig, DeviceInfo, resolve_device


@dataclass(frozen=True)
class ModelSpec:
    """Identifier and runtime options for a VLM checkpoint."""

    model_id: str
    revision: str | None = None
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    attn_implementation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class VLMModel(Protocol):
    """Minimal inference-facing model surface.

    Training loops must not be embedded here; adapters/trainers belong in
    ``training`` once those stages are implemented.
    """

    model_id: str

    def generate(
        self,
        prompt: str,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Return raw text output for a multimodal prompt."""


@dataclass
class EchoStubVLM:
    """Deterministic stub used for architecture tests without weight downloads."""

    model_id: str = "stub/echo"
    revision: str | None = None
    device: str = "cpu"
    last_call: dict[str, Any] = field(default_factory=dict)

    def generate(
        self,
        prompt: str,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        **kwargs: Any,
    ) -> str:
        n_images = 0 if images is None else len(images)
        n_video_items = 0 if videos is None else len(videos)
        n_frames = 0
        if videos:
            first = videos[0]
            if hasattr(first, "shape"):
                n_frames = int(first.shape[0]) if len(first.shape) == 4 else 1
            elif isinstance(first, (list, tuple)):
                n_frames = len(first)
        self.last_call = {
            "prompt": prompt,
            "n_images": n_images,
            "n_videos": n_video_items,
            "n_frames": n_frames,
            "kwargs": dict(kwargs),
        }
        return (
            f"STUB_RESPONSE images={n_images} videos={n_video_items} "
            f"frames={n_frames} :: {prompt.strip()}"
        )


def load_vlm(
    spec: ModelSpec,
    *,
    allow_download: bool = False,
    device: DeviceConfig | DeviceInfo | None = None,
) -> VLMModel:
    """Load a VLM for inference only (eval mode; weights not trained).

    ``stub/`` model ids never touch Hugging Face. Real checkpoints require
    ``magic-vlm[models]`` and either a local path or explicit ``allow_download=True``.
    """
    resolved: DeviceInfo | None
    if isinstance(device, DeviceInfo):
        resolved = device
    elif isinstance(device, DeviceConfig):
        resolved = resolve_device(device)
    else:
        resolved = None

    if spec.model_id.startswith("stub/"):
        stub_device = resolved.resolved if resolved is not None else "cpu"
        return EchoStubVLM(model_id=spec.model_id, revision=spec.revision, device=stub_device)

    try:
        from transformers import AutoProcessor  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for real model loading. Install magic-vlm[models]."
        ) from exc

    local_only = (not allow_download) and (not _looks_like_local_path(spec.model_id))
    if local_only:
        raise RuntimeError(
            "Refusing to download weights during load_vlm(allow_download=False). "
            "Pass allow_download=True once intentionally ready, or point model_id "
            "at a local directory."
        )

    local_files_only = not allow_download
    processor = AutoProcessor.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=local_files_only,
    )
    model = _load_qwen25_or_auto(spec, local_files_only=local_files_only)
    model.eval()
    device_str = resolved.resolved if resolved is not None else _infer_model_device_str(model)
    return TransformersVLM(
        model_id=spec.model_id,
        revision=spec.revision,
        model=model,
        processor=processor,
        device=device_str,
        spec=spec,
    )


def _load_qwen25_or_auto(spec: ModelSpec, *, local_files_only: bool) -> Any:
    """Load Qwen2.5-VL via the verified class, with Auto* fallback."""
    load_kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "trust_remote_code": spec.trust_remote_code,
        "local_files_only": local_files_only,
        "device_map": spec.device_map,
    }
    if spec.attn_implementation:
        load_kwargs["attn_implementation"] = spec.attn_implementation

    # Transformers 5.x docs use ``dtype``; older releases used ``torch_dtype``.
    dtype = _resolve_torch_dtype(spec.torch_dtype)

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

        model_cls = Qwen2_5_VLForConditionalGeneration
    except ImportError:  # pragma: no cover
        from transformers import AutoModelForImageTextToText  # type: ignore

        model_cls = AutoModelForImageTextToText

    try:
        return model_cls.from_pretrained(spec.model_id, dtype=dtype, **load_kwargs)
    except TypeError:
        return model_cls.from_pretrained(spec.model_id, torch_dtype=dtype, **load_kwargs)


def _resolve_torch_dtype(name: str) -> Any:
    try:
        import torch  # type: ignore
    except ImportError:  # pragma: no cover
        return name
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    return mapping.get(name, name)


def _infer_model_device_str(model: Any) -> str:
    if hasattr(model, "device"):
        return str(model.device)
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def _looks_like_local_path(model_id: str) -> bool:
    return Path(model_id).exists()


@dataclass
class TransformersVLM:
    """Inference wrapper: chat-template + generate. Does not train or mutate weights."""

    model_id: str
    model: Any
    processor: Any
    revision: str | None = None
    device: str = "cpu"
    spec: ModelSpec | None = None

    def generate(
        self,
        prompt: str,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        **kwargs: Any,
    ) -> str:
        generation = kwargs.pop("generation", None)
        if generation is not None and hasattr(generation, "to_generate_kwargs"):
            gen_kwargs = generation.to_generate_kwargs()
        else:
            gen_kwargs = {
                key: kwargs[key]
                for key in ("max_new_tokens", "do_sample", "temperature", "top_p", "top_k")
                if key in kwargs
            }
            if not gen_kwargs:
                gen_kwargs = {"max_new_tokens": 128, "do_sample": False}
            if not gen_kwargs.get("do_sample"):
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
                gen_kwargs.pop("top_k", None)

        messages = build_qwen_messages(prompt, images=images, videos=videos)
        model_inputs = self._prepare_inputs(messages, videos=videos, images=images)
        input_ids = model_inputs["input_ids"]
        output_ids = self.model.generate(**model_inputs, **gen_kwargs)
        trimmed = [out[len(inp) :] for inp, out in zip(input_ids, output_ids)]
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text

    def _prepare_inputs(
        self,
        messages: list[dict[str, Any]],
        *,
        videos: list[Any] | None,
        images: list[Any] | None,
    ) -> dict[str, Any]:
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        try:
            encoded = self.processor.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )
        encoded = _move_to_device(encoded, self.device)
        return encoded


def build_qwen_messages(
    prompt: str,
    *,
    images: list[Any] | None = None,
    videos: list[Any] | None = None,
    video_path: str | None = None,
) -> list[dict[str, Any]]:
    """Build a Qwen2.5-VL chat message list (verified HF 5.16 / docs pattern)."""
    content: list[dict[str, Any]] = []
    if videos:
        for video in videos:
            content.append({"type": "video", "video": video})
    elif video_path:
        content.append({"type": "video", "path": video_path})
    if images:
        for image in images:
            content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _move_to_device(encoded: Any, device: str) -> Any:
    if hasattr(encoded, "to"):
        return encoded.to(device)
    if isinstance(encoded, dict):
        moved = {}
        for key, value in encoded.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved
    return encoded
