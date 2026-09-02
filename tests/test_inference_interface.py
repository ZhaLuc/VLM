from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from magic_vlm.inference import (
    GenerationConfig,
    parse_answer,
    run_inference,
    run_inference_batch,
)
from magic_vlm.models import EchoStubVLM, ModelSpec, TransformersVLM, load_vlm
from magic_vlm.runtime import DeviceConfig, resolve_device
from magic_vlm.schemas import ExampleRecord, Provenance, Split, TaskType, VideoRef
from magic_vlm.video import SampledClip, preprocess_video_meta


def _example() -> ExampleRecord:
    return ExampleRecord(
        example_id="e1",
        clip_id="clip_e1",
        trick_id="cups",
        performer_id="a",
        camera_id="cam_front",
        video=VideoRef(path="e1.mp4", num_frames=8),
        task=TaskType.HIDDEN_STATE,
        question="Which cup contains the ball?",
        ground_truth="left",
        split=Split.TRAIN,
        provenance=Provenance(source="unit_test"),
    )


def test_generation_kwargs_exclude_metadata() -> None:
    greedy = GenerationConfig(max_new_tokens=32, temperature=0.0, do_sample=False)
    kwargs = greedy.to_generate_kwargs()
    assert "sampling_mode" not in kwargs
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    sampled = GenerationConfig(max_new_tokens=16, temperature=0.7, top_p=0.9, top_k=20, do_sample=True)
    sk = sampled.to_generate_kwargs()
    assert sk["temperature"] == 0.7
    assert sk["top_p"] == 0.9
    assert sk["top_k"] == 20


def test_parse_independent_of_raw() -> None:
    raw = "Reasoning about the cups.\nLeft"
    parsed = parse_answer(raw)
    assert parsed == "Left"
    assert raw == "Reasoning about the cups.\nLeft"


def test_stub_inference_preserves_raw_and_metadata() -> None:
    model = load_vlm(ModelSpec(model_id="stub/echo"), allow_download=False, device=DeviceConfig(preference="cpu"))
    example = _example()
    pre = preprocess_video_meta(example.video.path, num_frames=8)
    artifact = run_inference(
        model,
        example,
        preprocessed=pre,
        generation=GenerationConfig(max_new_tokens=16, do_sample=False),
        device=resolve_device(DeviceConfig(preference="cpu")),
        checkpoint_kind="stub",
    )
    assert artifact.raw_text.startswith("STUB_RESPONSE")
    assert artifact.parsed_answer
    assert artifact.raw_text != artifact.parsed_answer or "STUB" in artifact.raw_text
    assert artifact.clip_id == "clip_e1"
    assert artifact.question == example.question
    assert artifact.task == "hidden_state"
    assert artifact.generation["do_sample"] is False
    assert artifact.generation["max_new_tokens"] == 16
    assert artifact.latency_s is not None and artifact.latency_s >= 0
    assert artifact.device == "cpu"
    assert artifact.checkpoint_kind == "stub"
    assert artifact.preprocessing["frame_indices"] == list(pre.frame_indices)
    assert artifact.frame_indices == pre.frame_indices


def test_load_refuses_download() -> None:
    with pytest.raises(RuntimeError, match="Refusing to download"):
        load_vlm(ModelSpec(model_id="Qwen/Qwen2.5-VL-7B-Instruct"), allow_download=False)


def test_sampled_frames_passed_to_stub() -> None:
    import numpy as np

    frames = tuple(np.zeros((4, 4, 3), dtype="uint8") for _ in range(3))
    clip = SampledClip(
        source_path="e1.mp4",
        ordered_indices=(0, 4, 7),
        frame_indices=(0, 4, 7),
        temporal_shuffled=False,
        shuffle_seed=None,
        sample_strategy="uniform",
        max_frames=3,
        source_num_frames=8,
        frames=frames,
    )
    model = EchoStubVLM()
    artifact = run_inference(model, _example(), preprocessed=clip)
    assert model.last_call["n_frames"] == 3
    assert artifact.preprocessing["video_input_mode"] == "project_sampled_frames"
    assert "STUB_RESPONSE" in artifact.raw_text


def test_batch_is_sequential() -> None:
    model = EchoStubVLM()
    examples = [_example()]
    artifacts = run_inference_batch(model, examples, batch_size=4)
    assert len(artifacts) == 1
    assert artifacts[0].extras["batching"] == "sequential"
    assert artifacts[0].extras["requested_batch_size"] == 4


@dataclass
class _FakeProcessor:
    last_messages: Any = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        assert kwargs.get("add_generation_prompt") is True
        import torch

        return {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ["RAW MODEL TEXT\nleft"]


@dataclass
class _FakeModel:
    device = "cpu"
    last_generate_kwargs: dict[str, Any] | None = None

    def generate(self, **kwargs):
        import torch

        self.last_generate_kwargs = dict(kwargs)
        inp = kwargs["input_ids"]
        extra = torch.tensor([[99, 100]])
        return torch.cat([inp, extra], dim=1)

    def eval(self):
        return self

    def parameters(self):
        import torch

        yield torch.zeros(1)


def test_transformers_wrapper_mocked_qwen_path() -> None:
    fake_model = _FakeModel()
    fake_processor = _FakeProcessor()
    wrapper = TransformersVLM(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        model=fake_model,
        processor=fake_processor,
        device="cpu",
        revision=None,
    )
    text = wrapper.generate(
        "Which cup?",
        videos=[[[0, 0, 0]]],
        max_new_tokens=8,
        do_sample=False,
    )
    assert text == "RAW MODEL TEXT\nleft"
    assert fake_processor.last_messages[0]["content"][0]["type"] == "video"
    assert fake_model.last_generate_kwargs is not None
    assert fake_model.last_generate_kwargs["max_new_tokens"] == 8
    assert fake_model.last_generate_kwargs["do_sample"] is False
    assert "sampling_mode" not in fake_model.last_generate_kwargs

    example = _example()
    artifact = run_inference(
        wrapper,
        example,
        generation=GenerationConfig(max_new_tokens=8, do_sample=False),
        parse=parse_answer,
    )
    assert artifact.raw_text == "RAW MODEL TEXT\nleft"
    assert artifact.parsed_answer == "left"
    assert artifact.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
