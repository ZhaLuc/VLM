from __future__ import annotations

from magic_vlm.evaluation import exact_match, evaluate_exact_match, normalize_label
from magic_vlm.inference import GenerationConfig, build_prompt, parse_answer, run_inference
from magic_vlm.models import EchoStubVLM, ModelSpec, load_vlm
from magic_vlm.schemas import ExampleRecord, Provenance, Split, TaskType, VideoRef
from magic_vlm.video import preprocess_video_meta


def _ex() -> ExampleRecord:
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


def test_stub_load_and_raw_preservation() -> None:
    model = load_vlm(ModelSpec(model_id="stub/echo"), allow_download=False)
    assert isinstance(model, EchoStubVLM)
    example = _ex()
    pre = preprocess_video_meta(example.video.path, num_frames=8)
    artifact = run_inference(
        model,
        example,
        preprocessed=pre,
        generation=GenerationConfig(max_new_tokens=16),
    )
    assert "Which cup" in artifact.raw_text
    assert artifact.parsed_answer
    assert artifact.frame_indices == pre.frame_indices


def test_parse_and_exact_match() -> None:
    assert parse_answer("Reasoning...\nLeft") == "Left"
    assert exact_match(" Left ", "left")
    assert not exact_match("right", "left")
    # Normalization is compare-time only; authored gold stays intact elsewhere.
    assert normalize_label(" Top Pocket ") == "top pocket"


def test_evaluate_keeps_raw_and_ground_truth() -> None:
    example = _ex()
    from magic_vlm.schemas import InferenceArtifact

    artifact = InferenceArtifact(
        example_id="e1",
        model_id="stub/echo",
        prompt=build_prompt(example),
        raw_text="I think it is left",
        parsed_answer="left",
    )
    report = evaluate_exact_match([example], [artifact])
    assert report.accuracy == 1.0
    assert report.scores[0].raw_text == "I think it is left"
    assert report.scores[0].gold == "left"
