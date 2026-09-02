from magic_vlm.evaluation import exact_match, evaluate_exact_match
from magic_vlm.inference import GenerationConfig, build_prompt, parse_answer, run_inference
from magic_vlm.models import EchoStubVLM, ModelSpec, load_vlm
from magic_vlm.schemas import ExampleRecord, InferenceArtifact, Split, VideoRef
from magic_vlm.video import preprocess_video_meta


def _ex() -> ExampleRecord:
    return ExampleRecord(
        example_id="e1",
        split=Split.TRAIN,
        video=VideoRef(path="e1.mp4", num_frames=8),
        question="Which cup contains the ball?",
        answer="left",
        trick_id="cups",
        performer_id="a",
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
    assert isinstance(artifact, InferenceArtifact)
    assert "Which cup" in artifact.raw_text
    assert artifact.parsed_answer
    assert artifact.raw_text  # raw is first-class
    assert artifact.frame_indices == pre.frame_indices


def test_parse_and_exact_match() -> None:
    assert parse_answer("Reasoning...\nLeft") == "Left"
    assert exact_match(" Left ", "left")
    assert not exact_match("right", "left")


def test_evaluate_keeps_raw() -> None:
    example = _ex()
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
