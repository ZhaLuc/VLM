import pytest

from magic_vlm.rewards import ExactMatchReward, LengthPenaltyReward, score_batch
from magic_vlm.schemas import (
    ExampleRecord,
    InferenceArtifact,
    Provenance,
    Split,
    TaskType,
    VideoRef,
)
from magic_vlm.preferences import write_preference_pairs, load_preference_pairs
from magic_vlm.training import TrainingConfig, run_training, validate_training_split
from magic_vlm.dataset import SplitBoundaryError


def _ex(example_id: str = "e1", answer: str = "left", split: Split = Split.TRAIN) -> ExampleRecord:
    return ExampleRecord(
        example_id=example_id,
        clip_id=f"clip_{example_id}",
        trick_id="t",
        performer_id="p",
        camera_id="cam",
        video=VideoRef(path=f"{example_id}.mp4"),
        task=TaskType.HIDDEN_STATE,
        question="q",
        ground_truth=answer,
        split=split,
        provenance=Provenance(source="unit_test"),
    )


def test_rewards_independent_of_trainer() -> None:
    example = _ex()
    good = InferenceArtifact(
        example_id="e1",
        model_id="stub",
        prompt="p",
        raw_text="left",
        parsed_answer="left",
    )
    bad = InferenceArtifact(
        example_id="e1",
        model_id="stub",
        prompt="p",
        raw_text="right",
        parsed_answer="right",
    )
    reward = ExactMatchReward()
    assert reward.score(good, example) == 1.0
    assert reward.score(bad, example) == 0.0
    scores = score_batch(LengthPenaltyReward(), [good], [example])
    assert 0.0 < scores[0] <= 1.0


def test_preference_roundtrip(tmp_path) -> None:
    from magic_vlm.preferences import build_preference_pair
    from magic_vlm.schemas import PreferenceGenerationMeta

    pair = build_preference_pair(
        clip_id="clip_e1",
        example_id="e1",
        instruction="Explain the mechanism.",
        response_a="a",
        response_b="b",
        winner="a",
        annotator_id="ann_1",
        timestamp="2026-09-02T12:00:00+00:00",
        provenance=Provenance(source="unit_test"),
        generation_meta=PreferenceGenerationMeta(
            model_id_a="stub",
            model_id_b="stub",
        ),
        split=Split.TRAIN,
    )
    path = tmp_path / "prefs.jsonl"
    write_preference_pairs(path, [pair])
    loaded = load_preference_pairs(path)
    assert loaded[0].winner == "a"
    assert loaded[0].response_a == "a"
    assert loaded[0].pair_id == pair.pair_id


def test_training_refuses_held_out_and_algorithms() -> None:
    held = _ex("h", split=Split.HELD_OUT)
    with pytest.raises(SplitBoundaryError):
        validate_training_split([held])

    train = [_ex()]
    result = run_training(TrainingConfig(algorithm="none"), train)
    assert result.status == "skipped"
    with pytest.raises(NotImplementedError):
        run_training(TrainingConfig(algorithm="grpo"), train)
    with pytest.raises(ValueError, match="dpo_config"):
        run_training(TrainingConfig(algorithm="dpo"), train)
