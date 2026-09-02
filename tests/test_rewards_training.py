import pytest

from magic_vlm.rewards import ExactMatchReward, LengthPenaltyReward, score_batch
from magic_vlm.schemas import ExampleRecord, InferenceArtifact, PreferencePair, Split, VideoRef
from magic_vlm.preferences import write_preference_pairs, load_preference_pairs
from magic_vlm.training import TrainingConfig, run_training, validate_training_split
from magic_vlm.dataset import SplitBoundaryError


def _ex(example_id: str = "e1", answer: str = "left") -> ExampleRecord:
    return ExampleRecord(
        example_id=example_id,
        split=Split.TRAIN,
        video=VideoRef(path="e.mp4"),
        question="q",
        answer=answer,
        trick_id="t",
        performer_id="p",
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
    pair = PreferencePair(
        pair_id="p1",
        example_id="e1",
        response_a="a",
        response_b="b",
        winner="a",
        split=Split.TRAIN,
    )
    path = tmp_path / "prefs.jsonl"
    write_preference_pairs(path, [pair])
    loaded = load_preference_pairs(path)
    assert loaded[0].winner == "a"


def test_training_refuses_held_out_and_algorithms() -> None:
    held = _ex("h")
    held = ExampleRecord(
        example_id="h",
        split=Split.HELD_OUT,
        video=VideoRef(path="h.mp4"),
        question="q",
        answer="left",
        trick_id="t",
        performer_id="p",
    )
    with pytest.raises(SplitBoundaryError):
        validate_training_split([held])

    train = [_ex()]
    result = run_training(TrainingConfig(algorithm="none"), train)
    assert result.status == "skipped"
    with pytest.raises(NotImplementedError):
        run_training(TrainingConfig(algorithm="dpo"), train)
