"""Tests for modular objective rewards (used by GRPO; reward logic stays here)."""

from __future__ import annotations

import pytest

from magic_vlm.rewards import (
    HIDDEN_STATE_EXACT_MATCH_ID,
    HIDDEN_STATE_EXACT_MATCH_VERSION,
    HiddenStateExactMatchReward,
    RewardConfig,
    RewardError,
    SHORTCUT_RISKS,
    build_reward,
    canonicalize_label,
    compute_hidden_state_exact_match_value,
    evaluate_batch,
    extract_prediction,
    list_registered_rewards,
)
from magic_vlm.schemas import (
    ExampleRecord,
    InferenceArtifact,
    Provenance,
    Split,
    TaskType,
    VideoRef,
)


def _ex(gold: str = "left") -> ExampleRecord:
    return ExampleRecord(
        example_id="e1",
        clip_id="clip_e1",
        trick_id="t",
        performer_id="p",
        camera_id="cam",
        video=VideoRef(path="e1.mp4"),
        task=TaskType.HIDDEN_STATE,
        question="Which cup contains the ball?",
        ground_truth=gold,
        split=Split.TRAIN,
        provenance=Provenance(source="unit_test"),
    )


def _art(*, raw: str, parsed: str | None) -> InferenceArtifact:
    return InferenceArtifact(
        example_id="e1",
        model_id="stub",
        prompt="p",
        raw_text=raw,
        parsed_answer=parsed,
    )


def test_correct_answer_reward_one() -> None:
    reward = HiddenStateExactMatchReward()
    result = reward.evaluate(_art(raw="Answer: left", parsed="left"), _ex("left"))
    assert result.value == 1.0
    assert type(result.value) is float
    assert result.matched is True
    assert result.parse_failed is False
    assert result.reward_id == HIDDEN_STATE_EXACT_MATCH_ID
    assert result.version == HIDDEN_STATE_EXACT_MATCH_VERSION
    assert result.extras.get("is_reasoning_metric") is False


def test_incorrect_answer_reward_zero() -> None:
    reward = build_reward("hidden_state_exact_match")
    result = reward.evaluate(_art(raw="Answer: right", parsed="right"), _ex("left"))
    assert result.value == 0.0
    assert type(result.value) is float
    assert result.matched is False


def test_formatting_variations_still_match() -> None:
    reward = HiddenStateExactMatchReward()
    variants = [
        _art(raw="Left", parsed="Left"),
        _art(raw="  LEFT  ", parsed="  LEFT  "),
        _art(raw="Because...\nAnswer: left", parsed="left"),
        _art(raw="Final answer: Left", parsed="Left"),
    ]
    for art in variants:
        assert reward.score(art, _ex("left")) == 1.0


def test_malformed_never_correct() -> None:
    reward = HiddenStateExactMatchReward()
    # Empty parse / missing label
    bad_cases = [
        _art(raw="", parsed=""),
        _art(raw="I am unsure.", parsed=None),
        _art(raw="something", parsed=""),
        _art(raw="   ", parsed="   "),
    ]
    for art in bad_cases:
        result = reward.evaluate(art, _ex("left"))
        assert result.value == 0.0
        assert result.matched is False
    # Even if the raw string contains the gold token, a failed parse scores 0.
    forced = _art(raw="left ish commentary", parsed="")
    result = reward.evaluate(forced, _ex("left"))
    assert result.parse_failed is True
    assert result.value == 0.0


def test_deterministic_type_and_value() -> None:
    reward = HiddenStateExactMatchReward()
    art = _art(raw="Answer: center", parsed="center")
    ex = _ex("center")
    values = [reward.score(art, ex) for _ in range(5)]
    assert values == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert all(type(v) is float for v in values)
    results = [reward.evaluate(art, ex) for _ in range(3)]
    assert len({r.to_dict()["value"] for r in results}) == 1


def test_parse_canonicalize_score_separated() -> None:
    art = _art(raw="Answer: Right", parsed="Right")
    pred, failed = extract_prediction(art)
    assert pred == "Right"
    assert failed is False
    assert canonicalize_label(pred) == "right"
    assert compute_hidden_state_exact_match_value(
        prediction=pred, gold="right", parse_failed=False
    ) == 1.0
    assert compute_hidden_state_exact_match_value(
        prediction=pred, gold="right", parse_failed=True
    ) == 0.0


def test_config_and_registry() -> None:
    cfg = RewardConfig.from_yaml("configs/reward_hidden_state_exact_match.yaml")
    assert cfg.reward_id == HIDDEN_STATE_EXACT_MATCH_ID
    assert cfg.version == HIDDEN_STATE_EXACT_MATCH_VERSION
    reward = cfg.build()
    assert reward.version == HIDDEN_STATE_EXACT_MATCH_VERSION
    registered = list_registered_rewards()
    assert HIDDEN_STATE_EXACT_MATCH_ID in registered
    assert "temporal_iou" in registered
    with pytest.raises(RewardError):
        build_reward("hybrid_reward")
    temporal = build_reward("temporal_iou")
    assert temporal.version == "1.0.0"
    # Without causal annotation, temporal reward scores 0 (does not invent gold).
    result = temporal.evaluate(_art(raw="start_s=0 end_s=1", parsed="x"), _ex())
    assert result.value == 0.0
    assert result.extras.get("eligible") is False


def test_shortcut_risks_documented() -> None:
    assert any("answer_frequency" in r for r in SHORTCUT_RISKS)
    assert any("parser" in r for r in SHORTCUT_RISKS)
    assert any("camera" in r for r in SHORTCUT_RISKS)
    result = HiddenStateExactMatchReward().evaluate(
        _art(raw="Answer: left", parsed="left"), _ex()
    )
    assert "shortcut_risks" in result.extras


def test_evaluate_batch() -> None:
    reward = HiddenStateExactMatchReward()
    arts = [
        _art(raw="Answer: left", parsed="left"),
        _art(raw="Answer: right", parsed="right"),
    ]
    # second artifact needs matching example id for batch helper - adjust
    arts[1] = InferenceArtifact(
        example_id="e2",
        model_id="stub",
        prompt="p",
        raw_text="Answer: right",
        parsed_answer="right",
    )
    examples = [_ex("left"), _ex("left")]
    examples[1] = ExampleRecord(
        example_id="e2",
        clip_id="c2",
        trick_id="t",
        performer_id="p",
        camera_id="cam",
        video=VideoRef(path="e2.mp4"),
        task=TaskType.HIDDEN_STATE,
        question="q",
        ground_truth="left",
        split=Split.TRAIN,
        provenance=Provenance(source="unit_test"),
    )
    results = evaluate_batch(reward, arts, examples)
    assert [r.value for r in results] == [1.0, 0.0]


def test_legacy_exact_match_reward_compat() -> None:
    from magic_vlm.rewards import ExactMatchReward

    reward = ExactMatchReward()
    assert reward.score(_art(raw="left", parsed="left"), _ex("left")) == 1.0
    assert reward.evaluate(_art(raw="left", parsed="left"), _ex("left")).version == (
        HIDDEN_STATE_EXACT_MATCH_VERSION
    )
