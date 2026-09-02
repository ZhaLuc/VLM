"""Tests for pairwise preference representation and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from magic_vlm.preferences import (
    PreferenceValidationConfig,
    bradley_terry_rows,
    build_preference_pair,
    compute_content_pair_id,
    compute_judgment_id,
    dpo_training_rows,
    group_judgments_by_pair,
    load_preference_pairs,
    validate_preference_pairs,
    write_preference_pairs,
)
from magic_vlm.schemas import (
    PreferenceGenerationMeta,
    PreferencePair,
    Provenance,
    SchemaError,
    Split,
    TaskType,
    VideoRef,
)


def _gen(
    model: str = "stub/echo",
    *,
    temp: float = 0.7,
) -> PreferenceGenerationMeta:
    return PreferenceGenerationMeta(
        model_id_a=model,
        model_id_b=model,
        generation_a={"max_new_tokens": 128, "temperature": temp, "do_sample": True},
        generation_b={"max_new_tokens": 128, "temperature": temp, "do_sample": True},
        checkpoint_kind_a="base",
        checkpoint_kind_b="base",
        sampling_run_id="sample-run-1",
    )


def _prov() -> Provenance:
    return Provenance(source="human_preference_pilot", created_by="annotator_protocol_v1")


def _pair(
    *,
    response_a: str = "The ball is palmed during the third pass.",
    response_b: str = "A magnet holds the ball under the cup.",
    winner: str = "a",
    annotator_id: str = "ann_1",
    timestamp: str = "2026-09-02T20:00:00+00:00",
    rationale: str | None = "Clearer causal mechanism.",
    allow_ties: bool = False,
    clip_id: str = "clip_cups_01",
    instruction: str = "Explain the most likely hidden mechanism.",
    provenance: Provenance | None = None,
    generation_meta: PreferenceGenerationMeta | None = None,
    pair_id: str | None = None,
    judgment_id: str | None = None,
) -> PreferencePair:
    return build_preference_pair(
        clip_id=clip_id,
        example_id="ex_cups_01",
        video=VideoRef(path="data/videos/clip_cups_01.mp4"),
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
        winner=winner,
        annotator_id=annotator_id,
        timestamp=timestamp,
        rationale=rationale,
        provenance=provenance if provenance is not None else _prov(),
        generation_meta=generation_meta if generation_meta is not None else _gen(),
        allow_ties=allow_ties,
        split=Split.TRAIN,
        pair_id=pair_id,
        judgment_id=judgment_id,
        rubric_version="explanation_v1",
    )


def test_valid_preference_roundtrip(tmp_path: Path) -> None:
    pair = _pair()
    path = tmp_path / "prefs.jsonl"
    write_preference_pairs(path, [pair])
    loaded = load_preference_pairs(path)
    assert len(loaded) == 1
    got = loaded[0]
    assert got.winner == "a"
    assert got.response_a == pair.response_a
    assert got.response_b == pair.response_b
    assert got.generation_meta.model_id_a == "stub/echo"
    assert got.annotator_id == "ann_1"
    assert got.rationale == "Clearer causal mechanism."
    assert got.provenance.source == "human_preference_pilot"
    # Nested serialization keeps generation separate from annotation.
    raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "generation_meta" in raw
    assert "annotation" in raw
    assert "winner" in raw["annotation"]
    assert "winner" not in raw["generation_meta"]
    assert "model_id_a" not in raw["annotation"]


def test_raw_responses_not_normalized() -> None:
    a = "  Leading spaces matter.\n"
    b = "Trailing spaces matter.  "
    pair = _pair(response_a=a, response_b=b)
    assert pair.response_a == a
    assert pair.response_b == b
    blob = pair.to_dict()
    assert blob["response_a"] == a
    assert blob["response_b"] == b
    again = PreferencePair.from_dict(blob)
    assert again.response_a == a
    assert again.response_b == b


def test_stable_pair_id_duplicate_detection() -> None:
    p1 = _pair()
    p2 = _pair(annotator_id="ann_2", timestamp="2026-09-02T21:00:00+00:00")
    assert p1.pair_id == p2.pair_id
    assert p1.judgment_id != p2.judgment_id
    expected = compute_content_pair_id(
        clip_id=p1.clip_id,
        instruction=p1.instruction,
        response_a=p1.response_a,
        response_b=p1.response_b,
        task=p1.task,
    )
    assert p1.pair_id == expected


def test_identical_responses_fail_validation() -> None:
    twin = "same text"
    pair = _pair(response_a=twin, response_b=twin)
    report = validate_preference_pairs([pair])
    assert not report.passed
    assert any(f.code == "identical_responses" for f in report.findings)


def test_identical_responses_allowed_as_review() -> None:
    twin = "same text"
    pair = _pair(response_a=twin, response_b=twin)
    report = validate_preference_pairs(
        [pair],
        config=PreferenceValidationConfig(allow_identical_responses=True),
    )
    assert report.passed
    assert any(f.code == "identical_responses" for f in report.findings)


def test_invalid_winner() -> None:
    with pytest.raises(SchemaError, match="invalid winner"):
        _pair(winner="c")


def test_missing_winner() -> None:
    payload = _pair().to_dict()
    del payload["annotation"]["winner"]
    with pytest.raises(SchemaError, match="winner"):
        PreferencePair.from_dict(payload)


def test_tie_requires_allow_ties() -> None:
    with pytest.raises(SchemaError, match="allow_ties"):
        _pair(winner="tie", allow_ties=False)
    tied = _pair(winner="tie", allow_ties=True)
    assert tied.bradley_terry_label() == 0
    report = validate_preference_pairs(
        [tied],
        config=PreferenceValidationConfig(allow_ties=False),
    )
    assert not report.passed
    assert any(f.code == "tie_not_allowed" for f in report.findings)
    ok = validate_preference_pairs(
        [tied],
        config=PreferenceValidationConfig(allow_ties=True),
    )
    assert ok.passed


def test_missing_provenance() -> None:
    payload = _pair().to_dict()
    del payload["provenance"]
    with pytest.raises(SchemaError, match="provenance"):
        PreferencePair.from_dict(payload)


def test_optional_rationale() -> None:
    pair = _pair(rationale=None)
    assert pair.rationale is None
    report = validate_preference_pairs([pair])
    assert report.passed
    required = validate_preference_pairs(
        [pair],
        config=PreferenceValidationConfig(require_rationale=True),
    )
    assert not required.passed


def test_duplicate_pair_ids_policy() -> None:
    p1 = _pair(annotator_id="ann_1", timestamp="2026-09-02T20:00:00+00:00")
    p2 = _pair(annotator_id="ann_2", timestamp="2026-09-02T21:00:00+00:00")
    assert p1.pair_id == p2.pair_id
    multi = validate_preference_pairs([p1, p2])
    assert multi.passed
    assert any(f.code == "multiple_annotations" for f in multi.findings)
    strict = validate_preference_pairs(
        [p1, p2],
        config=PreferenceValidationConfig(allow_multiple_annotations_per_pair=False),
    )
    assert not strict.passed
    assert any(f.code == "duplicate_pair_id" for f in strict.findings)


def test_duplicate_judgment_ids() -> None:
    p1 = _pair()
    p2 = _pair(
        annotator_id="ann_other",
        timestamp="2026-09-02T22:00:00+00:00",
        judgment_id=p1.judgment_id,
        pair_id=compute_content_pair_id(
            clip_id="other_clip",
            instruction="other",
            response_a="x",
            response_b="y",
        ),
        response_a="x",
        response_b="y",
        clip_id="other_clip",
        instruction="other",
    )
    report = validate_preference_pairs([p1, p2])
    assert not report.passed
    assert any(f.code == "duplicate_judgment_id" for f in report.findings)


def test_multiple_annotations_grouping() -> None:
    p1 = _pair(annotator_id="ann_1", timestamp="2026-09-02T20:00:00+00:00", winner="a")
    p2 = _pair(annotator_id="ann_2", timestamp="2026-09-02T21:00:00+00:00", winner="b")
    grouped = group_judgments_by_pair([p1, p2])
    assert len(grouped[p1.pair_id]) == 2
    report = validate_preference_pairs([p1, p2])
    assert any(f.code == "annotator_disagreement" for f in report.findings)


def test_dpo_and_bradley_terry_projections() -> None:
    preferred = _pair(winner="a")
    other = _pair(
        winner="b",
        response_a="alpha",
        response_b="beta",
        annotator_id="ann_x",
        timestamp="2026-09-02T23:00:00+00:00",
    )
    dpo = dpo_training_rows([preferred, other])
    assert dpo[0]["chosen"] == preferred.response_a
    assert dpo[0]["rejected"] == preferred.response_b
    assert dpo[1]["chosen"] == "beta"
    assert dpo[1]["rejected"] == "alpha"
    bt = bradley_terry_rows([preferred, other])
    assert bt[0]["label"] == 1
    assert bt[1]["label"] == -1


def test_judgment_id_helper_stable() -> None:
    j1 = compute_judgment_id(
        pair_id="pref_abc",
        annotator_id="ann_1",
        timestamp="2026-09-02T20:00:00+00:00",
    )
    j2 = compute_judgment_id(
        pair_id="pref_abc",
        annotator_id="ann_1",
        timestamp="2026-09-02T20:00:00+00:00",
    )
    assert j1 == j2
    assert j1.startswith("judg_")


def test_toy_fixture_loads() -> None:
    path = Path("data/examples/toy_preferences.jsonl")
    pairs = load_preference_pairs(path)
    assert len(pairs) >= 2
    report = validate_preference_pairs(pairs)
    assert report.passed
    assert pairs[0].task is TaskType.EXPLANATION
