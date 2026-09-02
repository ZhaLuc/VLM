"""Tests for the minimal preference annotation workflow."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from magic_vlm.annotation import (
    AnnotationCandidate,
    AnnotationError,
    AnnotationSessionConfig,
    AnnotationStore,
    default_rubric,
    format_candidate_for_display,
    load_annotation_queue,
    load_rubric,
    record_judgment,
    resolve_video_path,
    run_annotation_session,
    write_annotation_queue,
)
from magic_vlm.preferences import load_preference_pairs
from magic_vlm.schemas import PreferenceGenerationMeta, Split, VideoRef


def _gen() -> PreferenceGenerationMeta:
    return PreferenceGenerationMeta(
        model_id_a="stub/echo",
        model_id_b="stub/echo",
        generation_a={"temperature": 0.7, "do_sample": True},
        generation_b={"temperature": 0.7, "do_sample": True},
    )


def _candidate(
    *,
    queue_id: str = "q1",
    clip_id: str = "clip_1",
    response_a: str = "Palm on the third pass.",
    response_b: str = "Magnet under the cup.",
    instruction: str = "Explain the hidden mechanism.",
) -> AnnotationCandidate:
    return AnnotationCandidate(
        queue_id=queue_id,
        clip_id=clip_id,
        example_id="ex_1",
        video=VideoRef(path="data/videos/missing.mp4"),
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
        generation_meta=_gen(),
        split=Split.TRAIN,
    )


def test_load_queue_and_preserve_raw_text(tmp_path: Path) -> None:
    a = "  keep leading\n"
    b = "keep trailing  "
    c = _candidate(response_a=a, response_b=b)
    path = tmp_path / "queue.jsonl"
    write_annotation_queue(path, [c])
    loaded = load_annotation_queue(path)
    assert loaded[0].response_a == a
    assert loaded[0].response_b == b
    assert loaded[0].pair_id == c.pair_id


def test_select_a_and_b_with_rationale(tmp_path: Path) -> None:
    store = AnnotationStore(tmp_path / "prefs.jsonl")
    c1 = _candidate(queue_id="q1", response_a="good", response_b="bad")
    c2 = _candidate(
        queue_id="q2",
        clip_id="clip_2",
        response_a="x",
        response_b="y",
    )
    p1 = record_judgment(
        c1,
        store=store,
        annotator_id="ann_1",
        winner="a",
        rationale="More specific mechanism.",
        timestamp="2026-09-02T20:00:00+00:00",
    )
    assert p1.winner == "a"
    assert p1.rationale == "More specific mechanism."
    assert p1.response_a == "good"
    p2 = record_judgment(
        c2,
        store=store,
        annotator_id="ann_1",
        winner="b",
        timestamp="2026-09-02T20:01:00+00:00",
    )
    assert p2.winner == "b"
    loaded = load_preference_pairs(tmp_path / "prefs.jsonl")
    assert [x.winner for x in loaded] == ["a", "b"]


def test_persistence_append_only(tmp_path: Path) -> None:
    path = tmp_path / "prefs.jsonl"
    store = AnnotationStore(path)
    record_judgment(
        _candidate(),
        store=store,
        annotator_id="ann_1",
        winner="a",
        timestamp="2026-09-02T20:00:00+00:00",
    )
    text1 = path.read_text(encoding="utf-8")
    store2 = AnnotationStore(path)
    other = _candidate(clip_id="clip_other", response_a="aa", response_b="bb")
    record_judgment(
        other,
        store=store2,
        annotator_id="ann_1",
        winner="b",
        timestamp="2026-09-02T20:02:00+00:00",
    )
    text2 = path.read_text(encoding="utf-8")
    assert text2.startswith(text1)
    assert len(load_preference_pairs(path)) == 2


def test_duplicate_prevention(tmp_path: Path) -> None:
    store = AnnotationStore(tmp_path / "prefs.jsonl")
    c = _candidate()
    record_judgment(
        c,
        store=store,
        annotator_id="ann_1",
        winner="a",
        timestamp="2026-09-02T20:00:00+00:00",
    )
    with pytest.raises(AnnotationError, match="duplicate"):
        record_judgment(
            c,
            store=store,
            annotator_id="ann_1",
            winner="b",
            timestamp="2026-09-02T21:00:00+00:00",
        )


def test_resume_skips_completed(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.jsonl"
    out_path = tmp_path / "prefs.jsonl"
    c1 = _candidate(queue_id="q1")
    c2 = _candidate(queue_id="q2", clip_id="clip_2", response_a="p", response_b="q")
    write_annotation_queue(queue_path, [c1, c2])

    cfg = AnnotationSessionConfig(
        annotator_id="ann_resume",
        queue_path=queue_path,
        judgments_path=out_path,
        rubric=default_rubric(),
        open_video=False,
    )
    first = run_annotation_session(
        cfg,
        winners=["a"],
        rationales=["first"],
        limit=1,
        stdout=StringIO(),
        open_video_fn=lambda _p: False,
    )
    assert first.n_recorded == 1
    second = run_annotation_session(
        cfg,
        winners=["b"],
        rationales=["second"],
        stdout=StringIO(),
        open_video_fn=lambda _p: False,
    )
    assert second.n_pending_before == 1
    assert second.n_recorded == 1
    loaded = load_preference_pairs(out_path)
    assert len(loaded) == 2
    assert {p.winner for p in loaded} == {"a", "b"}


def test_display_includes_raw_and_rubric() -> None:
    c = _candidate(response_a="RAW_A", response_b="RAW_B")
    text = format_candidate_for_display(
        c,
        rubric=default_rubric(),
        index=1,
        total=1,
        video_path=Path("data/videos/missing.mp4"),
    )
    assert "RAW_A" in text
    assert "RAW_B" in text
    assert "Verbosity" in text or "verbosity" in text.lower()
    assert "confidence" in text.lower()


def test_load_rubric_yaml() -> None:
    rubric = load_rubric("configs/annotation_rubric.yaml")
    assert rubric.version == "explanation_pref_v1"
    assert any("Factual correctness" in p for p in rubric.prioritize)
    assert any("Verbosity" in d for d in rubric.do_not_reward)


def test_toy_queue_loads() -> None:
    rows = load_annotation_queue("data/examples/toy_annotation_queue.jsonl")
    assert len(rows) >= 2
    assert all(isinstance(r.response_a, str) and isinstance(r.response_b, str) for r in rows)


def test_cli_manual_fixture_annotation(tmp_path: Path) -> None:
    from magic_vlm.cli import annotate_main

    out = tmp_path / "manual_smoke.jsonl"
    code = annotate_main(
        [
            "--annotator",
            "ann_manual_smoke",
            "--queue",
            "data/examples/toy_annotation_queue.jsonl",
            "--out",
            str(out),
            "--rubric",
            "configs/annotation_rubric.yaml",
            "--no-open-video",
            "--limit",
            "1",
            "--winner",
            "a",
            "--rationale",
            "Cites a concrete palm-and-load mechanism grounded in timing.",
        ]
    )
    assert code == 0
    pairs = load_preference_pairs(out)
    assert len(pairs) == 1
    assert pairs[0].winner == "a"
    assert pairs[0].response_a.startswith("The performer palms")
    # Resume: second run with same annotator should not duplicate first item.
    code2 = annotate_main(
        [
            "--annotator",
            "ann_manual_smoke",
            "--queue",
            "data/examples/toy_annotation_queue.jsonl",
            "--out",
            str(out),
            "--no-open-video",
            "--limit",
            "1",
            "--winner",
            "b",
            "--rationale",
            "Second item: less invented mechanism.",
        ]
    )
    assert code2 == 0
    pairs2 = load_preference_pairs(out)
    assert len(pairs2) == 2
    assert pairs2[0].judgment_id != pairs2[1].judgment_id


def test_resolve_video_path(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    c = AnnotationCandidate(
        clip_id="c",
        instruction="q",
        response_a="a",
        response_b="b",
        generation_meta=_gen(),
        video=VideoRef(path="clip.mp4"),
    )
    resolved = resolve_video_path(c, video_root=tmp_path)
    assert resolved == video
