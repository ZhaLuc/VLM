"""Temporal-order diagnostic: same sampled frames, shuffled presentation only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from magic_vlm.experiment import experiment_config_from_dict, load_experiment_config
from magic_vlm.inference import GenerationConfig, build_prompt, run_inference
from magic_vlm.models import EchoStubVLM
from magic_vlm.schemas import (
    ExampleRecord,
    InferenceArtifact,
    Provenance,
    Split,
    TaskType,
    VideoRef,
)
from magic_vlm.temporal import (
    TemporalShuffleError,
    assert_shared_sample,
    assert_shuffle_permutes,
    pair_example,
    run_temporal_shuffle_experiment,
)
from magic_vlm.temporal_metrics import (
    INTEGRITY_NOTE,
    SHUFFLE_METHOD,
    paired_outcome,
    summarize_pairs,
)
from magic_vlm.video import (
    VideoPreprocessConfig,
    apply_temporal_shuffle,
    as_ordered_clip,
    build_sample_plan,
    ordered_and_shuffled_pair,
)


def _example(*, example_id: str = "e1", n_frames: int = 16, gold: str = "left") -> ExampleRecord:
    return ExampleRecord(
        example_id=example_id,
        clip_id="clip_e1",
        trick_id="cups",
        performer_id="a",
        camera_id="cam_front",
        video=VideoRef(path="synthetic.mp4", num_frames=n_frames),
        task=TaskType.HIDDEN_STATE,
        question="Which cup contains the ball?",
        ground_truth=gold,
        split=Split.HELD_OUT,
        provenance=Provenance(source="unit_test"),
    )


def _artifact(
    example: ExampleRecord,
    *,
    raw: str,
    parsed: str,
    frame_indices: tuple[int, ...],
    shuffled: bool,
    prompt: str | None = None,
    generation: dict | None = None,
) -> InferenceArtifact:
    used = prompt if prompt is not None else build_prompt(example)
    gen = generation or GenerationConfig(max_new_tokens=16, do_sample=False).to_dict()
    return InferenceArtifact(
        example_id=example.example_id,
        clip_id=example.clip_id,
        task=example.task.value,
        question=example.question,
        model_id="stub/echo",
        prompt=used,
        raw_text=raw,
        parsed_answer=parsed,
        frame_indices=frame_indices,
        generation=gen,
        preprocessing={
            "ordered_indices": list(frame_indices) if not shuffled else [0, 5, 10, 15],
            "frame_indices": list(frame_indices),
            "temporal_shuffled": shuffled,
            "shuffle_seed": 7 if shuffled else None,
        },
    )


def test_shuffle_preserves_synthetic_frame_identities() -> None:
    """Order changes; the sampled frame identity set does not."""
    plan = build_sample_plan(
        "synthetic.mp4",
        num_frames=16,
        config=VideoPreprocessConfig(max_frames=4),
    )
    tokens = tuple(f"frame:{idx}" for idx in plan.ordered_indices)
    ordered = as_ordered_clip(plan, frames=tokens)
    shuffled = apply_temporal_shuffle(plan, seed=7, frames=tokens)

    assert ordered.ordered_indices == shuffled.ordered_indices == plan.ordered_indices
    assert set(shuffled.frame_indices) == set(ordered.frame_indices)
    assert shuffled.frame_indices != ordered.frame_indices
    assert ordered.frames is not None and shuffled.frames is not None
    by_index = {idx: frame for idx, frame in zip(ordered.frame_indices, ordered.frames)}
    for idx, frame in zip(shuffled.frame_indices, shuffled.frames):
        assert frame == by_index[idx]
    assert set(shuffled.frames) == set(ordered.frames)


def test_shuffle_seed_reproducible() -> None:
    plan = build_sample_plan(
        "synthetic.mp4",
        num_frames=16,
        config=VideoPreprocessConfig(max_frames=4),
    )
    a = apply_temporal_shuffle(plan, seed=7)
    b = apply_temporal_shuffle(plan, seed=7)
    c = apply_temporal_shuffle(plan, seed=8)
    assert a.frame_indices == b.frame_indices
    assert a.frame_indices != c.frame_indices
    assert set(c.frame_indices) == set(a.frame_indices)
    assert a.shuffle_seed == 7
    assert c.shuffle_seed == 8


def test_index_only_pair_metadata_consistent() -> None:
    cfg = VideoPreprocessConfig(max_frames=4, shuffle_seed=5)
    ordered, shuffled = ordered_and_shuffled_pair(
        "synthetic.mp4",
        config=cfg,
        num_frames=16,
        shuffle_seed=5,
        load_frames=False,
    )
    assert_shared_sample(ordered, shuffled)
    assert ordered.metadata.get("ordering") == "temporal"
    assert shuffled.metadata.get("ordering") == "shuffled"
    assert shuffled.shuffle_seed == 5
    assert ordered.shuffle_seed is None
    d = ordered.to_dict()
    assert d["ordered_indices"] == list(ordered.frame_indices)
    assert set(shuffled.to_dict()["frame_indices"]) == set(d["ordered_indices"])


def test_pair_example_records_responses_and_correctness() -> None:
    example = _example()
    plan = build_sample_plan(
        example.video.path,
        num_frames=16,
        config=VideoPreprocessConfig(max_frames=4),
    )
    ordered = as_ordered_clip(plan)
    shuffled = apply_temporal_shuffle(plan, seed=7)
    prompt = build_prompt(example)
    pair = pair_example(
        example,
        ordered=ordered,
        shuffled=shuffled,
        ordered_artifact=_artifact(
            example,
            raw="Answer: left",
            parsed="left",
            frame_indices=ordered.frame_indices,
            shuffled=False,
            prompt=prompt,
        ),
        shuffled_artifact=_artifact(
            example,
            raw="Answer: right",
            parsed="right",
            frame_indices=shuffled.frame_indices,
            shuffled=True,
            prompt=prompt,
        ),
        shuffle_seed=7,
        prompt=prompt,
    )
    assert pair.ordered_correct is True
    assert pair.shuffled_correct is False
    assert pair.outcome == "ordered_only"
    assert pair.same_frame_set is True
    assert pair.order_changed is True
    assert pair.shuffle_method == SHUFFLE_METHOD
    assert pair.question == example.question
    assert "Which cup contains the ball?" in pair.prompt


def test_refuses_prompt_or_generation_mismatch() -> None:
    example = _example()
    plan = build_sample_plan(
        example.video.path, num_frames=16, config=VideoPreprocessConfig(max_frames=4)
    )
    ordered = as_ordered_clip(plan)
    shuffled = apply_temporal_shuffle(plan, seed=7)
    prompt = build_prompt(example)
    other_prompt = prompt + " Extra instruction."
    with pytest.raises(TemporalShuffleError, match="prompt wording"):
        pair_example(
            example,
            ordered=ordered,
            shuffled=shuffled,
            ordered_artifact=_artifact(
                example,
                raw="x",
                parsed="x",
                frame_indices=ordered.frame_indices,
                shuffled=False,
                prompt=prompt,
            ),
            shuffled_artifact=_artifact(
                example,
                raw="x",
                parsed="x",
                frame_indices=shuffled.frame_indices,
                shuffled=True,
                prompt=other_prompt,
            ),
            shuffle_seed=7,
            prompt=prompt,
        )


def test_paired_metrics() -> None:
    example = _example()
    plan = build_sample_plan(
        example.video.path, num_frames=16, config=VideoPreprocessConfig(max_frames=4)
    )
    ordered = as_ordered_clip(plan)
    shuffled = apply_temporal_shuffle(plan, seed=7)
    prompt = build_prompt(example)

    def _pair(oid: str, o_ok: bool, s_ok: bool):
        gold_o = "left" if o_ok else "right"
        gold_s = "left" if s_ok else "center"
        ex = _example(example_id=oid)
        return pair_example(
            ex,
            ordered=ordered,
            shuffled=shuffled,
            ordered_artifact=_artifact(
                ex,
                raw=f"Answer: {gold_o}",
                parsed=gold_o,
                frame_indices=ordered.frame_indices,
                shuffled=False,
                prompt=prompt,
            ),
            shuffled_artifact=_artifact(
                ex,
                raw=f"Answer: {gold_s}",
                parsed=gold_s,
                frame_indices=shuffled.frame_indices,
                shuffled=True,
                prompt=prompt,
            ),
            shuffle_seed=7,
            prompt=prompt,
        )

    pairs = (
        _pair("both_ok", True, True),
        _pair("both_bad", False, False),
        _pair("ord_only", True, False),
        _pair("shf_only", False, True),
    )
    summary = summarize_pairs(pairs, shuffle_seed=7, split="held_out")
    assert summary.n_pairs == 4
    assert summary.ordered_accuracy == 0.5
    assert summary.shuffled_accuracy == 0.5
    assert summary.accuracy_difference == 0.0
    assert summary.n_both_correct == 1
    assert summary.n_both_incorrect == 1
    assert summary.n_ordered_only == 1
    assert summary.n_shuffled_only == 1
    assert summary.shuffle_method == SHUFFLE_METHOD
    assert "NOT proof of causal reasoning" in summary.integrity_note
    assert paired_outcome(True, False) == "ordered_only"


def test_refuses_identity_shuffle_seed() -> None:
    plan = build_sample_plan(
        "synthetic.mp4",
        num_frames=16,
        config=VideoPreprocessConfig(max_frames=4),
    )
    ordered = as_ordered_clip(plan)
    identity = apply_temporal_shuffle(plan, seed=0)

    assert tuple(identity.frame_indices) == tuple(ordered.frame_indices)
    with pytest.raises(TemporalShuffleError, match="identity permutation"):
        assert_shuffle_permutes(ordered, identity, seed=0)


def test_refuses_training_method() -> None:
    cfg = load_experiment_config("configs/temporal_shuffle_stub.yaml")
    payload = cfg.to_dict()
    payload["training_method"] = "dpo"
    payload["checkpoint"]["kind"] = "post_trained"
    trained = experiment_config_from_dict(payload)
    with pytest.raises(TemporalShuffleError, match="training"):
        run_temporal_shuffle_experiment(trained, run_id="should-fail")


def test_paired_smoke_stub(tmp_path: Path) -> None:
    cfg = load_experiment_config("configs/temporal_shuffle_stub.yaml")
    payload = cfg.to_dict()
    payload["output_dir"] = str(tmp_path / "runs")
    cfg = experiment_config_from_dict(payload)
    summary, pairs, run_dir = run_temporal_shuffle_experiment(
        cfg, run_id="temporal-stub", load_frames=False
    )
    assert summary.n_pairs == 1
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.same_frame_set is True
    assert pair.order_changed is True
    assert pair.question == "Which cup contains the ball?"
    assert pair.model_id == "stub/echo"
    assert pair.shuffle_seed == 7
    assert set(pair.ordered_indices) == set(pair.shuffled_frame_indices)
    assert "Which cup contains the ball?" in pair.prompt
    assert pair.ordered_correct is False
    assert pair.shuffled_correct is False
    assert summary.n_both_incorrect == 1
    assert summary.accuracy_difference == 0.0
    assert INTEGRITY_NOTE in summary.integrity_note

    meta = json.loads((run_dir / "temporal_shuffle_metadata.json").read_text(encoding="utf-8"))
    assert meta["diagnostic"] == "temporal_order"
    assert meta["shuffle_method"] == SHUFFLE_METHOD
    assert meta["shuffle_seed"] == 7
    assert meta["training_use"] is False
    assert meta["resampled"] is False
    assert meta["model_id"] == "stub/echo"
    assert meta["generation"]["do_sample"] is False
    assert meta["sampled_frames"][0]["ordered_indices"] == list(pair.ordered_indices)
    assert meta["prompt_template"] == "hidden_state_v1"

    rows = [
        json.loads(line)
        for line in (run_dir / "temporal_shuffle_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["ordered_raw"]
    assert rows[0]["shuffled_raw"]
    assert rows[0]["ordered_correct"] is False
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "NOT proof" in metrics["integrity_note"]


def test_cli_temporal_shuffle(tmp_path: Path) -> None:
    from magic_vlm.cli import temporal_shuffle_main

    cfg_text = Path("configs/temporal_shuffle_stub.yaml").read_text(encoding="utf-8")
    cfg_text = cfg_text.replace("output_dir: runs", f"output_dir: {tmp_path.as_posix()}/runs")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    code = temporal_shuffle_main(["--config", str(cfg_path), "--run-id", "cli-temporal"])
    assert code == 0
    run_dir = tmp_path / "runs" / "cli-temporal"
    assert (run_dir / "temporal_shuffle_pairs.jsonl").exists()
    assert (run_dir / "temporal_shuffle_summary.json").exists()
    assert (run_dir / "temporal_shuffle_metadata.json").exists()


def test_stub_inference_sees_same_prompt_different_order() -> None:
    example = _example()
    cfg = VideoPreprocessConfig(max_frames=4, shuffle_seed=3)
    ordered, shuffled = ordered_and_shuffled_pair(
        example.video.path, config=cfg, num_frames=16, load_frames=False
    )
    model = EchoStubVLM()
    prompt = build_prompt(example)
    gen = GenerationConfig(max_new_tokens=8, do_sample=False)
    o_art = run_inference(model, example, preprocessed=ordered, generation=gen, prompt=prompt)
    s_art = run_inference(model, example, preprocessed=shuffled, generation=gen, prompt=prompt)
    assert o_art.prompt == s_art.prompt == prompt
    assert o_art.generation == s_art.generation
    assert o_art.preprocessing["ordered_indices"] == s_art.preprocessing["ordered_indices"]
    assert o_art.preprocessing["frame_indices"] != s_art.preprocessing["frame_indices"]
    assert o_art.preprocessing["temporal_shuffled"] is False
    assert s_art.preprocessing["temporal_shuffled"] is True
