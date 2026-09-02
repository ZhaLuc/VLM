"""Tests for GRPO dataset adapter, reward wiring, stack probe, and smoke training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("datasets")

from magic_vlm.dataset import filter_split, filter_task, load_manifest
from magic_vlm.dpo import create_tiny_local_causal_lm
from magic_vlm.grpo import (
    GRPOConfigSpec,
    GRPOError,
    assert_baseline_not_overwritten,
    completion_to_text,
    examples_to_grpo_records,
    load_grpo_checkpoint_dir,
    make_objective_reward_fn,
    probe_grpo_stack,
    split_grpo_records,
    train_grpo,
)
from magic_vlm.rewards import HiddenStateExactMatchReward
from magic_vlm.schemas import ExampleRecord, Provenance, Split, TaskType, VideoRef

MANIFEST = Path("data/examples/toy_manifest.jsonl")


def test_probe_grpo_stack_records_versions() -> None:
    stack = probe_grpo_stack()
    assert stack.torch_version is not None
    assert stack.transformers_version is not None
    assert stack.trl_has_grpo_trainer is True
    assert isinstance(stack.ready_for_text_grpo, bool)
    assert isinstance(stack.ready_for_vlm_grpo, bool)
    assert isinstance(stack.limitations, tuple)
    # This host historically has CPU-only torch → VLM GRPO not ready.
    if not stack.cuda_available:
        assert stack.ready_for_vlm_grpo is False


def test_adapter_refuses_held_out() -> None:
    examples = filter_task(load_manifest(MANIFEST), TaskType.HIDDEN_STATE)
    held = [ex for ex in examples if ex.split is Split.HELD_OUT]
    assert held
    with pytest.raises(GRPOError, match="held_out"):
        examples_to_grpo_records(held)


def test_adapter_preserves_ground_truth() -> None:
    train = filter_task(
        filter_split(load_manifest(MANIFEST), Split.TRAIN), TaskType.HIDDEN_STATE
    )
    records = examples_to_grpo_records(train)
    by_id = {ex.example_id: ex for ex in train}
    for row in records:
        assert row["ground_truth"] == by_id[row["example_id"]].ground_truth
        assert row["question"] == by_id[row["example_id"]].question
        assert "prompt" in row and row["prompt"]


def test_split_train_val() -> None:
    examples = filter_task(load_manifest(MANIFEST), TaskType.HIDDEN_STATE)
    usable = [ex for ex in examples if ex.split is not Split.HELD_OUT]
    records = examples_to_grpo_records(usable)
    train, val = split_grpo_records(records)
    assert train
    assert all(r["split"] == "train" for r in train)
    assert all(r["split"] == "val" for r in val)


def test_reward_fn_scores_failed_rollout_zero() -> None:
    example = ExampleRecord(
        example_id="e1",
        clip_id="c1",
        trick_id="t1",
        performer_id="p",
        camera_id="cam",
        video=VideoRef(path="e1.mp4"),
        task=TaskType.HIDDEN_STATE,
        question="Where is the ball?",
        ground_truth="left",
        split=Split.TRAIN,
        provenance=Provenance(source="unit_test"),
    )
    reward = HiddenStateExactMatchReward()
    log: list[dict] = []
    fn = make_objective_reward_fn({"e1": example}, reward, completion_log=log)
    scores = fn(
        prompts=["q"],
        completions=["<<<unparseable>>>"],
        example_id=["e1"],
        ground_truth=["left"],
    )
    assert scores == [0.0]
    assert log[0]["raw_completion"] == "<<<unparseable>>>"
    assert log[0]["matched"] is False
    assert log[0]["reward"] == 0.0

    # Missing example still scores 0 (not dropped / not None)
    scores2 = fn(
        prompts=["q"],
        completions=["left"],
        example_id=["missing"],
        ground_truth=["left"],
    )
    assert scores2 == [0.0]


def test_completion_to_text_chat_payload() -> None:
    assert completion_to_text("plain") == "plain"
    assert "hello" in completion_to_text([{"role": "assistant", "content": "hello"}])


def test_baseline_protection(tmp_path: Path) -> None:
    base = tmp_path / "baseline"
    base.mkdir()
    (base / "BASELINE_IMMUTABLE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(GRPOError, match="baseline"):
        assert_baseline_not_overwritten(base, base)


def test_config_rejects_held_out_checkpoint_selection() -> None:
    with pytest.raises(GRPOError, match="held-out"):
        GRPOConfigSpec(
            manifest=str(MANIFEST),
            checkpoint_selection="held_out_best",
            num_generations=2,
        )


def test_grpo_smoke_train_checkpoint(tmp_path: Path) -> None:
    train = filter_task(
        filter_split(load_manifest(MANIFEST), Split.TRAIN), TaskType.HIDDEN_STATE
    )
    records = examples_to_grpo_records(train)
    texts: list[str] = []
    for r in records:
        texts.extend([r["prompt"], str(r["ground_truth"]), "left right center"])
    model_dir = create_tiny_local_causal_lm(texts, tmp_path / "tiny_lm", seed=0)

    baseline = tmp_path / "baseline_run"
    baseline.mkdir()
    (baseline / "BASELINE_IMMUTABLE.json").write_text(
        json.dumps({"immutable": True}), encoding="utf-8"
    )
    marker_before = (baseline / "BASELINE_IMMUTABLE.json").read_text(encoding="utf-8")

    cfg = GRPOConfigSpec(
        manifest=str(MANIFEST),
        output_dir=str(tmp_path / "grpo_runs"),
        run_id="smoke",
        model_id=model_dir,
        baseline_run_dir=str(baseline),
        dataset_version="toy-v0",
        reward_id="hidden_state_exact_match",
        reward_version="1.0.0",
        num_generations=2,
        learning_rate=1e-4,
        max_steps=2,
        per_device_train_batch_size=2,
        seed=0,
        max_completion_length=16,
        use_peft=True,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=("c_attn",),
        allow_download=False,
        modality="text",
        require_vlm_ready=False,
        eval_held_out_after_train=True,
    )
    result = train_grpo(cfg)
    assert result.status == "completed"
    assert Path(result.checkpoint_dir).is_dir()
    load_grpo_checkpoint_dir(result.checkpoint_dir)
    run = Path(result.run_dir)
    assert (run / "train_metadata.json").exists()
    assert (run / "DISCLAIMER.json").exists()
    assert (run / "raw_completions.jsonl").exists()
    assert (run / "reward_stats.json").exists()
    assert (run / "grpo_train_records.jsonl").exists()
    meta = json.loads((run / "train_metadata.json").read_text(encoding="utf-8"))
    assert meta["group_size_num_generations"] == 2
    assert meta["reward_id"] == "hidden_state_exact_match"
    assert meta["checkpoint_selection"] == "last_train_step"
    assert meta["peft"]["use_peft"] is True
    assert (baseline / "BASELINE_IMMUTABLE.json").read_text(encoding="utf-8") == marker_before
    train_rows = (run / "grpo_train_records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    for line in train_rows:
        assert json.loads(line)["split"] == "train"


def test_cli_probe() -> None:
    from magic_vlm.cli import train_grpo_main

    assert train_grpo_main(["--probe-only"]) == 0
