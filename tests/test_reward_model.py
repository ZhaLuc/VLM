"""Tests for Bradley-Terry preference reward model."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from magic_vlm.preferences import load_preference_pairs
from magic_vlm.reward_model import (
    INTEGRITY_DISCLAIMER,
    RewardModelConfig,
    RewardModelError,
    bradley_terry_loss,
    bradley_terry_loss_value,
    evaluate_preference_accuracy,
    load_checkpoint,
    preference_examples_from_pairs,
    score_preference_file,
    split_preference_examples,
    train_bradley_terry_reward_model,
)


FIXTURE = Path("tests/fixtures/reward_model/synthetic_bt_prefs.jsonl")
FIXTURE_HELD = Path("tests/fixtures/reward_model/synthetic_bt_prefs_with_heldout.jsonl")


def test_bradley_terry_numerical_loss() -> None:
    # -log(sigmoid(2)) ≈ 0.126928
    expected = bradley_terry_loss_value(2.0)
    assert expected == pytest.approx(0.126928, abs=1e-4)
    tw = torch.tensor([2.0, 0.0])
    tl = torch.tensor([0.0, 0.0])
    loss = bradley_terry_loss(tw, tl)
    assert float(loss[0]) == pytest.approx(expected, abs=1e-5)
    assert float(loss[1]) == pytest.approx(bradley_terry_loss_value(0.0), abs=1e-5)


def test_split_refuses_held_out() -> None:
    pairs = load_preference_pairs(FIXTURE_HELD)
    examples = preference_examples_from_pairs(pairs)
    with pytest.raises(RewardModelError, match="held_out"):
        split_preference_examples(examples)


def test_train_val_separation_uses_labels() -> None:
    pairs = load_preference_pairs(FIXTURE)
    examples = preference_examples_from_pairs(pairs)
    train, val = split_preference_examples(examples, seed=0)
    assert all(ex.split == "train" for ex in train)
    assert all(ex.split == "val" for ex in val)
    assert len(train) == 8
    assert len(val) == 3
    train_ids = {ex.judgment_id for ex in train}
    val_ids = {ex.judgment_id for ex in val}
    assert train_ids.isdisjoint(val_ids)


def test_smoke_training_checkpoint_and_ordering(tmp_path: Path) -> None:
    cfg = RewardModelConfig(
        prefs_path=str(FIXTURE),
        output_dir=str(tmp_path / "rm"),
        run_id="smoke",
        seed=0,
        embedding_dim=32,
        hidden_dim=32,
        max_length=64,
        learning_rate=0.05,
        epochs=40,
        batch_size=4,
        device="cpu",
        dataset_version="synthetic-bt-v0",
    )
    result = train_bradley_terry_reward_model(cfg)
    assert Path(result.checkpoint_path).exists()
    assert (Path(result.run_dir) / "train_metadata.json").exists()
    assert (Path(result.run_dir) / "metrics.jsonl").exists()
    assert (Path(result.run_dir) / "DISCLAIMER.json").exists()
    assert INTEGRITY_DISCLAIMER in result.disclaimer

    model, tokenizer, payload = load_checkpoint(result.checkpoint_path, device="cpu")
    assert payload["architecture"]["video_pixels"] is False
    pairs = load_preference_pairs(FIXTURE)
    examples = preference_examples_from_pairs(pairs)
    train, val = split_preference_examples(examples)
    train_acc, _, train_rows = evaluate_preference_accuracy(
        model,
        train,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        device="cpu",
    )
    val_acc, _, val_rows = evaluate_preference_accuracy(
        model,
        val,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        device="cpu",
    )
    # Should learn the synthetic preference pattern on train.
    assert train_acc >= 0.75
    assert all("reward_chosen" in r and "reward_rejected" in r for r in train_rows)
    # Score ordering: chosen reward > rejected when correct_order
    for row in train_rows:
        if row["correct_order"]:
            assert row["reward_chosen"] > row["reward_rejected"]

    scored = score_preference_file(
        FIXTURE,
        result.checkpoint_path,
        out_path=tmp_path / "scores.jsonl",
        device="cpu",
    )
    assert len(scored) == len(examples)
    assert result.best_val_preference_accuracy is not None
    # Val may be weak with tiny data; still recorded (no hiding).
    assert 0.0 <= float(result.best_val_preference_accuracy) <= 1.0
    assert 0.0 <= val_acc <= 1.0


def test_cli_train_and_score(tmp_path: Path) -> None:
    from magic_vlm.cli import score_reward_main, train_reward_main

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"prefs_path: {FIXTURE.as_posix()}",
                f"output_dir: {(tmp_path / 'runs').as_posix()}",
                "run_id: cli_smoke",
                "seed: 1",
                "embedding_dim: 32",
                "hidden_dim: 32",
                "max_length: 64",
                "learning_rate: 0.05",
                "epochs: 30",
                "batch_size: 4",
                "device: cpu",
                "dataset_version: synthetic-bt-v0",
            ]
        ),
        encoding="utf-8",
    )
    code = train_reward_main(["--config", str(cfg_path)])
    assert code == 0
    ckpt = tmp_path / "runs" / "cli_smoke" / "checkpoint_best.pt"
    assert ckpt.exists()
    out = tmp_path / "score.jsonl"
    code2 = score_reward_main(
        ["--prefs", str(FIXTURE), "--checkpoint", str(ckpt), "--out", str(out)]
    )
    assert code2 == 0
    assert out.exists()


def test_config_yaml_loads() -> None:
    cfg = RewardModelConfig.from_yaml("configs/reward_model_bt_synthetic.yaml")
    assert cfg.prefs_path.endswith("synthetic_bt_prefs.jsonl")
    assert cfg.embedding_dim == 64
