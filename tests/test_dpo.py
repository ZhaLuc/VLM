"""Tests for DPO dataset adapter, stack probe, and smoke training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("datasets")

from magic_vlm.dpo import (
    DPOConfigSpec,
    DPOError,
    assert_baseline_not_overwritten,
    create_tiny_local_causal_lm,
    load_dpo_checkpoint_dir,
    preferences_to_dpo_records,
    probe_dpo_stack,
    split_dpo_records,
    train_dpo,
)
from magic_vlm.preferences import load_preference_pairs
from magic_vlm.schemas import Split


FIXTURE = Path("tests/fixtures/reward_model/synthetic_bt_prefs.jsonl")
FIXTURE_HELD = Path("tests/fixtures/reward_model/synthetic_bt_prefs_with_heldout.jsonl")


def test_probe_dpo_stack_records_versions() -> None:
    stack = probe_dpo_stack()
    assert stack.torch_version is not None
    assert stack.transformers_version is not None
    assert stack.trl_has_dpo_trainer is True
    # Documented for this install path; do not invent support.
    assert isinstance(stack.ready_for_text_dpo, bool)
    assert isinstance(stack.ready_for_vlm_dpo, bool)
    assert isinstance(stack.limitations, tuple)


def test_preferences_adapter_preserves_raw_text() -> None:
    pairs = load_preference_pairs(FIXTURE)
    records = preferences_to_dpo_records(pairs)
    assert records
    # Raw chosen/rejected unchanged vs preference store
    by_jid = {p.judgment_id: p for p in pairs if p.winner != "tie"}
    for row in records:
        src = by_jid[row["judgment_id"]]
        chosen, rejected = src.chosen_rejected()
        assert row["chosen"] == chosen
        assert row["rejected"] == rejected
        assert row["prompt"] == src.instruction


def test_adapter_refuses_held_out() -> None:
    pairs = load_preference_pairs(FIXTURE_HELD)
    with pytest.raises(DPOError, match="held_out"):
        preferences_to_dpo_records(pairs)


def test_split_train_val() -> None:
    pairs = load_preference_pairs(FIXTURE)
    records = preferences_to_dpo_records(pairs)
    train, val = split_dpo_records(records)
    assert all(r["split"] == "train" for r in train)
    assert all(r["split"] == "val" for r in val)
    assert len(train) == 8
    assert len(val) == 3


def test_baseline_protection(tmp_path: Path) -> None:
    base = tmp_path / "baseline"
    base.mkdir()
    (base / "BASELINE_IMMUTABLE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DPOError, match="baseline"):
        assert_baseline_not_overwritten(base, base)
    with pytest.raises(DPOError, match="baseline"):
        assert_baseline_not_overwritten(base, base / "child")


def test_dpo_smoke_train_checkpoint(tmp_path: Path) -> None:
    pairs = load_preference_pairs(FIXTURE)
    records = preferences_to_dpo_records(pairs)
    texts = []
    for r in records:
        texts.extend([r["prompt"], r["chosen"], r["rejected"]])
    model_dir = create_tiny_local_causal_lm(texts, tmp_path / "tiny_lm", seed=0)

    baseline = tmp_path / "baseline_run"
    baseline.mkdir()
    (baseline / "BASELINE_IMMUTABLE.json").write_text(
        json.dumps({"immutable": True}), encoding="utf-8"
    )
    marker_before = (baseline / "BASELINE_IMMUTABLE.json").read_text(encoding="utf-8")

    cfg = DPOConfigSpec(
        prefs_path=str(FIXTURE),
        output_dir=str(tmp_path / "dpo_runs"),
        run_id="smoke",
        model_id=model_dir,
        baseline_run_dir=str(baseline),
        dataset_version="synthetic-bt-v0",
        beta=0.1,
        learning_rate=1e-3,
        max_steps=3,
        per_device_train_batch_size=1,
        seed=0,
        max_length=64,
        use_peft=True,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=("c_attn",),
        allow_download=False,
        modality="text",
        require_vlm_ready=False,
    )
    result = train_dpo(cfg)
    assert result.status == "completed"
    assert Path(result.checkpoint_dir).is_dir()
    schema = load_dpo_checkpoint_dir(result.checkpoint_dir)
    assert schema["schema_ok"] is True
    assert (Path(result.run_dir) / "train_metadata.json").exists()
    assert (Path(result.run_dir) / "DISCLAIMER.json").exists()
    assert (Path(result.run_dir) / "dpo_train_records.jsonl").exists()
    meta = json.loads((Path(result.run_dir) / "train_metadata.json").read_text(encoding="utf-8"))
    assert meta["beta"] == 0.1
    assert meta["splits"]["held_out_used"] is False
    assert meta["checkpoint_selection_rule"] == "last_train_step"
    # Baseline unchanged
    assert (baseline / "BASELINE_IMMUTABLE.json").read_text(encoding="utf-8") == marker_before


def test_cli_probe_and_smoke(tmp_path: Path) -> None:
    from magic_vlm.cli import train_dpo_main

    code = train_dpo_main(["--probe-only"])
    assert code == 0

    # Write a config pointing at tmp output
    cfg_path = tmp_path / "dpo.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"prefs_path: {FIXTURE.as_posix()}",
                f"output_dir: {(tmp_path / 'out').as_posix()}",
                "run_id: cli_smoke",
                "model_id: unused",
                "beta: 0.1",
                "max_steps: 2",
                "learning_rate: 0.001",
                "use_peft: true",
                "lora_r: 4",
                "lora_alpha: 8",
                "lora_target_modules: [c_attn]",
                "modality: text",
                "allow_download: false",
                "seed: 0",
                "max_length: 64",
            ]
        ),
        encoding="utf-8",
    )
    code2 = train_dpo_main(["--config", str(cfg_path), "--smoke-local-lm"])
    assert code2 == 0
    assert (tmp_path / "out" / "cli_smoke" / "checkpoint").is_dir()
