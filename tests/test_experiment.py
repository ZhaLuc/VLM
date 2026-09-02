from pathlib import Path

import pytest
import yaml

from magic_vlm.experiment import (
    CheckpointSpec,
    DatasetRunConfig,
    ExperimentConfig,
    build_run_manifest,
    experiment_config_from_dict,
    initialize_experiment,
    load_experiment_config,
)
from magic_vlm.inference import GenerationConfig
from magic_vlm.models import ModelSpec
from magic_vlm.runtime import DeviceConfig, SeedConfig
from magic_vlm.video import VideoPreprocessConfig


def test_load_stub_config() -> None:
    cfg = load_experiment_config(Path("configs/baseline_stub.yaml"))
    assert cfg.name == "baseline_stub"
    assert cfg.model.model_id.startswith("stub/")
    assert cfg.preserve_raw_outputs is True
    assert cfg.baseline_immutable is True
    assert cfg.allow_held_out_in_training is False
    assert cfg.video.temporal_shuffle is False
    assert cfg.is_zero_shot_baseline is True
    assert cfg.training_method == "none"
    assert cfg.generation.top_p == 1.0
    assert cfg.device.preference == "cpu"
    assert cfg.dataset.version == "toy-v0"


def test_config_roundtrip_serialization(tmp_path: Path) -> None:
    cfg = load_experiment_config(Path("configs/baseline_stub.yaml"))
    out = tmp_path / "roundtrip.yaml"
    cfg.save(out)
    reloaded = load_experiment_config(out)
    assert reloaded.name == cfg.name
    assert reloaded.generation.to_dict() == cfg.generation.to_dict()
    assert reloaded.dataset.to_dict() == cfg.dataset.to_dict()
    # dict dump/load via YAML preserves scientific knobs
    raw = yaml.safe_load(out.read_text(encoding="utf-8"))
    again = experiment_config_from_dict(raw)
    assert again.seed == cfg.seed
    assert again.reward_function == cfg.reward_function


def test_held_out_flag_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="bad",
            model=ModelSpec(model_id="stub/echo"),
            dataset=DatasetRunConfig(manifest="x.jsonl"),
            allow_held_out_in_training=True,
            video=VideoPreprocessConfig(),
        )


def test_post_trained_cannot_be_baseline_immutable() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="bad",
            model=ModelSpec(model_id="stub/echo"),
            dataset=DatasetRunConfig(manifest="x.jsonl"),
            training_method="dpo",
            checkpoint=CheckpointSpec(kind="post_trained", path="adapters/x"),
            baseline_immutable=True,
        )


def test_run_manifest_fingerprint_stable() -> None:
    cfg = load_experiment_config(Path("configs/baseline_stub.yaml"))
    a = build_run_manifest(cfg, run_id="fixed")
    b = build_run_manifest(cfg, run_id="fixed")
    assert a.config_hash == b.config_hash
    assert a.run_id == "fixed"


def test_initialize_experiment_writes_metadata(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        name="init_unit",
        model=ModelSpec(model_id="stub/echo"),
        dataset=DatasetRunConfig(manifest="data/examples/toy_manifest.jsonl", version="toy-v0"),
        output_dir=str(tmp_path / "runs"),
        generation=GenerationConfig(max_new_tokens=32, temperature=0.0, do_sample=False),
        device=DeviceConfig(preference="cpu"),
        seed_config=SeedConfig(seed=123),
        checkpoint=CheckpointSpec(kind="stub"),
        training_method="none",
        baseline_immutable=True,
    )
    ctx = initialize_experiment(cfg, run_id="unit-run")
    assert ctx.run_dir.exists()
    assert (ctx.run_dir / "metadata.json").exists()
    assert (ctx.run_dir / "environment.json").exists()
    assert (ctx.run_dir / "config.yaml").exists()
    assert (ctx.run_dir / "determinism.json").exists()
    assert (ctx.run_dir / "result_changing_parameters.json").exists()
    assert (ctx.run_dir / "run.log").exists()
    assert ctx.device.resolved == "cpu"
    assert ctx.determinism.level == "partially_controlled"
    assert ctx.record.identity["is_zero_shot_baseline"] is True
    assert ctx.record.identity["seed"] == 123
