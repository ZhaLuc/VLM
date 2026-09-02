from pathlib import Path

import pytest

from magic_vlm.experiment import ExperimentConfig, build_run_manifest, load_experiment_config
from magic_vlm.models import ModelSpec
from magic_vlm.video import VideoPreprocessConfig


def test_load_stub_config() -> None:
    cfg = load_experiment_config(Path("configs/baseline_stub.yaml"))
    assert cfg.name == "baseline_stub"
    assert cfg.model.model_id.startswith("stub/")
    assert cfg.preserve_raw_outputs is True
    assert cfg.baseline_immutable is True
    assert cfg.allow_held_out_in_training is False
    assert cfg.video.temporal_shuffle is False


def test_held_out_flag_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="bad",
            model=ModelSpec(model_id="stub/echo"),
            dataset_manifest="x.jsonl",
            allow_held_out_in_training=True,
            video=VideoPreprocessConfig(),
        )


def test_run_manifest_fingerprint_stable() -> None:
    cfg = load_experiment_config(Path("configs/baseline_stub.yaml"))
    a = build_run_manifest(cfg, run_id="fixed")
    b = build_run_manifest(cfg, run_id="fixed")
    assert a.config_hash == b.config_hash
    assert a.run_id == "fixed"
