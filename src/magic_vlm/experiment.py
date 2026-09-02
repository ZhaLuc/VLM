"""Experiment configuration and lightweight orchestration metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from magic_vlm.models import ModelSpec
from magic_vlm.utils import config_fingerprint, utc_now_iso
from magic_vlm.video import VideoPreprocessConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level run configuration.

    Research invariants encoded here:
    - ``preserve_raw_outputs`` defaults True
    - ``baseline_immutable`` marks zero-shot baseline runs as frozen references
    - ``allow_held_out_in_training`` defaults False and should stay False
    - ``video.temporal_shuffle`` defaults False; enable only for diagnostics
    """

    name: str
    model: ModelSpec
    dataset_manifest: str
    output_dir: str = "runs"
    video: VideoPreprocessConfig = field(default_factory=VideoPreprocessConfig)
    stage: str = "baseline"
    preserve_raw_outputs: bool = True
    baseline_immutable: bool = True
    allow_held_out_in_training: bool = False
    allow_model_download: bool = False
    seed: int = 0
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.allow_held_out_in_training:
            raise ValueError(
                "allow_held_out_in_training must remain False to protect evaluation validity"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": asdict(self.model),
            "dataset_manifest": self.dataset_manifest,
            "output_dir": self.output_dir,
            "video": asdict(self.video),
            "stage": self.stage,
            "preserve_raw_outputs": self.preserve_raw_outputs,
            "baseline_immutable": self.baseline_immutable,
            "allow_held_out_in_training": self.allow_held_out_in_training,
            "allow_model_download": self.allow_model_download,
            "seed": self.seed,
            "notes": self.notes,
            "extras": dict(self.extras),
        }


@dataclass(frozen=True)
class RunManifest:
    """Metadata written beside artifacts for reproducibility."""

    run_id: str
    created_at: str
    config: dict[str, Any]
    config_hash: str
    git_commit: str | None
    stage: str
    baseline_immutable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment config must be a mapping: {path}")
    model_raw = dict(raw.get("model") or {})
    video_raw = dict(raw.get("video") or {})
    return ExperimentConfig(
        name=str(raw["name"]),
        model=ModelSpec(**model_raw),
        dataset_manifest=str(raw["dataset_manifest"]),
        output_dir=str(raw.get("output_dir", "runs")),
        video=VideoPreprocessConfig(**video_raw),
        stage=str(raw.get("stage", "baseline")),
        preserve_raw_outputs=bool(raw.get("preserve_raw_outputs", True)),
        baseline_immutable=bool(raw.get("baseline_immutable", True)),
        allow_held_out_in_training=bool(raw.get("allow_held_out_in_training", False)),
        allow_model_download=bool(raw.get("allow_model_download", False)),
        seed=int(raw.get("seed", 0)),
        notes=str(raw.get("notes", "")),
        extras=dict(raw.get("extras") or {}),
    )


def build_run_manifest(config: ExperimentConfig, *, run_id: str | None = None) -> RunManifest:
    from magic_vlm.utils import git_commit_sha, make_run_id

    payload = config.to_dict()
    return RunManifest(
        run_id=run_id or make_run_id(config.name),
        created_at=utc_now_iso(),
        config=payload,
        config_hash=config_fingerprint(payload),
        git_commit=git_commit_sha(),
        stage=config.stage,
        baseline_immutable=config.baseline_immutable,
    )
