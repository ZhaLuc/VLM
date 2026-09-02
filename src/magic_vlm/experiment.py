"""Experiment configuration, initialization, and reproducibility metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from magic_vlm.inference import GenerationConfig
from magic_vlm.logging_utils import setup_run_logging
from magic_vlm.models import ModelSpec
from magic_vlm.runtime import (
    DeterminismReport,
    DeviceConfig,
    DeviceInfo,
    SeedConfig,
    capture_environment,
    resolve_device,
    set_seed,
)
from magic_vlm.utils import (
    config_fingerprint,
    git_commit_sha,
    make_run_id,
    utc_now_iso,
    write_json,
)
from magic_vlm.video import VideoPreprocessConfig

TrainingMethod = Literal["none", "sft", "dpo", "grpo", "ppo"]
CheckpointKind = Literal["base", "post_trained", "adapter", "stub"]


@dataclass(frozen=True)
class DatasetRunConfig:
    """Dataset identity captured for every run."""

    manifest: str
    version: str = "unspecified"
    split: str | None = None
    task: str = "hidden_state"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetRunConfig:
        return cls(
            manifest=str(data["manifest"]),
            version=str(data.get("version", "unspecified")),
            split=(None if data.get("split") in (None, "", "null") else str(data.get("split"))),
            task=str(data.get("task", "hidden_state")),
        )


@dataclass(frozen=True)
class CheckpointSpec:
    """Distinguishes untouched base checkpoints from post-trained artifacts."""

    kind: CheckpointKind = "base"
    path: str | None = None
    parent_run_id: str | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CheckpointSpec:
        raw = dict(data or {})
        kind = str(raw.get("kind", "base"))
        return cls(
            kind=kind,  # type: ignore[arg-type]
            path=raw.get("path"),
            parent_run_id=raw.get("parent_run_id"),
            label=raw.get("label"),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level serializable experiment configuration.

    Research invariants:
    - ``preserve_raw_outputs`` defaults True
    - ``baseline_immutable`` marks zero-shot baseline runs as frozen references
    - ``allow_held_out_in_training`` must stay False
    - ``training_method='none'`` identifies zero-shot / no post-training runs
    - ``checkpoint.kind='base'`` (or ``stub``) vs ``post_trained``/``adapter``
    """

    name: str
    model: ModelSpec
    dataset: DatasetRunConfig
    output_dir: str = "runs"
    video: VideoPreprocessConfig = field(default_factory=VideoPreprocessConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    seed_config: SeedConfig = field(default_factory=SeedConfig)
    checkpoint: CheckpointSpec = field(default_factory=CheckpointSpec)
    stage: str = "baseline"
    training_method: TrainingMethod = "none"
    reward_function: str = "exact_match"
    preserve_raw_outputs: bool = True
    baseline_immutable: bool = True
    allow_held_out_in_training: bool = False
    allow_model_download: bool = False
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.allow_held_out_in_training:
            raise ValueError(
                "allow_held_out_in_training must remain False to protect evaluation validity"
            )
        if self.is_zero_shot_baseline and self.checkpoint.kind in {"post_trained", "adapter"}:
            raise ValueError(
                "Zero-shot baseline runs cannot point at a post_trained/adapter checkpoint"
            )
        if (not self.is_zero_shot_baseline) and self.baseline_immutable:
            raise ValueError(
                "baseline_immutable=True is reserved for zero-shot baseline configs "
                f"(training_method={self.training_method!r}, checkpoint.kind={self.checkpoint.kind!r})"
            )

    @property
    def seed(self) -> int:
        return self.seed_config.seed

    @property
    def dataset_manifest(self) -> str:
        return self.dataset.manifest

    @property
    def is_zero_shot_baseline(self) -> bool:
        return self.training_method == "none" and self.checkpoint.kind in {"base", "stub"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": asdict(self.model),
            "dataset": self.dataset.to_dict(),
            "dataset_manifest": self.dataset.manifest,  # backward-compatible alias
            "output_dir": self.output_dir,
            "video": asdict(self.video),
            "generation": self.generation.to_dict(),
            "device": asdict(self.device),
            "seed": self.seed,
            "seed_config": asdict(self.seed_config),
            "checkpoint": self.checkpoint.to_dict(),
            "stage": self.stage,
            "training_method": self.training_method,
            "reward_function": self.reward_function,
            "task": self.dataset.task,
            "split": self.dataset.split,
            "dataset_version": self.dataset.version,
            "preserve_raw_outputs": self.preserve_raw_outputs,
            "baseline_immutable": self.baseline_immutable,
            "allow_held_out_in_training": self.allow_held_out_in_training,
            "allow_model_download": self.allow_model_download,
            "is_zero_shot_baseline": self.is_zero_shot_baseline,
            "notes": self.notes,
            "extras": dict(self.extras),
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=False)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_yaml(), encoding="utf-8")


@dataclass(frozen=True)
class RunManifest:
    """Legacy-compatible summary; prefer ``ExperimentRecord`` for new writers."""

    run_id: str
    created_at: str
    config: dict[str, Any]
    config_hash: str
    git_commit: str | None
    stage: str
    baseline_immutable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentRecord:
    """Full structured metadata written at run initialization."""

    run_id: str
    created_at: str
    output_dir: str
    config: dict[str, Any]
    config_hash: str
    identity: dict[str, Any]
    environment: dict[str, Any]
    determinism: dict[str, Any]
    device: dict[str, Any]
    git_commit: str | None
    code_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentContext:
    """Initialized run directory and metadata (no model load required)."""

    run_id: str
    run_dir: Path
    config: ExperimentConfig
    record: ExperimentRecord
    device: DeviceInfo
    determinism: DeterminismReport
    logger_name: str = "magic_vlm"


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment config must be a mapping: {path}")
    return experiment_config_from_dict(raw)


def experiment_config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    model_raw = dict(raw.get("model") or {})
    video_raw = dict(raw.get("video") or {})
    generation_raw = dict(raw.get("generation") or {})
    device_raw = dict(raw.get("device") or {})
    seed_raw = dict(raw.get("seed_config") or {})
    if "seed" in raw and "seed" not in seed_raw:
        seed_raw["seed"] = raw["seed"]
    checkpoint_raw = dict(raw.get("checkpoint") or {})

    if "dataset" in raw and isinstance(raw["dataset"], dict):
        dataset = DatasetRunConfig.from_dict(raw["dataset"])
    else:
        dataset = DatasetRunConfig(
            manifest=str(raw["dataset_manifest"]),
            version=str(raw.get("dataset_version", "unspecified")),
            split=raw.get("split"),
            task=str(raw.get("task", "hidden_state")),
        )

    if not checkpoint_raw and str(model_raw.get("model_id", "")).startswith("stub/"):
        checkpoint_raw = {"kind": "stub", "label": "architecture_stub"}

    training_method = str(raw.get("training_method", "none"))
    baseline_immutable = bool(raw.get("baseline_immutable", training_method == "none"))

    return ExperimentConfig(
        name=str(raw["name"]),
        model=ModelSpec(**model_raw),
        dataset=dataset,
        output_dir=str(raw.get("output_dir", "runs")),
        video=VideoPreprocessConfig(**video_raw),
        generation=GenerationConfig.from_dict(generation_raw),
        device=DeviceConfig(**device_raw),
        seed_config=SeedConfig(**seed_raw),
        checkpoint=CheckpointSpec.from_dict(checkpoint_raw),
        stage=str(raw.get("stage", "baseline")),
        training_method=training_method,  # type: ignore[arg-type]
        reward_function=str(raw.get("reward_function", "exact_match")),
        preserve_raw_outputs=bool(raw.get("preserve_raw_outputs", True)),
        baseline_immutable=baseline_immutable,
        allow_held_out_in_training=bool(raw.get("allow_held_out_in_training", False)),
        allow_model_download=bool(raw.get("allow_model_download", False)),
        notes=str(raw.get("notes", "")),
        extras=dict(raw.get("extras") or {}),
    )


def build_run_manifest(config: ExperimentConfig, *, run_id: str | None = None) -> RunManifest:
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


def build_identity_block(config: ExperimentConfig) -> dict[str, Any]:
    """Compact identity fields required for scientific traceability."""
    return {
        "model": config.model.model_id,
        "model_revision": config.model.revision,
        "checkpoint_kind": config.checkpoint.kind,
        "checkpoint_path": config.checkpoint.path,
        "checkpoint_label": config.checkpoint.label,
        "parent_run_id": config.checkpoint.parent_run_id,
        "dataset_manifest": config.dataset.manifest,
        "dataset_version": config.dataset.version,
        "split": config.dataset.split,
        "task": config.dataset.task,
        "training_method": config.training_method,
        "reward_function": config.reward_function,
        "generation": config.generation.to_dict(),
        "seed": config.seed,
        "stage": config.stage,
        "is_zero_shot_baseline": config.is_zero_shot_baseline,
        "baseline_immutable": config.baseline_immutable,
    }


def initialize_experiment(
    config: ExperimentConfig,
    *,
    run_id: str | None = None,
    apply_seeds: bool = True,
) -> ExperimentContext:
    """Create output dirs and write reproducibility metadata without loading a VLM."""
    rid = run_id or make_run_id(config.name)
    run_dir = Path(config.output_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config.device)
    determinism = set_seed(config.seed_config) if apply_seeds else DeterminismReport(
        level="unavailable",
        seed=config.seed,
        settings_applied={},
        notes=("Seed application skipped by caller.",),
    )
    environment = capture_environment(device=device)
    payload = config.to_dict()
    record = ExperimentRecord(
        run_id=rid,
        created_at=utc_now_iso(),
        output_dir=str(run_dir),
        config=payload,
        config_hash=config_fingerprint(payload),
        identity=build_identity_block(config),
        environment=environment,
        determinism=determinism.to_dict(),
        device=device.to_dict(),
        git_commit=environment.get("git_commit"),
        code_state={
            "git_commit": environment.get("git_commit"),
            "git_dirty": environment.get("git_dirty"),
            "package_version": environment.get("package_version"),
            "config_hash": config_fingerprint(payload),
        },
    )

    # Serializable config + metadata (shell history not required to reproduce knobs).
    config.save(run_dir / "config.yaml")
    write_json(run_dir / "config.json", payload)
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "determinism.json", determinism.to_dict())
    write_json(run_dir / "device.json", device.to_dict())
    write_json(run_dir / "metadata.json", record.to_dict())
    write_json(run_dir / "run_manifest.json", build_run_manifest(config, run_id=rid).to_dict())
    write_json(
        run_dir / "result_changing_parameters.json",
        list_result_changing_parameters(config),
    )

    logger = setup_run_logging(run_dir)
    logger.info(
        "Initialized experiment run_id=%s zero_shot_baseline=%s device=%s determinism=%s",
        rid,
        config.is_zero_shot_baseline,
        device.resolved,
        determinism.level,
    )

    return ExperimentContext(
        run_id=rid,
        run_dir=run_dir,
        config=config,
        record=record,
        device=device,
        determinism=determinism,
    )


def list_result_changing_parameters(config: ExperimentConfig) -> dict[str, Any]:
    """Enumerate knobs that can change scientific results if altered."""
    return {
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "torch_dtype": config.model.torch_dtype,
        "checkpoint": config.checkpoint.to_dict(),
        "dataset_manifest": config.dataset.manifest,
        "dataset_version": config.dataset.version,
        "split": config.dataset.split,
        "task": config.dataset.task,
        "training_method": config.training_method,
        "reward_function": config.reward_function,
        "generation": config.generation.to_dict(),
        "video": asdict(config.video),
        "seed": config.seed,
        "seed_config": asdict(config.seed_config),
        "device": asdict(config.device),
        "allow_model_download": config.allow_model_download,
        "notes": (
            "Changing any of these fields can invalidate comparison against prior runs. "
            "Hardware, CUDA/driver, and library versions in environment.json also matter."
        ),
    }
