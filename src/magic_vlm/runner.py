"""Configuration-driven experiment dispatcher.

Reuses existing runners (baseline, temporal-shuffle, DPO, reward-model).
Does not implement a plugin system, distributed scheduling, silent retries,
or overwrite of prior runs.

Research integrity
------------------
Every dispatch writes enough metadata to answer: "What exactly produced this
number?" Failed runs are recorded. Checkpoint selection must not use final
held-out test performance (enforced here for DPO; trainers keep their policies).
"""

from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from magic_vlm.experiment import experiment_config_from_dict
from magic_vlm.utils import (
    RunDirectoryError,
    allocate_run_directory,
    assert_run_directory_available,
    config_fingerprint,
    git_commit_sha,
    make_run_id,
    utc_now_iso,
    write_json,
)

ExperimentType = Literal[
    "baseline",
    "temporal_shuffle",
    "dpo",
    "grpo",
    "reward_model",
    "comparison",
    "reward_hacking",
]

SUPPORTED_EXPERIMENT_TYPES: tuple[str, ...] = (
    "baseline",
    "temporal_shuffle",
    "dpo",
    "grpo",
    "reward_model",
    "comparison",
    "reward_hacking",
)

INTEGRITY_NOTE = (
    "Common experiment runner: config-driven dispatch only. "
    "Does not silently retry failures or overwrite prior run directories. "
    "Does not choose checkpoints using final held-out test performance."
)


class ExperimentDispatchError(ValueError):
    """Invalid experiment configuration or unsupported experiment type."""


@dataclass(frozen=True)
class DispatchResult:
    """Outcome of one dispatched experiment."""

    experiment_type: str
    run_id: str
    run_dir: str
    status: str  # ok | failed
    config_path: str | None
    config_hash: str | None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    integrity_note: str = INTEGRITY_NOTE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_supported_experiments() -> dict[str, str]:
    return {
        "baseline": "Zero-shot immutable baseline (ExperimentConfig)",
        "temporal_shuffle": "Temporal-order diagnostic (ExperimentConfig)",
        "dpo": "DPO post-training (DPOConfigSpec)",
        "grpo": "GRPO post-training on objective reward (GRPOConfigSpec)",
        "reward_model": "Bradley-Terry text reward model (RewardModelConfig)",
        "comparison": "Cross-method locked held-out comparative evaluation",
        "reward_hacking": "Reward–quality divergence diagnostics (observational)",
    }


def load_raw_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ExperimentDispatchError(f"Config must be a mapping: {path}")
    return dict(raw)


def resolve_experiment_type(raw: dict[str, Any]) -> ExperimentType:
    """Resolve experiment type. Prefer explicit ``experiment_type``."""
    explicit = raw.get("experiment_type")
    if explicit not in (None, "", "null"):
        key = str(explicit).strip()
        if key not in SUPPORTED_EXPERIMENT_TYPES:
            raise ExperimentDispatchError(
                f"Unsupported experiment_type {key!r}. "
                f"Supported: {list(SUPPORTED_EXPERIMENT_TYPES)}"
            )
        return key  # type: ignore[return-value]

    if "prefs_path" in raw and "beta" in raw and "model_id" in raw:
        return "dpo"
    if (
        raw.get("reward_id")
        and "num_generations" in raw
        and "model_id" in raw
        and "manifest" in raw
    ):
        return "grpo"
    if "methods" in raw and ("protocol" in raw or raw.get("experiment_type") == "comparison"):
        return "comparison"
    if (
        "before" in raw
        and "after" in raw
        and ("quadrant_reward" in raw or raw.get("experiment_type") == "reward_hacking")
    ):
        return "reward_hacking"
    if "prefs_path" in raw and "embedding_dim" in raw:
        return "reward_model"
    stage = str(raw.get("stage") or "")
    training = str(raw.get("training_method") or "none")
    name = str(raw.get("name") or "")
    if stage == "diagnostic" or name.startswith("temporal_shuffle"):
        return "temporal_shuffle"
    if training == "none" and "model" in raw and "dataset" in raw:
        return "baseline"
    raise ExperimentDispatchError(
        "Could not resolve experiment_type. Set experiment_type explicitly to one of "
        f"{list(SUPPORTED_EXPERIMENT_TYPES)}"
    )


def validate_dispatch_config(raw: dict[str, Any], *, experiment_type: str) -> None:
    """Validate that required scientific knobs exist for the resolved type."""
    if experiment_type in {"baseline", "temporal_shuffle"}:
        missing = [k for k in ("name", "model", "dataset") if k not in raw]
        if missing:
            raise ExperimentDispatchError(
                f"{experiment_type} config missing required fields: {missing}"
            )
        if "model_id" not in dict(raw.get("model") or {}):
            raise ExperimentDispatchError(f"{experiment_type} config requires model.model_id")
        dataset = dict(raw.get("dataset") or {})
        if "manifest" not in dataset and "dataset_manifest" not in raw:
            raise ExperimentDispatchError(
                f"{experiment_type} config requires dataset.manifest"
            )
        if str(raw.get("training_method", "none")) != "none":
            raise ExperimentDispatchError(
                f"{experiment_type} requires training_method: none"
            )
        return

    if experiment_type == "dpo":
        for key in ("prefs_path", "model_id", "output_dir"):
            if key not in raw:
                raise ExperimentDispatchError(f"dpo config missing {key}")
        selection = str(raw.get("checkpoint_selection") or "last_train_step")
        lowered = selection.lower()
        if "held_out" in lowered or lowered in {"test", "final_test", "heldout"}:
            raise ExperimentDispatchError(
                "checkpoint_selection must not use final held-out/test performance "
                f"(got {selection!r})"
            )
        return

    if experiment_type == "grpo":
        for key in ("manifest", "model_id", "output_dir", "reward_id", "num_generations"):
            if key not in raw:
                raise ExperimentDispatchError(f"grpo config missing {key}")
        if int(raw["num_generations"]) < 2:
            raise ExperimentDispatchError("grpo num_generations (group size) must be >= 2")
        selection = str(raw.get("checkpoint_selection") or "last_train_step")
        lowered = selection.lower()
        if "held_out" in lowered or lowered in {"test", "final_test", "heldout"}:
            raise ExperimentDispatchError(
                "checkpoint_selection must not use final held-out/test performance "
                f"(got {selection!r})"
            )
        return

    if experiment_type == "reward_model":
        for key in ("prefs_path", "output_dir"):
            if key not in raw:
                raise ExperimentDispatchError(f"reward_model config missing {key}")
        return

    if experiment_type == "comparison":
        if "methods" not in raw:
            raise ExperimentDispatchError("comparison config missing methods")
        proto = dict(raw.get("protocol") or {})
        if "manifest" not in proto and "manifest" not in raw:
            raise ExperimentDispatchError("comparison config requires protocol.manifest")
        if not list(raw.get("methods") or []):
            raise ExperimentDispatchError("comparison config methods must be non-empty")
        selection = str(raw.get("checkpoint_selection") or "")
        if selection:
            lowered = selection.lower()
            if "held_out" in lowered or lowered in {"test", "final_test", "heldout"}:
                raise ExperimentDispatchError(
                    "comparison must not select checkpoints via final held-out/"
                    f"test performance (got {selection!r})"
                )
        return

    if experiment_type == "reward_hacking":
        for key in ("before", "after"):
            if key not in raw:
                raise ExperimentDispatchError(f"reward_hacking config missing {key}")
        if "manifest" not in raw and "manifest" not in dict(raw.get("protocol") or {}):
            raise ExperimentDispatchError("reward_hacking config requires manifest")
        return

    raise ExperimentDispatchError(f"Unsupported experiment_type: {experiment_type}")


def _resolve_run_id(raw: dict[str, Any], *, run_id: str | None, default_name: str) -> str:
    if run_id is not None and str(run_id).strip():
        return str(run_id).strip()
    if raw.get("run_id"):
        return str(raw["run_id"]).strip()
    name = str(raw.get("name") or default_name)
    return make_run_id(name)


def _output_dir_for(raw: dict[str, Any], experiment_type: str) -> str:
    if raw.get("output_dir"):
        return str(raw["output_dir"])
    if experiment_type == "dpo":
        return "runs/dpo"
    if experiment_type == "grpo":
        return "runs/grpo"
    if experiment_type == "reward_model":
        return "runs/reward_model"
    if experiment_type == "comparison":
        return "runs/comparison"
    if experiment_type == "reward_hacking":
        return "runs/reward_hacking"
    return "runs"


def _write_dispatch_header(
    run_dir: Path,
    *,
    experiment_type: str,
    config_path: str | Path | None,
    raw: dict[str, Any],
    run_id: str,
    started_at: str,
) -> str:
    payload = dict(raw)
    payload["experiment_type"] = experiment_type
    config_hash = config_fingerprint(payload)
    write_json(
        run_dir / "dispatch_config.json",
        {
            "experiment_type": experiment_type,
            "run_id": run_id,
            "config_path": None if config_path is None else str(config_path),
            "config_hash": config_hash,
            "config": payload,
            "git_commit": git_commit_sha(),
            "started_at": started_at,
            "integrity_note": INTEGRITY_NOTE,
        },
    )
    write_json(
        run_dir / "status.json",
        {
            "status": "running",
            "experiment_type": experiment_type,
            "run_id": run_id,
            "started_at": started_at,
        },
    )
    return config_hash


def _finalize(run_dir: Path, result: DispatchResult) -> None:
    write_json(run_dir / "status.json", result.to_dict())
    write_json(run_dir / "dispatch_result.json", result.to_dict())
    if result.status == "failed":
        write_json(
            run_dir / "failure.json",
            {
                "error_type": result.error_type,
                "error_message": result.error_message,
                "finished_at": result.finished_at,
                "experiment_type": result.experiment_type,
                "run_id": result.run_id,
                "traceback_path": "failure_traceback.txt",
                "note": "Failed run recorded; not retried silently.",
            },
        )


def _run_baseline(
    raw: dict[str, Any],
    *,
    run_id: str,
    load_frames: bool,
    allow_download: bool | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    from magic_vlm.baseline import run_zero_shot_baseline

    config = experiment_config_from_dict({**raw, "experiment_type": "baseline"})
    result = run_zero_shot_baseline(
        config,
        run_id=run_id,
        load_frames=load_frames,
        allow_download=allow_download,
    )
    metrics = result.summary.to_dict()
    artifacts = {
        "predictions": "predictions.jsonl",
        "metrics": "metrics.json",
        "baseline_summary": "baseline_summary.json",
        "immutable_marker": "BASELINE_IMMUTABLE.json",
    }
    return result.run_dir, metrics, artifacts


def _run_temporal_shuffle(
    raw: dict[str, Any],
    *,
    run_id: str,
    load_frames: bool,
    allow_download: bool | None,
    shuffle_seed: int | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    from magic_vlm.temporal import run_temporal_shuffle_experiment

    config = experiment_config_from_dict({**raw, "experiment_type": "temporal_shuffle"})
    summary, _pairs, run_dir = run_temporal_shuffle_experiment(
        config,
        run_id=run_id,
        load_frames=load_frames,
        allow_download=allow_download,
        shuffle_seed=shuffle_seed,
    )
    metrics = summary.to_dict()
    artifacts = {
        "pairs": "temporal_shuffle_pairs.jsonl",
        "summary": "temporal_shuffle_summary.json",
        "metadata": "temporal_shuffle_metadata.json",
        "metrics": "metrics.json",
    }
    return str(run_dir), metrics, artifacts


def _run_dpo(raw: dict[str, Any], *, run_id: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    from magic_vlm.dpo import DPOConfigSpec, train_dpo

    payload = dict(raw)
    payload["run_id"] = run_id
    config = DPOConfigSpec.from_dict(payload)
    result = train_dpo(config)
    metrics = dict(result.metrics)
    artifacts = {
        "checkpoint_dir": result.checkpoint_dir,
        "result": "result.json",
        "train_metadata": "train_metadata.json",
    }
    return result.run_dir, metrics, artifacts


def _run_grpo(raw: dict[str, Any], *, run_id: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    from magic_vlm.grpo import GRPOConfigSpec, train_grpo

    payload = dict(raw)
    payload["run_id"] = run_id
    config = GRPOConfigSpec.from_dict(payload)
    result = train_grpo(config)
    metrics = dict(result.metrics)
    artifacts = {
        "checkpoint_dir": result.checkpoint_dir,
        "result": "result.json",
        "train_metadata": "train_metadata.json",
        "raw_completions": "raw_completions.jsonl",
        "reward_stats": "reward_stats.json",
        "held_out_eval": "held_out_eval.json",
    }
    return result.run_dir, metrics, artifacts


def _run_reward_model(
    raw: dict[str, Any], *, run_id: str
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    from magic_vlm.reward_model import RewardModelConfig, train_bradley_terry_reward_model

    payload = dict(raw)
    payload["run_id"] = run_id
    config = RewardModelConfig.from_dict(payload)
    result = train_bradley_terry_reward_model(config)
    metrics = {
        "best_val_preference_accuracy": result.best_val_preference_accuracy,
        "best_epoch": result.best_epoch,
        "n_train": result.n_train,
        "n_val": result.n_val,
    }
    artifacts = {
        "checkpoint": result.checkpoint_path,
        "train_result": "train_result.json",
        "metrics": "metrics.jsonl",
    }
    return result.run_dir, metrics, artifacts


def _run_comparison(
    raw: dict[str, Any], *, run_id: str
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    from magic_vlm.comparison import ComparisonConfig, run_comparison

    payload = dict(raw)
    payload["run_id"] = run_id
    config = ComparisonConfig.from_dict(payload)
    result = run_comparison(config)
    metrics = {
        "n_locked": result.report["coverage"]["locked_n"],
        "aggregates": result.report.get("aggregates"),
        "protocol_compatible": result.report["protocol_compatibility"]["compatible"],
    }
    artifacts = {
        "aligned": "aligned_examples.jsonl",
        "metrics": "comparison_metrics.json",
        "report": "comparison_report.md",
    }
    return result.run_dir, metrics, artifacts


def _run_reward_hacking(
    raw: dict[str, Any], *, run_id: str
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    from magic_vlm.reward_hacking import RewardHackingConfig, run_reward_hacking

    payload = dict(raw)
    payload["run_id"] = run_id
    config = RewardHackingConfig.from_dict(payload)
    result = run_reward_hacking(config)
    metrics = {
        "n_aligned": result.report["n_aligned"],
        "before_after": result.report.get("before_after"),
        "frac_high_reward_low_accuracy_after": result.report.get(
            "frac_high_reward_low_accuracy_after"
        ),
        "human_evaluation_available": result.report["human_evaluation"]["available"],
    }
    artifacts = {
        "metrics": "reward_hacking_metrics.json",
        "report": "reward_hacking_report.md",
        "examples": "examples_inspectable.jsonl",
        "high_reward_low_accuracy": "high_reward_low_accuracy.jsonl",
    }
    return result.run_dir, metrics, artifacts


def run_experiment(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    experiment_type: str | None = None,
    load_frames: bool = False,
    allow_download: bool | None = None,
    shuffle_seed: int | None = None,
    propagate_errors: bool = True,
) -> DispatchResult:
    """Load a YAML config, validate, dispatch to an existing experiment runner.

    Refuses to overwrite an existing run directory. Does not silently retry.
    On failure, writes ``failure.json`` / ``status.json`` when a run dir exists,
    then re-raises unless ``propagate_errors=False``.
    """
    path = Path(config_path)
    raw = load_raw_config(path)
    etype = (
        str(experiment_type).strip()
        if experiment_type not in (None, "")
        else resolve_experiment_type(raw)
    )
    if etype not in SUPPORTED_EXPERIMENT_TYPES:
        raise ExperimentDispatchError(
            f"Unsupported experiment_type {etype!r}. "
            f"Supported: {list(SUPPORTED_EXPERIMENT_TYPES)}"
        )
    validate_dispatch_config(raw, experiment_type=etype)

    rid = _resolve_run_id(raw, run_id=run_id, default_name=etype)
    output_dir = _output_dir_for(raw, etype)
    # Refuse collisions before any runner starts.
    assert_run_directory_available(output_dir, rid, overwrite=False)

    started_at = utc_now_iso()
    intended_run_dir = Path(output_dir) / rid
    run_dir: Path | None = None
    config_hash: str | None = None

    try:
        if etype == "baseline":
            out_path, metrics, artifacts = _run_baseline(
                raw,
                run_id=rid,
                load_frames=load_frames,
                allow_download=allow_download,
            )
        elif etype == "temporal_shuffle":
            out_path, metrics, artifacts = _run_temporal_shuffle(
                raw,
                run_id=rid,
                load_frames=load_frames,
                allow_download=allow_download,
                shuffle_seed=shuffle_seed,
            )
        elif etype == "dpo":
            out_path, metrics, artifacts = _run_dpo(raw, run_id=rid)
        elif etype == "grpo":
            out_path, metrics, artifacts = _run_grpo(raw, run_id=rid)
        elif etype == "reward_model":
            out_path, metrics, artifacts = _run_reward_model(raw, run_id=rid)
        elif etype == "comparison":
            out_path, metrics, artifacts = _run_comparison(raw, run_id=rid)
        elif etype == "reward_hacking":
            out_path, metrics, artifacts = _run_reward_hacking(raw, run_id=rid)
        else:  # pragma: no cover
            raise ExperimentDispatchError(f"Unsupported experiment_type: {etype}")

        run_dir = Path(out_path)
        config_hash = _write_dispatch_header(
            run_dir,
            experiment_type=etype,
            config_path=path,
            raw=raw,
            run_id=rid,
            started_at=started_at,
        )
        finished = utc_now_iso()
        result = DispatchResult(
            experiment_type=etype,
            run_id=rid,
            run_dir=str(run_dir),
            status="ok",
            config_path=str(path),
            config_hash=config_hash,
            metrics=metrics,
            artifacts=artifacts,
            started_at=started_at,
            finished_at=finished,
        )
        _finalize(run_dir, result)
        return result
    except Exception as exc:
        finished = utc_now_iso()
        if intended_run_dir.exists():
            run_dir = intended_run_dir
        elif run_dir is None:
            try:
                run_dir = allocate_run_directory(output_dir, rid, overwrite=False)
            except Exception:  # noqa: BLE001
                run_dir = None

        if run_dir is not None and config_hash is None:
            try:
                config_hash = _write_dispatch_header(
                    run_dir,
                    experiment_type=etype,
                    config_path=path,
                    raw=raw,
                    run_id=rid,
                    started_at=started_at,
                )
            except Exception:  # noqa: BLE001
                pass

        result = DispatchResult(
            experiment_type=etype,
            run_id=rid,
            run_dir="" if run_dir is None else str(run_dir),
            status="failed",
            config_path=str(path),
            config_hash=config_hash,
            error_type=type(exc).__name__,
            error_message=str(exc),
            started_at=started_at,
            finished_at=finished,
        )
        if run_dir is not None:
            _finalize(run_dir, result)
            (run_dir / "failure_traceback.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        if propagate_errors:
            raise
        return result
