"""Training stage dispatch for VLM post-training algorithms.

GRPO and PPO on the VLM remain unimplemented here.
DPO lives in ``magic_vlm.dpo`` (TRL + optional PEFT). A separate Bradley-Terry
preference reward model lives in ``magic_vlm.reward_model``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from magic_vlm.dataset import SplitBoundaryError, assert_no_held_out
from magic_vlm.schemas import ExampleRecord, Split

AlgorithmName = Literal["none", "dpo", "grpo", "ppo", "sft"]


@dataclass(frozen=True)
class TrainingConfig:
    """Declarative training request.

    ``algorithm='none'`` is the baseline/no-op path used by early stages.
    ``algorithm='dpo'`` delegates to ``magic_vlm.dpo`` (requires a DPO config path
    in ``extras['dpo_config']``).
    """

    algorithm: AlgorithmName = "none"
    output_dir: str = "runs/train"
    learning_rate: float = 1e-5
    max_steps: int = 0
    lora: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    status: str
    algorithm: str
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)


def validate_training_split(examples: Sequence[ExampleRecord]) -> None:
    """Refuse held-out leakage before any future trainer is invoked."""
    assert_no_held_out(examples, context="training")
    if not examples:
        raise ValueError("Training requires at least one non-held-out example")
    if any(example.split is Split.HELD_OUT for example in examples):
        raise SplitBoundaryError("held_out examples cannot enter training")


def run_training(config: TrainingConfig, examples: Sequence[ExampleRecord]) -> TrainingResult:
    """Dispatch training.

    ``none`` skips. ``dpo`` requires ``extras['dpo_config']`` and ignores the
    ExampleRecord list (DPO uses preference JSONL). GRPO/PPO remain unimplemented.
    """
    if config.algorithm != "dpo":
        validate_training_split(examples)
    if config.algorithm == "none":
        return TrainingResult(
            status="skipped",
            algorithm=config.algorithm,
            message="No training requested (baseline architecture stage).",
            metrics={"n_examples": len(examples)},
        )
    if config.algorithm == "dpo":
        from magic_vlm.dpo import DPOConfigSpec, train_dpo

        dpo_cfg_path = config.extras.get("dpo_config")
        if not dpo_cfg_path:
            raise ValueError("DPO requires extras['dpo_config'] path to a YAML config")
        result = train_dpo(DPOConfigSpec.from_yaml(dpo_cfg_path))
        return TrainingResult(
            status=result.status,
            algorithm="dpo",
            message=f"DPO finished; checkpoint={result.checkpoint_dir}",
            metrics=result.metrics,
        )
    raise NotImplementedError(
        f"Training algorithm {config.algorithm!r} is not implemented yet. "
        "Use magic_vlm.dpo for DPO; GRPO/PPO are not available."
    )
