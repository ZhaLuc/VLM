"""Training stage placeholders.

DPO, GRPO, and PPO are intentionally unimplemented. This module only defines
configuration surfaces and boundary checks so later stages can plug in without
rewiring model loading or rewards.
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

    Only ``algorithm='none'`` is supported in this architecture stage.
    """
    validate_training_split(examples)
    if config.algorithm == "none":
        return TrainingResult(
            status="skipped",
            algorithm=config.algorithm,
            message="No training requested (baseline architecture stage).",
            metrics={"n_examples": len(examples)},
        )
    raise NotImplementedError(
        f"Training algorithm {config.algorithm!r} is not implemented yet. "
        "Model loading and reward functions remain usable independently."
    )
