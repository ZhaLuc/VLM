"""Standalone reward functions (testable without GRPO/DPO)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from magic_vlm.evaluation import exact_match
from magic_vlm.schemas import ExampleRecord, InferenceArtifact


@runtime_checkable
class RewardFunction(Protocol):
    """Score one prediction. Must not depend on a trainer loop."""

    name: str

    def score(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> float:
        ...


@dataclass(frozen=True)
class ExactMatchReward:
    """Programmatic 0/1 reward for hidden-state labels."""

    name: str = "exact_match"

    def score(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> float:
        prediction = artifact.parsed_answer if artifact.parsed_answer is not None else artifact.raw_text
        return 1.0 if exact_match(prediction, example.answer) else 0.0


@dataclass(frozen=True)
class LengthPenaltyReward:
    """Simple diagnostic reward; useful for reward-hacking smoke tests later."""

    name: str = "length_penalty"
    max_chars: int = 200
    base: RewardFunction = field(default_factory=ExactMatchReward)

    def score(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> float:
        base_score = self.base.score(artifact, example, **kwargs)
        penalty = min(len(artifact.raw_text) / float(self.max_chars), 1.0)
        return float(base_score) * (1.0 - 0.5 * penalty)


def score_batch(
    reward: RewardFunction,
    artifacts: list[InferenceArtifact],
    examples: list[ExampleRecord],
) -> list[float]:
    by_id = {example.example_id: example for example in examples}
    scores: list[float] = []
    for artifact in artifacts:
        example = by_id[artifact.example_id]
        scores.append(float(reward.score(artifact, example)))
    return scores
