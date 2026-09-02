"""Evaluation metrics that operate on preserved inference artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from magic_vlm.schemas import ExampleRecord, InferenceArtifact, Split


@dataclass(frozen=True)
class ExampleScore:
    example_id: str
    split: str
    correct: bool
    gold: str | None
    prediction: str | None
    raw_text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    metric: str
    n: int
    accuracy: float | None
    scores: tuple[ExampleScore, ...]
    by_split: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "n": self.n,
            "accuracy": self.accuracy,
            "by_split": dict(self.by_split),
            "scores": [score.to_dict() for score in self.scores],
        }


def normalize_label(value: str | None) -> str:
    """Compare-time normalization for predictions and gold *copies*.

    This must never be used to rewrite stored dataset ``ground_truth`` values.
    Manifests keep the authored label verbatim.
    """
    if value is None:
        return ""
    return " ".join(value.strip().lower().split())


def exact_match(prediction: str | None, gold: str | None) -> bool:
    return normalize_label(prediction) == normalize_label(gold) and normalize_label(gold) != ""


def evaluate_exact_match(
    examples: Sequence[ExampleRecord],
    artifacts: Sequence[InferenceArtifact],
) -> EvaluationReport:
    """Score Task-B style exact-match accuracy from raw artifacts."""
    by_id = {artifact.example_id: artifact for artifact in artifacts}
    scores: list[ExampleScore] = []
    for example in examples:
        artifact = by_id.get(example.example_id)
        if artifact is None:
            raise KeyError(f"Missing inference artifact for example {example.example_id!r}")
        pred = artifact.parsed_answer
        gold = example.ground_truth
        ok = exact_match(pred, gold)
        scores.append(
            ExampleScore(
                example_id=example.example_id,
                split=example.split.value,
                correct=ok,
                gold=gold,
                prediction=pred,
                raw_text=artifact.raw_text,
            )
        )

    accuracy = (sum(1 for s in scores if s.correct) / len(scores)) if scores else None
    by_split: dict[str, float] = {}
    for split in Split:
        subset = [s for s in scores if s.split == split.value]
        if subset:
            by_split[split.value] = sum(1 for s in subset if s.correct) / len(subset)
    return EvaluationReport(
        metric="exact_match",
        n=len(scores),
        accuracy=accuracy,
        scores=tuple(scores),
        by_split=by_split,
    )


def summarize_scores(scores: Iterable[ExampleScore]) -> dict[str, float | int]:
    material = list(scores)
    if not material:
        return {"n": 0, "accuracy": 0.0}
    return {
        "n": len(material),
        "accuracy": sum(1 for s in material if s.correct) / len(material),
    }
