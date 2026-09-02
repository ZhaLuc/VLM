"""Evaluation metrics that operate on preserved inference artifacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from magic_vlm.schemas import ExampleRecord, InferenceArtifact, Split


@dataclass(frozen=True)
class ExampleScore:
    example_id: str
    split: str
    correct: bool
    gold: str | None
    prediction: str | None
    raw_text: str
    trick_id: str | None = None
    parse_failed: bool = False
    latency_s: float | None = None

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


@dataclass(frozen=True)
class BaselineSummary:
    """Aggregate zero-shot baseline metrics."""

    metric: str
    n_examples: int
    n_correct: int
    overall_accuracy: float | None
    n_parse_failures: int
    parse_failure_rate: float | None
    per_trick_accuracy: dict[str, float]
    per_trick_counts: dict[str, int]
    mean_latency_s: float | None
    split: str
    example_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "n_examples": self.n_examples,
            "n_correct": self.n_correct,
            "overall_accuracy": self.overall_accuracy,
            "n_parse_failures": self.n_parse_failures,
            "parse_failure_rate": self.parse_failure_rate,
            "per_trick_accuracy": dict(self.per_trick_accuracy),
            "per_trick_counts": dict(self.per_trick_counts),
            "mean_latency_s": self.mean_latency_s,
            "split": self.split,
            "example_ids": list(self.example_ids),
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


def is_parse_failure(raw_text: str, parsed: str | None) -> bool:
    """True when parsing produced no usable label from a non-empty raw response."""
    if parsed is None:
        return True
    if raw_text.strip() and not str(parsed).strip():
        return True
    return False


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
        parse_failed = is_parse_failure(artifact.raw_text, pred)
        ok = (not parse_failed) and exact_match(pred, gold)
        scores.append(
            ExampleScore(
                example_id=example.example_id,
                split=example.split.value,
                correct=ok,
                gold=gold,
                prediction=pred,
                raw_text=artifact.raw_text,
                trick_id=example.trick_id,
                parse_failed=parse_failed,
                latency_s=artifact.latency_s,
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


def evaluate_baseline(
    examples: Sequence[ExampleRecord],
    artifacts: Sequence[InferenceArtifact],
    predictions: Sequence[Any],
) -> BaselineSummary:
    """Compute overall / per-trick accuracy and parse-failure counts.

    Every input example must appear in ``predictions`` (no silent exclusions).
    """
    if len(examples) != len(artifacts) or len(examples) != len(predictions):
        raise ValueError(
            "examples, artifacts, and predictions must have equal length "
            f"(got {len(examples)}, {len(artifacts)}, {len(predictions)})"
        )
    pred_ids = [getattr(p, "example_id") for p in predictions]
    example_ids = [ex.example_id for ex in examples]
    if pred_ids != example_ids:
        raise ValueError("prediction order/IDs must match examples exactly")

    n = len(predictions)
    n_correct = sum(1 for p in predictions if getattr(p, "correct"))
    n_parse_failures = sum(1 for p in predictions if getattr(p, "parse_failed"))
    overall = (n_correct / n) if n else None
    parse_rate = (n_parse_failures / n) if n else None

    trick_correct: dict[str, list[bool]] = defaultdict(list)
    for example, pred in zip(examples, predictions):
        trick_correct[example.trick_id].append(bool(getattr(pred, "correct")))
    per_trick_accuracy = {
        trick: (sum(flags) / len(flags) if flags else 0.0)
        for trick, flags in sorted(trick_correct.items())
    }
    per_trick_counts = {trick: len(flags) for trick, flags in sorted(trick_correct.items())}

    latencies = [
        float(getattr(p, "latency_s"))
        for p in predictions
        if getattr(p, "latency_s", None) is not None
    ]
    mean_latency = (sum(latencies) / len(latencies)) if latencies else None
    split = examples[0].split.value if examples else ""
    return BaselineSummary(
        metric="exact_match",
        n_examples=n,
        n_correct=n_correct,
        overall_accuracy=overall,
        n_parse_failures=n_parse_failures,
        parse_failure_rate=parse_rate,
        per_trick_accuracy=per_trick_accuracy,
        per_trick_counts=per_trick_counts,
        mean_latency_s=mean_latency,
        split=split,
        example_ids=tuple(example_ids),
    )


def summarize_scores(scores: Iterable[ExampleScore]) -> dict[str, float | int]:
    material = list(scores)
    if not material:
        return {"n": 0, "accuracy": 0.0}
    return {
        "n": len(material),
        "accuracy": sum(1 for s in material if s.correct) / len(material),
    }
