"""Paired comparison metrics for the temporal-order diagnostic.

These counts describe whether predictions change when the **same** sampled
frames are reordered. They are not a causal-reasoning proof.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence

# Matches ``magic_vlm.video._shuffle_indices`` (LCG Fisher-Yates).
SHUFFLE_METHOD = "lcg_fisher_yates_permutation_of_sampled_indices"

INTEGRITY_NOTE = (
    "Temporal-shuffle sensitivity is a temporal-order diagnostic: it shows "
    "whether predictions change when the same sampled frames are reordered. "
    "It is NOT proof of causal reasoning."
)


@dataclass(frozen=True)
class TemporalShuffleSummary:
    n_pairs: int
    ordered_accuracy: float | None
    shuffled_accuracy: float | None
    accuracy_difference: float | None  # ordered - shuffled
    n_both_correct: int
    n_both_incorrect: int
    n_ordered_only: int
    n_shuffled_only: int
    shuffle_seed: int
    shuffle_method: str
    split: str
    integrity_note: str = INTEGRITY_NOTE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_outcome(ordered_correct: bool, shuffled_correct: bool) -> str:
    if ordered_correct and shuffled_correct:
        return "both_correct"
    if (not ordered_correct) and (not shuffled_correct):
        return "both_incorrect"
    if ordered_correct and not shuffled_correct:
        return "ordered_only"
    return "shuffled_only"


def summarize_pairs(
    pairs: Sequence[Any],
    *,
    shuffle_seed: int,
    split: str,
) -> TemporalShuffleSummary:
    """Aggregate paired correctness into accuracies, difference, and outcome counts."""
    n = len(pairs)
    n_ord = sum(1 for p in pairs if p.ordered_correct)
    n_shf = sum(1 for p in pairs if p.shuffled_correct)
    counts = Counter(p.outcome for p in pairs)
    ordered_acc = (n_ord / n) if n else None
    shuffled_acc = (n_shf / n) if n else None
    diff = None
    if ordered_acc is not None and shuffled_acc is not None:
        diff = ordered_acc - shuffled_acc
    return TemporalShuffleSummary(
        n_pairs=n,
        ordered_accuracy=ordered_acc,
        shuffled_accuracy=shuffled_acc,
        accuracy_difference=diff,
        n_both_correct=counts.get("both_correct", 0),
        n_both_incorrect=counts.get("both_incorrect", 0),
        n_ordered_only=counts.get("ordered_only", 0),
        n_shuffled_only=counts.get("shuffled_only", 0),
        shuffle_seed=shuffle_seed,
        shuffle_method=SHUFFLE_METHOD,
        split=split,
    )
