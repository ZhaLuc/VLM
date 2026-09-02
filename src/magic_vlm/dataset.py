"""Dataset loading and split-boundary enforcement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from magic_vlm.schemas import ExampleRecord, Split


class SplitBoundaryError(ValueError):
    """Raised when held-out (or other forbidden) examples leak into a stage."""


def load_manifest(path: str | Path) -> list[ExampleRecord]:
    """Load a JSONL manifest of :class:`ExampleRecord` objects."""
    manifest_path = Path(path)
    records: list[ExampleRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(ExampleRecord.from_dict(json.loads(text)))
            except Exception as exc:  # noqa: BLE001 - surface line context
                raise ValueError(f"Invalid manifest entry at {manifest_path}:{line_no}") from exc
    return records


def write_manifest(path: str | Path, records: Sequence[ExampleRecord]) -> None:
    """Write records as JSONL (one example per line)."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")


def filter_split(records: Iterable[ExampleRecord], split: Split) -> list[ExampleRecord]:
    return [record for record in records if record.split == split]


def iter_for_stage(
    records: Sequence[ExampleRecord],
    *,
    stage: str,
    allow_held_out: bool = False,
) -> Iterator[ExampleRecord]:
    """Yield examples allowed for a named pipeline stage.

    Baseline evaluation may opt into held-out with ``allow_held_out=True``.
    Training / preference fitting / reward-model stages must leave it False.
    """
    stage_normalized = stage.strip().lower()
    training_like = stage_normalized in {
        "train",
        "training",
        "dpo",
        "grpo",
        "ppo",
        "reward_model",
        "preference",
        "sft",
    }
    for record in records:
        if record.split is Split.HELD_OUT and training_like and not allow_held_out:
            raise SplitBoundaryError(
                f"Refusing held-out example {record.example_id!r} for stage {stage!r}"
            )
        if training_like and record.split is Split.HELD_OUT:
            continue
        yield record


def assert_no_held_out(records: Iterable[ExampleRecord], *, context: str) -> None:
    leaked = [r.example_id for r in records if r.split is Split.HELD_OUT]
    if leaked:
        raise SplitBoundaryError(f"Held-out leakage in {context}: {leaked}")
