"""Dataset loading, validation, and split-boundary enforcement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from magic_vlm.schemas import ExampleRecord, SchemaError, Split, TaskType, validate_manifest_records


class SplitBoundaryError(ValueError):
    """Raised when held-out (or other forbidden) examples leak into a stage."""


def load_manifest(
    path: str | Path,
    *,
    validate: bool = True,
) -> list[ExampleRecord]:
    """Load a JSONL manifest of :class:`ExampleRecord` objects.

    Validation is on by default. This loader returns **all** splits present in
    the file; it does not mix them into a training set. Call
    :func:`filter_split` or :func:`load_split` explicitly for a single partition.
    """
    manifest_path = Path(path)
    records: list[ExampleRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
                records.append(ExampleRecord.from_dict(payload))
            except (json.JSONDecodeError, SchemaError, TypeError, ValueError) as exc:
                raise SchemaError(
                    f"Invalid manifest entry at {manifest_path}:{line_no}: {exc}"
                ) from exc
    if validate:
        validate_manifest_records(records)
    return records


def write_manifest(path: str | Path, records: Sequence[ExampleRecord]) -> None:
    """Write records as JSONL (one example per line).

    Does not alter ``ground_truth`` strings. Validates uniqueness before write.
    """
    validate_manifest_records(list(records))
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")


def load_split(
    path: str | Path,
    split: Split,
    *,
    validate: bool = True,
) -> list[ExampleRecord]:
    """Load a manifest and return only the requested split."""
    return filter_split(load_manifest(path, validate=validate), split)


def filter_split(records: Iterable[ExampleRecord], split: Split) -> list[ExampleRecord]:
    return [record for record in records if record.split == split]


def filter_task(records: Iterable[ExampleRecord], task: TaskType) -> list[ExampleRecord]:
    return [record for record in records if record.task == task]


def examples_for_clip(records: Iterable[ExampleRecord], clip_id: str) -> list[ExampleRecord]:
    """Return all question variants for one clip (may span only one split)."""
    return [record for record in records if record.clip_id == clip_id]


def iter_for_stage(
    records: Sequence[ExampleRecord],
    *,
    stage: str,
    allow_held_out: bool = False,
) -> Iterator[ExampleRecord]:
    """Yield examples allowed for a named pipeline stage.

    Baseline evaluation may opt into held-out with ``allow_held_out=True``.
    Training / preference fitting / reward-model stages must leave it False.

    This helper never silently merges held-out into training: it raises instead.
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
