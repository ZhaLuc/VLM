"""Preference-pair I/O (collection/training algorithms come later)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from magic_vlm.schemas import PreferencePair, Split


def load_preference_pairs(path: str | Path) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                pairs.append(PreferencePair.from_dict(json.loads(text)))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid preference row at {path}:{line_no}") from exc
    return pairs


def write_preference_pairs(path: str | Path, pairs: Sequence[PreferencePair]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_dict(), ensure_ascii=True) + "\n")


def filter_preference_split(pairs: Sequence[PreferencePair], split: Split) -> list[PreferencePair]:
    return [pair for pair in pairs if pair.split is split]
