"""Small shared helpers for hashing, time, and artifact I/O."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_fingerprint(data: Any) -> str:
    digest = hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()
    return digest[:16]


def make_run_id(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name).strip("-")
    return f"{stamp}_{safe or 'run'}"


class RunDirectoryError(FileExistsError):
    """Raised when a run directory already exists and overwrite is forbidden."""


def allocate_run_directory(
    output_dir: str | Path,
    run_id: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Create a unique run directory. Refuses to overwrite by default."""
    run_dir = assert_run_directory_available(output_dir, run_id, overwrite=overwrite)
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def assert_run_directory_available(
    output_dir: str | Path,
    run_id: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Return the intended run path; raise if it already exists (unless overwrite)."""
    if not str(run_id).strip():
        raise ValueError("run_id must be non-empty")
    run_dir = Path(output_dir) / run_id
    if run_dir.exists() and not overwrite:
        raise RunDirectoryError(
            f"Run directory already exists (refusing overwrite): {run_dir}. "
            "Choose a new --run-id or output_dir so prior results stay intact."
        )
    return run_dir


def git_commit_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
