"""Minimal human preference annotation workflow (research tooling).

Loads candidate explanation pairs, shows video path + task + raw A/B text,
records A/B judgments with optional rationale into an append-only JSONL store.
Does not modify candidate responses, train models, or sync to the cloud.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

import yaml

from magic_vlm.preferences import (
    build_preference_pair,
    compute_content_pair_id,
    load_preference_pairs,
)
from magic_vlm.schemas import (
    PreferenceGenerationMeta,
    PreferencePair,
    Provenance,
    SchemaError,
    Split,
    TaskType,
    VideoRef,
)
from magic_vlm.utils import utc_now_iso, write_json

DEFAULT_RUBRIC_VERSION = "explanation_pref_v1"
DEFAULT_RUBRIC_PATH = Path("configs/annotation_rubric.yaml")


class AnnotationError(ValueError):
    """Raised for annotation workflow / storage integrity failures."""


@dataclass(frozen=True)
class AnnotationRubric:
    """Configurable preference rubric (loaded from YAML or defaults)."""

    version: str
    title: str
    prioritize: tuple[str, ...]
    do_not_reward: tuple[str, ...]
    decision_rule: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "title": self.title,
            "prioritize": list(self.prioritize),
            "do_not_reward": list(self.do_not_reward),
            "decision_rule": self.decision_rule,
            "notes": list(self.notes),
        }

    def format_display(self) -> str:
        lines = [
            f"Rubric: {self.title} ({self.version})",
            "",
            "Prefer the response that is stronger on:",
        ]
        for item in self.prioritize:
            lines.append(f"  + {item}")
        lines.append("")
        lines.append("Do NOT prefer a response merely because it is:")
        for item in self.do_not_reward:
            lines.append(f"  - {item}")
        lines.append("")
        lines.append(f"Decision rule: {self.decision_rule}")
        for note in self.notes:
            lines.append(f"Note: {note}")
        return "\n".join(lines)


def default_rubric() -> AnnotationRubric:
    return AnnotationRubric(
        version=DEFAULT_RUBRIC_VERSION,
        title="Mechanism explanation preference",
        prioritize=(
            "Factual correctness relative to what the demonstration shows",
            "Evidence grounded in visible events in the clip",
            "Specificity about the hidden mechanism (not vague restatement)",
        ),
        do_not_reward=(
            "Verbosity or length by itself",
            "Confident tone or assertive phrasing by itself",
            "Fluent writing that invents unsupported details",
        ),
        decision_rule=(
            "Choose A or B based on correctness, evidence, and mechanism specificity. "
            "If both fail equally, still force a choice under the default protocol "
            "(ties are disabled unless explicitly enabled)."
        ),
        notes=(
            "Do not rewrite or edit the candidate responses.",
            "Optional rationales should cite evidence, not praise style.",
        ),
    )


def load_rubric(path: str | Path | None = None) -> AnnotationRubric:
    """Load rubric YAML; missing path falls back to the built-in default."""
    if path is None:
        path = DEFAULT_RUBRIC_PATH if DEFAULT_RUBRIC_PATH.exists() else None
    if path is None:
        return default_rubric()
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise AnnotationError(f"Rubric file must be a mapping: {path}")
    base = default_rubric()
    return AnnotationRubric(
        version=str(payload.get("version", base.version)),
        title=str(payload.get("title", base.title)),
        prioritize=tuple(payload.get("prioritize") or base.prioritize),
        do_not_reward=tuple(payload.get("do_not_reward") or base.do_not_reward),
        decision_rule=str(payload.get("decision_rule", base.decision_rule)),
        notes=tuple(payload.get("notes") or base.notes),
    )


@dataclass(frozen=True)
class AnnotationCandidate:
    """One pending pairwise comparison (no human judgment yet)."""

    clip_id: str
    instruction: str
    response_a: str
    response_b: str
    generation_meta: PreferenceGenerationMeta
    example_id: str | None = None
    video: VideoRef | None = None
    task: TaskType = TaskType.EXPLANATION
    split: Split = Split.TRAIN
    queue_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("clip_id", self.clip_id),
            ("instruction", self.instruction),
        ):
            if not str(value).strip():
                raise AnnotationError(f"{name} must be a non-empty string")
        if not isinstance(self.response_a, str) or not isinstance(self.response_b, str):
            raise AnnotationError("response_a and response_b must be raw strings")

    @property
    def pair_id(self) -> str:
        return compute_content_pair_id(
            clip_id=self.clip_id,
            instruction=self.instruction,
            response_a=self.response_a,
            response_b=self.response_b,
            task=self.task,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "clip_id": self.clip_id,
            "example_id": self.example_id,
            "video": None if self.video is None else self.video.to_dict(),
            "task": self.task.value,
            "instruction": self.instruction,
            "response_a": self.response_a,
            "response_b": self.response_b,
            "split": self.split.value,
            "generation_meta": self.generation_meta.to_dict(),
            "metadata": dict(self.metadata),
            "pair_id": self.pair_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationCandidate:
        if not isinstance(data, dict):
            raise AnnotationError("annotation candidate must be a JSON object")
        payload = dict(data)
        instruction = payload.get("instruction")
        if instruction is None:
            instruction = payload.get("question")
        if instruction is None:
            raise AnnotationError("candidate requires instruction (or question)")
        if "response_a" not in payload or "response_b" not in payload:
            raise AnnotationError("candidate requires response_a and response_b")
        # Preserve candidate text exactly.
        response_a = payload["response_a"]
        response_b = payload["response_b"]
        if not isinstance(response_a, str) or not isinstance(response_b, str):
            raise AnnotationError("response_a and response_b must be JSON strings")

        gen_raw = payload.get("generation_meta")
        if gen_raw is None:
            mid = str(payload.get("model_id") or "unknown")
            gen_raw = {
                "model_id_a": mid,
                "model_id_b": mid,
                "generation_a": dict(payload.get("generation_a") or payload.get("generation") or {}),
                "generation_b": dict(payload.get("generation_b") or payload.get("generation") or {}),
            }
        generation_meta = PreferenceGenerationMeta.from_dict(dict(gen_raw))

        video = None
        if payload.get("video") is not None:
            video = VideoRef.from_dict(payload["video"])
        try:
            task = TaskType(str(payload.get("task", TaskType.EXPLANATION.value)))
        except ValueError as exc:
            raise AnnotationError(f"invalid task: {payload.get('task')!r}") from exc
        try:
            split = Split(str(payload.get("split", Split.TRAIN.value)))
        except ValueError as exc:
            raise AnnotationError(f"invalid split: {payload.get('split')!r}") from exc

        return cls(
            clip_id=str(payload["clip_id"]),
            example_id=None if payload.get("example_id") is None else str(payload["example_id"]),
            video=video,
            task=task,
            instruction=str(instruction),
            response_a=response_a,
            response_b=response_b,
            generation_meta=generation_meta,
            split=split,
            queue_id=None if payload.get("queue_id") is None else str(payload["queue_id"]),
            metadata=dict(payload.get("metadata") or {}),
        )


def load_annotation_queue(path: str | Path) -> list[AnnotationCandidate]:
    rows: list[AnnotationCandidate] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(AnnotationCandidate.from_dict(json.loads(text)))
            except Exception as exc:  # noqa: BLE001
                raise AnnotationError(f"Invalid queue row at {path}:{line_no}") from exc
    return rows


def write_annotation_queue(path: str | Path, rows: list[AnnotationCandidate]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.to_dict()
            payload.pop("pair_id", None)  # derived; optional on reload
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


@dataclass
class AnnotationStore:
    """Append-only preference judgment store with resume helpers."""

    path: Path
    _judgment_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _by_annotator_pair: set[tuple[str, str]] = field(
        default_factory=set, init=False, repr=False
    )
    _pairs: list[PreferencePair] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.reload()

    def reload(self) -> None:
        self._judgment_ids = set()
        self._by_annotator_pair = set()
        self._pairs = []
        if not self.path.exists():
            return
        self._pairs = load_preference_pairs(self.path)
        for pair in self._pairs:
            self._judgment_ids.add(pair.judgment_id)
            self._by_annotator_pair.add((pair.annotator_id, pair.pair_id))

    @property
    def pairs(self) -> tuple[PreferencePair, ...]:
        return tuple(self._pairs)

    def has_judgment_id(self, judgment_id: str) -> bool:
        return judgment_id in self._judgment_ids

    def has_annotator_pair(self, annotator_id: str, pair_id: str) -> bool:
        return (annotator_id, pair_id) in self._by_annotator_pair

    def append(self, pair: PreferencePair) -> PreferencePair:
        """Persist one immutable judgment. Refuses duplicate judgment_id."""
        if pair.judgment_id in self._judgment_ids:
            raise AnnotationError(
                f"duplicate judgment_id refused: {pair.judgment_id!r}"
            )
        if (pair.annotator_id, pair.pair_id) in self._by_annotator_pair:
            raise AnnotationError(
                f"annotator {pair.annotator_id!r} already judged pair_id "
                f"{pair.pair_id!r}; resume skips this item"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(pair.to_dict(), ensure_ascii=True) + "\n")
        self._judgment_ids.add(pair.judgment_id)
        self._by_annotator_pair.add((pair.annotator_id, pair.pair_id))
        self._pairs.append(pair)
        return pair


@dataclass(frozen=True)
class AnnotationSessionConfig:
    annotator_id: str
    queue_path: Path
    judgments_path: Path
    rubric: AnnotationRubric = field(default_factory=default_rubric)
    video_root: Path | None = None
    open_video: bool = True
    allow_ties: bool = False
    provenance_source: str = "human_preference_annotation"
    skip_completed: bool = True


def resolve_video_path(
    candidate: AnnotationCandidate,
    *,
    video_root: Path | None = None,
) -> Path | None:
    if candidate.video is None or not str(candidate.video.path).strip():
        return None
    path = Path(candidate.video.path)
    if path.is_file():
        return path
    if video_root is not None:
        alt = video_root / path
        if alt.is_file():
            return alt
        alt2 = video_root / path.name
        if alt2.is_file():
            return alt2
    return path  # may be missing; still expose for the annotator


def open_video_for_annotator(path: Path) -> bool:
    """Best-effort local open (OS handler / browser). Returns True if attempted."""
    if not path.is_file():
        return False
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
            return True
        # Linux / other
        subprocess.run(["xdg-open", str(path)], check=False)
        return True
    except OSError:
        try:
            webbrowser.open(path.resolve().as_uri())
            return True
        except Exception:  # noqa: BLE001
            return False


def format_candidate_for_display(
    candidate: AnnotationCandidate,
    *,
    rubric: AnnotationRubric,
    index: int,
    total: int,
    video_path: Path | None,
) -> str:
    video_note = "missing"
    if video_path is not None:
        video_note = f"{video_path} ({'exists' if video_path.is_file() else 'NOT FOUND'})"
    lines = [
        "=" * 72,
        f"Annotation item {index}/{total}",
        f"queue_id={candidate.queue_id!r} pair_id={candidate.pair_id}",
        f"clip_id={candidate.clip_id} example_id={candidate.example_id}",
        f"video: {video_note}",
        f"task: {candidate.task.value}",
        "",
        rubric.format_display(),
        "",
        "Task / instruction:",
        candidate.instruction,
        "",
        "----- Response A (raw; do not edit) -----",
        candidate.response_a,
        "",
        "----- Response B (raw; do not edit) -----",
        candidate.response_b,
        "",
        "Choose the better explanation under the rubric (A or B).",
        "Do not prefer verbosity or confidence alone.",
        "=" * 72,
    ]
    return "\n".join(lines)


def pending_candidates(
    queue: Sequence[AnnotationCandidate],
    store: AnnotationStore,
    *,
    annotator_id: str,
    skip_completed: bool = True,
) -> list[AnnotationCandidate]:
    if not skip_completed:
        return list(queue)
    return [
        item
        for item in queue
        if not store.has_annotator_pair(annotator_id, item.pair_id)
    ]


def record_judgment(
    candidate: AnnotationCandidate,
    *,
    store: AnnotationStore,
    annotator_id: str,
    winner: str,
    rationale: str | None = None,
    rubric: AnnotationRubric | None = None,
    allow_ties: bool = False,
    provenance_source: str = "human_preference_annotation",
    timestamp: str | None = None,
) -> PreferencePair:
    """Build and append one preference judgment without altering responses."""
    rubric = rubric or default_rubric()
    winner_norm = str(winner).strip().lower()
    if winner_norm in {"a", "response_a", "1"}:
        winner_norm = "a"
    elif winner_norm in {"b", "response_b", "2"}:
        winner_norm = "b"
    elif winner_norm == "tie":
        winner_norm = "tie"
    else:
        raise AnnotationError(f"winner must be 'a' or 'b' (got {winner!r})")

    if store.has_annotator_pair(annotator_id, candidate.pair_id):
        raise AnnotationError(
            f"duplicate annotation refused for annotator={annotator_id!r} "
            f"pair_id={candidate.pair_id!r}"
        )

    pair = build_preference_pair(
        clip_id=candidate.clip_id,
        example_id=candidate.example_id,
        video=candidate.video,
        instruction=candidate.instruction,
        response_a=candidate.response_a,
        response_b=candidate.response_b,
        winner=winner_norm,
        annotator_id=annotator_id,
        timestamp=timestamp or utc_now_iso(),
        rationale=rationale,
        provenance=Provenance(
            source=provenance_source,
            created_by=annotator_id,
            created_at=timestamp or utc_now_iso(),
            collection_notes=f"rubric={rubric.version}",
        ),
        generation_meta=candidate.generation_meta,
        task=candidate.task,
        allow_ties=allow_ties,
        split=candidate.split,
        rubric_version=rubric.version,
        metadata=dict(candidate.metadata),
    )
    if store.has_judgment_id(pair.judgment_id):
        # Extremely unlikely collision on (pair, annotator, timestamp); refuse.
        raise AnnotationError(f"duplicate judgment_id refused: {pair.judgment_id!r}")
    return store.append(pair)


@dataclass(frozen=True)
class SessionResult:
    n_queue: int
    n_pending_before: int
    n_recorded: int
    n_skipped: int
    judgments_path: str
    recorded_judgment_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_annotation_session(
    config: AnnotationSessionConfig,
    *,
    winners: list[str] | None = None,
    rationales: list[str | None] | None = None,
    limit: int | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    open_video_fn: Callable[[Path], bool] | None = None,
) -> SessionResult:
    """Run annotation over pending queue items.

    Interactive when ``winners`` is None; otherwise applies provided winners
    in order (for tests / scripted smoke).
    """
    out = stdout or sys.stdout
    inp = stdin or sys.stdin
    open_fn = open_video_fn or open_video_for_annotator

    if not str(config.annotator_id).strip():
        raise AnnotationError("annotator_id is required")

    queue = load_annotation_queue(config.queue_path)
    store = AnnotationStore(config.judgments_path)
    pending = pending_candidates(
        queue,
        store,
        annotator_id=config.annotator_id,
        skip_completed=config.skip_completed,
    )
    if limit is not None:
        pending = pending[: max(0, limit)]

    out.write(f"Loaded queue n={len(queue)} pending={len(pending)} "
              f"store={config.judgments_path}\n")
    out.write(config.rubric.format_display() + "\n\n")

    recorded: list[str] = []
    skipped = 0
    scripted = winners is not None
    winner_iter = iter(winners or [])
    rationale_iter = iter(rationales or [])

    for index, candidate in enumerate(pending, start=1):
        video_path = resolve_video_path(candidate, video_root=config.video_root)
        out.write(
            format_candidate_for_display(
                candidate,
                rubric=config.rubric,
                index=index,
                total=len(pending),
                video_path=video_path,
            )
            + "\n"
        )
        if config.open_video and video_path is not None and video_path.is_file():
            opened = open_fn(video_path)
            out.write(f"[video open attempted={opened}]\n")
        elif video_path is not None:
            out.write(f"[video path exposed but file missing: {video_path}]\n")
        else:
            out.write("[no video path on candidate; inspect clip_id only]\n")

        if scripted:
            try:
                winner = next(winner_iter)
            except StopIteration as exc:
                raise AnnotationError("Not enough scripted winners for pending items") from exc
            rationale = next(rationale_iter, None)
        else:
            out.write("Enter choice [a/b/s=skip/q=quit], then optional rationale.\n")
            out.write("choice> ")
            out.flush()
            choice_line = inp.readline()
            if choice_line == "":
                break
            choice = choice_line.strip().lower()
            if choice in {"q", "quit"}:
                break
            if choice in {"s", "skip"}:
                skipped += 1
                continue
            winner = choice
            out.write("rationale (optional, empty to skip)> ")
            out.flush()
            rationale_line = inp.readline()
            rationale = rationale_line.rstrip("\n") if rationale_line else ""
            if rationale.strip() == "":
                rationale = None

        try:
            pair = record_judgment(
                candidate,
                store=store,
                annotator_id=config.annotator_id,
                winner=winner,
                rationale=rationale,
                rubric=config.rubric,
                allow_ties=config.allow_ties,
                provenance_source=config.provenance_source,
            )
        except (AnnotationError, SchemaError) as exc:
            raise AnnotationError(str(exc)) from exc
        recorded.append(pair.judgment_id)
        out.write(
            f"Saved judgment_id={pair.judgment_id} winner={pair.winner} "
            f"pair_id={pair.pair_id}\n\n"
        )

    result = SessionResult(
        n_queue=len(queue),
        n_pending_before=len(pending),
        n_recorded=len(recorded),
        n_skipped=skipped,
        judgments_path=str(config.judgments_path),
        recorded_judgment_ids=tuple(recorded),
    )
    write_json(
        Path(config.judgments_path).with_suffix(".session.json"),
        {
            **result.to_dict(),
            "annotator_id": config.annotator_id,
            "rubric_version": config.rubric.version,
            "queue_path": str(config.queue_path),
            "finished_at": utc_now_iso(),
        },
    )
    return result
