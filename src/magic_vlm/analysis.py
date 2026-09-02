"""Baseline failure analysis and research diagnostics.

Scientific posture
------------------
Accuracy alone does not measure reasoning. This module surfaces successes,
failures, group variation, answer-frequency structure, and *observable*
diagnostic tags. Tags are descriptive evidence hooks — they do **not** assert
that a miss is a "reasoning failure."

Do not retune evaluation protocols after inspecting held-out results.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from magic_vlm.dataset import load_manifest
from magic_vlm.evaluation import exact_match, is_parse_failure, normalize_label
from magic_vlm.schemas import ExampleRecord
from magic_vlm.utils import read_jsonl, write_json, write_jsonl

# Observational tags only — wording avoids unsupported causal claims.
TAG_PARSE_FAILURE = "parse_failure"
TAG_INFERENCE_ERROR = "inference_error"
TAG_EMPTY_RAW = "empty_raw_response"
TAG_WRONG_LABEL = "wrong_label"
TAG_VISUAL_INPUT_MISSING = "visual_input_missing"
TAG_INCORRECT_WITH_PIXELS = "incorrect_with_pixels_present"
TAG_POSSIBLE_FREQUENCY_SHORTCUT = "possible_answer_frequency_shortcut"
TAG_AMBIGUITY_OR_TASK_NOTE = "ambiguity_or_task_design_note"
TAG_CORRECT = "correct"

AMBIGUITY_HINTS = (
    "ambig",
    "unclear",
    "uncertain",
    "multipl",
    "debatable",
    "task-design",
    "task design",
    "label disagree",
)


@dataclass(frozen=True)
class GroupStats:
    key: str
    n: int
    n_correct: int
    accuracy: float | None
    n_parse_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerDistributionStats:
    gold_counts: dict[str, int]
    predicted_counts: dict[str, int]
    majority_gold_label: str | None
    majority_gold_count: int
    majority_gold_fraction: float | None
    majority_class_baseline_accuracy: float | None
    predicted_mode_label: str | None
    predicted_mode_fraction: float | None
    gold_entropy_bits: float | None
    predicted_entropy_bits: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExampleDiagnosis:
    """Fully inspectable per-example record for baseline review."""

    example_id: str
    clip_id: str
    trick_id: str
    performer_id: str | None
    camera_id: str | None
    split: str
    question: str
    ground_truth: str | None
    raw_text: str
    parsed_answer: str | None
    parse_failed: bool
    correct: bool
    latency_s: float | None
    error: str | None
    tags: tuple[str, ...]
    notes: str | None = None
    question_variant: str | None = None
    video_input_mode: str | None = None
    prompt: str | None = None
    model_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        return payload


@dataclass(frozen=True)
class BaselineAnalysis:
    """Machine-readable aggregate diagnostics for one baseline run."""

    run_id: str | None
    split: str | None
    n_examples: int
    n_correct: int
    n_incorrect: int
    overall_accuracy: float | None
    n_parse_failures: int
    parse_failure_rate: float | None
    per_trick: tuple[GroupStats, ...]
    per_performer: tuple[GroupStats, ...]
    per_camera: tuple[GroupStats, ...]
    answer_distribution: AnswerDistributionStats
    tag_counts: dict[str, int]
    example_ids_correct: tuple[str, ...]
    example_ids_incorrect: tuple[str, ...]
    example_ids_parse_failed: tuple[str, ...]
    diagnoses: tuple[ExampleDiagnosis, ...]
    integrity: dict[str, Any]
    interpretation_caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "split": self.split,
            "n_examples": self.n_examples,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
            "overall_accuracy": self.overall_accuracy,
            "n_parse_failures": self.n_parse_failures,
            "parse_failure_rate": self.parse_failure_rate,
            "per_trick": [g.to_dict() for g in self.per_trick],
            "per_performer": [g.to_dict() for g in self.per_performer],
            "per_camera": [g.to_dict() for g in self.per_camera],
            "answer_distribution": self.answer_distribution.to_dict(),
            "tag_counts": dict(self.tag_counts),
            "example_ids_correct": list(self.example_ids_correct),
            "example_ids_incorrect": list(self.example_ids_incorrect),
            "example_ids_parse_failed": list(self.example_ids_parse_failed),
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "integrity": dict(self.integrity),
            "interpretation_caveats": list(self.interpretation_caveats),
        }


def _shannon_entropy_bits(counts: Mapping[str, int]) -> float | None:
    total = sum(counts.values())
    if total <= 0:
        return None
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _group_stats(
    rows: Sequence[tuple[str, bool, bool]],
) -> tuple[GroupStats, ...]:
    """rows: (group_key, correct, parse_failed)."""
    buckets: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for key, correct, parse_failed in rows:
        buckets[key].append((correct, parse_failed))
    out: list[GroupStats] = []
    for key in sorted(buckets):
        items = buckets[key]
        n = len(items)
        n_correct = sum(1 for c, _ in items if c)
        n_parse = sum(1 for _, pf in items if pf)
        out.append(
            GroupStats(
                key=key,
                n=n,
                n_correct=n_correct,
                accuracy=(n_correct / n) if n else None,
                n_parse_failures=n_parse,
            )
        )
    return tuple(out)


def _text_suggests_ambiguity(*parts: str | None) -> bool:
    blob = " ".join(p for p in parts if p).lower()
    return any(hint in blob for hint in AMBIGUITY_HINTS)


def assign_diagnostic_tags(
    *,
    correct: bool,
    parse_failed: bool,
    raw_text: str,
    parsed_answer: str | None,
    error: str | None,
    video_input_mode: str | None,
    majority_gold_label: str | None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Assign observational tags; never asserts 'reasoning failure'."""
    tags: list[str] = []
    if correct:
        tags.append(TAG_CORRECT)
        return tuple(tags)

    if error:
        tags.append(TAG_INFERENCE_ERROR)
    if not (raw_text or "").strip():
        tags.append(TAG_EMPTY_RAW)
    if parse_failed or is_parse_failure(raw_text, parsed_answer):
        tags.append(TAG_PARSE_FAILURE)
    elif parsed_answer is not None and str(parsed_answer).strip():
        tags.append(TAG_WRONG_LABEL)

    mode = (video_input_mode or "").lower()
    if "indices_only" in mode or mode in {"no_pixels", "meta_only"}:
        tags.append(TAG_VISUAL_INPUT_MISSING)
    elif mode and "pixel" in mode:
        tags.append(TAG_INCORRECT_WITH_PIXELS)

    if (
        majority_gold_label
        and parsed_answer is not None
        and normalize_label(parsed_answer) == normalize_label(majority_gold_label)
        and not correct
    ):
        tags.append(TAG_POSSIBLE_FREQUENCY_SHORTCUT)

    meta_blob = ""
    if metadata:
        meta_blob = json.dumps(dict(metadata), sort_keys=True)
    if _text_suggests_ambiguity(notes, meta_blob):
        tags.append(TAG_AMBIGUITY_OR_TASK_NOTE)

    return tuple(dict.fromkeys(tags))  # stable unique order


def _prediction_rows(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in predictions]


def analyze_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    examples: Sequence[ExampleRecord] | None = None,
    run_id: str | None = None,
    split: str | None = None,
) -> BaselineAnalysis:
    """Analyze baseline prediction rows (every row included; nothing suppressed)."""
    rows = _prediction_rows(predictions)
    if not rows:
        empty_dist = AnswerDistributionStats(
            gold_counts={},
            predicted_counts={},
            majority_gold_label=None,
            majority_gold_count=0,
            majority_gold_fraction=None,
            majority_class_baseline_accuracy=None,
            predicted_mode_label=None,
            predicted_mode_fraction=None,
            gold_entropy_bits=None,
            predicted_entropy_bits=None,
        )
        return BaselineAnalysis(
            run_id=run_id,
            split=split,
            n_examples=0,
            n_correct=0,
            n_incorrect=0,
            overall_accuracy=None,
            n_parse_failures=0,
            parse_failure_rate=None,
            per_trick=(),
            per_performer=(),
            per_camera=(),
            answer_distribution=empty_dist,
            tag_counts={},
            example_ids_correct=(),
            example_ids_incorrect=(),
            example_ids_parse_failed=(),
            diagnoses=(),
            integrity={"n_predictions": 0, "n_examples_joined": 0},
            interpretation_caveats=_default_caveats(),
        )

    by_example: dict[str, ExampleRecord] = {}
    if examples is not None:
        by_example = {ex.example_id: ex for ex in examples}
        pred_ids = [str(r["example_id"]) for r in rows]
        missing = [eid for eid in pred_ids if eid not in by_example]
        if missing:
            raise KeyError(
                "Prediction example_id(s) missing from joined manifest: "
                + ", ".join(missing[:5])
                + ("..." if len(missing) > 5 else "")
            )

    gold_labels = [
        normalize_label(r.get("ground_truth"))
        for r in rows
        if r.get("ground_truth") is not None and str(r.get("ground_truth")).strip()
    ]
    gold_counts_raw = Counter(gold_labels)
    majority_gold_label = None
    majority_gold_count = 0
    if gold_counts_raw:
        majority_gold_label, majority_gold_count = gold_counts_raw.most_common(1)[0]

    diagnoses: list[ExampleDiagnosis] = []
    for row in rows:
        eid = str(row["example_id"])
        ex = by_example.get(eid)
        performer = ex.performer_id if ex else row.get("performer_id")
        camera = ex.camera_id if ex else row.get("camera_id")
        notes = ex.notes if ex else row.get("notes")
        qvariant = ex.question_variant if ex else row.get("question_variant")
        metadata = dict(ex.metadata) if ex else dict(row.get("metadata") or {})
        preprocessing = row.get("preprocessing") or {}
        video_mode = None
        if isinstance(preprocessing, dict):
            video_mode = preprocessing.get("video_input_mode")
        raw_text = str(row.get("raw_text") or "")
        parsed = row.get("parsed_answer")
        parse_failed = bool(row.get("parse_failed"))
        if "parse_failed" not in row:
            parse_failed = is_parse_failure(raw_text, parsed)
        error = row.get("error")
        correct = bool(row.get("correct"))
        # Recompute correctness when joining gold from manifest for consistency checks.
        gold = row.get("ground_truth")
        if ex is not None and gold is None:
            gold = ex.ground_truth
        if "correct" not in row:
            correct = (not parse_failed) and (error is None) and exact_match(
                None if parsed is None else str(parsed),
                None if gold is None else str(gold),
            )
            if parse_failed or error:
                correct = False

        tags = assign_diagnostic_tags(
            correct=correct,
            parse_failed=parse_failed,
            raw_text=raw_text,
            parsed_answer=None if parsed is None else str(parsed),
            error=None if error is None else str(error),
            video_input_mode=None if video_mode is None else str(video_mode),
            majority_gold_label=majority_gold_label,
            notes=None if notes is None else str(notes),
            metadata=metadata,
        )
        diagnoses.append(
            ExampleDiagnosis(
                example_id=eid,
                clip_id=str(row.get("clip_id") or (ex.clip_id if ex else "")),
                trick_id=str(row.get("trick_id") or (ex.trick_id if ex else "")),
                performer_id=None if performer is None else str(performer),
                camera_id=None if camera is None else str(camera),
                split=str(row.get("split") or (ex.split.value if ex else "") or (split or "")),
                question=str(row.get("question") or (ex.question if ex else "")),
                ground_truth=None if gold is None else str(gold),
                raw_text=raw_text,
                parsed_answer=None if parsed is None else str(parsed),
                parse_failed=parse_failed,
                correct=correct,
                latency_s=row.get("latency_s"),
                error=None if error is None else str(error),
                tags=tags,
                notes=None if notes is None else str(notes),
                question_variant=None if qvariant is None else str(qvariant),
                video_input_mode=None if video_mode is None else str(video_mode),
                prompt=None if row.get("prompt") is None else str(row.get("prompt")),
                model_id=None if row.get("model_id") is None else str(row.get("model_id")),
                metadata=metadata,
            )
        )

    n = len(diagnoses)
    n_correct = sum(1 for d in diagnoses if d.correct)
    n_incorrect = n - n_correct
    n_parse = sum(1 for d in diagnoses if d.parse_failed)
    overall = (n_correct / n) if n else None
    parse_rate = (n_parse / n) if n else None

    per_trick = _group_stats([(d.trick_id, d.correct, d.parse_failed) for d in diagnoses])
    per_performer = _group_stats(
        [
            (d.performer_id or "unknown", d.correct, d.parse_failed)
            for d in diagnoses
            if d.performer_id is not None or examples is not None
        ]
    )
    # If no performer metadata anywhere, leave empty rather than all-"unknown".
    if all(d.performer_id is None for d in diagnoses):
        per_performer = ()
    per_camera = _group_stats(
        [
            (d.camera_id or "unknown", d.correct, d.parse_failed)
            for d in diagnoses
            if d.camera_id is not None
        ]
    )
    if all(d.camera_id is None for d in diagnoses):
        per_camera = ()

    # Display counts use normalized labels for aggregation; originals stay on diagnoses.
    gold_display = Counter(
        normalize_label(d.ground_truth) or "<missing>"
        for d in diagnoses
    )
    pred_display = Counter(
        normalize_label(d.parsed_answer) if d.parsed_answer is not None else "<unparsed>"
        for d in diagnoses
    )
    pred_mode_label = None
    pred_mode_frac = None
    if pred_display:
        pred_mode_label, pred_mode_n = pred_display.most_common(1)[0]
        pred_mode_frac = pred_mode_n / n

    majority_baseline = None
    majority_frac = None
    if majority_gold_label is not None and n:
        majority_frac = majority_gold_count / n
        majority_baseline = sum(
            1
            for d in diagnoses
            if normalize_label(d.ground_truth) == majority_gold_label
        ) / n

    answer_distribution = AnswerDistributionStats(
        gold_counts=dict(sorted(gold_display.items())),
        predicted_counts=dict(sorted(pred_display.items())),
        majority_gold_label=majority_gold_label,
        majority_gold_count=majority_gold_count,
        majority_gold_fraction=majority_frac,
        majority_class_baseline_accuracy=majority_baseline,
        predicted_mode_label=pred_mode_label,
        predicted_mode_fraction=pred_mode_frac,
        gold_entropy_bits=_shannon_entropy_bits(gold_display),
        predicted_entropy_bits=_shannon_entropy_bits(pred_display),
    )

    tag_counts: Counter[str] = Counter()
    for d in diagnoses:
        tag_counts.update(d.tags)

    # Integrity: group totals must match n; accuracy math must agree.
    trick_n = sum(g.n for g in per_trick)
    integrity = {
        "n_predictions": n,
        "n_examples_joined": len(by_example) if examples is not None else None,
        "sum_per_trick_n": trick_n,
        "per_trick_n_matches_total": trick_n == n,
        "n_correct_plus_incorrect": n_correct + n_incorrect == n,
        "overall_equals_n_correct_over_n": (
            overall == (n_correct / n) if n else overall is None
        ),
        "parse_failures_visible": n_parse == sum(1 for d in diagnoses if TAG_PARSE_FAILURE in d.tags or d.parse_failed),
        "all_incorrect_exported": n_incorrect
        == sum(1 for d in diagnoses if not d.correct),
        "raw_text_present_for_all": all(isinstance(d.raw_text, str) for d in diagnoses),
    }

    resolved_split = split or next((d.split for d in diagnoses if d.split), None)
    return BaselineAnalysis(
        run_id=run_id,
        split=resolved_split,
        n_examples=n,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        overall_accuracy=overall,
        n_parse_failures=n_parse,
        parse_failure_rate=parse_rate,
        per_trick=per_trick,
        per_performer=per_performer,
        per_camera=per_camera,
        answer_distribution=answer_distribution,
        tag_counts=dict(sorted(tag_counts.items())),
        example_ids_correct=tuple(d.example_id for d in diagnoses if d.correct),
        example_ids_incorrect=tuple(d.example_id for d in diagnoses if not d.correct),
        example_ids_parse_failed=tuple(d.example_id for d in diagnoses if d.parse_failed),
        diagnoses=tuple(diagnoses),
        integrity=integrity,
        interpretation_caveats=_default_caveats(),
    )


def _default_caveats() -> tuple[str, ...]:
    return (
        "Exact-match accuracy does not measure reasoning quality.",
        "Tags are observational; 'wrong_label' is not evidence of a reasoning failure.",
        "possible_answer_frequency_shortcut only notes that a wrong prediction matches the majority gold label.",
        "visual_input_missing means pixels were not provided to the model for that row.",
        "Do not retune prompts or protocol against held_out after inspecting these results.",
    )


def format_analysis_report(analysis: BaselineAnalysis) -> str:
    """Human-readable markdown report (non-overclaiming language)."""
    lines: list[str] = []
    lines.append("# Baseline evaluation analysis")
    lines.append("")
    if analysis.run_id:
        lines.append(f"- **run_id:** `{analysis.run_id}`")
    if analysis.split:
        lines.append(f"- **split:** `{analysis.split}`")
    lines.append(f"- **n_examples:** {analysis.n_examples}")
    lines.append(f"- **n_correct / n_incorrect:** {analysis.n_correct} / {analysis.n_incorrect}")
    acc = analysis.overall_accuracy
    lines.append(f"- **overall exact-match accuracy:** {acc if acc is not None else 'n/a'}")
    lines.append(
        f"- **parse failures:** {analysis.n_parse_failures} "
        f"(rate={analysis.parse_failure_rate if analysis.parse_failure_rate is not None else 'n/a'})"
    )
    lines.append("")
    lines.append("## Interpretation caveats")
    lines.append("")
    for caveat in analysis.interpretation_caveats:
        lines.append(f"- {caveat}")
    lines.append("")
    lines.append("## Per-trick accuracy")
    lines.append("")
    if not analysis.per_trick:
        lines.append("_No trick groups._")
    else:
        lines.append("| trick_id | n | correct | accuracy | parse_failures |")
        lines.append("|---|---:|---:|---:|---:|")
        for g in analysis.per_trick:
            lines.append(
                f"| `{g.key}` | {g.n} | {g.n_correct} | {g.accuracy} | {g.n_parse_failures} |"
            )
    lines.append("")
    if analysis.per_performer:
        lines.append("## Per-performer accuracy")
        lines.append("")
        lines.append("| performer_id | n | correct | accuracy | parse_failures |")
        lines.append("|---|---:|---:|---:|---:|")
        for g in analysis.per_performer:
            lines.append(
                f"| `{g.key}` | {g.n} | {g.n_correct} | {g.accuracy} | {g.n_parse_failures} |"
            )
        lines.append("")
    if analysis.per_camera:
        lines.append("## Per-camera accuracy")
        lines.append("")
        lines.append("| camera_id | n | correct | accuracy | parse_failures |")
        lines.append("|---|---:|---:|---:|---:|")
        for g in analysis.per_camera:
            lines.append(
                f"| `{g.key}` | {g.n} | {g.n_correct} | {g.accuracy} | {g.n_parse_failures} |"
            )
        lines.append("")

    dist = analysis.answer_distribution
    lines.append("## Answer distribution")
    lines.append("")
    lines.append(f"- gold counts: `{dist.gold_counts}`")
    lines.append(f"- predicted counts: `{dist.predicted_counts}`")
    lines.append(
        f"- majority gold label: `{dist.majority_gold_label}` "
        f"(fraction={dist.majority_gold_fraction})"
    )
    lines.append(
        f"- majority-class baseline accuracy (always predict majority gold): "
        f"{dist.majority_class_baseline_accuracy}"
    )
    lines.append(
        f"- predicted mode: `{dist.predicted_mode_label}` "
        f"(fraction={dist.predicted_mode_fraction})"
    )
    lines.append(f"- gold entropy (bits): {dist.gold_entropy_bits}")
    lines.append(f"- predicted entropy (bits): {dist.predicted_entropy_bits}")
    lines.append("")
    lines.append("## Diagnostic tag counts")
    lines.append("")
    if not analysis.tag_counts:
        lines.append("_None._")
    else:
        for tag, count in analysis.tag_counts.items():
            lines.append(f"- `{tag}`: {count}")
    lines.append("")
    lines.append("## Incorrect examples (all)")
    lines.append("")
    incorrect = [d for d in analysis.diagnoses if not d.correct]
    if not incorrect:
        lines.append("_None._")
    else:
        for d in incorrect:
            lines.append(f"### `{d.example_id}`")
            lines.append("")
            lines.append(f"- trick=`{d.trick_id}` performer=`{d.performer_id}` camera=`{d.camera_id}`")
            lines.append(f"- question: {d.question}")
            lines.append(f"- gold: `{d.ground_truth}`")
            lines.append(f"- parsed: `{d.parsed_answer}`")
            lines.append(f"- parse_failed: {d.parse_failed}")
            lines.append(f"- tags: {', '.join(d.tags) if d.tags else '(none)'}")
            lines.append(f"- video_input_mode: `{d.video_input_mode}`")
            if d.error:
                lines.append(f"- error: `{d.error}`")
            raw_preview = d.raw_text.replace("\n", "\\n")
            if len(raw_preview) > 400:
                raw_preview = raw_preview[:400] + "…"
            lines.append(f"- raw_text: `{raw_preview}`")
            lines.append("")
    lines.append("## Correct examples")
    lines.append("")
    correct = [d for d in analysis.diagnoses if d.correct]
    if not correct:
        lines.append("_None._")
    else:
        for d in correct:
            lines.append(
                f"- `{d.example_id}` trick=`{d.trick_id}` gold=`{d.ground_truth}` "
                f"parsed=`{d.parsed_answer}`"
            )
    lines.append("")
    lines.append("## Integrity checks")
    lines.append("")
    for key, value in analysis.integrity.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def write_analysis_outputs(
    analysis: BaselineAnalysis,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write metrics JSON, markdown report, and example-level exports."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": root / "analysis_metrics.json",
        "report": root / "analysis_report.md",
        "errors": root / "errors.jsonl",
        "successes": root / "successes.jsonl",
        "inspectable": root / "examples_inspectable.jsonl",
    }
    write_json(paths["metrics"], analysis.to_dict())
    paths["report"].write_text(format_analysis_report(analysis), encoding="utf-8")
    write_jsonl(paths["errors"], [d.to_dict() for d in analysis.diagnoses if not d.correct])
    write_jsonl(paths["successes"], [d.to_dict() for d in analysis.diagnoses if d.correct])
    write_jsonl(paths["inspectable"], [d.to_dict() for d in analysis.diagnoses])
    return paths


def _load_run_meta(run_dir: Path) -> tuple[str | None, str | None, Path | None]:
    run_id = run_dir.name
    split = None
    manifest: Path | None = None
    summary_path = run_dir / "baseline_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_id = summary.get("run_id") or run_id
        split = summary.get("split")
    config_path = run_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        ds = cfg.get("dataset") or {}
        if ds.get("manifest"):
            manifest = Path(ds["manifest"])
        if split is None:
            split = ds.get("split")
    return run_id, split, manifest


def analyze_baseline_run(
    run_dir: str | Path,
    *,
    manifest: str | Path | None = None,
    write: bool = True,
    out_dir: str | Path | None = None,
) -> BaselineAnalysis:
    """Load a baseline run directory and produce full diagnostics."""
    root = Path(run_dir)
    pred_path = root / "predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions.jsonl under {root}")
    predictions = read_jsonl(pred_path)
    run_id, split, manifest_from_cfg = _load_run_meta(root)
    manifest_path = Path(manifest) if manifest is not None else manifest_from_cfg
    examples = None
    if manifest_path is not None and Path(manifest_path).exists():
        examples = load_manifest(manifest_path)
        if split:
            from magic_vlm.schemas import Split
            from magic_vlm.dataset import filter_split

            try:
                examples = filter_split(examples, Split(split))
            except ValueError:
                pass

    analysis = analyze_predictions(
        predictions,
        examples=examples,
        run_id=run_id,
        split=split,
    )
    if write:
        write_analysis_outputs(analysis, out_dir or root)
    return analysis


def analyze_from_baseline_result(
    predictions: Sequence[Any],
    examples: Sequence[ExampleRecord],
    *,
    run_id: str | None = None,
    split: str | None = None,
) -> BaselineAnalysis:
    """Analyze in-memory baseline predictions (objects with ``to_dict``)."""
    rows = [p.to_dict() if hasattr(p, "to_dict") else dict(p) for p in predictions]
    return analyze_predictions(rows, examples=examples, run_id=run_id, split=split)
