"""Cross-method comparative evaluation under a shared held-out protocol.

Scientific posture
------------------
Compares implemented methods (zero-shot baseline, DPO, GRPO, temporal-shuffle
diagnostics, …) on the **same** held-out example set with the **same** task and
metric definitions. Does **not**:

* invent a single "reasoning score"
* hide missing predictions
* select checkpoints using final test performance
* equate accuracy / reward / temporal sensitivity / preference agreement with
  reasoning improvement

SFT is listed only if explicitly implemented (currently it is not).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from magic_vlm.analysis import GroupStats, _group_stats
from magic_vlm.dataset import filter_split, filter_task, load_manifest
from magic_vlm.evaluation import exact_match, is_parse_failure
from magic_vlm.schemas import ExampleRecord, Split, TaskType
from magic_vlm.utils import (
    allocate_run_directory,
    config_fingerprint,
    read_jsonl,
    utc_now_iso,
    write_json,
    write_jsonl,
)

MethodKind = Literal[
    "zero_shot",
    "baseline",
    "sft",
    "dpo",
    "grpo",
    "temporal_shuffle",
    "reward_model",
    "other",
]

INTEGRITY_DISCLAIMER = (
    "Comparative evaluation reports accuracy, generalization slices, temporal "
    "sensitivity, and reward deltas as separate dimensions. None of these is "
    "automatically labeled as reasoning improvement. Missing predictions are "
    "visible. Incompatible generation protocols are labeled, not silently merged."
)

IMPLEMENTED_METHODS: tuple[str, ...] = (
    "zero_shot",
    "baseline",
    "dpo",
    "grpo",
    "temporal_shuffle",
    "reward_model",
)
UNIMPLEMENTED_METHODS: tuple[str, ...] = ("sft", "ppo")


class ComparisonError(ValueError):
    """Invalid comparison configuration, alignment failure, or protocol mismatch."""


@dataclass(frozen=True)
class ProtocolSpec:
    """Locked scientific protocol for a comparison."""

    manifest: str
    split: str = Split.HELD_OUT.value
    task: str = TaskType.HIDDEN_STATE.value
    metric: str = "exact_match"
    require_full_coverage: bool = True
    allow_incompatible_protocols: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MethodSpec:
    """One method arm in the comparison."""

    method_id: str
    kind: str
    run_dir: str | None = None
    predictions_path: str | None = None
    temporal_pairs_path: str | None = None
    temporal_summary_path: str | None = None
    reward_stats_path: str | None = None
    generation_policy: dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonConfig:
    """YAML-serializable comparative evaluation request."""

    protocol: ProtocolSpec
    methods: tuple[MethodSpec, ...]
    output_dir: str = "runs/comparison"
    run_id: str | None = None
    reference_method_id: str | None = None
    include_unseen_slices: bool = True
    include_temporal: bool = True
    include_reward: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.to_dict(),
            "methods": [m.to_dict() for m in self.methods],
            "output_dir": self.output_dir,
            "run_id": self.run_id,
            "reference_method_id": self.reference_method_id,
            "include_unseen_slices": self.include_unseen_slices,
            "include_temporal": self.include_temporal,
            "include_reward": self.include_reward,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ComparisonConfig:
        raw = dict(data)
        proto_raw = dict(raw.get("protocol") or {})
        if "manifest" in raw and "manifest" not in proto_raw:
            proto_raw["manifest"] = raw["manifest"]
        known_p = ProtocolSpec.__dataclass_fields__  # type: ignore[attr-defined]
        protocol = ProtocolSpec(**{k: v for k, v in proto_raw.items() if k in known_p})
        methods_raw = list(raw.get("methods") or [])
        if not methods_raw:
            raise ComparisonError("comparison config requires at least one method")
        methods: list[MethodSpec] = []
        known_m = MethodSpec.__dataclass_fields__  # type: ignore[attr-defined]
        for item in methods_raw:
            if not isinstance(item, dict):
                raise ComparisonError("each method entry must be a mapping")
            payload = {k: v for k, v in item.items() if k in known_m}
            if "generation_policy" in payload and payload["generation_policy"] is None:
                payload["generation_policy"] = {}
            methods.append(MethodSpec(**payload))
        return cls(
            protocol=protocol,
            methods=tuple(methods),
            output_dir=str(raw.get("output_dir") or "runs/comparison"),
            run_id=None if raw.get("run_id") in (None, "") else str(raw["run_id"]),
            reference_method_id=(
                None
                if raw.get("reference_method_id") in (None, "")
                else str(raw["reference_method_id"])
            ),
            include_unseen_slices=bool(raw.get("include_unseen_slices", True)),
            include_temporal=bool(raw.get("include_temporal", True)),
            include_reward=bool(raw.get("include_reward", True)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ComparisonConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ComparisonError(f"Comparison config must be a mapping: {path}")
        return cls.from_dict(raw)


@dataclass(frozen=True)
class NormalizedPrediction:
    """Example-level prediction normalized across run formats."""

    example_id: str
    correct: bool | None
    parse_failed: bool
    raw_text: str
    parsed_answer: str | None
    ground_truth: str | None
    reward: float | None = None
    missing: bool = False
    source_path: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_predictions_path(method: MethodSpec) -> Path:
    """Locate prediction JSONL for a method arm."""
    if method.predictions_path:
        path = Path(method.predictions_path)
        if not path.exists():
            raise ComparisonError(
                f"predictions_path not found for {method.method_id}: {path}"
            )
        return path
    if not method.run_dir:
        raise ComparisonError(
            f"method {method.method_id!r} needs run_dir or predictions_path"
        )
    run = Path(method.run_dir)
    candidates = [
        run / "predictions.jsonl",
        run / "held_out_eval_rows.jsonl",
        run / "examples_inspectable.jsonl",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise ComparisonError(
        f"No predictions JSONL found under {run} for method {method.method_id!r}. "
        f"Tried: {[str(c.name) for c in candidates]}"
    )


def _row_to_normalized(row: Mapping[str, Any], *, source: str) -> NormalizedPrediction:
    eid = str(row["example_id"])
    raw_text = str(row.get("raw_text") or "")
    parsed = row.get("parsed_answer")
    if parsed is not None:
        parsed = str(parsed)
    gold = row.get("ground_truth")
    if gold is not None:
        gold = str(gold)
    if "parse_failed" in row:
        parse_failed = bool(row["parse_failed"])
    else:
        parse_failed = is_parse_failure(raw_text, parsed)
    correct: bool | None
    if "correct" in row:
        correct = bool(row["correct"])
    elif "matched" in row:
        correct = bool(row["matched"])
    elif gold is not None:
        correct = (not parse_failed) and exact_match(parsed, gold)
    else:
        correct = None
    reward = row.get("reward")
    if reward is not None:
        reward = float(reward)
    elif correct is not None:
        reward = 1.0 if correct else 0.0
    return NormalizedPrediction(
        example_id=eid,
        correct=correct,
        parse_failed=parse_failed,
        raw_text=raw_text,
        parsed_answer=parsed,
        ground_truth=gold,
        reward=reward,
        missing=False,
        source_path=source,
        extras={
            k: row[k]
            for k in ("model_id", "error", "latency_s", "split")
            if k in row
        },
    )


def load_method_predictions(method: MethodSpec) -> dict[str, NormalizedPrediction]:
    path = resolve_predictions_path(method)
    rows = read_jsonl(path)
    out: dict[str, NormalizedPrediction] = {}
    for row in rows:
        if "example_id" not in row:
            raise ComparisonError(f"Row missing example_id in {path}")
        pred = _row_to_normalized(row, source=str(path))
        if pred.example_id in out:
            raise ComparisonError(
                f"Duplicate example_id {pred.example_id!r} in {path} "
                f"for {method.method_id}"
            )
        out[pred.example_id] = pred
    return out


def load_temporal_summary(method: MethodSpec) -> dict[str, Any] | None:
    if method.temporal_summary_path:
        path = Path(method.temporal_summary_path)
    elif method.run_dir:
        path = Path(method.run_dir) / "temporal_shuffle_summary.json"
    else:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_temporal_pairs(method: MethodSpec) -> list[dict[str, Any]] | None:
    if method.temporal_pairs_path:
        path = Path(method.temporal_pairs_path)
    elif method.run_dir:
        path = Path(method.run_dir) / "temporal_shuffle_pairs.jsonl"
    else:
        return None
    if not path.exists():
        return None
    return list(read_jsonl(path))


def load_reward_stats(method: MethodSpec) -> dict[str, Any] | None:
    if method.reward_stats_path:
        path = Path(method.reward_stats_path)
    elif method.run_dir:
        path = Path(method.run_dir) / "reward_stats.json"
    else:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def locked_held_out_examples(
    examples: Sequence[ExampleRecord],
    *,
    split: str,
    task: str,
) -> list[ExampleRecord]:
    try:
        split_enum = Split(split)
    except ValueError as exc:
        raise ComparisonError(f"Invalid split {split!r}") from exc
    try:
        task_enum = TaskType(task)
    except ValueError as exc:
        raise ComparisonError(f"Invalid task {task!r}") from exc
    locked = filter_task(filter_split(list(examples), split_enum), task_enum)
    if not locked:
        raise ComparisonError(f"No examples for split={split!r} task={task!r}")
    return list(locked)


def compute_seen_sets(examples: Sequence[ExampleRecord]) -> dict[str, set[str]]:
    """Identity values observed on non-held_out splits (train ∪ val)."""
    seen = {
        "trick_id": set(),
        "performer_id": set(),
        "camera_id": set(),
        "prop_id": set(),
        "question_variant": set(),
        "clip_id": set(),
    }
    for ex in examples:
        if ex.split is Split.HELD_OUT:
            continue
        seen["trick_id"].add(ex.trick_id)
        seen["performer_id"].add(ex.performer_id)
        seen["camera_id"].add(ex.camera_id)
        if ex.prop_id:
            seen["prop_id"].add(ex.prop_id)
        if ex.question_variant:
            seen["question_variant"].add(ex.question_variant)
        seen["clip_id"].add(ex.clip_id)
    return seen


def example_axis_flags(
    example: ExampleRecord,
    seen: Mapping[str, set[str]],
) -> dict[str, bool]:
    """Boolean generalization axes for one held-out example."""
    prop_unseen = False
    if example.prop_id:
        prop_unseen = example.prop_id not in seen["prop_id"]
    wording_unseen = False
    if example.question_variant:
        wording_unseen = example.question_variant not in seen["question_variant"]
    known_trick = example.trick_id in seen["trick_id"]
    return {
        "unseen_trick": example.trick_id not in seen["trick_id"],
        "unseen_performer": example.performer_id not in seen["performer_id"],
        "unseen_camera": example.camera_id not in seen["camera_id"],
        "unseen_prop": prop_unseen,
        "unseen_wording": wording_unseen,
        "known_trick_variation": known_trick and wording_unseen,
        "has_prop_id": example.prop_id is not None,
        "has_question_variant": example.question_variant is not None,
    }


def generation_policy_fingerprint(policy: Mapping[str, Any] | None) -> str:
    return config_fingerprint(dict(policy or {}))


def check_protocol_compatibility(
    methods: Sequence[MethodSpec],
    *,
    allow_incompatible: bool,
) -> dict[str, Any]:
    fps = {
        m.method_id: generation_policy_fingerprint(m.generation_policy) for m in methods
    }
    unique = sorted(set(fps.values()))
    compatible = len(unique) <= 1
    report = {
        "compatible": compatible,
        "fingerprints": fps,
        "n_distinct_policies": len(unique),
        "note": (
            "All methods share the same generation_policy fingerprint."
            if compatible
            else (
                "Methods use different generation policies; "
                "comparisons are labeled incompatible."
            )
        ),
    }
    if not compatible and not allow_incompatible:
        raise ComparisonError(
            "Incompatible generation protocols across methods. "
            "Set protocol.allow_incompatible_protocols: true to proceed with labels, "
            f"or align generation_policy. fingerprints={fps}"
        )
    return report


def _accuracy(correct_flags: Sequence[bool | None]) -> float | None:
    usable = [c for c in correct_flags if c is not None]
    if not usable:
        return None
    return sum(1 for c in usable if c) / len(usable)


def _mean(values: Sequence[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def align_methods(
    locked: Sequence[ExampleRecord],
    method_preds: Mapping[str, Mapping[str, NormalizedPrediction]],
    *,
    require_full_coverage: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build example-aligned rows; never silently drop missing predictions."""
    locked_ids = [ex.example_id for ex in locked]
    coverage: dict[str, Any] = {"locked_n": len(locked_ids), "methods": {}}
    aligned: list[dict[str, Any]] = []

    for ex in locked:
        row: dict[str, Any] = {
            "example_id": ex.example_id,
            "clip_id": ex.clip_id,
            "trick_id": ex.trick_id,
            "performer_id": ex.performer_id,
            "camera_id": ex.camera_id,
            "prop_id": ex.prop_id,
            "question_variant": ex.question_variant,
            "split": ex.split.value,
            "task": ex.task.value,
            "question": ex.question,
            "ground_truth": ex.ground_truth,
            "methods": {},
        }
        for mid, preds in method_preds.items():
            pred = preds.get(ex.example_id)
            if pred is None:
                row["methods"][mid] = {
                    "missing": True,
                    "correct": None,
                    "parse_failed": None,
                    "raw_text": None,
                    "parsed_answer": None,
                    "reward": None,
                }
            else:
                gold = ex.ground_truth if ex.ground_truth is not None else pred.ground_truth
                correct = pred.correct
                if correct is None and gold is not None:
                    correct = (not pred.parse_failed) and exact_match(
                        pred.parsed_answer, gold
                    )
                row["methods"][mid] = {
                    "missing": False,
                    "correct": correct,
                    "parse_failed": pred.parse_failed,
                    "raw_text": pred.raw_text,
                    "parsed_answer": pred.parsed_answer,
                    "reward": pred.reward,
                    "source_path": pred.source_path,
                }
        aligned.append(row)

    for mid, preds in method_preds.items():
        present = sum(1 for eid in locked_ids if eid in preds)
        missing_ids = [eid for eid in locked_ids if eid not in preds]
        extra_ids = [eid for eid in preds if eid not in set(locked_ids)]
        coverage["methods"][mid] = {
            "n_present": present,
            "n_missing": len(missing_ids),
            "missing_example_ids": missing_ids,
            "n_extra_outside_lock": len(extra_ids),
            "extra_example_ids": extra_ids[:20],
            "full_coverage": len(missing_ids) == 0,
        }
        if require_full_coverage and missing_ids:
            raise ComparisonError(
                f"Method {mid!r} missing {len(missing_ids)} locked held-out "
                f"example(s): {missing_ids[:5]}"
                f"{'...' if len(missing_ids) > 5 else ''}. "
                "Set protocol.require_full_coverage: false to allow incomplete arms "
                "(missing rows remain visible)."
            )
    return aligned, coverage


def aggregate_method_metrics(
    aligned: Sequence[Mapping[str, Any]],
    method_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for mid in method_ids:
        corrects: list[bool | None] = []
        rewards: list[float | None] = []
        n_missing = 0
        n_parse = 0
        for row in aligned:
            cell = row["methods"][mid]
            if cell.get("missing"):
                n_missing += 1
                corrects.append(None)
                rewards.append(None)
                continue
            corrects.append(cell.get("correct"))
            rewards.append(cell.get("reward"))
            if cell.get("parse_failed"):
                n_parse += 1
        scored = [c for c in corrects if c is not None]
        out[mid] = {
            "n_locked": len(aligned),
            "n_scored": len(scored),
            "n_missing": n_missing,
            "n_correct": sum(1 for c in scored if c),
            "accuracy": _accuracy(corrects),
            "n_parse_failures": n_parse,
            "parse_failure_rate": (n_parse / len(scored)) if scored else None,
            "mean_reward": _mean(rewards),
            "dimension": {
                "task_performance": "exact_match accuracy on locked held-out set",
                "reward": "mean reward when present (separate from reasoning claims)",
            },
        }
    return out


def _slice_rows(
    aligned: Sequence[Mapping[str, Any]],
    method_id: str,
    predicate: Callable[[Mapping[str, Any]], bool],
) -> list[tuple[str, bool, bool]]:
    rows: list[tuple[str, bool, bool]] = []
    for row in aligned:
        if not predicate(row):
            continue
        cell = row["methods"][method_id]
        if cell.get("missing") or cell.get("correct") is None:
            continue
        rows.append(("__slice__", bool(cell["correct"]), bool(cell.get("parse_failed"))))
    return rows


def slice_accuracy(
    aligned: Sequence[Mapping[str, Any]],
    method_ids: Sequence[str],
    *,
    name: str,
    predicate: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for mid in method_ids:
        rows = _slice_rows(aligned, mid, predicate)
        n = len(rows)
        n_correct = sum(1 for _, c, _ in rows if c)
        methods[mid] = {
            "n": n,
            "n_correct": n_correct,
            "accuracy": (n_correct / n) if n else None,
        }
    return {"slice": name, "methods": methods}


def per_identity_metrics(
    aligned: Sequence[Mapping[str, Any]],
    method_id: str,
    field_name: str,
) -> tuple[GroupStats, ...]:
    rows: list[tuple[str, bool, bool]] = []
    for row in aligned:
        cell = row["methods"][method_id]
        if cell.get("missing") or cell.get("correct") is None:
            continue
        key = row.get(field_name)
        if key is None or key == "":
            key = "unknown"
        rows.append((str(key), bool(cell["correct"]), bool(cell.get("parse_failed"))))
    return _group_stats(rows)


def compute_deltas(
    aggregates: Mapping[str, Mapping[str, Any]],
    *,
    reference_method_id: str | None,
) -> dict[str, Any]:
    if not reference_method_id:
        return {"reference_method_id": None, "deltas": {}}
    if reference_method_id not in aggregates:
        raise ComparisonError(
            f"reference_method_id {reference_method_id!r} not in methods"
        )
    ref = aggregates[reference_method_id]
    ref_acc = ref.get("accuracy")
    ref_reward = ref.get("mean_reward")
    deltas: dict[str, Any] = {}
    for mid, stats in aggregates.items():
        if mid == reference_method_id:
            continue
        acc = stats.get("accuracy")
        rew = stats.get("mean_reward")
        deltas[mid] = {
            "accuracy_delta_vs_reference": (
                None if ref_acc is None or acc is None else acc - ref_acc
            ),
            "mean_reward_delta_vs_reference": (
                None if ref_reward is None or rew is None else rew - ref_reward
            ),
            "note": (
                "accuracy_delta is independent task-performance change; "
                "mean_reward_delta is reward change. Neither is reasoning improvement."
            ),
        }
    return {"reference_method_id": reference_method_id, "deltas": deltas}


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def render_comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Comparative evaluation",
        "",
        str(report.get("integrity_disclaimer", INTEGRITY_DISCLAIMER)),
        "",
        f"- Protocol split: `{report['protocol']['split']}`",
        f"- Task: `{report['protocol']['task']}`",
        f"- Metric: `{report['protocol']['metric']}`",
        f"- Locked examples: **{report['coverage']['locked_n']}**",
        (
            "- Generation protocols compatible: "
            f"**{report['protocol_compatibility']['compatible']}**"
        ),
        "",
        "## Aggregate task performance (exact-match)",
        "",
        "| method | n_scored | n_missing | accuracy | mean_reward |",
        "|---|---:|---:|---:|---:|",
    ]
    for mid, stats in report["aggregates"].items():
        lines.append(
            f"| `{mid}` | {stats['n_scored']} | {stats['n_missing']} | "
            f"{_fmt(stats['accuracy'])} | {_fmt(stats['mean_reward'])} |"
        )
    deltas = report.get("deltas") or {}
    if deltas.get("deltas"):
        lines.extend(
            [
                "",
                f"## Deltas vs reference `{deltas['reference_method_id']}`",
                "",
                "| method | accuracy Δ | reward Δ |",
                "|---|---:|---:|",
            ]
        )
        for mid, d in deltas["deltas"].items():
            lines.append(
                f"| `{mid}` | {_fmt(d['accuracy_delta_vs_reference'])} | "
                f"{_fmt(d['mean_reward_delta_vs_reference'])} |"
            )
    slices = report.get("generalization_slices") or []
    if slices:
        lines.extend(["", "## Generalization slices", ""])
        for sl in slices:
            lines.append(f"### {sl['slice']}")
            lines.append("")
            lines.append("| method | n | accuracy |")
            lines.append("|---|---:|---:|")
            for mid, st in sl["methods"].items():
                lines.append(f"| `{mid}` | {st['n']} | {_fmt(st['accuracy'])} |")
            lines.append("")
    temporal = report.get("temporal") or {}
    if temporal:
        lines.extend(["", "## Temporal-order diagnostic (separate dimension)", ""])
        for mid, t in temporal.items():
            if t is None:
                lines.append(f"- `{mid}`: no temporal summary attached")
                continue
            lines.append(
                f"- `{mid}`: ordered={_fmt(t.get('ordered_accuracy'))} "
                f"shuffled={_fmt(t.get('shuffled_accuracy'))} "
                f"diff={_fmt(t.get('accuracy_difference'))}"
            )
        lines.append("")
        lines.append(
            "_Temporal sensitivity is not causal-reasoning proof and is not "
            "collapsed into task accuracy._"
        )
    lines.extend(
        [
            "",
            "## Dimension legend",
            "",
            "- **accuracy** — independent task performance (exact match)",
            "- **generalization slices** — unseen trick/performer/camera/prop/wording",
            "- **temporal** — ordered vs shuffled frame-order sensitivity",
            "- **reward** — training/objective reward statistics when present",
            "- **human quality** — not computed here (preference RM / annotators)",
            "",
            "Do **not** collapse these into one unsupported reasoning score.",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class ComparisonResult:
    run_dir: str
    report: dict[str, Any]
    aligned_path: str
    metrics_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "aligned_path": self.aligned_path,
            "metrics_path": self.metrics_path,
            "markdown_path": self.markdown_path,
            "disclaimer": INTEGRITY_DISCLAIMER,
        }


def run_comparison(config: ComparisonConfig) -> ComparisonResult:
    """Execute locked held-out comparative evaluation and write artifacts."""
    if not config.methods:
        raise ComparisonError("No methods to compare")
    method_ids = [m.method_id for m in config.methods]
    if len(set(method_ids)) != len(method_ids):
        raise ComparisonError("method_id values must be unique")
    for m in config.methods:
        if m.kind in UNIMPLEMENTED_METHODS:
            raise ComparisonError(
                f"Method kind {m.kind!r} is not implemented in this repository; "
                f"implemented kinds include {list(IMPLEMENTED_METHODS)}"
            )

    all_examples = load_manifest(config.protocol.manifest)
    locked = locked_held_out_examples(
        all_examples,
        split=config.protocol.split,
        task=config.protocol.task,
    )
    seen = compute_seen_sets(all_examples)
    proto_compat = check_protocol_compatibility(
        config.methods,
        allow_incompatible=config.protocol.allow_incompatible_protocols,
    )

    method_preds: dict[str, dict[str, NormalizedPrediction]] = {}
    for m in config.methods:
        if m.kind == "temporal_shuffle" and not m.predictions_path and m.run_dir:
            run = Path(m.run_dir)
            has_preds = (run / "predictions.jsonl").exists() or (
                run / "held_out_eval_rows.jsonl"
            ).exists()
            if not has_preds:
                method_preds[m.method_id] = {}
                continue
        method_preds[m.method_id] = load_method_predictions(m)

    aligned, coverage = align_methods(
        locked,
        method_preds,
        require_full_coverage=config.protocol.require_full_coverage,
    )

    by_id = {ex.example_id: ex for ex in locked}
    for row in aligned:
        ex = by_id[row["example_id"]]
        row["axes"] = example_axis_flags(ex, seen)

    aggregates = aggregate_method_metrics(aligned, method_ids)
    deltas = compute_deltas(aggregates, reference_method_id=config.reference_method_id)

    per_group: dict[str, Any] = {}
    for mid in method_ids:
        per_group[mid] = {
            "per_trick": [g.to_dict() for g in per_identity_metrics(aligned, mid, "trick_id")],
            "per_performer": [
                g.to_dict() for g in per_identity_metrics(aligned, mid, "performer_id")
            ],
            "per_camera": [
                g.to_dict() for g in per_identity_metrics(aligned, mid, "camera_id")
            ],
            "per_prop": [g.to_dict() for g in per_identity_metrics(aligned, mid, "prop_id")],
            "per_question_variant": [
                g.to_dict()
                for g in per_identity_metrics(aligned, mid, "question_variant")
            ],
        }

    generalization_slices: list[dict[str, Any]] = []
    if config.include_unseen_slices:
        slice_defs: list[tuple[str, Callable[[Mapping[str, Any]], bool]]] = [
            ("unseen_trick", lambda r: r["axes"]["unseen_trick"]),
            ("unseen_performer", lambda r: r["axes"]["unseen_performer"]),
            ("unseen_camera", lambda r: r["axes"]["unseen_camera"]),
            (
                "unseen_prop",
                lambda r: r["axes"]["unseen_prop"] and r["axes"]["has_prop_id"],
            ),
            (
                "unseen_wording",
                lambda r: r["axes"]["unseen_wording"]
                and r["axes"]["has_question_variant"],
            ),
            ("known_trick_variation", lambda r: r["axes"]["known_trick_variation"]),
            ("all_locked", lambda r: True),
        ]
        for name, pred in slice_defs:
            generalization_slices.append(
                slice_accuracy(aligned, method_ids, name=name, predicate=pred)
            )

    temporal_block: dict[str, Any] = {}
    if config.include_temporal:
        for m in config.methods:
            summary = load_temporal_summary(m)
            pairs = load_temporal_pairs(m) if summary is None else None
            if summary is None and pairs is not None:
                n = len(pairs)
                n_ord = sum(1 for p in pairs if p.get("ordered_correct"))
                n_shf = sum(1 for p in pairs if p.get("shuffled_correct"))
                summary = {
                    "n_pairs": n,
                    "ordered_accuracy": (n_ord / n) if n else None,
                    "shuffled_accuracy": (n_shf / n) if n else None,
                    "accuracy_difference": (((n_ord / n) - (n_shf / n)) if n else None),
                    "note": "Recomputed from temporal_shuffle_pairs.jsonl",
                }
            temporal_block[m.method_id] = summary

    reward_block: dict[str, Any] = {}
    if config.include_reward:
        for m in config.methods:
            reward_block[m.method_id] = {
                "held_out_mean_reward": aggregates[m.method_id].get("mean_reward"),
                "training_reward_stats": load_reward_stats(m),
                "note": (
                    "Training reward_stats are not held-out task performance. "
                    "Do not treat reward gains as reasoning improvement."
                ),
            }

    run_id = config.run_id or f"compare_{utc_now_iso().replace(':', '').replace('+', '_')}"
    run_dir = allocate_run_directory(config.output_dir, run_id, overwrite=False)

    report = {
        "created_at": utc_now_iso(),
        "integrity_disclaimer": INTEGRITY_DISCLAIMER,
        "protocol": config.protocol.to_dict(),
        "protocol_compatibility": proto_compat,
        "methods": [m.to_dict() for m in config.methods],
        "reference_method_id": config.reference_method_id,
        "coverage": coverage,
        "aggregates": aggregates,
        "deltas": deltas,
        "per_group": per_group,
        "generalization_slices": generalization_slices,
        "temporal": temporal_block,
        "reward": reward_block,
        "seen_identity_sizes": {k: len(v) for k, v in seen.items()},
        "dimension_legend": {
            "accuracy_improvement": "Change in exact-match on locked held-out set",
            "generalization_improvement": "Slice accuracies on unseen identities/wording",
            "temporal_sensitivity": "Ordered vs shuffled diagnostic (separate)",
            "reward_improvement": "Objective/training reward change (separate)",
            "human_quality_improvement": "Not computed in this pipeline",
            "reasoning_improvement": "NOT inferred automatically from any of the above",
        },
        "unimplemented_methods_excluded": list(UNIMPLEMENTED_METHODS),
    }

    write_json(run_dir / "comparison_config.json", config.to_dict())
    write_json(run_dir / "DISCLAIMER.json", {"disclaimer": INTEGRITY_DISCLAIMER})
    write_jsonl(run_dir / "aligned_examples.jsonl", aligned)
    write_json(run_dir / "comparison_metrics.json", report)
    md_path = run_dir / "comparison_report.md"
    md_path.write_text(render_comparison_markdown(report), encoding="utf-8")
    write_json(
        run_dir / "result.json",
        {
            "run_dir": str(run_dir),
            "n_methods": len(method_ids),
            "n_locked": len(locked),
            "protocol_compatible": proto_compat["compatible"],
            "disclaimer": INTEGRITY_DISCLAIMER,
        },
    )
    return ComparisonResult(
        run_dir=str(run_dir),
        report=report,
        aligned_path=str(run_dir / "aligned_examples.jsonl"),
        metrics_path=str(run_dir / "comparison_metrics.json"),
        markdown_path=str(md_path),
    )
