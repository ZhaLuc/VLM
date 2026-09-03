"""Research-quality experiment report generation.

Aggregates stored run artifacts into deterministic markdown + JSON summaries
suitable for lab meetings and later paper prep. Read-only: does not re-run
models or invent missing numbers.

Integrity
---------
* Missing metadata is marked ``unavailable`` (never fabricated).
* Negative and inconclusive findings stay visible.
* Representative examples follow a documented selection rule (not cherry-picking).
* Observations stay precise (e.g. "accuracy increased"); never auto-translated
  to "reasoning improved."
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from magic_vlm.utils import (
    allocate_run_directory,
    config_fingerprint,
    read_jsonl,
    stable_json,
    write_json,
)

INTEGRITY_DISCLAIMER = (
    "This report aggregates stored experiment artifacts only. "
    "Observations such as accuracy or reward changes are not automatically "
    "labeled as reasoning improvement. Missing fields are marked unavailable. "
    "Failed and inconclusive results remain visible. Representative examples "
    "follow a documented selection rule and are not cherry-picked."
)

EXAMPLE_SELECTION_RULE = (
    "Pool examples from examples_inspectable.jsonl, else aligned_examples.jsonl, "
    "else predictions.jsonl / held_out_eval_rows.jsonl. "
    "Success = correct is True; failure = correct is False (parse failures stay "
    "in failures). Sort by (trick_id, example_id). Round-robin by trick_id until "
    "max_successes / max_failures. Never pad. Truncate raw_text for display; "
    "full rows remain in linked JSONL."
)

FORBIDDEN_CLAIM_PATTERNS = (
    re.compile(r"reasoning\s+improved", re.I),
    re.compile(r"improved\s+reasoning", re.I),
    re.compile(r"better\s+reasoning", re.I),
    re.compile(r"this\s+proves\s+reward\s+hacking", re.I),
    re.compile(r"confirmed\s+reward\s+hacking", re.I),
    re.compile(r"causal\s+understanding\s+improved", re.I),
)

UNAVAILABLE = "unavailable"


class ReportingError(ValueError):
    """Invalid report configuration or artifact load failure."""


@dataclass(frozen=True)
class ExampleSelectionConfig:
    max_successes: int = 5
    max_failures: int = 5
    sort_keys: tuple[str, ...] = ("trick_id", "example_id")
    diversity_key: str = "trick_id"
    raw_text_max_chars: int = 400

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_successes": self.max_successes,
            "max_failures": self.max_failures,
            "sort_keys": list(self.sort_keys),
            "diversity_key": self.diversity_key,
            "raw_text_max_chars": self.raw_text_max_chars,
            "rule": EXAMPLE_SELECTION_RULE,
        }


@dataclass(frozen=True)
class ReportConfig:
    """YAML-serializable research report request."""

    output_dir: str = "runs/reports"
    run_id: str | None = None
    primary_run_dir: str | None = None
    artifacts: dict[str, str | None] = field(default_factory=dict)
    protocol: dict[str, Any] = field(default_factory=dict)
    example_selection: ExampleSelectionConfig = field(
        default_factory=ExampleSelectionConfig
    )
    generated_at: str | None = None  # freeze for deterministic tests
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "run_id": self.run_id,
            "primary_run_dir": self.primary_run_dir,
            "artifacts": dict(self.artifacts),
            "protocol": dict(self.protocol),
            "example_selection": self.example_selection.to_dict(),
            "generated_at": self.generated_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReportConfig:
        raw = dict(data)
        sel_raw = dict(raw.get("example_selection") or {})
        sel = ExampleSelectionConfig(
            max_successes=int(sel_raw.get("max_successes", 5)),
            max_failures=int(sel_raw.get("max_failures", 5)),
            sort_keys=tuple(sel_raw.get("sort_keys") or ("trick_id", "example_id")),
            diversity_key=str(sel_raw.get("diversity_key") or "trick_id"),
            raw_text_max_chars=int(sel_raw.get("raw_text_max_chars", 400)),
        )
        arts = dict(raw.get("artifacts") or {})
        # Normalize empty strings to None
        arts = {k: (None if v in ("", None) else str(v)) for k, v in arts.items()}
        return cls(
            output_dir=str(raw.get("output_dir") or "runs/reports"),
            run_id=None if raw.get("run_id") in (None, "") else str(raw["run_id"]),
            primary_run_dir=(
                None
                if raw.get("primary_run_dir") in (None, "")
                else str(raw["primary_run_dir"])
            ),
            artifacts=arts,
            protocol=dict(raw.get("protocol") or {}),
            example_selection=sel,
            generated_at=(
                None if raw.get("generated_at") in (None, "") else str(raw["generated_at"])
            ),
            notes=str(raw.get("notes") or ""),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ReportConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ReportingError(f"Report config must be a mapping: {path}")
        return cls.from_dict(raw)

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | Path,
        *,
        output_dir: str = "runs/reports",
        run_id: str | None = None,
        generated_at: str | None = None,
    ) -> ReportConfig:
        path = Path(run_dir)
        if not path.exists():
            raise ReportingError(f"run_dir not found: {path}")
        kind = detect_run_kind(path)
        artifacts: dict[str, str | None] = {
            "baseline": None,
            "temporal": None,
            "dpo": None,
            "grpo": None,
            "reward_model": None,
            "comparison": None,
            "reward_hacking": None,
        }
        if kind in artifacts:
            artifacts[kind] = str(path)
        elif kind == "unknown":
            artifacts["baseline"] = str(path)  # best-effort primary
        return cls(
            output_dir=output_dir,
            run_id=run_id or f"report_{path.name}",
            primary_run_dir=str(path),
            artifacts=artifacts,
            generated_at=generated_at,
        )


def detect_run_kind(run_dir: str | Path) -> str:
    """Fingerprint a run directory. Returns unknown if unclear."""
    p = Path(run_dir)
    if (p / "reward_hacking_metrics.json").exists():
        return "reward_hacking"
    if (p / "comparison_metrics.json").exists():
        return "comparison"
    if (
        (p / "grpo_train_records.jsonl").exists()
        or (p / "held_out_eval.json").exists()
        or (p / "held_out_eval_rows.jsonl").exists()
        or (
            (p / "reward_stats.json").exists()
            and (p / "train_metadata.json").exists()
        )
    ):
        return "grpo"
    if (p / "dpo_train_records.jsonl").exists():
        return "dpo"
    if (p / "checkpoint_best.pt").exists() and (p / "train_metadata.json").exists():
        return "reward_model"
    if (p / "temporal_shuffle_summary.json").exists():
        return "temporal"
    if (p / "BASELINE_IMMUTABLE.json").exists() or (p / "predictions.jsonl").exists():
        return "baseline"
    dispatch = _read_json(p / "dispatch_config.json")
    if dispatch and dispatch.get("experiment_type"):
        et = str(dispatch["experiment_type"])
        if et == "temporal_shuffle":
            return "temporal"
        if et in {
            "baseline",
            "dpo",
            "grpo",
            "comparison",
            "reward_hacking",
            "reward_model",
        }:
            return et
    return "unknown"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _section(status: str, **payload: Any) -> dict[str, Any]:
    out = {"status": status}
    out.update(payload)
    return out


def _unavailable(reason: str) -> dict[str, Any]:
    return _section(UNAVAILABLE, reason=reason)


def _rel_link(path: Path | None, report_dir: Path) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        return str(path.resolve().relative_to(report_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def load_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def collect_prediction_rows(run_dir: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Load example rows for representative selection."""
    for name in (
        "examples_inspectable.jsonl",
        "aligned_examples.jsonl",
        "predictions.jsonl",
        "held_out_eval_rows.jsonl",
    ):
        path = run_dir / name
        rows = load_optional_jsonl(path)
        if rows:
            return rows, str(path)
    return [], None


def _normalize_example_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten comparison / RH / baseline row shapes for selection."""
    if "before" in row and "after" in row:
        # Reward-hacking row: prefer after for correctness display
        after = dict(row.get("after") or {})
        return {
            "example_id": row.get("example_id"),
            "trick_id": row.get("trick_id") or "",
            "clip_id": row.get("clip_id"),
            "correct": after.get("correct"),
            "parse_failed": after.get("parse_failed"),
            "raw_text": after.get("raw_text"),
            "parsed_answer": after.get("parsed_answer"),
            "ground_truth": row.get("ground_truth"),
            "source": "reward_hacking_after",
        }
    if "methods" in row and isinstance(row["methods"], dict):
        # Aligned comparison: pick first method cell with correct set
        cell = None
        for _mid, c in row["methods"].items():
            if not c.get("missing"):
                cell = c
                break
        cell = cell or {}
        return {
            "example_id": row.get("example_id"),
            "trick_id": row.get("trick_id") or "",
            "clip_id": row.get("clip_id"),
            "correct": cell.get("correct"),
            "parse_failed": cell.get("parse_failed"),
            "raw_text": cell.get("raw_text"),
            "parsed_answer": cell.get("parsed_answer"),
            "ground_truth": row.get("ground_truth"),
            "source": "aligned_methods",
        }
    return {
        "example_id": row.get("example_id"),
        "trick_id": row.get("trick_id") or "",
        "clip_id": row.get("clip_id"),
        "correct": row.get("correct", row.get("matched")),
        "parse_failed": row.get("parse_failed"),
        "raw_text": row.get("raw_text"),
        "parsed_answer": row.get("parsed_answer"),
        "ground_truth": row.get("ground_truth"),
        "source": "predictions",
    }


def select_representative_examples(
    rows: Sequence[Mapping[str, Any]],
    selection: ExampleSelectionConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Apply documented selection rule; deterministic."""
    normalized = [_normalize_example_row(r) for r in rows]
    successes = [r for r in normalized if r.get("correct") is True]
    failures = [r for r in normalized if r.get("correct") is False]

    def _key(r: dict[str, Any]) -> tuple:
        return tuple(str(r.get(k) or "") for k in selection.sort_keys)

    successes = sorted(successes, key=_key)
    failures = sorted(failures, key=_key)

    def _round_robin(pool: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if limit <= 0 or not pool:
            return []
        buckets: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for row in pool:
            key = str(row.get(selection.diversity_key) or "unknown")
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(row)
        out: list[dict[str, Any]] = []
        idx = 0
        while len(out) < limit and any(buckets[k] for k in order):
            key = order[idx % len(order)]
            idx += 1
            if buckets[key]:
                out.append(buckets[key].pop(0))
        return out

    def _preview(row: dict[str, Any]) -> dict[str, Any]:
        raw = str(row.get("raw_text") or "")
        if len(raw) > selection.raw_text_max_chars:
            raw = raw[: selection.raw_text_max_chars] + "…"
        return {
            "example_id": row.get("example_id"),
            "trick_id": row.get("trick_id"),
            "clip_id": row.get("clip_id"),
            "correct": row.get("correct"),
            "parse_failed": row.get("parse_failed"),
            "parsed_answer": row.get("parsed_answer"),
            "ground_truth": row.get("ground_truth"),
            "raw_text_preview": raw,
            "source": row.get("source"),
        }

    return {
        "successes": [_preview(r) for r in _round_robin(successes, selection.max_successes)],
        "failures": [_preview(r) for r in _round_robin(failures, selection.max_failures)],
        "n_success_pool": len(successes),
        "n_failure_pool": len(failures),
        "selection": selection.to_dict(),
    }


def _load_run_bundle(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None or not run_dir.exists():
        return {"present": False, "path": None if run_dir is None else str(run_dir)}
    files = {
        "config": _read_json(run_dir / "config.json")
        or _read_json(run_dir / "dispatch_config.json"),
        "metadata": _read_json(run_dir / "metadata.json"),
        "train_metadata": _read_json(run_dir / "train_metadata.json"),
        "metrics": _read_json(run_dir / "metrics.json")
        or _read_json(run_dir / "baseline_summary.json"),
        "held_out_eval": _read_json(run_dir / "held_out_eval.json"),
        "reward_stats": _read_json(run_dir / "reward_stats.json"),
        "temporal_summary": _read_json(run_dir / "temporal_shuffle_summary.json"),
        "temporal_metadata": _read_json(run_dir / "temporal_shuffle_metadata.json"),
        "comparison_metrics": _read_json(run_dir / "comparison_metrics.json"),
        "reward_hacking_metrics": _read_json(run_dir / "reward_hacking_metrics.json"),
        "disclaimer": _read_json(run_dir / "DISCLAIMER.json"),
        "result": _read_json(run_dir / "result.json"),
        "failure": _read_json(run_dir / "failure.json"),
        "split_lock": _read_json(run_dir / "split_lock.json"),
        "stack_probe": _read_json(run_dir / "stack_probe.json"),
    }
    pred_rows, pred_source = collect_prediction_rows(run_dir)
    links = {
        name: str(run_dir / name)
        for name in (
            "predictions.jsonl",
            "held_out_eval_rows.jsonl",
            "examples_inspectable.jsonl",
            "aligned_examples.jsonl",
            "analysis_report.md",
            "comparison_report.md",
            "reward_hacking_report.md",
            "train_metadata.json",
            "held_out_eval.json",
            "reward_stats.json",
            "temporal_shuffle_summary.json",
            "temporal_shuffle_pairs.jsonl",
            "DISCLAIMER.json",
            "failure.json",
            "checkpoint",
        )
        if (run_dir / name).exists()
    }
    return {
        "present": True,
        "path": str(run_dir),
        "kind": detect_run_kind(run_dir),
        "files": files,
        "prediction_rows": pred_rows,
        "prediction_source": pred_source,
        "artifact_links": links,
        "has_checkpoint": (run_dir / "checkpoint").exists()
        or (run_dir / "checkpoint_best.pt").exists(),
    }


def build_experiment_report(config: ReportConfig) -> dict[str, Any]:
    """Assemble a full experiment report dict (deterministic given inputs)."""
    artifacts: dict[str, dict[str, Any]] = {}
    for key in (
        "baseline",
        "temporal",
        "dpo",
        "grpo",
        "reward_model",
        "comparison",
        "reward_hacking",
    ):
        path = config.artifacts.get(key)
        artifacts[key] = _load_run_bundle(None if path is None else Path(path))

    if config.primary_run_dir:
        primary = _load_run_bundle(Path(config.primary_run_dir))
    else:
        primary = next(
            (b for b in artifacts.values() if b.get("present")),
            {"present": False, "path": None, "kind": "unknown"},
        )

    unresolved: list[str] = []
    for key, bundle in artifacts.items():
        if not bundle.get("present"):
            unresolved.append(f"artifact.{key}: {UNAVAILABLE}")
        elif bundle.get("files", {}).get("failure"):
            unresolved.append(f"artifact.{key}: failed run recorded in failure.json")

    # --- sections ---
    summary = _build_summary(artifacts, primary, config, unresolved)
    configuration = _build_configuration(artifacts, primary, config)
    dataset = _build_dataset(artifacts, primary, config)
    model = _build_model(artifacts, primary)
    training = _build_training(artifacts)
    reward = _build_reward(artifacts)
    evaluation = _build_evaluation(artifacts, primary)
    generalization = _build_generalization(artifacts)
    temporal = _build_temporal(artifacts)
    hacking = _build_reward_hacking(artifacts)
    examples = _build_examples(artifacts, primary, config.example_selection)

    # Human evaluation visibility from RH if present
    if hacking.get("status") == "available":
        he = (hacking.get("human_evaluation") or {})
        if he.get("available") is False:
            unresolved.append("human_evaluation: unavailable (recorded explicitly)")
    else:
        unresolved.append("reward_hacking_analysis: unavailable")

    if generalization.get("status") != "available":
        unresolved.append("generalization_slices: unavailable")
    if temporal.get("status") != "available":
        unresolved.append("temporal_shuffle: unavailable")

    report = {
        "schema_version": "1.0.0",
        "generated_at": config.generated_at,  # may be None → deterministic without clock
        "integrity_disclaimer": INTEGRITY_DISCLAIMER,
        "example_selection_rule": EXAMPLE_SELECTION_RULE,
        "config_fingerprint": config_fingerprint(config.to_dict()),
        "protocol": dict(config.protocol) if config.protocol else UNAVAILABLE,
        "notes": config.notes or UNAVAILABLE,
        "summary": summary,
        "configuration": configuration,
        "dataset_split": dataset,
        "model_checkpoint": model,
        "training_method": training,
        "reward": reward,
        "evaluation": evaluation,
        "generalization": generalization,
        "temporal_shuffle": temporal,
        "reward_hacking_analysis": hacking,
        "representative_successes": examples["successes"],
        "representative_failures": examples["failures"],
        "example_selection": examples["meta"],
        "unresolved_issues": unresolved,
        "artifact_index": {
            k: {
                "present": v.get("present"),
                "path": v.get("path"),
                "kind": v.get("kind"),
                "links": v.get("artifact_links"),
            }
            for k, v in artifacts.items()
        },
        "dimension_legend": {
            "accuracy": "Exact-match task performance when available",
            "reward": "Programmatic or training reward statistics",
            "generalization": "Unseen-identity / wording slices when compared",
            "temporal": "Ordered vs shuffled diagnostic (not causal proof)",
            "reward_hacking": "Possible reward–quality divergence (observational)",
            "reasoning_improvement": "NOT inferred automatically from any metric",
        },
    }
    _assert_no_forbidden_claims(report)
    return report


def _build_summary(
    artifacts: Mapping[str, dict[str, Any]],
    primary: Mapping[str, Any],
    config: ReportConfig,
    unresolved: Sequence[str],
) -> dict[str, Any]:
    present = [k for k, v in artifacts.items() if v.get("present")]
    eval_bits: dict[str, Any] = {}
    base = artifacts.get("baseline") or {}
    grpo = artifacts.get("grpo") or {}
    cmp = artifacts.get("comparison") or {}
    if base.get("present") and base.get("files", {}).get("metrics"):
        m = base["files"]["metrics"]
        eval_bits["baseline_accuracy"] = m.get("overall_accuracy", m.get("accuracy"))
    if grpo.get("present") and grpo.get("files", {}).get("held_out_eval"):
        eval_bits["grpo_held_out_accuracy"] = grpo["files"]["held_out_eval"].get(
            "accuracy"
        )
    if cmp.get("present") and cmp.get("files", {}).get("comparison_metrics"):
        aggs = cmp["files"]["comparison_metrics"].get("aggregates") or {}
        eval_bits["comparison_accuracies"] = {
            mid: st.get("accuracy") for mid, st in aggs.items()
        }
    return _section(
        "available",
        present_artifacts=present,
        primary_kind=primary.get("kind"),
        primary_path=primary.get("path"),
        headline_metrics=eval_bits or UNAVAILABLE,
        n_unresolved=len(unresolved),
        statement=(
            "Report aggregates stored artifacts only; no scientific conclusion "
            "is invented here."
        ),
    )


def _build_configuration(
    artifacts: Mapping[str, dict[str, Any]],
    primary: Mapping[str, Any],
    config: ReportConfig,
) -> dict[str, Any]:
    cfg = None
    if primary.get("files"):
        cfg = primary["files"].get("config")
    if cfg is None:
        for bundle in artifacts.values():
            if bundle.get("files", {}).get("config"):
                cfg = bundle["files"]["config"]
                break
    if cfg is None and not config.protocol:
        return _unavailable("No config.json / dispatch_config.json / protocol found")
    return _section(
        "available",
        run_config=cfg if cfg is not None else UNAVAILABLE,
        report_protocol=config.protocol or UNAVAILABLE,
    )


def _build_dataset(
    artifacts: Mapping[str, dict[str, Any]],
    primary: Mapping[str, Any],
    config: ReportConfig,
) -> dict[str, Any]:
    proto = dict(config.protocol)
    meta = (primary.get("files") or {}).get("metadata") or {}
    train_meta = None
    for key in ("grpo", "dpo", "reward_model", "baseline"):
        tm = (artifacts.get(key) or {}).get("files", {}).get("train_metadata")
        if tm:
            train_meta = tm
            break
    split_lock = (primary.get("files") or {}).get("split_lock")

    manifest = proto.get("manifest")
    if not manifest and train_meta:
        manifest = train_meta.get("manifest")
    if not manifest and isinstance(meta.get("config"), dict):
        manifest = (meta.get("config") or {}).get("dataset", {}).get("manifest")

    dataset_version = proto.get("dataset_version")
    if dataset_version is None and train_meta:
        dataset_version = train_meta.get("dataset_version")

    dataset = {
        "manifest": manifest if manifest else UNAVAILABLE,
        "dataset_version": dataset_version if dataset_version else UNAVAILABLE,
        "task": proto.get("task") or UNAVAILABLE,
        "split": proto.get("split") or UNAVAILABLE,
        "split_lock": split_lock if split_lock is not None else UNAVAILABLE,
    }
    if all(
        dataset[k] == UNAVAILABLE for k in ("manifest", "dataset_version", "task", "split")
    ):
        return _unavailable("Dataset/split metadata not found")
    return _section("available", **dataset)


def _build_model(artifacts: Mapping[str, dict[str, Any]], primary: Mapping[str, Any]) -> dict[str, Any]:
    model_id = UNAVAILABLE
    checkpoint = UNAVAILABLE
    hardware = UNAVAILABLE
    seed = UNAVAILABLE
    for key in ("grpo", "dpo", "baseline", "reward_model", "temporal"):
        bundle = artifacts.get(key) or {}
        if not bundle.get("present"):
            continue
        tm = bundle.get("files", {}).get("train_metadata") or {}
        cfg = bundle.get("files", {}).get("config") or {}
        meta = bundle.get("files", {}).get("metadata") or {}
        if model_id == UNAVAILABLE:
            model_id = (
                tm.get("base_checkpoint")
                or tm.get("base_model_id")
                or cfg.get("model_id")
                or (cfg.get("model") or {}).get("model_id")
                or (meta.get("identity") or {}).get("model_id")
                or UNAVAILABLE
            )
        if checkpoint == UNAVAILABLE and bundle.get("has_checkpoint"):
            links = bundle.get("artifact_links") or {}
            checkpoint = links.get("checkpoint") or str(
                Path(bundle["path"]) / "checkpoint"
            )
        if hardware == UNAVAILABLE:
            hardware = tm.get("hardware") or bundle.get("files", {}).get("stack_probe") or UNAVAILABLE
        if seed == UNAVAILABLE:
            seed = tm.get("seed")
            if seed is None:
                seed = cfg.get("seed", UNAVAILABLE)
    if model_id == UNAVAILABLE and checkpoint == UNAVAILABLE:
        return _unavailable("Model/checkpoint metadata not found")
    return _section(
        "available",
        model=model_id,
        checkpoint=checkpoint,
        hardware=hardware,
        seed=seed,
    )


def _build_training(artifacts: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    for key, method in (("grpo", "grpo"), ("dpo", "dpo"), ("reward_model", "reward_model")):
        bundle = artifacts.get(key) or {}
        tm = bundle.get("files", {}).get("train_metadata")
        if not tm:
            continue
        return _section(
            "available",
            method=method,
            hyperparameters={
                k: tm.get(k)
                for k in (
                    "learning_rate",
                    "max_steps",
                    "training_steps",
                    "batch_size",
                    "per_device_train_batch_size",
                    "gradient_accumulation_steps",
                    "beta",
                    "num_generations",
                    "group_size_num_generations",
                    "peft",
                    "optimizer",
                    "seed",
                    "checkpoint_selection",
                    "checkpoint_selection_rule",
                )
                if k in tm
            },
            train_metadata_path=str(Path(bundle["path"]) / "train_metadata.json"),
            note=(
                "Training loss/reward reduction is not reasoning improvement."
            ),
        )
    # Zero-shot baseline
    if (artifacts.get("baseline") or {}).get("present"):
        return _section(
            "available",
            method="none_zero_shot",
            hyperparameters=UNAVAILABLE,
            note="Baseline / zero-shot: no post-training method applied.",
        )
    return _unavailable("No training metadata found")


def _build_reward(artifacts: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("grpo", "comparison", "reward_hacking", "reward_model"):
        bundle = artifacts.get(key) or {}
        if not bundle.get("present"):
            continue
        rs = bundle.get("files", {}).get("reward_stats")
        tm = bundle.get("files", {}).get("train_metadata") or {}
        if rs:
            payload[f"{key}_reward_stats"] = rs
        if tm.get("reward_id"):
            payload[f"{key}_reward_id"] = tm.get("reward_id")
            payload[f"{key}_reward_version"] = tm.get("reward_version")
    if not payload:
        return _unavailable("No reward function / reward_stats found")
    payload["note"] = (
        "Reward statistics are separate from independent task accuracy."
    )
    return _section("available", **payload)


def _accuracy_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    normalized = [_normalize_example_row(r) for r in rows]
    scored = [r for r in normalized if r.get("correct") is not None]
    if not scored:
        return None
    n_correct = sum(1 for r in scored if r.get("correct") is True)
    return {
        "n_examples": len(scored),
        "n_correct": n_correct,
        "accuracy": n_correct / len(scored),
        "source": "derived_from_prediction_rows",
        "note": "Derived from stored prediction rows; not a new evaluation run.",
    }


def _build_evaluation(
    artifacts: Mapping[str, dict[str, Any]],
    primary: Mapping[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in ("baseline", "grpo", "comparison", "temporal", "reward_hacking"):
        bundle = artifacts.get(key) or {}
        if not bundle.get("present"):
            continue
        files = bundle.get("files") or {}
        if files.get("metrics"):
            metrics[f"{key}_metrics"] = files["metrics"]
        if files.get("held_out_eval"):
            metrics[f"{key}_held_out_eval"] = files["held_out_eval"]
        if files.get("comparison_metrics"):
            cm = files["comparison_metrics"]
            metrics["comparison_aggregates"] = cm.get("aggregates")
            metrics["comparison_deltas"] = cm.get("deltas")
        derived = _accuracy_from_rows(bundle.get("prediction_rows") or [])
        if derived and f"{key}_derived_accuracy" not in metrics:
            # Prefer explicit metrics/held_out_eval when present
            if not files.get("metrics") and not files.get("held_out_eval"):
                metrics[f"{key}_derived_accuracy"] = derived
    if not metrics:
        return _unavailable("No evaluation metrics found")
    return _section(
        "available",
        metrics=metrics,
        note=(
            "Report exact-match / listed metrics only. Do not infer reasoning "
            "improvement from these numbers."
        ),
    )


def _build_generalization(artifacts: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    cmp = artifacts.get("comparison") or {}
    cm = (cmp.get("files") or {}).get("comparison_metrics")
    if not cm:
        return _unavailable("No comparison_metrics.json with generalization slices")
    slices = cm.get("generalization_slices")
    if not slices:
        return _unavailable("comparison_metrics.json lacks generalization_slices")
    return _section(
        "available",
        slices=slices,
        per_group=cm.get("per_group", UNAVAILABLE),
        note="Slice metrics describe identity/wording coverage, not reasoning.",
    )


def _build_temporal(artifacts: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    for key in ("temporal", "comparison", "baseline", "grpo"):
        bundle = artifacts.get(key) or {}
        summary = (bundle.get("files") or {}).get("temporal_summary")
        if summary:
            return _section(
                "available",
                summary=summary,
                source_path=bundle.get("path"),
                note=(
                    "Temporal-shuffle sensitivity is an order diagnostic, not "
                    "proof of causal reasoning."
                ),
            )
        # comparison may embed temporal block
        cm = (bundle.get("files") or {}).get("comparison_metrics")
        if cm and cm.get("temporal"):
            return _section(
                "available",
                summary=cm["temporal"],
                source_path=bundle.get("path"),
                note=(
                    "Temporal-shuffle sensitivity is an order diagnostic, not "
                    "proof of causal reasoning."
                ),
            )
    return _unavailable("No temporal_shuffle_summary found")


def _build_reward_hacking(artifacts: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    bundle = artifacts.get("reward_hacking") or {}
    rh = (bundle.get("files") or {}).get("reward_hacking_metrics")
    if not rh:
        return _unavailable("No reward_hacking_metrics.json found")
    return _section(
        "available",
        before_after=rh.get("before_after"),
        quadrant_counts_after=rh.get("quadrant_counts_after"),
        findings=rh.get("findings"),
        human_evaluation=rh.get("human_evaluation"),
        single_example_is_not_proof=rh.get("single_example_is_not_proof", True),
        source_path=bundle.get("path"),
        note=(
            "High reward with low independent accuracy is a possible divergence "
            "signal in aggregate; no single example proves reward hacking."
        ),
    )


def _build_examples(
    artifacts: Mapping[str, dict[str, Any]],
    primary: Mapping[str, Any],
    selection: ExampleSelectionConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source = None
    # Prefer comparison aligned, then RH, then baseline/grpo predictions
    for key in ("comparison", "reward_hacking", "baseline", "grpo", "temporal"):
        bundle = artifacts.get(key) or {}
        if bundle.get("prediction_rows"):
            rows = list(bundle["prediction_rows"])
            source = bundle.get("prediction_source")
            break
    if not rows and primary.get("prediction_rows"):
        rows = list(primary["prediction_rows"])
        source = primary.get("prediction_source")
    if not rows:
        empty = {
            "successes": [],
            "failures": [],
            "meta": {
                "status": UNAVAILABLE,
                "reason": "No prediction / inspectable example rows found",
                "selection": selection.to_dict(),
            },
        }
        return empty
    selected = select_representative_examples(rows, selection)
    return {
        "successes": selected["successes"],
        "failures": selected["failures"],
        "meta": {
            "status": "available",
            "pool_source": source,
            "n_success_pool": selected["n_success_pool"],
            "n_failure_pool": selected["n_failure_pool"],
            "selection": selected["selection"],
        },
    }


def _assert_no_forbidden_claims(report: Mapping[str, Any]) -> None:
    blob = stable_json(report)
    for pat in FORBIDDEN_CLAIM_PATTERNS:
        if pat.search(blob):
            raise ReportingError(
                f"Forbidden scientific claim language detected in report: {pat.pattern}"
            )


def render_experiment_report_markdown(report: Mapping[str, Any]) -> str:
    """Deterministic markdown rendering."""
    lines: list[str] = [
        "# Research experiment report",
        "",
        str(report["integrity_disclaimer"]),
        "",
    ]
    if report.get("generated_at"):
        lines.append(f"- Generated at: `{report['generated_at']}`")
    lines.append(f"- Config fingerprint: `{report['config_fingerprint']}`")
    lines.append("")

    def _status_line(title: str, section: Mapping[str, Any] | Any) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not isinstance(section, dict):
            lines.append(f"- {section}")
            lines.append("")
            return
        status = section.get("status", "available")
        lines.append(f"- Status: **{status}**")
        if status == UNAVAILABLE:
            lines.append(f"- Reason: {section.get('reason', UNAVAILABLE)}")
            lines.append("")
            return
        for key, value in section.items():
            if key == "status":
                continue
            if isinstance(value, (dict, list)):
                lines.append(f"- `{key}`:")
                lines.append("```json")
                lines.append(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))
                lines.append("```")
            else:
                lines.append(f"- `{key}`: {value}")
        lines.append("")

    summary = report.get("summary") or {}
    _status_line("1. Experiment summary", summary)
    _status_line("2. Configuration", report.get("configuration") or {})
    _status_line("3. Dataset / split", report.get("dataset_split") or {})
    _status_line("4. Model / checkpoint", report.get("model_checkpoint") or {})
    _status_line("5. Training method", report.get("training_method") or {})
    _status_line("6. Reward", report.get("reward") or {})
    _status_line("7. Evaluation", report.get("evaluation") or {})
    _status_line("8. Generalization", report.get("generalization") or {})
    _status_line("9. Temporal shuffle", report.get("temporal_shuffle") or {})
    _status_line("10. Reward-hacking analysis", report.get("reward_hacking_analysis") or {})

    lines.append("## 11. Representative successes")
    lines.append("")
    lines.append(f"Selection rule: {report.get('example_selection_rule')}")
    lines.append("")
    successes = report.get("representative_successes") or []
    if not successes:
        lines.append("- None available (or pool empty).")
    for row in successes:
        lines.append(
            f"- `{row.get('example_id')}` trick=`{row.get('trick_id')}` "
            f"pred=`{row.get('parsed_answer')}` gold=`{row.get('ground_truth')}`"
        )
        preview = row.get("raw_text_preview")
        if preview:
            lines.append(f"  - preview: {preview}")
    lines.append("")

    lines.append("## 12. Representative failures")
    lines.append("")
    failures = report.get("representative_failures") or []
    if not failures:
        lines.append("- None available (or pool empty).")
    for row in failures:
        lines.append(
            f"- `{row.get('example_id')}` trick=`{row.get('trick_id')}` "
            f"pred=`{row.get('parsed_answer')}` gold=`{row.get('ground_truth')}`"
        )
        preview = row.get("raw_text_preview")
        if preview:
            lines.append(f"  - preview: {preview}")
    lines.append("")

    lines.append("## 13. Unresolved issues")
    lines.append("")
    unresolved = report.get("unresolved_issues") or []
    if not unresolved:
        lines.append("- None recorded.")
    for item in unresolved:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Artifact index")
    lines.append("")
    for key, info in (report.get("artifact_index") or {}).items():
        lines.append(
            f"- `{key}`: present={info.get('present')} path=`{info.get('path')}`"
        )
    lines.append("")
    lines.append("## Dimension legend")
    lines.append("")
    for key, val in (report.get("dimension_legend") or {}).items():
        lines.append(f"- **{key}**: {val}")
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class ReportWriteResult:
    run_dir: str
    markdown_path: str
    json_path: str
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "markdown_path": self.markdown_path,
            "json_path": self.json_path,
            "disclaimer": INTEGRITY_DISCLAIMER,
        }


def write_experiment_report(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path,
    run_id: str,
    config: ReportConfig | None = None,
) -> ReportWriteResult:
    run_dir = allocate_run_directory(output_dir, run_id, overwrite=False)
    md = render_experiment_report_markdown(report)
    md_path = run_dir / "experiment_report.md"
    md_path.write_text(md, encoding="utf-8")
    # Deterministic JSON (sorted keys)
    json_path = run_dir / "experiment_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_json(
        run_dir / "DISCLAIMER.json",
        {
            "disclaimer": INTEGRITY_DISCLAIMER,
            "example_selection_rule": EXAMPLE_SELECTION_RULE,
            "never_claim_reasoning_improvement": True,
        },
    )
    if config is not None:
        write_json(run_dir / "experiment_report_config.json", config.to_dict())
    write_json(
        run_dir / "result.json",
        {
            "run_dir": str(run_dir),
            "config_fingerprint": report.get("config_fingerprint"),
            "n_unresolved": len(report.get("unresolved_issues") or []),
            "disclaimer": INTEGRITY_DISCLAIMER,
        },
    )
    return ReportWriteResult(
        run_dir=str(run_dir),
        markdown_path=str(md_path),
        json_path=str(json_path),
        report=dict(report),
    )


def generate_experiment_report(config: ReportConfig) -> ReportWriteResult:
    """Build + write a research report from config."""
    report = build_experiment_report(config)
    run_id = config.run_id or f"report_{config_fingerprint(config.to_dict())}"
    return write_experiment_report(
        report,
        output_dir=config.output_dir,
        run_id=run_id,
        config=config,
    )
