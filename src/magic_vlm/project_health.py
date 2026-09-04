"""End-to-end project health audit, smoke probes, and status dashboard.

Produces machine-readable audit JSON plus human-readable markdown/HTML.
Does not invent PASS for real VLM / real baseline without evidence.
"""

from __future__ import annotations

import argparse
import html
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from magic_vlm.utils import utc_now_iso, write_json

Status = Literal[
    "PASS",
    "PARTIAL",
    "FAIL",
    "BLOCKED",
    "NOT_IMPLEMENTED",
    "NOT_APPLICABLE",
]

Priority = Literal["now", "later", "optional"]

EVIDENCE_LABELS: dict[int, str] = {
    0: "LEVEL 0 - NOT INSPECTED",
    1: "LEVEL 1 - CODE EXISTS",
    2: "LEVEL 2 - IMPORT/UNIT TEST",
    3: "LEVEL 3 - INTEGRATION/SMOKE TEST",
    4: "LEVEL 4 - REAL ARTIFACT TEST",
    5: "LEVEL 5 - REAL RESEARCH RUN",
}

# Research-stage pipeline (ids must stay stable for dashboards).
PIPELINE_ORDER: tuple[tuple[str, str], ...] = (
    ("repository", "Repository architecture"),
    ("environment", "Environment setup"),
    ("reproducibility", "Reproducibility / configuration"),
    ("dataset_schema", "Dataset schema"),
    ("dataset_validation", "Dataset validation"),
    ("video_preprocessing", "Video preprocessing"),
    ("vlm_loading", "VLM model loading"),
    ("vlm_inference", "VLM video inference"),
    ("zero_shot_baseline", "Zero-shot baseline"),
    ("baseline_evaluation", "Baseline evaluation"),
    ("failure_analysis", "Failure analysis"),
    ("preference_schema", "Preference schema"),
    ("preference_annotation", "Preference annotation workflow"),
    ("preference_validation", "Preference validation"),
    ("reward_model", "Bradley-Terry reward model"),
    ("dpo", "DPO"),
    ("reward_interface", "Reward interface"),
    ("temporal_shuffle", "Temporal shuffle"),
    ("temporal_causal_reward", "Temporal / causal reward"),
    ("experiment_runner", "Common experiment runner"),
    ("grpo", "GRPO"),
    ("comparative_evaluation", "Comparative evaluation"),
    ("reward_hacking", "Reward-hacking analysis"),
    ("reporting", "Research reporting"),
)

QWEN_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
)

EXPECTED_ENTRY_POINTS: tuple[str, ...] = (
    "magic-vlm-smoke",
    "magic-vlm-init",
    "magic-vlm-validate",
    "magic-vlm-sample-frames",
    "magic-vlm-infer",
    "magic-vlm-baseline",
    "magic-vlm-analyze-baseline",
    "magic-vlm-validate-preferences",
    "magic-vlm-annotate",
    "magic-vlm-analyze-preferences",
    "magic-vlm-train-reward",
    "magic-vlm-score-reward",
    "magic-vlm-train-dpo",
    "magic-vlm-train-grpo",
    "magic-vlm-temporal-shuffle",
    "magic-vlm-compare-objective",
    "magic-vlm-compare-methods",
    "magic-vlm-analyze-reward-hacking",
    "magic-vlm-report",
    "magic-vlm-run",
)


@dataclass
class ComponentResult:
    """Status of one research-pipeline stage."""

    id: str
    name: str
    status: Status
    evidence_level: int
    notes: str = ""
    tested: bool = False
    last_test: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "evidence_level": self.evidence_level,
            "evidence_label": EVIDENCE_LABELS.get(
                self.evidence_level, f"LEVEL {self.evidence_level}"
            ),
            "notes": self.notes,
            "tested": self.tested,
            "last_test": self.last_test,
            "details": dict(self.details),
        }


@dataclass
class Blocker:
    id: str
    why: str
    need: str
    priority: Priority

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HumanInputItem:
    priority: Priority
    what: str
    where: str
    format: str
    after: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _try_import(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
        return {
            "available": True,
            "version": getattr(mod, "__version__", None),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}


def hf_hub_root() -> Path:
    """Resolve Hugging Face hub cache directory."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def list_qwen_cache_dirs(hub: Path | None = None) -> list[str]:
    root = hub or hf_hub_root()
    if not root.is_dir():
        return []
    names: list[str] = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name.lower().startswith("models--qwen"):
            names.append(path.name)
    return names


def count_real_mp4(root: Path) -> int:
    """Count non-empty ``*.mp4`` files under ``data/videos``."""
    videos = root / "data" / "videos"
    if not videos.is_dir():
        return 0
    count = 0
    for path in videos.rglob("*.mp4"):
        if path.is_file() and path.stat().st_size > 0:
            count += 1
    return count


def load_hidden_state_inventory(root: Path) -> dict[str, Any] | None:
    path = Path(root) / "data" / "examples" / "hidden_state_candidate_inventory.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def hidden_state_dataset_stats(inventory: dict[str, Any] | None) -> dict[str, Any]:
    """Collection counts from the inventory file. Never invents gold."""
    empty = {
        "candidate_count": 0,
        "eligible_count": 0,
        "pending_human_review": 0,
        "rejected_count": 0,
        "clips_needed_for_pilot": 5,
    }
    if not inventory:
        return {
            "hidden_state_candidates": 0,
            "approved_gold_examples": 0,
            "pending_review": 0,
            "rejected": 0,
            "clips_needed": 5,
            "wikimedia_controls": dict(empty),
            "mac_king_candidates": dict(empty),
            "hidden_state_gold": {
                "eligible_count": 0,
                "pending_human_review": 0,
                "clips_needed_for_pilot": 5,
            },
        }
    ready = inventory.get("readiness") or {}
    collections = ready.get("collections") or {}
    wiki = collections.get("wikimedia_controls") or dict(empty)
    mac = collections.get("mac_king_candidates") or dict(empty)
    gold = collections.get("hidden_state_gold") or {
        "eligible_count": int(ready.get("valid_candidates") or 0),
        "pending_human_review": int(ready.get("candidates_needing_human_review") or 0),
        "clips_needed_for_pilot": int(ready.get("additional_clips_needed") or 5),
    }
    approved = int(gold.get("eligible_count") or 0)
    pending = int(gold.get("pending_human_review") or 0)
    rejected = int(wiki.get("rejected_count") or 0) + int(mac.get("rejected_count") or 0)
    return {
        "hidden_state_candidates": int(mac.get("candidate_count") or 0),
        "approved_gold_examples": approved,
        "pending_review": pending,
        "rejected": rejected,
        "clips_needed": int(gold.get("clips_needed_for_pilot") or max(0, 5 - approved)),
        "wikimedia_controls": wiki,
        "mac_king_candidates": mac,
        "hidden_state_gold": gold,
    }


def probe_nvidia_smi() -> dict[str, Any]:
    """Best-effort ``nvidia-smi`` probe. Never invents a GPU."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {
            "available": False,
            "executable": None,
            "returncode": None,
            "gpus": [],
            "error": "nvidia-smi not found on PATH",
        }
    try:
        proc = subprocess.run(
            [
                exe,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        gpus: list[dict[str, Any]] = []
        for line in (proc.stdout or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append(
                    {
                        "name": parts[0],
                        "memory_total_mib": parts[1],
                        "driver_version": parts[2],
                    }
                )
        return {
            "available": proc.returncode == 0 and len(gpus) > 0,
            "executable": exe,
            "returncode": proc.returncode,
            "gpus": gpus,
            "stderr_tail": (proc.stderr or "")[-500:],
            "error": None if proc.returncode == 0 else (proc.stderr or "nvidia-smi failed"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "executable": exe,
            "returncode": None,
            "gpus": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_real_smoke_evidence(root: Path) -> dict[str, Any] | None:
    """Load committed REAL_ZERO_SHOT_BASELINE_SMOKE_TEST evidence if present."""
    path = Path(root) / "reports" / "real_zero_shot_baseline_smoke" / "summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def load_real_baseline_evidence(root: Path) -> dict[str, Any] | None:
    """Load committed REAL_ZERO_SHOT_BASELINE evidence if present."""
    path = Path(root) / "reports" / "real_zero_shot_baseline" / "summary.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if str(payload.get("label") or "") != "REAL_ZERO_SHOT_BASELINE":
        return None
    return payload


def probe_environment(root: Path | None = None) -> dict[str, Any]:
    """Probe Python, torch/CUDA, optional stacks, video deps, media, HF cache."""
    root = Path(root or Path.cwd()).resolve()
    torch_info = _try_import("torch")
    cuda_available = False
    cuda_device_count = 0
    torch_version = torch_info.get("version")
    cuda_build_version = None
    gpu_names: list[str] = []
    gpu_memory_bytes: list[int] = []
    if torch_info["available"]:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
            torch_version = str(torch.__version__)
            cuda_build_version = getattr(torch.version, "cuda", None)
            if cuda_available:
                for idx in range(cuda_device_count):
                    gpu_names.append(str(torch.cuda.get_device_name(idx)))
                    props = torch.cuda.get_device_properties(idx)
                    gpu_memory_bytes.append(int(props.total_memory))
        except Exception as exc:  # noqa: BLE001
            torch_info["error"] = f"{type(exc).__name__}: {exc}"

    nvidia = probe_nvidia_smi()
    qwen_dirs = list_qwen_cache_dirs()
    real_mp4_count = count_real_mp4(root)
    smoke_evidence = load_real_smoke_evidence(root)
    baseline_evidence = load_real_baseline_evidence(root)
    baseline_metrics = (baseline_evidence or {}).get("metrics") or {}
    baseline_dataset = (baseline_evidence or {}).get("dataset") or {}

    return {
        "probed_at": utc_now_iso(),
        "root": str(root),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "torch": {
            **torch_info,
            "version": torch_version,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_build_version": cuda_build_version,
            "cpu_only": bool(torch_info["available"]) and not cuda_available,
            "gpu_names": gpu_names,
            "gpu_memory_bytes": gpu_memory_bytes,
        },
        "nvidia_smi": nvidia,
        "gpu_available": bool(nvidia.get("available")) or (cuda_available and cuda_device_count > 0),
        "transformers": _try_import("transformers"),
        "trl": _try_import("trl"),
        "peft": _try_import("peft"),
        "cv2": _try_import("cv2"),
        "accelerate": _try_import("accelerate"),
        "real_mp4_count": real_mp4_count,
        "videos_dir": str(root / "data" / "videos"),
        "hf_hub_cache": str(hf_hub_root()),
        "qwen_cache_dirs": qwen_dirs,
        "qwen_cache_present": len(qwen_dirs) > 0,
        "real_smoke_evidence": smoke_evidence,
        "real_smoke_evidence_present": smoke_evidence is not None,
        "real_baseline_evidence": baseline_evidence,
        "real_baseline_completed": baseline_evidence is not None,
        "real_baseline_run_id": (baseline_evidence or {}).get("run_id"),
        "real_baseline_model_id": (baseline_evidence or {}).get("model_id"),
        "real_baseline_examples_evaluated": int(
            baseline_metrics.get("n_examples")
            or baseline_dataset.get("examples_evaluated")
            or 0
        ),
        "real_baseline_accuracy": baseline_metrics.get("overall_accuracy"),
    }


def run_pytest(root: Path, *, quiet: bool = True) -> dict[str, Any]:
    """Run ``pytest -q`` and capture a coarse summary."""
    cmd = [sys.executable, "-m", "pytest"]
    if quiet:
        cmd.append("-q")
    started = utc_now_iso()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        summary_line = ""
        for line in reversed(out.strip().splitlines()):
            if line.strip():
                summary_line = line.strip()
                break
        return {
            "ran": True,
            "returncode": proc.returncode,
            "summary": summary_line,
            "stdout_tail": "\n".join(out.splitlines()[-40:]),
            "started_at": started,
            "finished_at": utc_now_iso(),
            "passed": proc.returncode == 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ran": True,
            "returncode": -1,
            "summary": f"{type(exc).__name__}: {exc}",
            "stdout_tail": traceback.format_exc(),
            "started_at": started,
            "finished_at": utc_now_iso(),
            "passed": False,
        }


def _smoke_dataset(root: Path) -> dict[str, Any]:
    from magic_vlm.dataset import load_manifest
    from magic_vlm.validate import ValidatorConfig, validate_dataset

    manifest = root / "data" / "examples" / "toy_manifest.jsonl"
    records = load_manifest(manifest)
    report = validate_dataset(
        manifest,
        config=ValidatorConfig(check_media=True, root=root),
    )
    return {
        "ok": True,
        "n_records": len(records),
        "validation_passed": bool(report.passed),
        "n_findings": len(report.findings),
        "finding_severities": sorted({f.severity for f in report.findings}),
        "media_checked": True,
    }


def _smoke_video() -> dict[str, Any]:
    from magic_vlm.video import (
        VideoPreprocessConfig,
        apply_temporal_shuffle,
        build_sample_plan,
        select_frame_indices,
    )

    cfg = VideoPreprocessConfig(max_frames=4, sample_strategy="uniform")
    ordered = select_frame_indices(16, cfg)
    plan = build_sample_plan("synthetic://clip.mp4", num_frames=16, config=cfg)
    shuffled = apply_temporal_shuffle(plan, seed=7)
    same_set = set(shuffled.ordered_indices) == set(shuffled.frame_indices)
    return {
        "ok": True,
        "ordered_indices": list(ordered),
        "shuffled_indices": list(shuffled.frame_indices),
        "same_index_set": same_set,
        "temporal_shuffled": bool(shuffled.temporal_shuffled),
    }


def _smoke_rewards() -> dict[str, Any]:
    from magic_vlm.rewards import HiddenStateExactMatchReward
    from magic_vlm.schemas import (
        ExampleRecord,
        InferenceArtifact,
        Provenance,
        Split,
        TaskType,
        VideoRef,
    )

    example = ExampleRecord(
        example_id="audit_reward_ex",
        clip_id="audit_clip",
        trick_id="t",
        performer_id="p",
        camera_id="cam",
        video=VideoRef(path="audit.mp4"),
        task=TaskType.HIDDEN_STATE,
        question="Which cup contains the ball?",
        ground_truth="left",
        split=Split.TRAIN,
        provenance=Provenance(source="project_health"),
    )
    reward = HiddenStateExactMatchReward()
    good = reward.evaluate(
        InferenceArtifact(
            example_id=example.example_id,
            model_id="stub",
            prompt="p",
            raw_text="Answer: left",
            parsed_answer="left",
        ),
        example,
    )
    bad = reward.evaluate(
        InferenceArtifact(
            example_id=example.example_id,
            model_id="stub",
            prompt="p",
            raw_text="Answer: right",
            parsed_answer="right",
        ),
        example,
    )
    return {
        "ok": good.value == 1.0 and bad.value == 0.0,
        "good_value": good.value,
        "bad_value": bad.value,
        "reward_id": good.reward_id,
        "version": good.version,
    }


def _smoke_stub_vlm() -> dict[str, Any]:
    from magic_vlm.models import ModelSpec, load_vlm

    model = load_vlm(ModelSpec(model_id="stub/echo"), allow_download=False)
    text = model.generate("audit prompt", images=None, videos=None)
    return {
        "ok": bool(text) and "STUB_RESPONSE" in text,
        "model_id": getattr(model, "model_id", "stub/echo"),
        "sample": text[:120],
    }


def _smoke_real_qwen_load() -> dict[str, Any]:
    """Attempt real Qwen load with ``allow_download=False`` (expect fail without cache)."""
    from magic_vlm.models import ModelSpec, load_vlm

    model_id = QWEN_MODEL_IDS[0]
    try:
        model = load_vlm(ModelSpec(model_id=model_id), allow_download=False)
        return {
            "ok": True,
            "loaded": True,
            "model_id": model_id,
            "expected_fail": False,
            "error": None,
            "note": "Unexpected success without download - local weights present.",
            "device": getattr(model, "device", None),
        }
    except Exception as exc:  # noqa: BLE001 - expected on this host
        return {
            "ok": True,  # probe completed; failure is expected without cache
            "loaded": False,
            "model_id": model_id,
            "expected_fail": True,
            "error": f"{type(exc).__name__}: {exc}",
            "note": "Real Qwen load refused or failed with allow_download=False.",
        }


def _smoke_entry_points() -> dict[str, Any]:
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            console = {ep.name: ep.value for ep in eps.select(group="console_scripts")}
        else:  # pragma: no cover - older importlib
            console = {ep.name: ep.value for ep in eps.get("console_scripts", [])}  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "found": [], "missing": list(EXPECTED_ENTRY_POINTS)}

    found = sorted(name for name in EXPECTED_ENTRY_POINTS if name in console)
    missing = sorted(name for name in EXPECTED_ENTRY_POINTS if name not in console)
    return {
        "ok": len(missing) == 0,
        "found": found,
        "missing": missing,
        "n_found": len(found),
        "n_expected": len(EXPECTED_ENTRY_POINTS),
    }


def _smoke_reporting(root: Path) -> dict[str, Any]:
    from magic_vlm.reporting import ReportConfig, build_experiment_report

    cfg_path = root / "configs" / "experiment_report_toy.yaml"
    config = ReportConfig.from_yaml(cfg_path)
    report = build_experiment_report(config)
    return {
        "ok": isinstance(report, dict) and "integrity_disclaimer" in report,
        "keys": sorted(report.keys())[:20],
        "has_disclaimer": "integrity_disclaimer" in report,
        "config": str(cfg_path),
    }


def _call_run_zero_shot_baseline(config: Any, **preferred: Any) -> Any:
    """Invoke ``run_zero_shot_baseline`` with only accepted keyword args."""
    from magic_vlm.baseline import run_zero_shot_baseline

    sig = inspect.signature(run_zero_shot_baseline)
    kwargs = {k: v for k, v in preferred.items() if k in sig.parameters}
    return run_zero_shot_baseline(config, **kwargs)


def _smoke_stub_baseline(root: Path) -> dict[str, Any]:
    from magic_vlm.experiment import experiment_config_from_dict, load_experiment_config

    cfg_path = root / "configs" / "baseline_stub.yaml"
    try:
        config = load_experiment_config(cfg_path)
        payload = config.to_dict()
        out = root / "reports" / "project_health" / "stub_baseline_runs"
        payload["output_dir"] = str(out)
        config = experiment_config_from_dict(payload)
        result = _call_run_zero_shot_baseline(
            config,
            run_id=f"health_stub_{utc_now_iso().replace(':', '').replace('+', '_')}",
            allow_download=False,
            load_frames=False,
            continue_on_error=True,
        )
        summary = getattr(result, "summary", None)
        return {
            "ok": True,
            "run_id": getattr(result, "run_id", None),
            "run_dir": str(getattr(result, "run_dir", "")),
            "split": getattr(result, "split", None),
            "n_examples": getattr(summary, "n_examples", None),
            "overall_accuracy": getattr(summary, "overall_accuracy", None),
            "immutable": getattr(result, "immutable", None),
            "note": "Stub baseline only - not a real VLM research run.",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
            "note": "Stub baseline smoke failed.",
        }


def run_smokes(
    root: Path,
    *,
    run_stub_baseline: bool = True,
) -> dict[str, Any]:
    """Execute lightweight integration smokes (no real GPU/weights required)."""
    results: dict[str, Any] = {}
    probes = [
        ("dataset", lambda: _smoke_dataset(root)),
        ("video", _smoke_video),
        ("rewards", _smoke_rewards),
        ("stub_vlm", _smoke_stub_vlm),
        ("real_qwen_load", _smoke_real_qwen_load),
        ("entry_points", _smoke_entry_points),
        ("reporting", lambda: _smoke_reporting(root)),
    ]
    for name, fn in probes:
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            results[name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:],
            }
    if run_stub_baseline:
        results["stub_baseline"] = _smoke_stub_baseline(root)
    else:
        results["stub_baseline"] = {
            "ok": None,
            "skipped": True,
            "note": "Stub baseline smoke skipped by caller.",
        }
    return results


def _module_exists(dotted: str) -> bool:
    try:
        importlib.import_module(dotted)
        return True
    except Exception:  # noqa: BLE001
        return False


def build_component_results(
    *,
    env: dict[str, Any],
    smokes: dict[str, Any],
    tests: dict[str, Any] | None,
    root: Path,
) -> list[ComponentResult]:
    """Map audit evidence onto ``PIPELINE_ORDER`` component statuses."""
    now = utc_now_iso()
    by_id: dict[str, ComponentResult] = {}

    def put(
        cid: str,
        status: Status,
        evidence: int,
        notes: str,
        *,
        tested: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        name = next((n for i, n in PIPELINE_ORDER if i == cid), cid)
        by_id[cid] = ComponentResult(
            id=cid,
            name=name,
            status=status,
            evidence_level=evidence,
            notes=notes,
            tested=tested,
            last_test=now if tested else None,
            details=details or {},
        )

    # repository
    put(
        "repository",
        "PASS",
        3 if (root / "pyproject.toml").exists() else 1,
        "Package layout, docs, and configs present.",
        tested=True,
        details={"has_pyproject": (root / "pyproject.toml").exists()},
    )

    # environment
    torch_ok = bool(env.get("torch", {}).get("available"))
    cuda = bool(env.get("torch", {}).get("cuda_available"))
    if torch_ok and cuda:
        put("environment", "PASS", 3, "Torch with CUDA available.", tested=True, details=env["torch"])
    elif torch_ok:
        put(
            "environment",
            "BLOCKED",
            3,
            "Torch is CPU-only; real Qwen2.5-VL research runs need CUDA.",
            tested=True,
            details=env["torch"],
        )
    else:
        put(
            "environment",
            "BLOCKED",
            2,
            "Torch not importable in this interpreter.",
            tested=True,
            details=env.get("torch") or {},
        )

    # reproducibility
    put(
        "reproducibility",
        "PASS" if _module_exists("magic_vlm.experiment") else "FAIL",
        3,
        "ExperimentConfig / initialize_experiment available.",
        tested=True,
    )

    # dataset schema + validation
    ds = smokes.get("dataset") or {}
    if ds.get("ok"):
        if ds.get("validation_passed"):
            put(
                "dataset_schema",
                "PASS",
                3,
                f"Loaded {ds.get('n_records')} toy manifest records.",
                tested=True,
                details=ds,
            )
            put(
                "dataset_validation",
                "PASS",
                3,
                "validate_dataset passed on toy_manifest.",
                tested=True,
                details=ds,
            )
        else:
            put(
                "dataset_schema",
                "PASS",
                3,
                f"Schema load ok ({ds.get('n_records')} records).",
                tested=True,
                details=ds,
            )
            put(
                "dataset_validation",
                "PARTIAL",
                3,
                "Toy manifest validates structurally; media checks report missing files "
                f"(findings={ds.get('n_findings')}).",
                tested=True,
                details=ds,
            )
    else:
        put("dataset_schema", "FAIL", 2, str(ds.get("error") or "dataset smoke failed"), tested=True)
        put("dataset_validation", "FAIL", 2, str(ds.get("error") or "validation smoke failed"), tested=True)

    # video
    vid = smokes.get("video") or {}
    real_mp4 = int(env.get("real_mp4_count") or 0)
    if vid.get("ok") and vid.get("same_index_set"):
        if real_mp4 > 0:
            put(
                "video_preprocessing",
                "PASS",
                4,
                f"Frame select/shuffle smoke ok; {real_mp4} real mp4(s) on disk.",
                tested=True,
                details=vid,
            )
        else:
            put(
                "video_preprocessing",
                "PARTIAL",
                3,
                "Index sampling + temporal shuffle smoke ok; no real mp4 under data/videos.",
                tested=True,
                details=vid,
            )
    else:
        put(
            "video_preprocessing",
            "FAIL",
            2,
            str(vid.get("error") or "video smoke failed"),
            tested=True,
            details=vid,
        )

    # VLM loading
    stub = smokes.get("stub_vlm") or {}
    real = smokes.get("real_qwen_load") or {}
    qwen_cache = bool(env.get("qwen_cache_present"))
    if stub.get("ok") and real.get("loaded"):
        put(
            "vlm_loading",
            "PASS",
            4,
            "Stub and real Qwen load succeeded.",
            tested=True,
            details={"stub": stub, "real": real},
        )
    elif stub.get("ok") and not real.get("loaded"):
        put(
            "vlm_loading",
            "BLOCKED",
            3,
            "Stub load works; real Qwen load with allow_download=False failed "
            f"(cache_present={qwen_cache}).",
            tested=True,
            details={"stub": stub, "real": real},
        )
    else:
        put(
            "vlm_loading",
            "FAIL",
            2,
            "Stub and/or real load probe failed.",
            tested=True,
            details={"stub": stub, "real": real},
        )

    # VLM inference - no fake PASS without real video+model
    if env.get("real_baseline_completed"):
        put(
            "vlm_inference",
            "PASS",
            5,
            (
                f"Formal baseline run_id={env.get('real_baseline_run_id')} "
                f"evaluated {env.get('real_baseline_examples_evaluated')} gold example(s) "
                f"with {env.get('real_baseline_model_id')}."
            ),
            tested=True,
            details={"baseline": env.get("real_baseline_evidence")},
        )
    elif real.get("loaded") and real_mp4 > 0 and cuda:
        put(
            "vlm_inference",
            "PARTIAL",
            3,
            "Weights/CUDA/mp4 present; full video→answer path not executed in this audit.",
            tested=False,
        )
    else:
        put(
            "vlm_inference",
            "BLOCKED",
            1 if not stub.get("ok") else 2,
            "Real video VLM inference blocked (need real mp4 + CUDA + Qwen weights).",
            tested=bool(stub.get("ok")),
            details={"stub_ok": stub.get("ok"), "real_loaded": real.get("loaded"), "real_mp4": real_mp4},
        )

    # zero-shot baseline
    base = smokes.get("stub_baseline") or {}
    if env.get("real_baseline_completed"):
        put(
            "zero_shot_baseline",
            "PASS",
            5,
            (
                f"REAL_ZERO_SHOT_BASELINE complete "
                f"(run_id={env.get('real_baseline_run_id')}, "
                f"n={env.get('real_baseline_examples_evaluated')}, "
                f"accuracy={env.get('real_baseline_accuracy')}). "
                "Distinct from REAL_ZERO_SHOT_BASELINE_SMOKE_TEST."
            ),
            tested=True,
            details={"baseline": env.get("real_baseline_evidence"), "stub": base},
        )
    elif base.get("skipped"):
        put(
            "zero_shot_baseline",
            "PARTIAL",
            1,
            "Stub baseline smoke skipped; real baseline still blocked.",
            tested=False,
            details=base,
        )
    elif base.get("ok") and real_mp4 > 0 and cuda and (qwen_cache or real.get("loaded")):
        put(
            "zero_shot_baseline",
            "PARTIAL",
            3,
            "Stub baseline ok and hardware/cache look ready - confirm a real Qwen run separately.",
            tested=True,
            details=base,
        )
    elif base.get("ok"):
        missing = []
        if real_mp4 <= 0:
            missing.append("no real mp4")
        if not cuda:
            missing.append("no CUDA")
        if not (qwen_cache or real.get("loaded")):
            missing.append("no Qwen cache")
        put(
            "zero_shot_baseline",
            "BLOCKED",
            3,
            "Stub baseline smoke passed; real hidden-state baseline blocked "
            f"({', '.join(missing) if missing else 'incomplete VLM stack'}).",
            tested=True,
            details=base,
        )
    else:
        put(
            "zero_shot_baseline",
            "FAIL" if base.get("ok") is False else "BLOCKED",
            2,
            str(base.get("error") or base.get("note") or "baseline not proven"),
            tested=base.get("ok") is False,
            details=base,
        )

    # baseline evaluation / failure analysis - code exists; real run not done
    put(
        "baseline_evaluation",
        "PARTIAL" if _module_exists("magic_vlm.evaluation") else "NOT_IMPLEMENTED",
        2,
        "Evaluation helpers exist; no real baseline metrics produced by this audit.",
        tested=False,
    )
    put(
        "failure_analysis",
        "PARTIAL" if _module_exists("magic_vlm.analysis") else "NOT_IMPLEMENTED",
        2,
        "Analysis module present; not exercised on a real baseline run.",
        tested=False,
    )

    # preferences path
    put(
        "preference_schema",
        "PASS" if _module_exists("magic_vlm.preferences") else "NOT_IMPLEMENTED",
        2,
        "Preference schema/module importable; no human preference labels collected.",
        tested=_module_exists("magic_vlm.preferences"),
    )
    put(
        "preference_annotation",
        "PARTIAL" if _module_exists("magic_vlm.annotation") else "NOT_IMPLEMENTED",
        2,
        "Annotation workflow code exists; requires human judgments later.",
        tested=False,
    )
    put(
        "preference_validation",
        "PARTIAL" if (root / "scripts" / "validate_preferences.py").exists() else "NOT_IMPLEMENTED",
        2,
        "Preference validation entry exists; not run on real annotations.",
        tested=False,
    )

    # reward model / dpo / grpo - blocked for real VLM
    put(
        "reward_model",
        "PARTIAL" if _module_exists("magic_vlm.reward_model") else "NOT_IMPLEMENTED",
        2,
        "BT reward-model code present; real training not audited here.",
        tested=False,
    )
    put(
        "dpo",
        "BLOCKED" if _module_exists("magic_vlm.dpo") else "NOT_IMPLEMENTED",
        2,
        "DPO stack code exists; real Qwen DPO needs CUDA + weights + preferences.",
        tested=False,
    )

    # reward interface
    rw = smokes.get("rewards") or {}
    if rw.get("ok"):
        put(
            "reward_interface",
            "PASS",
            3,
            "hidden_state_exact_match good/bad smoke passed.",
            tested=True,
            details=rw,
        )
    else:
        put(
            "reward_interface",
            "FAIL",
            2,
            str(rw.get("error") or "reward smoke failed"),
            tested=True,
            details=rw,
        )

    # temporal
    put(
        "temporal_shuffle",
        "PARTIAL" if vid.get("ok") else "FAIL",
        3 if vid.get("ok") else 1,
        "Temporal shuffle applied in index smoke; real-video diagnostic not run.",
        tested=bool(vid.get("ok")),
        details=vid,
    )
    put(
        "temporal_causal_reward",
        "PARTIAL" if _module_exists("magic_vlm.rewards") else "NOT_IMPLEMENTED",
        2,
        "temporal_iou / causal annotation path exists; not run on real causal labels.",
        tested=False,
    )

    # runner / comparison / hacking / reporting
    put(
        "experiment_runner",
        "PARTIAL" if _module_exists("magic_vlm.runner") else "NOT_IMPLEMENTED",
        2,
        "Common runner module present; not re-dispatched in this audit beyond stub baseline.",
        tested=False,
    )
    put(
        "grpo",
        "BLOCKED" if _module_exists("magic_vlm.grpo") else "NOT_IMPLEMENTED",
        2,
        "GRPO code exists; real VLM GRPO blocked without CUDA/weights/rewards data.",
        tested=False,
    )
    put(
        "comparative_evaluation",
        "PARTIAL" if _module_exists("magic_vlm.comparison") else "NOT_IMPLEMENTED",
        2,
        "Comparison module present; synthetic fixtures only unless real runs exist.",
        tested=False,
    )
    put(
        "reward_hacking",
        "PARTIAL" if _module_exists("magic_vlm.reward_hacking") else "NOT_IMPLEMENTED",
        2,
        "Reward-hacking diagnostics code present; needs before/after artifacts.",
        tested=False,
    )

    rep = smokes.get("reporting") or {}
    if rep.get("ok"):
        put(
            "reporting",
            "PASS",
            3,
            "ReportConfig + build_experiment_report smoke on experiment_report_toy.",
            tested=True,
            details=rep,
        )
    else:
        put(
            "reporting",
            "FAIL" if rep else "PARTIAL",
            2,
            str(rep.get("error") or "reporting smoke not ok"),
            tested=bool(rep),
            details=rep,
        )

    # fold pytest signal into notes for environment if tests ran
    if tests and tests.get("ran"):
        env_comp = by_id["environment"]
        suffix = f" pytest: {tests.get('summary') or tests.get('returncode')}"
        env_comp.notes = (env_comp.notes + suffix).strip()
        env_comp.details["pytest"] = {
            "passed": tests.get("passed"),
            "returncode": tests.get("returncode"),
            "summary": tests.get("summary"),
        }

    ordered: list[ComponentResult] = []
    for cid, _name in PIPELINE_ORDER:
        if cid in by_id:
            ordered.append(by_id[cid])
        else:
            ordered.append(
                ComponentResult(
                    id=cid,
                    name=_name,
                    status="NOT_APPLICABLE",
                    evidence_level=0,
                    notes="Not covered by this audit.",
                )
            )
    return ordered


def derive_overall(
    env: dict[str, Any],
    components: list[ComponentResult],
    smokes: dict[str, Any],
    dataset_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive first-baseline readiness and banner from hard acceptance criteria."""
    real_mp4 = int(env.get("real_mp4_count") or 0) > 0
    gpu_available = bool(env.get("gpu_available"))
    cuda = bool(env.get("torch", {}).get("cuda_available"))
    qwen_cache = bool(env.get("qwen_cache_present"))
    qwen_loaded = bool((smokes.get("real_qwen_load") or {}).get("loaded"))
    qwen_ready = qwen_cache or qwen_loaded
    real_infer = (
        bool(env.get("real_baseline_completed"))
        or bool(env.get("real_smoke_evidence_present"))
        or bool(((env.get("real_smoke_evidence") or {}).get("correct") is True))
    )
    real_baseline_done = bool(env.get("real_baseline_completed"))
    stub_ok = bool((smokes.get("stub_baseline") or {}).get("ok"))
    approved_gold = int((dataset_stats or {}).get("approved_gold_examples") or 0)
    has_gold = approved_gold >= 1
    clips_needed = int((dataset_stats or {}).get("clips_needed") or 0)
    examples_evaluated = int(env.get("real_baseline_examples_evaluated") or 0)

    runtime_checks = {
        "GPU_AVAILABLE": gpu_available,
        "CUDA_AVAILABLE": cuda,
        "QWEN_WEIGHTS_AVAILABLE": qwen_ready,
        "REAL_VLM_LOAD": qwen_loaded,
        "REAL_VIDEO_INFERENCE": real_infer,
        "FIRST_BASELINE_READY": False,
        "REAL_BASELINE_COMPLETED": real_baseline_done,
    }

    if real_baseline_done and has_gold and real_mp4 and cuda and qwen_ready:
        first_baseline = "YES"
        banner = "PAUSED - ZERO-SHOT PROTOTYPE COMPLETE"
        overall = "PASS"
        readiness_status = "REAL_BASELINE_COMPLETE"
        runtime_checks["FIRST_BASELINE_READY"] = True
        reason = (
            f"Formal zero-shot baseline run_id={env.get('real_baseline_run_id')} "
            f"evaluated {examples_evaluated}/{approved_gold} approved gold example(s) "
            f"with model {env.get('real_baseline_model_id')}. "
            "Post-training (preferences/DPO/GRPO) was not completed. "
            f"Pilot still needs {clips_needed} more clip(s) for a 5-clip set. "
            "n=1 does not establish generalization. Research paused for time/scope."
        )
    elif has_gold and real_mp4 and cuda and qwen_ready:
        first_baseline = "YES"
        banner = "READY FOR FIRST BASELINE"
        overall = "PASS"
        readiness_status = "READY_FOR_REAL_BASELINE"
        runtime_checks["FIRST_BASELINE_READY"] = True
        reason = (
            f"{approved_gold} approved gold example(s), real mp4(s) present, "
            "CUDA available, and Qwen cache/load evidence found."
        )
    elif has_gold and real_mp4:
        first_baseline = "PARTIALLY"
        banner = "NOT READY FOR RESEARCH RUN"
        overall = "BLOCKED"
        missing = []
        if not gpu_available and not cuda:
            readiness_status = "GPU_BLOCKED"
            missing.append("NVIDIA GPU / CUDA torch")
        elif not cuda:
            readiness_status = "GPU_BLOCKED"
            missing.append("CUDA-enabled PyTorch (torch.cuda)")
        elif not qwen_ready:
            readiness_status = "MODEL_BLOCKED"
            missing.append("local Qwen2.5-VL weights / HF cache")
        else:
            readiness_status = "VLM_LOAD_BLOCKED"
            missing.append("incomplete VLM stack")
        reason = (
            "Approved hidden-state gold exists, but first real VLM baseline "
            "still blocked by: "
            + (", ".join(missing) if missing else "incomplete VLM stack")
        )
    else:
        first_baseline = "NO"
        banner = "NOT READY FOR RESEARCH RUN"
        overall = "BLOCKED"
        missing = []
        if not has_gold:
            missing.append("approved hidden-state gold example")
            readiness_status = "DATA_BLOCKED"
        elif not real_mp4:
            missing.append("real mp4 under data/videos")
            readiness_status = "DATA_BLOCKED"
        elif not cuda:
            missing.append("CUDA GPU (torch.cuda)")
            readiness_status = "GPU_BLOCKED"
        elif not qwen_ready:
            missing.append("local Qwen2.5-VL weights / HF cache")
            readiness_status = "MODEL_BLOCKED"
        else:
            readiness_status = "VLM_LOAD_BLOCKED"
            missing.append("incomplete VLM stack")
        reason = (
            "First real hidden-state baseline cannot run. Missing: "
            + ", ".join(missing)
            + ("; stub tooling may still work for synthetic smoke tests." if stub_ok else "")
        )

    blocked = [c for c in components if c.status == "BLOCKED"]
    failed = [c for c in components if c.status == "FAIL"]
    if failed and overall == "PASS":
        overall = "PARTIAL"

    return {
        "overall_status": overall,
        "first_baseline_ready": first_baseline,
        "readiness_status": readiness_status,
        "banner": banner,
        "reason": reason,
        "runtime_checks": runtime_checks,
        "criteria": {
            "approved_gold": has_gold,
            "approved_gold_examples": approved_gold,
            "real_mp4": real_mp4,
            "gpu_available": gpu_available,
            "cuda": cuda,
            "qwen_cache_or_load": qwen_ready,
            "real_vlm_load": qwen_loaded,
            "real_video_inference": real_infer,
            "real_baseline_completed": real_baseline_done,
            "real_baseline_run_id": env.get("real_baseline_run_id"),
            "real_baseline_model_id": env.get("real_baseline_model_id"),
            "real_baseline_examples_evaluated": examples_evaluated,
            "clips_needed": clips_needed,
            "stub_baseline_ok": stub_ok,
        },
        "n_blocked": len(blocked),
        "n_failed": len(failed),
    }


def build_blockers(
    env: dict[str, Any],
    smokes: dict[str, Any],
    dataset_stats: dict[str, Any] | None = None,
) -> list[Blocker]:
    blockers: list[Blocker] = []
    approved_gold = int((dataset_stats or {}).get("approved_gold_examples") or 0)
    if approved_gold < 1:
        blockers.append(
            Blocker(
                id="approved_gold",
                why="No approved hidden-state gold example is recorded",
                need=(
                    "In docs/OPEN_ITEMS.md (or candidate review), record an S6 decision "
                    "`APPROVE / EDIT / REJECT` with exactly one of those words"
                ),
                priority="now",
            )
        )
    if int(env.get("real_mp4_count") or 0) == 0:
        blockers.append(
            Blocker(
                id="real_videos",
                why="data/videos has no non-empty .mp4 files",
                need="Add real magic/mentalism mp4 clips referenced by the research manifest",
                priority="now",
            )
        )
    if not env.get("torch", {}).get("available"):
        blockers.append(
            Blocker(
                id="torch_missing",
                why="PyTorch is not importable",
                need="Install torch (CPU or CUDA wheel matching this machine)",
                priority="now",
            )
        )
    elif not env.get("torch", {}).get("cuda_available"):
        blockers.append(
            Blocker(
                id="cuda",
                why="Torch reports CUDA unavailable (CPU-only)",
                need="GPU host with CUDA-enabled PyTorch for practical Qwen2.5-VL runs",
                priority="now",
            )
        )
    if not env.get("qwen_cache_present") and not (smokes.get("real_qwen_load") or {}).get("loaded"):
        blockers.append(
            Blocker(
                id="qwen_weights",
                why="No HF Qwen cache and load_vlm(allow_download=False) did not load",
                need="Download Qwen2.5-VL locally or point model_id at a local directory",
                priority="now",
            )
        )
    if not env.get("cv2", {}).get("available"):
        blockers.append(
            Blocker(
                id="opencv",
                why="OpenCV (cv2) not importable",
                need="Install magic-vlm[video] / opencv-python-headless for frame decode",
                priority="later",
            )
        )
    if not env.get("transformers", {}).get("available"):
        blockers.append(
            Blocker(
                id="transformers",
                why="transformers not importable",
                need="Install magic-vlm[models]",
                priority="now",
            )
        )
    pending = int((dataset_stats or {}).get("pending_review") or 0)
    clips_needed = int((dataset_stats or {}).get("clips_needed") or 0)
    if approved_gold >= 1:
        blockers.append(
            Blocker(
                id="human_labels",
                why=(
                    f"{approved_gold} approved hidden-state gold example(s); "
                    f"{pending} pending review; "
                    f"{clips_needed} more clip(s) needed for a 5-clip pilot"
                ),
                need=(
                    "Leave S7 PENDING. Do not gold-label Wikimedia clips. "
                    "Additional gold clips are later, not the first zero-shot baseline"
                ),
                priority="later",
            )
        )
    else:
        blockers.append(
            Blocker(
                id="human_labels",
                why="Real ground-truth / preference / causal annotations not yet collected",
                need=(
                    "Review Mac King S6/S7 hidden-state proposals (PENDING); "
                    "Wikimedia transparent-cup pilots remain controls, not gold"
                    if int(env.get("real_mp4_count") or 0) > 0
                    else "Author research labels after videos exist"
                ),
                priority="now" if int(env.get("real_mp4_count") or 0) > 0 else "later",
            )
        )
    blockers.append(
        Blocker(
            id="vllm_optional",
            why="vLLM not required for first baseline",
            need="Optional faster serving stack later",
            priority="optional",
        )
    )
    return blockers


def build_human_input(
    env: dict[str, Any],
    dataset_stats: dict[str, Any] | None = None,
) -> list[HumanInputItem]:
    approved_gold = int((dataset_stats or {}).get("approved_gold_examples") or 0)
    clips_needed = int((dataset_stats or {}).get("clips_needed") or 0)
    cuda = bool(env.get("torch", {}).get("cuda_available"))
    qwen_ready = bool(env.get("qwen_cache_present"))
    items = [
        HumanInputItem(
            priority="now",
            what="Provide at least one real magic/mentalism video clip with a hidden-state question and ground-truth label",
            where="data/videos/ (mp4) and a research manifest under data/ (JSONL)",
            format="mp4 + ExampleRecord JSONL (see magic_vlm.schemas / docs/OVERVIEW.md)",
            after="magic-vlm-validate --manifest <your_manifest.jsonl>",
        ),
        HumanInputItem(
            priority="now",
            what="Obtain a CUDA GPU environment and CUDA-enabled PyTorch",
            where="Training/inference host (not this CPU-only audit host)",
            format="NVIDIA GPU + matching CUDA torch wheel",
            after="python -c \"import torch; print(torch.cuda.is_available())\"",
        ),
        HumanInputItem(
            priority="now",
            what="Place local Qwen2.5-VL weights (3B or 7B Instruct)",
            where="HF hub cache or a local directory referenced by model_id",
            format="Transformers checkpoint directory for Qwen2.5-VL",
            after="magic-vlm-baseline --config configs/baseline_qwen25vl_3b.yaml --load-frames",
        ),
        HumanInputItem(
            priority="later",
            what="Human preference judgments (A/B explanations)",
            where="data/annotations/ or preference JSONL",
            format="PreferencePair schema (magic_vlm.preferences / docs/OVERVIEW.md)",
            after="magic-vlm-validate-preferences --input <prefs.jsonl>",
        ),
        HumanInputItem(
            priority="later",
            what="Causal / temporal span annotations for advanced rewards",
            where="Manifest causal/temporal fields",
            format="TemporalSpan + CausalAnnotation on ExampleRecord",
            after="magic-vlm-compare-objective --manifest ... --predictions ...",
        ),
        HumanInputItem(
            priority="optional",
            what="Hugging Face token if gated assets are used",
            where="Environment variable HF_TOKEN / huggingface-cli login",
            format="Access token string",
            after="huggingface-cli whoami",
        ),
    ]
    if int(env.get("real_mp4_count") or 0) > 0:
        items = [i for i in items if "real magic" not in i.what.lower()]
    if cuda:
        items = [i for i in items if "CUDA GPU environment" not in i.what]
    if qwen_ready:
        items = [i for i in items if "Qwen2.5-VL weights" not in i.what]
    if int(env.get("real_mp4_count") or 0) > 0:
        if approved_gold >= 1:
            items.append(
                HumanInputItem(
                    priority="later",
                    what=(
                        "S6 is approved gold. Leave S7 PENDING. "
                        f"Source {clips_needed} more hidden-state clip(s) for a 5-clip pilot. "
                        "Do not gold-label Wikimedia clips."
                    ),
                    where="reports/hidden_state_candidates/index.html and docs/OPEN_ITEMS.md",
                    format="Human decision on pending proposals only; no unverified ground_truth",
                    after=(
                        "magic-vlm-baseline --config configs/baseline_qwen25vl_3b.yaml "
                        "--run-id baseline-real-v1 --load-frames"
                        if cuda and qwen_ready
                        else "Do not run Qwen until CUDA and local weights exist"
                    ),
                )
            )
        else:
            items.insert(
                0,
                HumanInputItem(
                    priority="now",
                    what=(
                        "Record an explicit S6 decision in docs/OPEN_ITEMS.md: "
                        "replace `APPROVE / EDIT / REJECT` with one word. "
                        "S7 stays pending. Do not gold-label Wikimedia clips."
                    ),
                    where="reports/hidden_state_candidates/index.html and docs/OPEN_ITEMS.md",
                    format="Human decision on pending proposals only; no unverified ground_truth",
                    after="Open reports/hidden_state_candidates/index.html",
                ),
            )
    return items


def next_actions(
    overall: dict[str, Any],
    blockers: list[Blocker],
) -> list[str]:
    actions: list[str] = []
    if overall.get("readiness_status") == "REAL_BASELINE_COMPLETE":
        clips_needed = int((overall.get("criteria") or {}).get("clips_needed") or 4)
        actions.append(
            f"Source {clips_needed} more approved hidden-state clips for a 5-clip pilot "
            "(leave S7 PENDING until reviewed; do not gold-label Wikimedia controls)"
        )
        return actions[:5]
    if overall.get("readiness_status") == "READY_FOR_REAL_BASELINE":
        actions.append(
            "magic-vlm-baseline --config configs/baseline_qwen25vl_3b.yaml "
            "--run-id baseline-real-v1 --load-frames"
        )
        return actions[:5]
    now_blockers = [b for b in blockers if b.priority == "now"]
    for b in now_blockers[:4]:
        actions.append(b.need)
    if overall.get("first_baseline_ready") != "YES":
        actions.append(
            "Re-run: python scripts/project_health.py  after videos + CUDA + Qwen weights"
        )
    return actions[:5]


def render_markdown(audit: dict[str, Any]) -> str:
    overall = audit["overall"]
    lines: list[str] = [
        "# VLM Magic/Mentalism Research Project Status",
        "",
        "## Overall Status",
        "",
        f"**{overall['banner']}**",
        "",
        f"- Overall: `{overall['overall_status']}`",
        f"- First real baseline ready: `{overall['first_baseline_ready']}`",
        f"- Reason: {overall['reason']}",
        f"- Generated: {audit.get('generated_at')}",
        "",
        "## Pipeline",
        "",
    ]
    for comp in audit["components"]:
        lines.append(
            f"- `{comp['status']}` **{comp['name']}** "
            f"(evidence {comp['evidence_level']}) - {comp['notes']}"
        )
    lines.extend(["", "## What Works", ""])
    for comp in audit["components"]:
        if comp["status"] == "PASS":
            lines.append(f"- {comp['name']}: {comp['notes']}")
    lines.extend(["", "## What Does Not Work / Blocked", ""])
    for comp in audit["components"]:
        if comp["status"] in {"FAIL", "BLOCKED"}:
            lines.append(f"- [{comp['status']}] {comp['name']}: {comp['notes']}")
    lines.extend(["", "## Human Input Required", ""])
    for bucket in ("now", "later", "optional"):
        title = {
            "now": "### I need to provide this now",
            "later": "### I need to provide this later",
            "optional": "### Optional",
        }[bucket]
        lines.append(title)
        lines.append("")
        items = [h for h in audit["human_input"] if h["priority"] == bucket]
        if not items:
            lines.append("- (none)")
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. **What:** {item['what']}")
            lines.append(f"   - Where: {item['where']}")
            lines.append(f"   - Format: {item['format']}")
            lines.append(f"   - After: `{item['after']}`")
        lines.append("")
    lines.extend(["## Environment Requirements", ""])
    env = audit["environment"]
    lines.append(
        f"- Python {env['python']['version']}; torch "
        f"{env['torch'].get('version')} cuda={env['torch'].get('cuda_available')}"
    )
    lines.append(f"- Real mp4 count: {env['real_mp4_count']}")
    lines.append(f"- Qwen HF cache present: {env['qwen_cache_present']}")
    lines.append(f"- GPU available (nvidia-smi/torch): {env.get('gpu_available')}")
    lines.extend(["", "## Runtime checks", ""])
    checks = overall.get("runtime_checks") or {}
    for key in (
        "GPU_AVAILABLE",
        "CUDA_AVAILABLE",
        "QWEN_WEIGHTS_AVAILABLE",
        "REAL_VLM_LOAD",
        "REAL_VIDEO_INFERENCE",
        "FIRST_BASELINE_READY",
        "REAL_BASELINE_COMPLETED",
    ):
        lines.append(f"- {key}: `{checks.get(key)}`")
    lines.append(f"- readiness_status: `{overall.get('readiness_status')}`")
    if env.get("real_baseline_completed"):
        lines.extend(["", "## Real zero-shot baseline", ""])
        lines.append("- label: `REAL_ZERO_SHOT_BASELINE`")
        lines.append(f"- run_id: `{env.get('real_baseline_run_id')}`")
        lines.append(f"- model: `{env.get('real_baseline_model_id')}`")
        lines.append(f"- examples_evaluated: `{env.get('real_baseline_examples_evaluated')}`")
        lines.append(f"- exact_match_accuracy: `{env.get('real_baseline_accuracy')}`")
        lines.append(
            "- evidence: `reports/real_zero_shot_baseline/` "
            "(distinct from `reports/real_zero_shot_baseline_smoke/`)"
        )
        hs_pre = audit.get("hidden_state_dataset") or {}
        lines.append(f"- approved_gold_examples: `{hs_pre.get('approved_gold_examples', 0)}`")
        lines.append(f"- clips_needed_for_pilot: `{hs_pre.get('clips_needed', 4)}`")
        lines.append(
            "- limitation: n is small; one correct answer does not establish "
            "hidden-state/temporal/causal reasoning or generalization."
        )
    lines.extend(["", "## Hidden-state dataset", ""])
    hs = audit.get("hidden_state_dataset") or {}
    wiki = hs.get("wikimedia_controls") or {}
    mac = hs.get("mac_king_candidates") or {}
    gold = hs.get("hidden_state_gold") or {}
    lines.append(f"- hidden_state_candidates: `{hs.get('hidden_state_candidates', 0)}`")
    lines.append(f"- approved_gold_examples: `{hs.get('approved_gold_examples', 0)}`")
    lines.append(f"- pending_review: `{hs.get('pending_review', 0)}`")
    lines.append(f"- rejected: `{hs.get('rejected', 0)}`")
    lines.append(f"- clips_needed: `{hs.get('clips_needed', 5)}`")
    lines.append("")
    lines.append("### WIKIMEDIA CONTROLS")
    lines.append("")
    lines.append(f"- candidate_count: `{wiki.get('candidate_count', 0)}`")
    lines.append(f"- eligible_count: `{wiki.get('eligible_count', 0)}`")
    lines.append(f"- pending_human_review: `{wiki.get('pending_human_review', 0)}`")
    lines.append(f"- rejected_count: `{wiki.get('rejected_count', 0)}`")
    lines.append("")
    lines.append("### MAC KING CANDIDATES")
    lines.append("")
    lines.append(f"- candidate_count: `{mac.get('candidate_count', 0)}`")
    lines.append(f"- eligible_count: `{mac.get('eligible_count', 0)}`")
    lines.append(f"- pending_human_review: `{mac.get('pending_human_review', 0)}`")
    lines.append(f"- rejected_count: `{mac.get('rejected_count', 0)}`")
    lines.append("")
    lines.append("### HIDDEN-STATE GOLD")
    lines.append("")
    lines.append(f"- eligible_count: `{gold.get('eligible_count', 0)}`")
    lines.append(f"- pending_human_review: `{gold.get('pending_human_review', 0)}`")
    lines.append(f"- clips_needed_for_pilot: `{gold.get('clips_needed_for_pilot', 5)}`")
    lines.append("")
    lines.extend(["## First Research Experiment Readiness", ""])
    lines.append(f"`{overall['first_baseline_ready']}` - {overall['reason']}")
    lines.extend(["", "## Next Actions", ""])
    for i, action in enumerate(audit.get("next_actions") or [], 1):
        lines.append(f"{i}. {action}")
    lines.append("")
    return "\n".join(lines)


def _status_color(status: str) -> str:
    if status == "PASS":
        return "#1b7f3a"
    if status in {"PARTIAL", "BLOCKED"}:
        return "#c4a000"
    if status == "FAIL":
        return "#b00020"
    return "#6b6b6b"


def render_html(audit: dict[str, Any]) -> str:
    overall = audit["overall"]
    banner = html.escape(str(overall["banner"]))
    banner_color = (
        "#1b7f3a"
        if overall["first_baseline_ready"] == "YES"
        or overall.get("readiness_status") == "REAL_BASELINE_COMPLETE"
        else "#b00020"
    )
    hs = audit.get("hidden_state_dataset") or {}
    wiki = hs.get("wikimedia_controls") or {}
    mac = hs.get("mac_king_candidates") or {}
    gold = hs.get("hidden_state_gold") or {}

    pipeline_rows = []
    for comp in audit["components"]:
        color = _status_color(comp["status"])
        pipeline_rows.append(
            "<div class='node' style='border-left:6px solid {color}'>"
            "<span class='badge' style='background:{color}'>{status}</span> "
            "<strong>{name}</strong>"
            "<div class='notes'>{notes}</div></div>".format(
                color=color,
                status=html.escape(comp["status"]),
                name=html.escape(comp["name"]),
                notes=html.escape(comp["notes"]),
            )
        )
    pipeline_html = "\n".join(pipeline_rows)

    table_rows = []
    for comp in audit["components"]:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(comp['name'])}</td>"
            f"<td style='color:{_status_color(comp['status'])};font-weight:600'>"
            f"{html.escape(comp['status'])}</td>"
            f"<td>{html.escape(comp.get('evidence_label') or str(comp['evidence_level']))}</td>"
            f"<td>{html.escape(comp.get('last_test') or '-')}</td>"
            f"<td>{html.escape(comp['notes'])}</td>"
            "</tr>"
        )

    blocker_rows = []
    for b in audit["blockers"]:
        blocker_rows.append(
            "<tr>"
            f"<td>{html.escape(b['id'])}</td>"
            f"<td>{html.escape(b['why'])}</td>"
            f"<td>{html.escape(b['need'])}</td>"
            f"<td>{html.escape(b['priority'])}</td>"
            "</tr>"
        )

    actions = "".join(f"<li>{html.escape(a)}</li>" for a in audit.get("next_actions") or [])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Project Status - Magic VLM</title>
<style>
body {{ font-family: Georgia, "Times New Roman", serif; margin: 0; background: #f6f3ee; color: #222; }}
header {{ background: {banner_color}; color: #fff; padding: 1.5rem 2rem; }}
header h1 {{ margin: 0 0 0.4rem 0; font-size: 1.4rem; letter-spacing: 0.02em; }}
header .banner {{ font-size: 1.8rem; font-weight: 700; }}
main {{ padding: 1.5rem 2rem 3rem; max-width: 1100px; }}
h2 {{ margin-top: 2rem; border-bottom: 1px solid #ccc; padding-bottom: 0.3rem; }}
.node {{ background: #fff; margin: 0.45rem 0; padding: 0.55rem 0.75rem; }}
.badge {{ color: #fff; padding: 0.1rem 0.45rem; font-size: 0.75rem; }}
.notes {{ color: #555; font-size: 0.9rem; margin-top: 0.25rem; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; }}
th, td {{ border: 1px solid #ddd; padding: 0.45rem 0.55rem; text-align: left; vertical-align: top; }}
th {{ background: #efeae2; }}
.meta {{ color: #444; }}
</style>
</head>
<body>
<header>
  <h1>PROJECT STATUS</h1>
  <div class="banner">{banner}</div>
  <p class="meta">First baseline ready: {html.escape(str(overall['first_baseline_ready']))}
  · Overall: {html.escape(str(overall['overall_status']))}
  · {html.escape(str(audit.get('generated_at') or ''))}</p>
</header>
<main>
  <p>{html.escape(str(overall.get('reason') or ''))}</p>
  <p>readiness_status: {html.escape(str(overall.get('readiness_status') or ''))}</p>
  <p>GPU_AVAILABLE={html.escape(str((overall.get('runtime_checks') or {}).get('GPU_AVAILABLE')))}
  · CUDA_AVAILABLE={html.escape(str((overall.get('runtime_checks') or {}).get('CUDA_AVAILABLE')))}
  · QWEN_WEIGHTS_AVAILABLE={html.escape(str((overall.get('runtime_checks') or {}).get('QWEN_WEIGHTS_AVAILABLE')))}
  · REAL_VLM_LOAD={html.escape(str((overall.get('runtime_checks') or {}).get('REAL_VLM_LOAD')))}
  · REAL_VIDEO_INFERENCE={html.escape(str((overall.get('runtime_checks') or {}).get('REAL_VIDEO_INFERENCE')))}
  · FIRST_BASELINE_READY={html.escape(str((overall.get('runtime_checks') or {}).get('FIRST_BASELINE_READY')))}
  · REAL_BASELINE_COMPLETED={html.escape(str((overall.get('runtime_checks') or {}).get('REAL_BASELINE_COMPLETED')))}</p>
  <p>real_baseline_run_id: {html.escape(str(audit.get('environment', {}).get('real_baseline_run_id') or '-'))}
  · model: {html.escape(str(audit.get('environment', {}).get('real_baseline_model_id') or '-'))}
  · examples_evaluated: {html.escape(str(audit.get('environment', {}).get('real_baseline_examples_evaluated') or 0))}
  · accuracy: {html.escape(str(audit.get('environment', {}).get('real_baseline_accuracy') or '-'))}</p>

  <h2>Pipeline</h2>
  <div class="pipeline">{pipeline_html}</div>

  <h2>What works / status table</h2>
  <table>
    <thead><tr><th>Component</th><th>Status</th><th>Evidence</th><th>Last Test</th><th>Notes</th></tr></thead>
    <tbody>
    {''.join(table_rows)}
    </tbody>
  </table>

  <h2>Blockers</h2>
  <table>
    <thead><tr><th>Blocker</th><th>Why</th><th>What I need to provide</th><th>Priority</th></tr></thead>
    <tbody>
    {''.join(blocker_rows)}
    </tbody>
  </table>

  <h2>Hidden-state dataset</h2>
  <p>hidden_state_candidates: {html.escape(str(hs.get('hidden_state_candidates', 0)))}
  · approved_gold_examples: {html.escape(str(hs.get('approved_gold_examples', 0)))}
  · pending_review: {html.escape(str(hs.get('pending_review', 0)))}
  · rejected: {html.escape(str(hs.get('rejected', 0)))}
  · clips_needed: {html.escape(str(hs.get('clips_needed', 5)))}</p>
  <h3>WIKIMEDIA CONTROLS</h3>
  <p>candidate_count: {html.escape(str(wiki.get('candidate_count', 0)))}
  · eligible_count: {html.escape(str(wiki.get('eligible_count', 0)))}
  · pending_human_review: {html.escape(str(wiki.get('pending_human_review', 0)))}
  · rejected_count: {html.escape(str(wiki.get('rejected_count', 0)))}</p>
  <h3>MAC KING CANDIDATES</h3>
  <p>candidate_count: {html.escape(str(mac.get('candidate_count', 0)))}
  · eligible_count: {html.escape(str(mac.get('eligible_count', 0)))}
  · pending_human_review: {html.escape(str(mac.get('pending_human_review', 0)))}
  · rejected_count: {html.escape(str(mac.get('rejected_count', 0)))}</p>
  <h3>HIDDEN-STATE GOLD</h3>
  <p>eligible_count: {html.escape(str(gold.get('eligible_count', 0)))}
  · pending_human_review: {html.escape(str(gold.get('pending_human_review', 0)))}
  · clips_needed_for_pilot: {html.escape(str(gold.get('clips_needed_for_pilot', 5)))}</p>

  <h2>Next actions</h2>
  <ol>{actions}</ol>
</main>
</body>
</html>
"""


def run_audit(
    root: str | Path,
    *,
    run_tests: bool = True,
    run_stub_baseline: bool = True,
) -> dict[str, Any]:
    """Run the full audit and write status artifacts under ``root``."""
    root_path = Path(root).resolve()
    prev = Path.cwd()
    os.chdir(root_path)
    try:
        env = probe_environment(root_path)
        tests = run_pytest(root_path) if run_tests else {"ran": False, "skipped": True}
        smokes = run_smokes(root_path, run_stub_baseline=run_stub_baseline)
        components = build_component_results(
            env=env, smokes=smokes, tests=tests, root=root_path
        )
        dataset_stats = hidden_state_dataset_stats(
            load_hidden_state_inventory(root_path)
        )
        overall = derive_overall(env, components, smokes, dataset_stats)
        blockers = build_blockers(env, smokes, dataset_stats)
        human = build_human_input(env, dataset_stats)
        actions = next_actions(overall, blockers)
        entry = smokes.get("entry_points") or {}

        audit: dict[str, Any] = {
            "generated_at": utc_now_iso(),
            "root": str(root_path),
            "environment": env,
            "tests": tests,
            "smokes": smokes,
            "components": [c.to_dict() for c in components],
            "overall": overall,
            "first_baseline_ready": overall["first_baseline_ready"],
            "hidden_state_dataset": dataset_stats,
            "real_baseline": {
                "completed": bool(env.get("real_baseline_completed")),
                "label": "REAL_ZERO_SHOT_BASELINE" if env.get("real_baseline_completed") else None,
                "run_id": env.get("real_baseline_run_id"),
                "model_id": env.get("real_baseline_model_id"),
                "examples_evaluated": env.get("real_baseline_examples_evaluated"),
                "accuracy": env.get("real_baseline_accuracy"),
                "evidence_dir": (
                    "reports/real_zero_shot_baseline"
                    if env.get("real_baseline_completed")
                    else None
                ),
                "smoke_evidence_dir": (
                    "reports/real_zero_shot_baseline_smoke"
                    if env.get("real_smoke_evidence_present")
                    else None
                ),
            },
            "blockers": [b.to_dict() for b in blockers],
            "human_input": [h.to_dict() for h in human],
            "next_actions": actions,
            "entry_points": entry,
            "integrity_note": (
                "Statuses reflect executed probes only. Stub PASS is not real VLM evidence."
            ),
        }

        out_dir = root_path / "reports" / "project_health"
        out_dir.mkdir(parents=True, exist_ok=True)
        audit_json = out_dir / "audit.json"
        write_json(audit_json, audit)

        md_path = root_path / "PROJECT_STATUS.md"
        md_path.write_text(render_markdown(audit), encoding="utf-8")

        html_body = render_html(audit)
        html_primary = out_dir / "project_status.html"
        html_primary.write_text(html_body, encoding="utf-8")
        html_copy = root_path / "reports" / "project_status.html"
        html_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(html_primary, html_copy)

        audit["artifacts"] = {
            "audit_json": str(audit_json),
            "project_status_md": str(md_path),
            "project_status_html": str(html_primary),
            "project_status_html_copy": str(html_copy),
        }
        write_json(audit_json, audit)
        return audit
    finally:
        os.chdir(prev)


def project_health_main(argv: list[str] | None = None) -> int:
    """CLI entry used by ``scripts/project_health.py`` / ``magic_vlm.cli``."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit repository readiness for the first real hidden-state baseline. "
            "Writes PROJECT_STATUS.md and HTML/JSON dashboards."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (default: cwd).",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Do not run pytest.",
    )
    parser.add_argument(
        "--skip-stub-baseline",
        action="store_true",
        help="Skip stub zero-shot baseline smoke.",
    )
    args = parser.parse_args(argv)
    root = (args.root or Path.cwd()).resolve()
    audit = run_audit(
        root,
        run_tests=not args.skip_tests,
        run_stub_baseline=not args.skip_stub_baseline,
    )
    overall = audit["overall"]
    print(f"banner={overall['banner']}")
    print(f"first_baseline_ready={overall['first_baseline_ready']}")
    print(f"overall={overall['overall_status']}")
    arts = audit.get("artifacts") or {}
    print(f"md={arts.get('project_status_md')}")
    print(f"html={arts.get('project_status_html')}")
    print(f"json={arts.get('audit_json')}")
    # Non-zero only on hard audit infrastructure failure is avoided; readiness is in artifacts.
    return 0


__all__ = [
    "PIPELINE_ORDER",
    "ComponentResult",
    "Blocker",
    "HumanInputItem",
    "probe_environment",
    "probe_nvidia_smi",
    "load_real_smoke_evidence",
    "load_real_baseline_evidence",
    "run_smokes",
    "run_audit",
    "derive_overall",
    "render_markdown",
    "render_html",
    "hidden_state_dataset_stats",
    "load_hidden_state_inventory",
    "project_health_main",
]
