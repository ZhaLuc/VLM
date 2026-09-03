"""Tests for project health audit (no fake real-baseline claims)."""

from __future__ import annotations

import json
from pathlib import Path

from magic_vlm.project_health import derive_overall, probe_environment, render_html, run_audit


def test_probe_environment_has_python(tmp_path: Path) -> None:
    env = probe_environment(tmp_path)
    assert "python" in env
    assert env["python"]["version"]
    assert "real_mp4_count" in env
    assert "torch" in env
    assert "qwen_cache_present" in env


def test_run_audit_quick_writes_artifacts() -> None:
    repo = Path(__file__).resolve().parents[1]
    audit = run_audit(repo, run_tests=False, run_stub_baseline=False)

    assert (repo / "PROJECT_STATUS.md").exists()
    assert (repo / "reports" / "project_health" / "project_status.html").exists()
    assert (repo / "reports" / "project_status.html").exists()
    audit_path = repo / "reports" / "project_health" / "audit.json"
    assert audit_path.exists()

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    ready = payload["first_baseline_ready"]
    assert ready in {"NO", "PARTIALLY", "YES"}
    assert audit["first_baseline_ready"] == ready

    real_mp4 = int(payload["environment"]["real_mp4_count"])
    gold = int(payload["hidden_state_dataset"]["approved_gold_examples"])
    if real_mp4 == 0 or gold == 0:
        assert ready != "YES"
        assert "READY FOR FIRST BASELINE" not in payload["overall"]["banner"]


def test_derive_overall_requires_approved_gold() -> None:
    env = {
        "real_mp4_count": 12,
        "gpu_available": True,
        "torch": {"cuda_available": True},
        "qwen_cache_present": True,
    }
    smokes = {"real_qwen_load": {"loaded": True}, "stub_baseline": {"ok": True}}
    blocked = derive_overall(env, [], smokes, {"approved_gold_examples": 0})
    assert blocked["first_baseline_ready"] == "NO"
    assert blocked["readiness_status"] == "DATA_BLOCKED"
    ready = derive_overall(env, [], smokes, {"approved_gold_examples": 1})
    assert ready["first_baseline_ready"] == "YES"
    assert ready["readiness_status"] == "READY_FOR_REAL_BASELINE"
    assert ready["runtime_checks"]["FIRST_BASELINE_READY"] is True


def test_derive_overall_gpu_blocked_when_no_cuda() -> None:
    env = {
        "real_mp4_count": 12,
        "gpu_available": False,
        "torch": {"cuda_available": False},
        "qwen_cache_present": True,
    }
    smokes = {"real_qwen_load": {"loaded": False}, "stub_baseline": {"ok": True}}
    out = derive_overall(env, [], smokes, {"approved_gold_examples": 1})
    assert out["first_baseline_ready"] == "PARTIALLY"
    assert out["readiness_status"] == "GPU_BLOCKED"


def test_derive_overall_marks_real_baseline_complete() -> None:
    env = {
        "real_mp4_count": 12,
        "gpu_available": True,
        "torch": {"cuda_available": True},
        "qwen_cache_present": True,
        "real_baseline_completed": True,
        "real_baseline_run_id": "baseline-real-v1",
        "real_baseline_model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "real_baseline_examples_evaluated": 1,
        "real_baseline_accuracy": 1.0,
    }
    smokes = {"real_qwen_load": {"loaded": True}, "stub_baseline": {"ok": True}}
    out = derive_overall(env, [], smokes, {"approved_gold_examples": 1, "clips_needed": 4})
    assert out["readiness_status"] == "REAL_BASELINE_COMPLETE"
    assert out["runtime_checks"]["REAL_BASELINE_COMPLETED"] is True
    assert out["banner"] == "PAUSED - ZERO-SHOT PROTOTYPE COMPLETE"


def test_html_contains_status_banner() -> None:
    audit = {
        "generated_at": "2026-09-02T00:00:00+00:00",
        "overall": {
            "banner": "NOT READY FOR RESEARCH RUN",
            "first_baseline_ready": "NO",
            "overall_status": "BLOCKED",
            "reason": "blocked for test",
        },
        "components": [
            {
                "id": "environment",
                "name": "Environment setup",
                "status": "BLOCKED",
                "evidence_level": 3,
                "evidence_label": "LEVEL 3 - INTEGRATION/SMOKE TEST",
                "notes": "CPU-only",
                "last_test": None,
            }
        ],
        "blockers": [
            {
                "id": "cuda",
                "why": "no cuda",
                "need": "gpu",
                "priority": "now",
            }
        ],
        "next_actions": ["Add real videos."],
    }
    body = render_html(audit)
    assert "NOT READY FOR RESEARCH RUN" in body
    assert "PROJECT STATUS" in body
    assert "Pipeline" in body


def test_html_and_markdown_include_hidden_state_collections() -> None:
    repo = Path(__file__).resolve().parents[1]
    audit = run_audit(repo, run_tests=False, run_stub_baseline=False)
    md = (repo / "PROJECT_STATUS.md").read_text(encoding="utf-8")
    html_body = (repo / "reports" / "project_status.html").read_text(encoding="utf-8")
    for blob in (md, html_body):
        assert "hidden_state_candidates" in blob
        assert "approved_gold_examples" in blob
        assert "pending_review" in blob
        assert "clips_needed" in blob
        assert "WIKIMEDIA CONTROLS" in blob
        assert "MAC KING CANDIDATES" in blob
        assert "HIDDEN-STATE GOLD" in blob
    hs = audit["hidden_state_dataset"]
    assert hs["approved_gold_examples"] == 1
    assert hs["pending_review"] == 1
    assert hs["hidden_state_candidates"] == 7
    assert hs["clips_needed"] == 4
    assert hs["wikimedia_controls"]["candidate_count"] == 5
    assert hs["mac_king_candidates"]["candidate_count"] == 7
    assert hs["hidden_state_gold"]["eligible_count"] == 1
    cuda = bool(audit["environment"]["torch"].get("cuda_available"))
    qwen = bool(audit["environment"].get("qwen_cache_present")) or bool(
        (audit.get("smokes") or {}).get("real_qwen_load", {}).get("loaded")
    )
    assert "runtime_checks" in audit["overall"]
    assert "readiness_status" in audit["overall"]
    if cuda and qwen:
        assert audit["first_baseline_ready"] == "YES"
        assert audit["overall"]["readiness_status"] in {
            "READY_FOR_REAL_BASELINE",
            "REAL_BASELINE_COMPLETE",
        }
        if audit["environment"].get("real_baseline_completed"):
            assert audit["overall"]["readiness_status"] == "REAL_BASELINE_COMPLETE"
            assert audit["real_baseline"]["run_id"] == "baseline-real-v1"
            assert audit["real_baseline"]["examples_evaluated"] == 1
            assert "REAL ZERO-SHOT BASELINE COMPLETE" in md or "PAUSED - ZERO-SHOT PROTOTYPE COMPLETE" in md
    else:
        assert audit["first_baseline_ready"] == "PARTIALLY"
