"""Tests for project health audit (no fake real-baseline claims)."""

from __future__ import annotations

import json
from pathlib import Path

from magic_vlm.project_health import probe_environment, render_html, run_audit


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
    if real_mp4 == 0:
        assert ready != "YES"
        assert "READY FOR FIRST BASELINE" not in payload["overall"]["banner"]


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
