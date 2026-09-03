from __future__ import annotations

import json
from pathlib import Path

from magic_vlm.dataset import load_manifest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "data" / "examples" / "wikimedia_pilot_manifest.template.jsonl"
REVIEW = ROOT / "data" / "examples" / "wikimedia_pilot_review.jsonl"
PROPOSALS = ROOT / "data" / "examples" / "wikimedia_pilot_annotation_proposals.json"
HTML = ROOT / "reports" / "wikimedia_clip_review" / "index.html"
STILLS = ROOT / "reports" / "wikimedia_clip_review" / "stills"
FILL = "HUMAN_FILL_REQUIRED"


def test_original_template_not_replaced_with_gold_labels() -> None:
    for record in load_manifest(TEMPLATE):
        assert record.question == FILL
        assert record.ground_truth == FILL


def test_review_manifest_keeps_unresolved_gold() -> None:
    records = load_manifest(REVIEW)
    assert len(records) == 5
    for record in records:
        assert record.question == FILL
        assert record.ground_truth == FILL
        assert record.metadata.get("human_approval") == "PENDING"
        proposal = record.metadata["annotation_proposal"]
        assert proposal["human_decision"] == "PENDING"
        assert proposal["human_review_required"] is True
        assert proposal["candidate_task_type"] == "NOT_SUITABLE_FOR_HIDDEN_STATE_TASK"


def test_annotation_proposals_are_pending() -> None:
    payload = json.loads(PROPOSALS.read_text(encoding="utf-8"))
    assert payload["human_approval"] == "PENDING"
    assert payload["best_first_baseline_clip"]["clip_id"] == "peerj_01_19_s006"
    clip_ids = {row["clip_id"] for row in payload["clips"]}
    assert clip_ids == {
        "peerj_01_19_s003",
        "peerj_01_19_s004",
        "peerj_01_19_s005",
        "peerj_01_19_s006",
        "peerj_01_19_s007",
    }


def test_review_html_and_stills_exist() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert "HUMAN APPROVAL: PENDING" in html
    assert "[APPROVE]" in html
    assert STILLS.is_dir()
    expected = [
        "s003_start.jpg",
        "s004_start.jpg",
        "s005_third_cup.jpg",
        "s006_four_balls.jpg",
        "s007_third_cup.jpg",
    ]
    for name in expected:
        assert (STILLS / name).is_file()
        assert (STILLS / name).stat().st_size > 0
    for clip in ("s003", "s004", "s005", "s006", "s007"):
        assert f"{clip}.mp4" in html
        assert (STILLS / f"{clip}_start.jpg").is_file()
