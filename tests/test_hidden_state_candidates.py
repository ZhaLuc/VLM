from __future__ import annotations

import json
from pathlib import Path

from magic_vlm.dataset import load_manifest
from magic_vlm.project_health import build_human_input
from magic_vlm.video import VideoPreprocessConfig, preprocess_video, probe_video

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "data" / "examples" / "hidden_state_candidate_inventory.json"
HTML = ROOT / "reports" / "hidden_state_candidates" / "index.html"
REVIEW = ROOT / "data" / "examples" / "wikimedia_pilot_review.jsonl"
TEMPLATE = ROOT / "data" / "examples" / "wikimedia_pilot_manifest.template.jsonl"
HUMAN = ROOT / "HUMAN_INPUT_REQUIRED.md"
FILL = "HUMAN_FILL_REQUIRED"
ALLOWED_STATUS = {
    "QUALIFIES",
    "QUALIFIES_WITH_HUMAN_REVIEW",
    "NOT_SUITABLE",
    "INSUFFICIENT_EVIDENCE",
    "MISSING_FROM_REPOSITORY",
}
WIKIMEDIA_IDS = {
    "peerj_01_19_s003",
    "peerj_01_19_s004",
    "peerj_01_19_s005",
    "peerj_01_19_s006",
    "peerj_01_19_s007",
}


def test_inventory_has_zero_qualifying_and_no_gold() -> None:
    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    ready = payload["readiness"]
    assert payload["gold_labels_written"] is False
    assert ready["valid_candidates"] == 0
    assert ready["candidates_needing_human_review"] == 0
    assert ready["invalid_candidates"] == 5
    assert ready["additional_clips_needed"] == 5
    assert ready["first_baseline_dataset"] == "NOT READY"
    statuses = {row["status"] for row in payload["candidates"]}
    assert statuses <= ALLOWED_STATUS
    assert "QUALIFIES" not in statuses
    assert "QUALIFIES_WITH_HUMAN_REVIEW" not in statuses
    local = [row for row in payload["candidates"] if row["candidate"] in WIKIMEDIA_IDS]
    assert {row["candidate"] for row in local} == WIKIMEDIA_IDS
    for row in local:
        assert row["status"] == "NOT_SUITABLE"
        assert row["hidden_state_class"] == "NOT_SUITABLE_FOR_HIDDEN_STATE"
        assert row["genuine_occlusion"] == "NO"
        assert row["keep_in_repository"] is True
        assert row["human_review_flag"] is False


def test_review_html_banner_matches_inventory() -> None:
    html = HTML.read_text(encoding="utf-8")
    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    ready = payload["readiness"]
    assert "HIDDEN-STATE DATASET READINESS" in html
    assert f"Valid candidates: {ready['valid_candidates']}" in html
    assert f"Candidates needing human review: {ready['candidates_needing_human_review']}" in html
    assert f"Invalid candidates: {ready['invalid_candidates']}" in html
    assert f"Additional clips needed: {ready['additional_clips_needed']}" in html
    assert "RECOMMENDED NEXT ACTION" in html
    assert "Do not gold-label" in html
    for clip in ("s003", "s004", "s005", "s006", "s007"):
        assert f"{clip}.mp4" in html


def test_pilot_manifests_still_have_no_hidden_state_gold() -> None:
    for path in (TEMPLATE, REVIEW):
        for record in load_manifest(path):
            assert record.question == FILL
            assert record.ground_truth == FILL
            assert record.clip_id in WIKIMEDIA_IDS


def test_human_input_required_is_short_and_actionable() -> None:
    text = HUMAN.read_text(encoding="utf-8")
    assert "Do not replace" in text or "Do not" in text
    assert "5" in text
    assert "HIDDEN_STATE_VIDEO_SOURCING_GUIDE.md" in text
    assert len(text.splitlines()) < 40


def test_health_does_not_ask_to_gold_label_wikimedia() -> None:
    items = build_human_input({"real_mp4_count": 5, "root": str(ROOT)})
    blob = " ".join(item.what for item in items)
    assert "Do not gold-label" in blob
    assert "Replace HUMAN_FILL_REQUIRED fields on the Wikimedia pilot" not in blob


def test_local_wikimedia_clips_still_preprocess() -> None:
    videos = sorted((ROOT / "data" / "videos").glob("*.mp4"))
    if len(videos) < 5:
        return
    cfg = VideoPreprocessConfig(max_frames=8, sample_strategy="uniform")
    for path in videos:
        info = probe_video(path)
        assert info["num_frames"] >= 1
        sampled = preprocess_video(path, config=cfg, load_frames=True)
        assert sampled.frames is not None
        assert len(sampled.frames) == 8
