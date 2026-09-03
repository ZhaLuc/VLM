from __future__ import annotations

import hashlib
import json
from pathlib import Path

from magic_vlm.dataset import load_manifest
from magic_vlm.hidden_state_eligibility import evaluate_gold_manifest
from magic_vlm.project_health import build_human_input, hidden_state_dataset_stats, load_hidden_state_inventory
from magic_vlm.video import VideoPreprocessConfig, preprocess_video, probe_video

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "data" / "examples" / "hidden_state_candidate_inventory.json"
HTML = ROOT / "reports" / "hidden_state_candidates" / "index.html"
REVIEW = ROOT / "data" / "examples" / "wikimedia_pilot_review.jsonl"
TEMPLATE = ROOT / "data" / "examples" / "wikimedia_pilot_manifest.template.jsonl"
MAC_REVIEW = ROOT / "data" / "examples" / "mac_king_review.jsonl"
MAC_PROPOSALS = ROOT / "data" / "examples" / "mac_king_annotation_proposals.json"
MAC_PROVENANCE = ROOT / "data" / "provenance" / "mac_king_cui_2011.json"
WIKI_PROVENANCE = ROOT / "data" / "provenance" / "wikimedia_peerj_01_19.json"
HUMAN = ROOT / "HUMAN_INPUT_REQUIRED.md"
STILLS = ROOT / "reports" / "mac_king_clip_review" / "stills"
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
MAC_IDS = {f"mac_king_s00{i}" for i in range(1, 8)}
MOVIES = [f"Movie{i}.MP4" for i in range(1, 8)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_inventory_has_s6_gold_and_s7_pending() -> None:
    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    ready = payload["readiness"]
    assert payload["gold_labels_written"] is True
    assert ready["valid_candidates"] == 1
    assert ready["candidates_needing_human_review"] == 1
    assert ready["invalid_candidates"] == 10
    assert ready["additional_clips_needed"] == 4
    assert ready["first_baseline_dataset"] == "DATA READY"
    assert ready["best_current_hidden_state_candidate"] == "mac_king_s006"
    collections = ready["collections"]
    assert collections["wikimedia_controls"]["candidate_count"] == 5
    assert collections["wikimedia_controls"]["eligible_count"] == 0
    assert collections["wikimedia_controls"]["pending_human_review"] == 0
    assert collections["wikimedia_controls"]["rejected_count"] == 5
    assert collections["mac_king_candidates"]["candidate_count"] == 7
    assert collections["mac_king_candidates"]["eligible_count"] == 1
    assert collections["mac_king_candidates"]["pending_human_review"] == 1
    assert collections["mac_king_candidates"]["rejected_count"] == 5
    assert collections["hidden_state_gold"]["eligible_count"] == 1
    assert collections["hidden_state_gold"]["pending_human_review"] == 1
    assert collections["hidden_state_gold"]["clips_needed_for_pilot"] == 4
    statuses = {row["status"] for row in payload["candidates"]}
    assert statuses <= ALLOWED_STATUS
    assert "QUALIFIES" in statuses
    local = [row for row in payload["candidates"] if row["candidate"] in WIKIMEDIA_IDS]
    assert {row["candidate"] for row in local} == WIKIMEDIA_IDS
    for row in local:
        assert row["status"] == "NOT_SUITABLE"
        assert row["hidden_state_class"] == "NOT_SUITABLE_FOR_HIDDEN_STATE"
        assert row["genuine_occlusion"] == "NO"
        assert row["keep_in_repository"] is True
        assert row["human_review_flag"] is False
        assert row["collection"] == "wikimedia_controls"
    by_id = {row["candidate"]: row for row in payload["candidates"]}
    assert by_id["mac_king_s006"]["status"] == "QUALIFIES"
    assert by_id["mac_king_s006"]["human_review_flag"] is False
    pending = [row for row in payload["candidates"] if row["status"] == "QUALIFIES_WITH_HUMAN_REVIEW"]
    assert {row["candidate"] for row in pending} == {"mac_king_s007"}
    for row in pending:
        assert row["human_review_flag"] is True
        assert row["hidden_state_class"] == "HIDDEN_STATE_CANDIDATE"
        assert row["reveal_status"] == "NO_REVEAL"


def test_review_html_banner_matches_inventory() -> None:
    html = HTML.read_text(encoding="utf-8")
    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    ready = payload["readiness"]
    assert "BEST CURRENT HIDDEN-STATE CANDIDATE" in html
    assert "mac_king_s006" in html
    assert "HOW MANY VALID CLIPS DO WE HAVE?" in html
    assert "Fewer than five" in html
    assert "HIDDEN-STATE DATASET READINESS" in html
    assert f"Valid candidates: {ready['valid_candidates']}" in html
    assert f"Candidates needing human review: {ready['candidates_needing_human_review']}" in html
    assert f"Invalid candidates: {ready['invalid_candidates']}" in html
    assert f"Additional clips needed: {ready['additional_clips_needed']}" in html
    assert "RECOMMENDED NEXT ACTION" in html
    assert "Do not gold-label" in html
    assert "APPROVE / EDIT / REJECT" in html
    assert "Human approval" in html
    assert "PENDING" in html
    for name in MOVIES:
        assert name in html
    for clip in ("s003", "s004", "s005", "s006", "s007"):
        assert f"{clip}.mp4" in html
    for label in ("Occlusion", "Temporal dependence", "Reveal", "Ground truth", "Benchmark status"):
        assert f"<dt>{label}</dt>" in html


def test_pilot_manifests_still_have_no_hidden_state_gold() -> None:
    for path in (TEMPLATE, REVIEW):
        for record in load_manifest(path):
            assert record.question == FILL
            assert record.ground_truth == FILL
            assert record.clip_id in WIKIMEDIA_IDS


def test_mac_king_review_records_s6_approval_and_keeps_s7_pending() -> None:
    records = load_manifest(MAC_REVIEW)
    assert {record.clip_id for record in records} == MAC_IDS
    assert len(records) == 7
    by_id = {record.clip_id: record for record in records}
    s6 = by_id["mac_king_s006"]
    s7 = by_id["mac_king_s007"]
    assert s6.question == "Which hand contains the coin after the apparent transfer?"
    assert s6.ground_truth == "right"
    assert s6.metadata.get("human_approval") == "APPROVED"
    assert s6.metadata.get("human_decision") == "APPROVE"
    assert s6.metadata.get("approved_by") == "human_researcher"
    assert s6.metadata.get("leakage_resolved_by_human") is True
    assert s6.metadata.get("unresolved_leakage_warning") is False
    assert s6.metadata["original_proposal"]["occlusion_status"] == "PARTIAL"
    assert s6.metadata["original_proposal"]["answer_leakage_status"] == "PARTIAL"
    assert s6.metadata["annotation_proposal"]["occlusion_status"] == "PARTIAL"
    assert s6.provenance.created_by == "human_researcher"
    assert s7.question == FILL
    assert s7.ground_truth == FILL
    assert s7.metadata.get("human_approval") == "PENDING"
    assert s7.metadata["annotation_proposal"]["human_decision"] == "PENDING"
    assert s7.metadata["annotation_proposal"]["candidate_ground_truth"] == "left"
    assert s7.metadata["annotation_proposal"]["reveal_status"] == "NO_REVEAL"
    for record in records:
        assert record.provenance.license == "HUMAN_LEGAL_REVIEW_REQUIRED"
        if record.clip_id not in {"mac_king_s006"}:
            assert record.question == FILL
            assert record.ground_truth == FILL
            assert record.metadata.get("human_approval") == "PENDING"
    proposals = json.loads(MAC_PROPOSALS.read_text(encoding="utf-8"))
    assert proposals["human_approval"] == "PARTIAL"
    assert proposals["gold_labels_written"] is True
    clips = {row["clip_id"]: row for row in proposals["clips"]}
    assert clips["mac_king_s006"]["human_decision"] == "APPROVE"
    assert clips["mac_king_s006"]["candidate_ground_truth"] == "right"
    assert clips["mac_king_s007"]["human_approval"] == "PENDING"
    assert clips["mac_king_s007"]["candidate_ground_truth"] == "left"


def test_human_input_required_is_short_and_actionable() -> None:
    text = HUMAN.read_text(encoding="utf-8")
    assert "Movie6.MP4" in text
    assert "Movie7.MP4" in text
    assert "APPROVE / EDIT / REJECT" in text
    assert "Do not gold-label Wikimedia" in text
    assert "5" in text
    assert len(text.splitlines()) < 40


def test_health_does_not_ask_to_gold_label_wikimedia() -> None:
    stats = hidden_state_dataset_stats(load_hidden_state_inventory(ROOT))
    items = build_human_input({"real_mp4_count": 5, "root": str(ROOT)}, stats)
    blob = " ".join(item.what for item in items)
    assert "Do not gold-label" in blob
    assert "Leave S7 PENDING" in blob
    assert "Replace HUMAN_FILL_REQUIRED fields on the Wikimedia pilot" not in blob
    assert stats["approved_gold_examples"] == 1
    assert stats["pending_review"] == 1
    assert stats["clips_needed"] == 4
    assert stats["hidden_state_candidates"] == 7


def test_hidden_state_pilot_contains_only_approved_s6() -> None:
    gold = ROOT / "data" / "examples" / "hidden_state_pilot.jsonl"
    records = load_manifest(gold)
    assert len(records) == 1
    record = records[0]
    assert record.clip_id == "mac_king_s006"
    assert record.question == "Which hand contains the coin after the apparent transfer?"
    assert record.ground_truth == "right"
    assert record.trick_id == "coin_fake_toss"
    assert record.performer_id == "mac_king"
    assert record.camera_id == "cui_2011_supplementary"
    assert record.split.value == "held_out"
    assert record.provenance.created_by == "human_researcher"
    assert record.metadata.get("human_approval") == "APPROVED"
    assert record.metadata.get("approved_by") == "human_researcher"
    assert record.metadata.get("leakage_resolved_by_human") is True
    assert record.metadata.get("unresolved_leakage_warning") is False
    assert record.metadata["original_proposal"]["occlusion_status"] == "PARTIAL"
    report = evaluate_gold_manifest(gold, ROOT)
    assert report["passed"] is True
    assert report["n"] == 1


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


def test_mac_king_movies_match_provenance_and_are_readable() -> None:
    provenance = json.loads(MAC_PROVENANCE.read_text(encoding="utf-8"))
    by_name = {row["local_filename"]: row for row in provenance["clips"]}
    cfg = VideoPreprocessConfig(max_frames=8, sample_strategy="uniform")
    for name in MOVIES:
        path = ROOT / "data" / "videos" / name
        assert path.is_file(), name
        meta = by_name[name]
        assert path.stat().st_size == meta["bytes"]
        assert _sha256(path) == meta["sha256"]
        info = probe_video(path)
        assert info["num_frames"] == meta["num_frames"]
        assert info["width"] == meta["width"]
        assert info["height"] == meta["height"]
        sampled = preprocess_video(path, config=cfg, load_frames=True)
        assert sampled.frames is not None
        assert len(sampled.frames) == 8
    wiki = json.loads(WIKI_PROVENANCE.read_text(encoding="utf-8"))
    for clip in wiki["clips"]:
        mp4 = ROOT / clip["local_mp4"]
        if mp4.is_file() and clip.get("mp4_sha256"):
            assert _sha256(mp4) == clip["mp4_sha256"]


def test_mac_king_review_stills_exist() -> None:
    expected = [
        "movie1_start.jpg",
        "movie1_end.jpg",
        "movie6_coin_in_right.jpg",
        "movie6_end.jpg",
        "movie7_coin_in_right.jpg",
        "movie7_end.jpg",
    ]
    for name in expected:
        path = STILLS / name
        assert path.is_file(), name
        assert path.stat().st_size > 0
