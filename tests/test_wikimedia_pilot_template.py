from __future__ import annotations

import json
from pathlib import Path

import pytest

from magic_vlm.dataset import load_manifest
from magic_vlm.schemas import ExampleRecord
from magic_vlm.video import VideoPreprocessConfig, preprocess_video, probe_video

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "data" / "examples" / "wikimedia_pilot_manifest.template.jsonl"
PROVENANCE = ROOT / "data" / "provenance" / "wikimedia_peerj_01_19.json"
FILL = "HUMAN_FILL_REQUIRED"


def test_pilot_template_uses_real_schema_without_invented_labels() -> None:
    records = load_manifest(TEMPLATE)
    assert 3 <= len(records) <= 5
    for record in records:
        assert isinstance(record, ExampleRecord)
        assert record.task.value == "hidden_state"
        assert record.split.value == "held_out"
        assert record.trick_id == FILL
        assert record.performer_id == FILL
        assert record.camera_id == FILL
        assert record.question == FILL
        assert record.ground_truth == FILL
        assert record.metadata.get("research_labels_complete") is False
        assert record.video.path.endswith(".mp4")
        assert record.video.num_frames and record.video.num_frames > 0
        assert record.video.duration_s and record.video.duration_s > 0
        assert record.provenance.license == "CC BY 3.0"
        assert "commons.wikimedia.org" in record.provenance.source


def test_provenance_inventory_covers_template_clips() -> None:
    payload = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    clip_ids = {row["clip_id"] for row in payload["clips"]}
    records = load_manifest(TEMPLATE)
    assert {r.clip_id for r in records} == clip_ids
    assert all(row["ogv_sha1_matches_commons"] for row in payload["clips"])


@pytest.mark.skipif(
    not any((ROOT / "data" / "videos").glob("*.mp4")),
    reason="local converted Wikimedia MP4s are not present",
)
def test_local_wikimedia_mp4_preprocess_smoke() -> None:
    records = load_manifest(TEMPLATE)
    path = ROOT / records[0].video.path
    info = probe_video(path)
    assert info["num_frames"] >= 1
    assert info["width"] >= 1 and info["height"] >= 1
    sampled = preprocess_video(
        path,
        config=VideoPreprocessConfig(max_frames=8, sample_strategy="uniform"),
        load_frames=True,
    )
    assert sampled.frames is not None
    assert len(sampled.frames) == sampled.n_frames == 8
    assert sampled.frames[0].shape[0] == info["height"]
    assert sampled.frames[0].shape[1] == info["width"]
