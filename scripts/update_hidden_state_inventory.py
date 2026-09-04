#!/usr/bin/env python
"""Refresh hidden-state inventory counts from Wikimedia + Mac King proposals."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "data" / "examples" / "hidden_state_candidate_inventory.json"
PROPOSALS = ROOT / "data" / "examples" / "mac_king_annotation_proposals.json"
PROVENANCE = ROOT / "data" / "provenance" / "mac_king_cui_2011.json"
TARGET = 5


def stats(rows: list[dict], *, collection: str | None = None) -> dict:
    subset = rows if collection is None else [r for r in rows if r.get("collection") == collection]
    return {
        "candidate_count": len(subset),
        "eligible_count": sum(1 for r in subset if r["status"] == "QUALIFIES"),
        "pending_human_review": sum(
            1 for r in subset if r["status"] == "QUALIFIES_WITH_HUMAN_REVIEW"
        ),
        "rejected_count": sum(1 for r in subset if r["status"] == "NOT_SUITABLE"),
    }


def main() -> int:
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    proposals = json.loads(PROPOSALS.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    tech = {row["clip_id"]: row for row in provenance["clips"]}

    for row in inv["candidates"]:
        cid = str(row.get("candidate", ""))
        if cid.startswith("mac_king_"):
            continue
        if cid.startswith("toy_"):
            row["collection"] = "synthetic_fixtures"
        elif cid == "peerj_01_19_s008" or "opaque" in cid:
            row["collection"] = "peerj_documented_missing"
        elif cid.startswith("peerj_"):
            row["collection"] = "wikimedia_controls"

    inv["candidates"] = [
        row
        for row in inv["candidates"]
        if not str(row.get("candidate", "")).startswith("mac_king_")
    ]

    for clip in proposals["clips"]:
        meta = tech[clip["clip_id"]]
        inv["candidates"].append(
            {
                "candidate": clip["clip_id"],
                "collection": "mac_king_candidates",
                "source": (
                    "Cui et al. 2011 Front. Hum. Neurosci. 5:103 supplementary "
                    f"{clip['paper_id']}; local {meta['local_filename']}"
                ),
                "duration_s": meta["duration_s"],
                "trick_or_condition": clip["source_condition"],
                "genuine_occlusion": clip["occlusion_status"],
                "temporal_dependence": clip["temporal_status"],
                "ground_truth": clip["ground_truth_basis"],
                "leakage_risk": clip["answer_leakage_status"],
                "status": clip["inventory_status"],
                "human_review_flag": bool(clip["human_review_required"]),
                "hidden_state_class": clip["hidden_state_class"],
                "control_roles": clip["control_roles"],
                "keep_in_repository": True,
                "local_present": True,
                "video_relpath": meta["local_mp4"],
                "thumbnail_relpath": (
                    f"reports/mac_king_clip_review/stills/{meta['local_filename'].replace('.MP4','').lower()}_end.jpg"
                ),
                "rank": clip["rank"],
                "reveal_status": clip["reveal_status"],
                "candidate_question": clip["candidate_question"],
                "candidate_ground_truth": clip["candidate_ground_truth"],
                "revealed_counterpart_clip_id": clip.get("revealed_counterpart_clip_id"),
                "provenance": {
                    "source_url": provenance["paper"]["urls"]["pmc"],
                    "title": clip["source_condition"],
                    "author": "Cui J, Otero-Millan J, Macknik SL, King M, Martinez-Conde S",
                    "performer": "Mac King (named in the paper)",
                    "license": provenance["license"]["short_name"],
                    "date_accessed": provenance["access"]["date_accessed"],
                    "original_filename": meta["source_filename"],
                    "local_filename": meta["local_filename"],
                    "conversion": None,
                    "source_notes": clip["what_happens"],
                    "ground_truth_source": clip["ground_truth_basis"],
                },
            }
        )

    all_rows = inv["candidates"]
    wiki = stats(all_rows, collection="wikimedia_controls")
    mac = stats(all_rows, collection="mac_king_candidates")
    gold_eligible = stats(all_rows)["eligible_count"]
    pending = stats(all_rows)["pending_human_review"]
    invalid = sum(
        1
        for r in all_rows
        if r["status"] == "NOT_SUITABLE"
        and r.get("collection") in {"wikimedia_controls", "mac_king_candidates"}
    )
    inv["gold_labels_written"] = gold_eligible > 0
    inv["notes"] = (
        "Wikimedia cups remain controls. Mac King S6 is human-approved gold. "
        "S7 remains PENDING. S1-S5 are reveal/no-object controls."
    )
    inv["readiness"] = {
        "first_baseline_dataset": "DATA READY" if gold_eligible else "NOT READY",
        "valid_candidates": gold_eligible,
        "candidates_needing_human_review": pending,
        "invalid_candidates": invalid,
        "additional_clips_needed": max(0, TARGET - gold_eligible),
        "recommended_next_milestone_clips": TARGET,
        "minimum_smoke_clips": 3,
        "later_dataset_b_clips": "15-25 (research plan; not the next step)",
        "best_current_hidden_state_candidate": "mac_king_s006",
        "how_many_valid_clips": (
            f"{gold_eligible} approved gold; {pending} pending human review; fewer than five"
        ),
        "recommended_next_action": (
            "S6 is approved gold. Obtain CUDA + local Qwen2.5-VL weights for the "
            "zero-shot baseline. S7 stays PENDING. Do not gold-label Wikimedia clips."
            if gold_eligible
            else (
                "Review mac_king_s006 and mac_king_s007 in "
                "reports/hidden_state_candidates/index.html (APPROVE / EDIT / REJECT). "
                "Do not gold-label Wikimedia clips."
            )
        ),
        "collections": {
            "wikimedia_controls": wiki,
            "mac_king_candidates": mac,
            "hidden_state_gold": {
                "eligible_count": gold_eligible,
                "pending_human_review": pending,
                "clips_needed_for_pilot": max(0, TARGET - gold_eligible),
            },
        },
    }
    videos = sorted(
        p.name
        for p in (ROOT / "data" / "videos").iterdir()
        if p.suffix.lower() in {".mp4", ".ogv"} and p.name != ".gitkeep"
    )
    inv["local_media_search"]["data_videos"] = videos
    inv["inspected_at"] = "2026-09-03"
    INVENTORY.write_text(json.dumps(inv, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("inventory", inv["readiness"]["collections"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
