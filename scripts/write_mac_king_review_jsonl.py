#!/usr/bin/env python
"""Write Mac King review JSONL from proposals + provenance. No gold labels."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSALS = ROOT / "data" / "examples" / "mac_king_annotation_proposals.json"
PROVENANCE = ROOT / "data" / "provenance" / "mac_king_cui_2011.json"
OUT = ROOT / "data" / "examples" / "mac_king_review.jsonl"
FILL = "HUMAN_FILL_REQUIRED"


def main() -> int:
    proposals = json.loads(PROPOSALS.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    tech = {row["clip_id"]: row for row in provenance["clips"]}
    rows = []
    for clip in proposals["clips"]:
        clip_id = clip["clip_id"]
        meta = tech[clip_id]
        record = {
            "example_id": f"{clip_id}_q1",
            "clip_id": clip_id,
            "trick_id": FILL,
            "performer_id": FILL,
            "camera_id": FILL,
            "video": {
                "path": meta["local_mp4"],
                "content_hash": meta["sha256"],
                "duration_s": meta["duration_s"],
                "fps": meta["fps"],
                "num_frames": meta["num_frames"],
            },
            "task": "hidden_state",
            "question": FILL,
            "ground_truth": FILL,
            "justification": None,
            "question_variant": "canonical",
            "temporal": None,
            "causal": None,
            "split": "held_out",
            "provenance": {
                "source": provenance["paper"]["urls"]["pmc"],
                "created_by": "local_ingest",
                "created_at": "2026-09-03",
                "license": provenance["license"]["short_name"],
                "collection_notes": (
                    f"Supplementary {clip['paper_id']} ({clip['source_condition']}) from "
                    "Cui et al. 2011 Front. Hum. Neurosci. 5:103. Local file "
                    f"{meta['local_filename']} sha256={meta['sha256']}. "
                    "No conversion. Overlay copyright is not a research label. "
                    "question/ground_truth remain HUMAN_FILL_REQUIRED."
                ),
            },
            "notes": (
                f"HUMAN APPROVAL: PENDING. Candidate task: {clip['candidate_task_type']}. "
                f"Inventory status: {clip['inventory_status']}. "
                "Do not score this row until a human replaces the sentinels."
            ),
            "prop_id": None,
            "metadata": {
                "research_labels_complete": False,
                "placeholder_fields": [
                    "trick_id",
                    "performer_id",
                    "camera_id",
                    "question",
                    "ground_truth",
                ],
                "placeholder_fields_are_not_labels": True,
                "paper_condition": clip["source_condition"],
                "paper_id": clip["paper_id"],
                "paper_performer": "Mac King",
                "source_filename": meta["source_filename"],
                "human_approval": "PENDING",
                "do_not_gold_label_as_hidden_state": clip["inventory_status"]
                != "QUALIFIES",
                "hidden_state_class": clip["hidden_state_class"],
                "control_roles": clip["control_roles"],
                "revealed_counterpart_clip_id": clip.get("revealed_counterpart_clip_id"),
                "annotation_proposal": {
                    "source_condition": clip["source_condition"],
                    "observed_event": clip["what_happens"],
                    "candidate_task_type": clip["candidate_task_type"],
                    "candidate_question": clip["candidate_question"],
                    "candidate_ground_truth": clip["candidate_ground_truth"],
                    "ground_truth_basis": clip["ground_truth_basis"],
                    "occlusion_status": clip["occlusion_status"],
                    "temporal_status": clip["temporal_status"],
                    "reveal_status": clip["reveal_status"],
                    "answer_leakage_status": clip["answer_leakage_status"],
                    "confidence": clip["confidence"],
                    "human_review_required": clip["human_review_required"],
                    "human_decision": "PENDING",
                    "reason": clip["reason"],
                    "hidden_state_class": clip["hidden_state_class"],
                    "control_roles": clip["control_roles"],
                },
            },
        }
        rows.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
