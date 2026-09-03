#!/usr/bin/env python
"""Write the approved hidden-state gold manifest from human-approved proposals."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSALS = ROOT / "data" / "examples" / "mac_king_annotation_proposals.json"
PROVENANCE = ROOT / "data" / "provenance" / "mac_king_cui_2011.json"
OUT = ROOT / "data" / "examples" / "hidden_state_pilot.jsonl"
APPROVED = {"APPROVE", "APPROVED"}


def main() -> int:
    proposals = json.loads(PROPOSALS.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    tech = {row["clip_id"]: row for row in provenance["clips"]}
    rows = []
    for clip in proposals["clips"]:
        if clip.get("human_approval") not in APPROVED and clip.get("human_decision") not in APPROVED:
            continue
        if clip.get("inventory_status") != "QUALIFIES":
            continue
        meta = tech[clip["clip_id"]]
        original = clip.get("original_proposal") or {}
        record = {
            "example_id": f"{clip['clip_id']}_q1",
            "clip_id": clip["clip_id"],
            "trick_id": clip.get("trick_id") or "coin_fake_toss",
            "performer_id": clip.get("performer_id") or "mac_king",
            "camera_id": clip.get("camera_id") or "cui_2011_supplementary",
            "video": {
                "path": meta["local_mp4"],
                "content_hash": meta["sha256"],
                "duration_s": meta["duration_s"],
                "fps": meta["fps"],
                "num_frames": meta["num_frames"],
            },
            "task": "hidden_state",
            "question": clip["candidate_question"],
            "ground_truth": clip["candidate_ground_truth"],
            "justification": (
                "Human researcher approved S6. Paper: fake toss retains the coin "
                "in the right hand. Evaluation clip omits the S1 reveal."
            ),
            "question_variant": "canonical",
            "temporal": None,
            "causal": None,
            "split": "held_out",
            "provenance": {
                "source": provenance["paper"]["urls"]["pmc"],
                "created_by": "human_researcher",
                "created_at": "2026-09-03",
                "license": provenance["license"]["short_name"],
                "collection_notes": (
                    f"Supplementary {clip['paper_id']} ({clip['source_condition']}) from "
                    "Cui et al. 2011 Front. Hum. Neurosci. 5:103. Local file "
                    f"{meta['local_filename']} sha256={meta['sha256']}. No conversion. "
                    "Human APPROVE recorded 2026-09-03. Camera is the study supplementary "
                    "recording; the paper does not name a camera_id. Original candidate "
                    "proposal retained in metadata.original_proposal."
                ),
            },
            "notes": (
                "HUMAN APPROVAL: APPROVED by the human researcher. "
                "First hidden-state gold example. S7 remains PENDING."
            ),
            "prop_id": "coin",
            "metadata": {
                "research_labels_complete": True,
                "placeholder_fields": [],
                "placeholder_fields_are_not_labels": False,
                "paper_condition": clip["source_condition"],
                "paper_id": clip["paper_id"],
                "paper_performer": "Mac King",
                "source_filename": meta["source_filename"],
                "human_approval": "APPROVED",
                "human_decision": "APPROVE",
                "approved_by": "human_researcher",
                "do_not_gold_label_as_hidden_state": False,
                "hidden_state_class": "HIDDEN_STATE_CANDIDATE",
                "control_roles": [],
                "revealed_counterpart_clip_id": clip.get("revealed_counterpart_clip_id"),
                "reveal_status": clip["reveal_status"],
                "answer_leakage_status": "RESOLVED_BY_HUMAN",
                "leakage_resolved_by_human": True,
                "unresolved_leakage_warning": False,
                "occlusion_status": "PASS",
                "original_proposal": original,
                "annotation_proposal": {
                    "source_condition": clip["source_condition"],
                    "observed_event": clip["what_happens"],
                    "candidate_task_type": clip["candidate_task_type"],
                    "candidate_question": clip["candidate_question"],
                    "candidate_ground_truth": clip["candidate_ground_truth"],
                    "ground_truth_basis": clip["ground_truth_basis"],
                    "occlusion_status": original.get("occlusion_status", clip["occlusion_status"]),
                    "temporal_status": clip["temporal_status"],
                    "reveal_status": clip["reveal_status"],
                    "answer_leakage_status": original.get(
                        "answer_leakage_status", clip["answer_leakage_status"]
                    ),
                    "confidence": clip["confidence"],
                    "human_review_required": False,
                    "human_decision": "APPROVE",
                    "reason": clip["reason"],
                    "hidden_state_class": clip["hidden_state_class"],
                    "control_roles": clip["control_roles"],
                },
            },
        }
        rows.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    OUT.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")
    print(f"Wrote {OUT} ({len(rows)} gold rows)")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
