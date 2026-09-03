#!/usr/bin/env python
"""Write Mac King review JSONL from proposals + provenance."""

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
        approved = clip.get("human_approval") in {"APPROVE", "APPROVED"}
        question = clip["candidate_question"] if approved and clip.get("candidate_question") else FILL
        ground_truth = (
            clip["candidate_ground_truth"] if approved and clip.get("candidate_ground_truth") else FILL
        )
        trick_id = clip.get("trick_id") if approved and clip.get("trick_id") else FILL
        performer_id = clip.get("performer_id") if approved and clip.get("performer_id") else FILL
        camera_id = clip.get("camera_id") if approved and clip.get("camera_id") else FILL
        record = {
            "example_id": f"{clip_id}_q1",
            "clip_id": clip_id,
            "trick_id": trick_id,
            "performer_id": performer_id,
            "camera_id": camera_id,
            "video": {
                "path": meta["local_mp4"],
                "content_hash": meta["sha256"],
                "duration_s": meta["duration_s"],
                "fps": meta["fps"],
                "num_frames": meta["num_frames"],
            },
            "task": "hidden_state",
            "question": question,
            "ground_truth": ground_truth,
            "justification": None,
            "question_variant": "canonical",
            "temporal": None,
            "causal": None,
            "split": "held_out",
            "provenance": {
                "source": provenance["paper"]["urls"]["pmc"],
                "created_by": "human_researcher" if approved else "local_ingest",
                "created_at": "2026-09-03",
                "license": provenance["license"]["short_name"],
                "collection_notes": (
                    f"Supplementary {clip['paper_id']} ({clip['source_condition']}) from "
                    "Cui et al. 2011 Front. Hum. Neurosci. 5:103. Local file "
                    f"{meta['local_filename']} sha256={meta['sha256']}. "
                    "No conversion. Overlay copyright is not a research label. "
                    + (
                        "Human APPROVE recorded; gold copy is data/examples/hidden_state_pilot.jsonl."
                        if approved
                        else "question/ground_truth remain HUMAN_FILL_REQUIRED."
                    )
                ),
            },
            "notes": (
                f"HUMAN APPROVAL: {'APPROVED' if approved else 'PENDING'}. "
                f"Candidate task: {clip['candidate_task_type']}. "
                f"Inventory status: {clip['inventory_status']}. "
                + (
                    "Scored gold lives in hidden_state_pilot.jsonl."
                    if approved
                    else "Do not score this row until a human replaces the sentinels."
                )
            ),
            "prop_id": "coin" if approved else None,
            "metadata": {
                "research_labels_complete": approved,
                "placeholder_fields": (
                    []
                    if approved
                    else [
                        "trick_id",
                        "performer_id",
                        "camera_id",
                        "question",
                        "ground_truth",
                    ]
                ),
                "placeholder_fields_are_not_labels": not approved,
                "paper_condition": clip["source_condition"],
                "paper_id": clip["paper_id"],
                "paper_performer": "Mac King",
                "source_filename": meta["source_filename"],
                "human_approval": "APPROVED" if approved else "PENDING",
                "human_decision": clip.get("human_decision") or ("APPROVE" if approved else "PENDING"),
                "approved_by": "human_researcher" if approved else None,
                "do_not_gold_label_as_hidden_state": clip["inventory_status"]
                != "QUALIFIES",
                "hidden_state_class": clip["hidden_state_class"],
                "control_roles": clip["control_roles"],
                "revealed_counterpart_clip_id": clip.get("revealed_counterpart_clip_id"),
                "reveal_status": clip["reveal_status"],
                "leakage_resolved_by_human": bool(clip.get("leakage_resolved_by_human")),
                "unresolved_leakage_warning": bool(clip.get("unresolved_leakage_warning")),
                "original_proposal": clip.get("original_proposal"),
                "annotation_proposal": {
                    "source_condition": clip["source_condition"],
                    "observed_event": clip["what_happens"],
                    "candidate_task_type": clip["candidate_task_type"],
                    "candidate_question": clip["candidate_question"],
                    "candidate_ground_truth": clip["candidate_ground_truth"],
                    "ground_truth_basis": clip["ground_truth_basis"],
                    "occlusion_status": (
                        (clip.get("original_proposal") or {}).get("occlusion_status")
                        or clip["occlusion_status"]
                    ),
                    "temporal_status": clip["temporal_status"],
                    "reveal_status": clip["reveal_status"],
                    "answer_leakage_status": (
                        (clip.get("original_proposal") or {}).get("answer_leakage_status")
                        or clip["answer_leakage_status"]
                    ),
                    "confidence": clip["confidence"],
                    "human_review_required": clip["human_review_required"],
                    "human_decision": clip.get("human_decision")
                    or ("APPROVE" if approved else "PENDING"),
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
