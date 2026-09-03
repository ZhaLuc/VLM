#!/usr/bin/env python
"""Validate approved hidden-state gold against benchmark eligibility checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from magic_vlm.hidden_state_eligibility import evaluate_gold_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "examples" / "hidden_state_pilot.jsonl",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = evaluate_gold_manifest(args.manifest, args.root)
    print(f"Hidden-state eligibility: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"  manifest: {report['manifest']}")
    print(f"  rows: {report['n']}")
    for row in report["rows"]:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"  [{status}] {row['clip_id']}")
        for err in row["errors"]:
            print(f"      {err}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
