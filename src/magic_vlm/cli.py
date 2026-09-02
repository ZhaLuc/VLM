"""Lightweight CLI entrypoints that do not download model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from magic_vlm.dataset import load_manifest
from magic_vlm.evaluation import evaluate_exact_match
from magic_vlm.experiment import initialize_experiment, load_experiment_config
from magic_vlm.inference import run_inference
from magic_vlm.models import load_vlm
from magic_vlm.rewards import ExactMatchReward, score_batch
from magic_vlm.utils import write_json, write_jsonl
from magic_vlm.video import preprocess_video_meta


def init_main(argv: list[str] | None = None) -> int:
    """Initialize an experiment directory and metadata without loading a VLM."""
    parser = argparse.ArgumentParser(
        description="Initialize experiment output + reproducibility metadata (no model load)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_stub.yaml"),
        help="Experiment YAML path.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    ctx = initialize_experiment(config, run_id=args.run_id)
    print(
        f"initialized run_id={ctx.run_id} dir={ctx.run_dir} "
        f"device={ctx.device.resolved} determinism={ctx.determinism.level} "
        f"zero_shot_baseline={config.is_zero_shot_baseline}"
    )
    return 0


def smoke_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Architecture smoke run (stub model only).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_stub.yaml"),
        help="Experiment YAML (should use a stub/ model id).",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run id.")
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    if not config.model.model_id.startswith("stub/"):
        raise SystemExit(
            "smoke_main refuses non-stub models to avoid accidental weight downloads."
        )

    ctx = initialize_experiment(config, run_id=args.run_id)
    examples = load_manifest(config.dataset.manifest)
    model = load_vlm(config.model, allow_download=False)
    artifacts = []
    for example in examples:
        num_frames = example.video.num_frames or 16
        preprocessed = preprocess_video_meta(
            example.video.path,
            num_frames=num_frames,
            config=config.video,
        )
        artifacts.append(
            run_inference(
                model,
                example,
                preprocessed=preprocessed,
                generation=config.generation,
            )
        )

    report = evaluate_exact_match(examples, artifacts)
    reward_scores = score_batch(ExactMatchReward(), artifacts, examples)
    if config.preserve_raw_outputs:
        write_jsonl(ctx.run_dir / "predictions.jsonl", [a.to_dict() for a in artifacts])
    write_json(ctx.run_dir / "metrics.json", report.to_dict())
    write_json(
        ctx.run_dir / "reward_scores.json",
        {"reward": config.reward_function, "scores": reward_scores},
    )
    print(f"smoke ok: run_id={ctx.run_id} accuracy={report.accuracy}")
    return 0


def validate_main(argv: list[str] | None = None) -> int:
    """Validate a dataset manifest for quality and train/held-out leakage."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate dataset manifests. Fails on hard errors and scientific "
            "leakage by default. Does not repair or rewrite data."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest path.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for machine-readable ValidationReport JSON.",
    )
    parser.add_argument(
        "--no-media-check",
        action="store_true",
        help="Skip video existence/readability checks.",
    )
    parser.add_argument(
        "--allow-leakage",
        action="store_true",
        help="Do not fail the process on leakage findings (still reported).",
    )
    parser.add_argument(
        "--answer-vocab",
        type=Path,
        default=None,
        help="Optional text file of allowed ground_truth strings (exact match).",
    )
    parser.add_argument(
        "--expected-fps",
        type=float,
        default=None,
        help="Optional expected FPS for review warnings.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repo/data root used to resolve relative video paths.",
    )
    args = parser.parse_args(argv)

    from magic_vlm.validate import ValidatorConfig, validate_dataset, write_report

    allowed = None
    if args.answer_vocab is not None:
        # Exact stored strings; blank lines ignored; no case-folding.
        allowed = frozenset(
            line
            for line in args.answer_vocab.read_text(encoding="utf-8").splitlines()
            if line != ""
        )

    report = validate_dataset(
        args.manifest,
        config=ValidatorConfig(
            root=args.root,
            check_media=not args.no_media_check,
            fail_on_leakage=not args.allow_leakage,
            allowed_answers=allowed,
            expected_fps=args.expected_fps,
        ),
    )
    print(report.format_human(), end="")
    if args.json_out is not None:
        write_report(report, args.json_out)
        print(f"Wrote JSON report: {args.json_out}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(smoke_main())
