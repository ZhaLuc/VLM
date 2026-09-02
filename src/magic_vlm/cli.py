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


def sample_main(argv: list[str] | None = None) -> int:
    """Sample ordered (+ optional shuffled) frames and write reproducible metadata."""
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic video frame sampling. Writes a JSON plan; does not "
            "modify source videos. Shuffle uses the same sampled frames."
        )
    )
    parser.add_argument("--video", type=Path, required=True, help="Source video path.")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument(
        "--strategy",
        choices=("uniform", "first_n"),
        default="uniform",
    )
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument(
        "--also-shuffled",
        action="store_true",
        help="Also emit a shuffled presentation of the same sample set.",
    )
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument(
        "--load-frames",
        action="store_true",
        help="Decode frames (requires OpenCV). Default is indices/metadata only.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=None, help="Override probed frame count.")
    args = parser.parse_args(argv)

    from magic_vlm.utils import write_json
    from magic_vlm.video import (
        ResizeConfig,
        VideoPreprocessConfig,
        ordered_and_shuffled_pair,
        preprocess_video,
    )

    resize = None
    if args.resize_width is not None or args.resize_height is not None:
        resize = ResizeConfig(width=args.resize_width, height=args.resize_height)
    cfg = VideoPreprocessConfig(
        max_frames=args.max_frames,
        sample_strategy=args.strategy,
        temporal_shuffle=False,
        shuffle_seed=args.shuffle_seed,
        resize=resize,
    )
    if args.also_shuffled:
        ordered, shuffled = ordered_and_shuffled_pair(
            args.video,
            config=cfg,
            num_frames=args.num_frames,
            shuffle_seed=args.shuffle_seed,
            load_frames=args.load_frames,
        )
        payload = {
            "ordered": ordered.to_dict(),
            "shuffled": shuffled.to_dict(),
            "same_ordered_indices": list(ordered.ordered_indices)
            == list(shuffled.ordered_indices),
            "integrity_note": (
                "Temporal-shuffle sensitivity is a temporal-order diagnostic, "
                "not proof of causal reasoning."
            ),
        }
    else:
        clip = preprocess_video(
            args.video,
            config=cfg,
            num_frames=args.num_frames,
            load_frames=args.load_frames,
        )
        payload = {"ordered": clip.to_dict()}
    write_json(args.json_out, payload)
    print(f"Wrote sample metadata: {args.json_out}")
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
