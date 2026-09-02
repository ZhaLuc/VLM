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


def validate_preferences_main(argv: list[str] | None = None) -> int:
    """Validate pairwise preference JSONL (no training / no AI labels)."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate preference judgments. Checks identity, winners, identical "
            "responses, and multi-annotation policy. Does not train or rewrite text."
        )
    )
    parser.add_argument("--prefs", type=Path, required=True, help="Preference JSONL path.")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--allow-ties",
        action="store_true",
        help="Permit winner=tie when records also set allow_ties.",
    )
    parser.add_argument(
        "--allow-identical",
        action="store_true",
        help="Downgrade identical A/B responses to review (default: error).",
    )
    parser.add_argument(
        "--forbid-multi-annotation",
        action="store_true",
        help="Error when multiple judgments share the same content pair_id.",
    )
    parser.add_argument(
        "--require-rationale",
        action="store_true",
        help="Require non-empty rationale on every judgment.",
    )
    args = parser.parse_args(argv)

    from magic_vlm.preferences import (
        PreferenceValidationConfig,
        load_preference_pairs,
        validate_preference_pairs,
        write_preference_validation_report,
    )

    pairs = load_preference_pairs(args.prefs)
    report = validate_preference_pairs(
        pairs,
        config=PreferenceValidationConfig(
            allow_ties=args.allow_ties,
            allow_identical_responses=args.allow_identical,
            allow_multiple_annotations_per_pair=not args.forbid_multi_annotation,
            require_rationale=args.require_rationale,
        ),
    )
    print(report.format_human(), end="")
    if args.json_out is not None:
        write_preference_validation_report(report, args.json_out)
        print(f"Wrote JSON report: {args.json_out}")
    return 0 if report.passed else 1


def analyze_preferences_main(argv: list[str] | None = None) -> int:
    """Quality-control analysis for preference JSONL (non-destructive)."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze preference-data quality before DPO/reward-model training. "
            "Reports errors, warnings, and possible biases. Never deletes records."
        )
    )
    parser.add_argument("--prefs", type=Path, required=True, help="Preference JSONL.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for JSON/MD/JSONL reports (default: alongside --prefs).",
    )
    args = parser.parse_args(argv)

    from magic_vlm.preference_quality import (
        analyze_preference_file,
        format_quality_report,
        write_quality_outputs,
    )

    report = analyze_preference_file(args.prefs)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.prefs.parent / f"{args.prefs.stem}_quality"
    write_quality_outputs(report, out_dir)
    print(format_quality_report(report), end="")
    print(
        f"quality ok parsed={report.n_parsed} malformed={report.n_malformed} "
        f"errors={report.n_errors} warnings={report.n_warnings} out={out_dir}"
    )
    return 0


def annotate_main(argv: list[str] | None = None) -> int:
    """Minimal human preference annotation over explanation pairs."""
    parser = argparse.ArgumentParser(
        description=(
            "Annotate pairwise explanation preferences. Shows video path, task, "
            "and raw A/B responses; appends immutable judgments. Resume-safe. "
            "No accounts, cloud sync, or AI ranking."
        )
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("data/examples/toy_annotation_queue.jsonl"),
        help="Candidate pair JSONL (no winners).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/annotations/preferences.jsonl"),
        help="Append-only judgment store.",
    )
    parser.add_argument("--annotator", type=str, required=True, help="Annotator ID.")
    parser.add_argument(
        "--rubric",
        type=Path,
        default=Path("configs/annotation_rubric.yaml"),
        help="Rubric YAML (correctness / evidence / specificity).",
    )
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument(
        "--no-open-video",
        action="store_true",
        help="Do not attempt to open the video file in an OS viewer.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max new judgments this run.")
    parser.add_argument(
        "--winner",
        action="append",
        default=None,
        help="Scripted winner (a/b); repeatable. Enables non-interactive mode.",
    )
    parser.add_argument(
        "--rationale",
        action="append",
        default=None,
        help="Optional rationale aligned with --winner order.",
    )
    args = parser.parse_args(argv)

    from magic_vlm.annotation import (
        AnnotationSessionConfig,
        load_rubric,
        run_annotation_session,
    )

    config = AnnotationSessionConfig(
        annotator_id=args.annotator,
        queue_path=args.queue,
        judgments_path=args.out,
        rubric=load_rubric(args.rubric if args.rubric.exists() else None),
        video_root=args.video_root,
        open_video=not args.no_open_video,
    )
    result = run_annotation_session(
        config,
        winners=args.winner,
        rationales=args.rationale,
        limit=args.limit,
    )
    print(
        f"annotate ok recorded={result.n_recorded} skipped={result.n_skipped} "
        f"pending_before={result.n_pending_before} out={result.judgments_path}"
    )
    return 0


def train_reward_main(argv: list[str] | None = None) -> int:
    """Train a small Bradley-Terry preference reward model (no GRPO/DPO)."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit a small text Bradley-Terry reward model on preference pairs. "
            "Validates with preference agreement (not reasoning accuracy). "
            "Refuses held_out preferences. Does not train the VLM."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reward_model_bt_synthetic.yaml"),
    )
    args = parser.parse_args(argv)

    from magic_vlm.reward_model import RewardModelConfig, train_bradley_terry_reward_model

    config = RewardModelConfig.from_yaml(args.config)
    result = train_bradley_terry_reward_model(config)
    print(
        f"reward-train ok run_dir={result.run_dir} "
        f"best_val_pref_acc={result.best_val_preference_accuracy} "
        f"checkpoint={result.checkpoint_path}"
    )
    print(result.disclaimer)
    return 0


def score_reward_main(argv: list[str] | None = None) -> int:
    """Score preference pairs with a trained reward checkpoint (inference only)."""
    parser = argparse.ArgumentParser(
        description=(
            "Reward-model inference only. Writes per-example chosen/rejected "
            "scores. Preference agreement is not factual correctness."
        )
    )
    parser.add_argument("--prefs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    from magic_vlm.reward_model import INTEGRITY_DISCLAIMER, score_preference_file

    rows = score_preference_file(
        args.prefs,
        args.checkpoint,
        out_path=args.out,
        device=args.device,
    )
    n_ok = sum(1 for r in rows if r.get("correct_order"))
    acc = (n_ok / len(rows)) if rows else float("nan")
    print(f"reward-score ok n={len(rows)} preference_accuracy={acc} out={args.out}")
    print(INTEGRITY_DISCLAIMER)
    return 0


def analyze_main(argv: list[str] | None = None) -> int:
    """Analyze a baseline run: metrics, report, and inspectable error exports."""
    parser = argparse.ArgumentParser(
        description=(
            "Baseline failure analysis. Reads predictions.jsonl; writes "
            "analysis_metrics.json, analysis_report.md, errors.jsonl, "
            "successes.jsonl, examples_inspectable.jsonl. Does not retune protocol."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Baseline run directory containing predictions.jsonl.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional dataset manifest for performer/camera/notes join.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as --run-dir).",
    )
    args = parser.parse_args(argv)

    from magic_vlm.analysis import analyze_baseline_run

    analysis = analyze_baseline_run(
        args.run_dir,
        manifest=args.manifest,
        write=True,
        out_dir=args.out_dir,
    )
    print(
        f"analysis ok run_id={analysis.run_id} split={analysis.split} "
        f"n={analysis.n_examples} accuracy={analysis.overall_accuracy} "
        f"incorrect={analysis.n_incorrect} parse_failures={analysis.n_parse_failures}"
    )
    return 0


def baseline_main(argv: list[str] | None = None) -> int:
    """Run the immutable zero-shot baseline on a fixed split (default: held_out)."""
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot baseline: untouched checkpoint, deterministic decoding, "
            "raw responses preserved. No fine-tuning / LoRA / DPO / GRPO."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_stub.yaml"))
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Evaluation split (default: config.dataset.split or held_out).",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--load-frames",
        action="store_true",
        help="Decode video frames when media files exist (requires OpenCV).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record per-example inference errors instead of aborting.",
    )
    args = parser.parse_args(argv)

    from magic_vlm.baseline import run_zero_shot_baseline

    config = load_experiment_config(args.config)
    result = run_zero_shot_baseline(
        config,
        split=args.split,
        run_id=args.run_id,
        allow_download=args.allow_download or None,
        load_frames=args.load_frames,
        continue_on_error=args.continue_on_error,
    )
    print(
        f"baseline ok run_id={result.run_id} split={result.split} "
        f"n={result.summary.n_examples} accuracy={result.summary.overall_accuracy} "
        f"parse_failures={result.summary.n_parse_failures} dir={result.run_dir}"
    )
    return 0


def infer_main(argv: list[str] | None = None) -> int:
    """Run inference for one example (stub by default; no weight download)."""
    parser = argparse.ArgumentParser(
        description=(
            "VLM inference for one dataset example. Does not train or download "
            "weights unless --allow-download is set."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_stub.yaml"))
    parser.add_argument("--example-id", type=str, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    if (not config.model.model_id.startswith("stub/")) and not args.allow_download:
        if not Path(config.model.model_id).exists():
            raise SystemExit(
                "Refusing to download model weights. Use a stub config, a local "
                "checkpoint path, or pass --allow-download."
            )
    ctx = initialize_experiment(config, run_id=args.run_id)
    examples = load_manifest(config.dataset.manifest)
    if args.example_id:
        examples = [ex for ex in examples if ex.example_id == args.example_id]
        if not examples:
            raise SystemExit(f"No example with id {args.example_id!r}")
    example = examples[0]
    num_frames = example.video.num_frames or 16
    preprocessed = preprocess_video_meta(
        example.video.path,
        num_frames=num_frames,
        config=config.video,
    )
    model = load_vlm(
        config.model,
        allow_download=args.allow_download or config.allow_model_download,
        device=config.device,
    )
    artifact = run_inference(
        model,
        example,
        preprocessed=preprocessed,
        generation=config.generation,
        device=ctx.device,
        checkpoint_kind=config.checkpoint.kind,
        checkpoint_path=config.checkpoint.path,
    )
    write_json(ctx.run_dir / "inference.json", artifact.to_dict())
    if args.json_out is not None:
        write_json(args.json_out, artifact.to_dict())
    print(
        f"infer ok example_id={artifact.example_id} clip_id={artifact.clip_id} "
        f"latency_s={artifact.latency_s} raw_chars={len(artifact.raw_text)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(smoke_main())
