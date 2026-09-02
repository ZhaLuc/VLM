"""Lightweight CLI entrypoints that do not download model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from magic_vlm.dataset import load_manifest
from magic_vlm.evaluation import evaluate_exact_match
from magic_vlm.experiment import build_run_manifest, load_experiment_config
from magic_vlm.inference import run_inference
from magic_vlm.models import load_vlm
from magic_vlm.rewards import ExactMatchReward, score_batch
from magic_vlm.utils import write_json, write_jsonl
from magic_vlm.video import preprocess_video_meta


def smoke_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Architecture smoke run (stub model only).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_stub.yaml"),
        help="Experiment YAML (should use a stub/ model id).",
    )
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    if not config.model.model_id.startswith("stub/"):
        raise SystemExit(
            "smoke_main refuses non-stub models to avoid accidental weight downloads."
        )

    examples = load_manifest(config.dataset_manifest)
    model = load_vlm(config.model, allow_download=False)
    artifacts = []
    for example in examples:
        num_frames = example.video.num_frames or 16
        preprocessed = preprocess_video_meta(
            example.video.path,
            num_frames=num_frames,
            config=config.video,
        )
        artifacts.append(run_inference(model, example, preprocessed=preprocessed))

    report = evaluate_exact_match(examples, artifacts)
    reward_scores = score_batch(ExactMatchReward(), artifacts, examples)
    run = build_run_manifest(config)

    out_dir = Path(config.output_dir) / run.run_id
    write_json(out_dir / "run_manifest.json", run.to_dict())
    if config.preserve_raw_outputs:
        write_jsonl(out_dir / "predictions.jsonl", [a.to_dict() for a in artifacts])
    write_json(out_dir / "metrics.json", report.to_dict())
    write_json(out_dir / "reward_scores.json", {"reward": "exact_match", "scores": reward_scores})
    print(f"smoke ok: run_id={run.run_id} accuracy={report.accuracy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(smoke_main())
