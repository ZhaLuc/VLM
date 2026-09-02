"""Zero-shot baseline experiment runner (untouched checkpoint only).

Scientific invariants
---------------------
- No fine-tuning, LoRA, SFT, DPO, GRPO, or weight updates
- Held-out membership comes from the manifest and is never rewritten here
- Failures are recorded, never silently dropped
- ``baseline_immutable`` runs are the frozen reference for later comparisons
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from magic_vlm.dataset import filter_split, load_manifest
from magic_vlm.evaluation import (
    BaselineSummary,
    evaluate_baseline,
    exact_match,
    is_parse_failure,
)
from magic_vlm.experiment import ExperimentConfig, ExperimentContext, initialize_experiment
from magic_vlm.inference import build_prompt, parse_answer, run_inference
from magic_vlm.models import load_vlm
from magic_vlm.schemas import ExampleRecord, InferenceArtifact, Split
from magic_vlm.utils import write_json, write_jsonl
from magic_vlm.video import SampledClip, preprocess_video, preprocess_video_meta

logger = logging.getLogger("magic_vlm")

PROMPT_TEMPLATE_ID = "hidden_state_v1"


class BaselineConfigError(ValueError):
    """Raised when a config is not a valid zero-shot baseline."""


@dataclass(frozen=True)
class BaselinePrediction:
    """One scored baseline example (prediction + correctness + metadata)."""

    example_id: str
    clip_id: str
    trick_id: str
    split: str
    question: str
    ground_truth: str | None
    prompt: str
    prompt_template_id: str
    raw_text: str
    parsed_answer: str | None
    parse_failed: bool
    correct: bool
    latency_s: float | None
    model_id: str
    generation: dict[str, Any]
    preprocessing: dict[str, Any]
    device: str | None
    frame_indices: tuple[int, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frame_indices"] = list(self.frame_indices)
        return payload


@dataclass(frozen=True)
class BaselineResult:
    run_id: str
    run_dir: str
    split: str
    summary: BaselineSummary
    predictions: tuple[BaselinePrediction, ...]
    artifacts: tuple[InferenceArtifact, ...]
    prompt_template: str
    prompt_template_id: str = PROMPT_TEMPLATE_ID
    immutable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "split": self.split,
            "immutable": self.immutable,
            "prompt_template_id": self.prompt_template_id,
            "prompt_template": self.prompt_template,
            "summary": self.summary.to_dict(),
            "n_predictions": len(self.predictions),
        }


def assert_zero_shot_baseline_config(config: ExperimentConfig) -> None:
    """Refuse any config that would train, adapt, or sample non-deterministically."""
    if config.training_method != "none":
        raise BaselineConfigError(
            f"Zero-shot baseline forbids training_method={config.training_method!r}"
        )
    if config.checkpoint.kind not in {"base", "stub"}:
        raise BaselineConfigError(
            f"Zero-shot baseline requires untouched checkpoint kind "
            f"(got {config.checkpoint.kind!r})"
        )
    if not config.is_zero_shot_baseline:
        raise BaselineConfigError("Config is not classified as a zero-shot baseline")
    if not config.baseline_immutable:
        raise BaselineConfigError(
            "baseline_immutable must be true so this run remains the frozen reference"
        )
    if config.generation.do_sample:
        raise BaselineConfigError(
            "Deterministic baseline requires generation.do_sample=false"
        )
    if config.video.temporal_shuffle:
        raise BaselineConfigError(
            "Zero-shot baseline uses ordered frames only (temporal_shuffle must be false)"
        )
    if config.dataset.task not in {"hidden_state", "hidden-state"}:
        raise BaselineConfigError(
            f"Baseline runner currently supports hidden_state task only "
            f"(got {config.dataset.task!r})"
        )


def resolve_eval_split(config: ExperimentConfig, split_override: str | None = None) -> Split:
    """Default evaluation split is held_out (fixed in the manifest)."""
    raw = split_override if split_override is not None else config.dataset.split
    if raw is None or raw == "":
        return Split.HELD_OUT
    try:
        return Split(raw)
    except ValueError as exc:
        raise BaselineConfigError(f"Invalid evaluation split: {raw!r}") from exc


def select_baseline_examples(
    records: Sequence[ExampleRecord],
    split: Split,
) -> list[ExampleRecord]:
    """Select examples for the baseline split without mutating the manifest."""
    selected = filter_split(records, split)
    if not selected:
        raise BaselineConfigError(
            f"No examples found for split={split.value!r}. "
            "Held-out membership must be fixed in the manifest before running baseline."
        )
    return selected


def _preprocess_example(
    example: ExampleRecord,
    config: ExperimentConfig,
    *,
    load_frames: bool,
) -> SampledClip:
    num_frames = example.video.num_frames
    if load_frames:
        video_path = Path(example.video.path)
        if not video_path.is_absolute():
            video_path = Path.cwd() / video_path
        if video_path.exists():
            return preprocess_video(
                video_path,
                config=config.video,
                num_frames=num_frames,
                source_fps=example.video.fps,
                source_content_hash=example.video.content_hash,
                load_frames=True,
            )
        logger.warning(
            "Video missing for %s (%s); falling back to index-only preprocessing",
            example.example_id,
            example.video.path,
        )
    if num_frames is None or num_frames < 1:
        num_frames = config.video.max_frames
    return preprocess_video_meta(
        example.video.path,
        num_frames=int(num_frames),
        config=config.video,
    )


def run_zero_shot_baseline(
    config: ExperimentConfig,
    *,
    split: str | None = None,
    run_id: str | None = None,
    allow_download: bool | None = None,
    load_frames: bool = False,
    continue_on_error: bool = False,
) -> BaselineResult:
    """Execute the immutable zero-shot baseline on one fixed split."""
    assert_zero_shot_baseline_config(config)
    eval_split = resolve_eval_split(config, split)
    ctx = initialize_experiment(config, run_id=run_id)
    records = load_manifest(config.dataset.manifest)
    examples = select_baseline_examples(records, eval_split)

    download = config.allow_model_download if allow_download is None else allow_download
    model = load_vlm(config.model, allow_download=download, device=config.device)

    prompt_template = (
        "You are analyzing a short magic or mentalism demonstration.\n"
        "Question: {question}\n"
        "Answer briefly and concretely."
    )
    (ctx.run_dir / "prompt_template.txt").write_text(
        f"template_id={PROMPT_TEMPLATE_ID}\n{prompt_template}\n",
        encoding="utf-8",
    )
    write_json(
        ctx.run_dir / "split_lock.json",
        {
            "split": eval_split.value,
            "n_examples": len(examples),
            "example_ids": [ex.example_id for ex in examples],
            "clip_ids": sorted({ex.clip_id for ex in examples}),
            "trick_ids": sorted({ex.trick_id for ex in examples}),
            "note": (
                "Held-out membership is taken from the manifest and not modified "
                "by the baseline runner."
            ),
        },
    )

    artifacts: list[InferenceArtifact] = []
    predictions: list[BaselinePrediction] = []
    for example in examples:
        prompt = build_prompt(example)
        try:
            preprocessed = _preprocess_example(example, config, load_frames=load_frames)
            artifact = run_inference(
                model,
                example,
                preprocessed=preprocessed,
                generation=config.generation,
                prompt=prompt,
                parse=parse_answer,
                device=ctx.device,
                checkpoint_kind=config.checkpoint.kind,
                checkpoint_path=config.checkpoint.path,
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - record or abort
            if not continue_on_error:
                raise
            logger.exception("Inference failed for %s", example.example_id)
            artifact = InferenceArtifact(
                example_id=example.example_id,
                clip_id=example.clip_id,
                task=example.task.value,
                question=example.question,
                model_id=getattr(model, "model_id", config.model.model_id),
                prompt=prompt,
                raw_text="",
                parsed_answer=None,
                generation=config.generation.to_dict(),
                device=ctx.device.resolved,
                latency_s=None,
                extras={"error": str(exc)},
            )
            error = str(exc)

        artifacts.append(artifact)
        parsed = artifact.parsed_answer
        parse_failed = is_parse_failure(artifact.raw_text, parsed) or error is not None
        correct = (error is None) and (not parse_failed) and exact_match(
            parsed, example.ground_truth
        )
        # Exact-match may still score when parse_failed is False; if parse failed,
        # force incorrect so failures are never silently dropped from the denominator.
        if parse_failed or error is not None:
            correct = False
        predictions.append(
            BaselinePrediction(
                example_id=example.example_id,
                clip_id=example.clip_id,
                trick_id=example.trick_id,
                split=example.split.value,
                question=example.question,
                ground_truth=example.ground_truth,
                prompt=prompt,
                prompt_template_id=PROMPT_TEMPLATE_ID,
                raw_text=artifact.raw_text,
                parsed_answer=parsed,
                parse_failed=parse_failed,
                correct=correct,
                latency_s=artifact.latency_s,
                model_id=artifact.model_id,
                generation=artifact.generation,
                preprocessing=dict(artifact.preprocessing),
                device=artifact.device,
                frame_indices=artifact.frame_indices,
                error=error,
            )
        )

    summary = evaluate_baseline(examples, artifacts, predictions)
    result = BaselineResult(
        run_id=ctx.run_id,
        run_dir=str(ctx.run_dir),
        split=eval_split.value,
        summary=summary,
        predictions=tuple(predictions),
        artifacts=tuple(artifacts),
        prompt_template=prompt_template,
        immutable=config.baseline_immutable,
    )
    _write_baseline_outputs(ctx, config, result)
    logger.info(
        "Baseline complete run_id=%s split=%s n=%s accuracy=%s parse_failures=%s",
        result.run_id,
        result.split,
        summary.n_examples,
        summary.overall_accuracy,
        summary.n_parse_failures,
    )
    return result


def _write_baseline_outputs(
    ctx: ExperimentContext,
    config: ExperimentConfig,
    result: BaselineResult,
) -> None:
    write_jsonl(ctx.run_dir / "predictions.jsonl", [p.to_dict() for p in result.predictions])
    if config.preserve_raw_outputs:
        write_jsonl(
            ctx.run_dir / "raw_responses.jsonl",
            [
                {
                    "example_id": a.example_id,
                    "clip_id": a.clip_id,
                    "raw_text": a.raw_text,
                    "parsed_answer": a.parsed_answer,
                    "generation": a.generation,
                    "latency_s": a.latency_s,
                }
                for a in result.artifacts
            ],
        )
    write_json(ctx.run_dir / "metrics.json", result.summary.to_dict())
    write_json(ctx.run_dir / "baseline_summary.json", result.to_dict())
    write_json(
        ctx.run_dir / "BASELINE_IMMUTABLE.json",
        {
            "immutable": True,
            "message": (
                "This zero-shot baseline is the frozen reference. Do not overwrite; "
                "compare later post-training runs against this run_id."
            ),
            "run_id": result.run_id,
            "split": result.split,
            "model_id": config.model.model_id,
            "dataset_version": config.dataset.version,
            "config_hash": ctx.record.config_hash,
        },
    )
