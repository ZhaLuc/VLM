"""Temporal-order diagnostic: ordered vs shuffled presentation of the *same* frames.

Not a causal-reasoning proof. Not a training dataset. Does not resample.

For each example:
1. Sample frames once (existing video pipeline).
2. Present them in temporal order.
3. Present the **same** frames shuffled with a reproducible seed.
4. Use the identical question, model, prompt, generation settings, and scorer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from magic_vlm.dataset import filter_split, load_manifest
from magic_vlm.experiment import ExperimentConfig, initialize_experiment
from magic_vlm.inference import build_prompt, run_inference
from magic_vlm.models import load_vlm
from magic_vlm.rewards import HiddenStateExactMatchReward
from magic_vlm.schemas import ExampleRecord, InferenceArtifact, Split
from magic_vlm.temporal_metrics import (
    INTEGRITY_NOTE,
    SHUFFLE_METHOD,
    TemporalShuffleSummary,
    paired_outcome,
    summarize_pairs,
)
from magic_vlm.utils import write_json, write_jsonl
from magic_vlm.video import (
    SampledClip,
    VideoPreprocessConfig,
    ordered_and_shuffled_pair,
)

PROMPT_TEMPLATE_ID = "hidden_state_v1"


class TemporalShuffleError(ValueError):
    """Invalid temporal-shuffle experiment configuration or pairing."""


@dataclass(frozen=True)
class PairedTemporalResult:
    """One example under ordered vs shuffled presentation."""

    example_id: str
    clip_id: str
    question: str
    ground_truth: str | None
    prompt: str
    prompt_template_id: str
    model_id: str
    generation: dict[str, Any]
    shuffle_seed: int
    shuffle_method: str
    ordered_indices: tuple[int, ...]
    shuffled_frame_indices: tuple[int, ...]
    ordered_raw: str
    shuffled_raw: str
    ordered_parsed: str | None
    shuffled_parsed: str | None
    ordered_correct: bool
    shuffled_correct: bool
    ordered_reward: float
    shuffled_reward: float
    same_frame_set: bool
    order_changed: bool
    ordered_preprocessing: dict[str, Any]
    shuffled_preprocessing: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ordered_indices"] = list(self.ordered_indices)
        payload["shuffled_frame_indices"] = list(self.shuffled_frame_indices)
        payload["outcome"] = self.outcome
        payload["diagnostic"] = "temporal_order"
        return payload

    @property
    def outcome(self) -> str:
        return paired_outcome(self.ordered_correct, self.shuffled_correct)


def assert_shared_sample(ordered: SampledClip, shuffled: SampledClip) -> None:
    """Same underlying sampled frames; only presentation order may change."""
    if tuple(ordered.ordered_indices) != tuple(shuffled.ordered_indices):
        raise TemporalShuffleError(
            "ordered_indices differ; shuffle must not resample"
        )
    if set(ordered.frame_indices) != set(shuffled.frame_indices):
        raise TemporalShuffleError("shuffled presentation does not use the same frame set")
    if ordered.temporal_shuffled:
        raise TemporalShuffleError("ordered clip must have temporal_shuffled=False")
    if not shuffled.temporal_shuffled:
        raise TemporalShuffleError("shuffled clip must have temporal_shuffled=True")
    if ordered.sample_strategy != shuffled.sample_strategy:
        raise TemporalShuffleError("sample_strategy must match across conditions")
    if ordered.max_frames != shuffled.max_frames:
        raise TemporalShuffleError("max_frames must match across conditions")


def assert_shuffle_permutes(ordered: SampledClip, shuffled: SampledClip, *, seed: int) -> None:
    """Refuse a no-op shuffle when more than one frame was sampled."""
    if len(ordered.frame_indices) <= 1:
        return
    if tuple(ordered.frame_indices) == tuple(shuffled.frame_indices):
        raise TemporalShuffleError(
            f"shuffle_seed={seed} left presentation order unchanged "
            f"(identity permutation of {list(ordered.ordered_indices)}). "
            "Choose a seed that permutes the sampled frames; seed 0 is identity "
            "for some short sample sets with this LCG Fisher-Yates shuffle."
        )


def pair_example(
    example: ExampleRecord,
    *,
    ordered: SampledClip,
    shuffled: SampledClip,
    ordered_artifact: InferenceArtifact,
    shuffled_artifact: InferenceArtifact,
    shuffle_seed: int,
    prompt: str,
    reward: HiddenStateExactMatchReward | None = None,
) -> PairedTemporalResult:
    assert_shared_sample(ordered, shuffled)
    if ordered_artifact.prompt != shuffled_artifact.prompt:
        raise TemporalShuffleError("prompt wording must be identical across conditions")
    if ordered_artifact.prompt != prompt:
        raise TemporalShuffleError("artifact prompt does not match experiment prompt")
    if ordered_artifact.generation != shuffled_artifact.generation:
        raise TemporalShuffleError("generation settings must be identical")
    if ordered_artifact.model_id != shuffled_artifact.model_id:
        raise TemporalShuffleError("model_id must be identical")
    if example.question not in prompt:
        raise TemporalShuffleError("prompt must contain the original question wording")

    scorer = reward or HiddenStateExactMatchReward()
    ord_res = scorer.evaluate(ordered_artifact, example)
    shf_res = scorer.evaluate(shuffled_artifact, example)
    return PairedTemporalResult(
        example_id=example.example_id,
        clip_id=example.clip_id,
        question=example.question,
        ground_truth=example.ground_truth,
        prompt=prompt,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        model_id=ordered_artifact.model_id,
        generation=dict(ordered_artifact.generation),
        shuffle_seed=shuffle_seed,
        shuffle_method=SHUFFLE_METHOD,
        ordered_indices=tuple(ordered.ordered_indices),
        shuffled_frame_indices=tuple(shuffled.frame_indices),
        ordered_raw=ordered_artifact.raw_text,
        shuffled_raw=shuffled_artifact.raw_text,
        ordered_parsed=ordered_artifact.parsed_answer,
        shuffled_parsed=shuffled_artifact.parsed_answer,
        ordered_correct=bool(ord_res.matched),
        shuffled_correct=bool(shf_res.matched),
        ordered_reward=float(ord_res.value),
        shuffled_reward=float(shf_res.value),
        same_frame_set=set(ordered.frame_indices) == set(shuffled.frame_indices),
        order_changed=tuple(ordered.frame_indices) != tuple(shuffled.frame_indices),
        ordered_preprocessing=dict(ordered_artifact.preprocessing),
        shuffled_preprocessing=dict(shuffled_artifact.preprocessing),
    )


def _require_num_frames(example: ExampleRecord) -> int:
    n = example.video.num_frames
    if n is None:
        raise TemporalShuffleError(
            f"{example.example_id}: video.num_frames is required so sampling "
            "does not probe/resample a different frame count"
        )
    return int(n)


def run_temporal_shuffle_experiment(
    config: ExperimentConfig,
    *,
    split: str | None = None,
    run_id: str | None = None,
    shuffle_seed: int | None = None,
    load_frames: bool = False,
    allow_download: bool | None = None,
) -> tuple[TemporalShuffleSummary, tuple[PairedTemporalResult, ...], Path]:
    """Paired diagnostic on one split. Shuffle outputs are never for training."""
    if config.training_method != "none":
        raise TemporalShuffleError(
            "Temporal-shuffle diagnostic forbids training_method != none "
            "(shuffle data must not be used for training)"
        )
    eval_split = Split(split or config.dataset.split or Split.HELD_OUT.value)
    seed = int(config.video.shuffle_seed if shuffle_seed is None else shuffle_seed)
    # Sampling is independent of presentation shuffle (existing pipeline).
    video_cfg = VideoPreprocessConfig(
        max_frames=config.video.max_frames,
        sample_strategy=config.video.sample_strategy,
        temporal_shuffle=False,
        shuffle_seed=seed,
        resize=config.video.resize,
    )

    ctx = initialize_experiment(config, run_id=run_id)
    examples = filter_split(load_manifest(config.dataset.manifest), eval_split)
    if not examples:
        raise TemporalShuffleError(f"No examples in split {eval_split.value}")

    download = config.allow_model_download if allow_download is None else allow_download
    model = load_vlm(config.model, allow_download=download, device=config.device)
    scorer = HiddenStateExactMatchReward()
    pairs: list[PairedTemporalResult] = []

    for example in examples:
        num_frames = _require_num_frames(example)
        ordered, shuffled = ordered_and_shuffled_pair(
            example.video.path,
            config=video_cfg,
            num_frames=num_frames,
            shuffle_seed=seed,
            load_frames=load_frames,
        )
        assert_shuffle_permutes(ordered, shuffled, seed=seed)
        prompt = build_prompt(example)
        ordered_art = run_inference(
            model,
            example,
            preprocessed=ordered,
            generation=config.generation,
            prompt=prompt,
            device=ctx.device,
            checkpoint_kind=config.checkpoint.kind,
            checkpoint_path=config.checkpoint.path,
        )
        shuffled_art = run_inference(
            model,
            example,
            preprocessed=shuffled,
            generation=config.generation,
            prompt=prompt,
            device=ctx.device,
            checkpoint_kind=config.checkpoint.kind,
            checkpoint_path=config.checkpoint.path,
        )
        pairs.append(
            pair_example(
                example,
                ordered=ordered,
                shuffled=shuffled,
                ordered_artifact=ordered_art,
                shuffled_artifact=shuffled_art,
                shuffle_seed=seed,
                prompt=prompt,
                reward=scorer,
            )
        )

    summary = summarize_pairs(pairs, shuffle_seed=seed, split=eval_split.value)
    _write_outputs(ctx.run_dir, config, summary, pairs, seed)
    return summary, tuple(pairs), ctx.run_dir


def _write_outputs(
    run_dir: Path,
    config: ExperimentConfig,
    summary: TemporalShuffleSummary,
    pairs: Sequence[PairedTemporalResult],
    seed: int,
) -> None:
    write_json(run_dir / "temporal_shuffle_summary.json", summary.to_dict())
    write_jsonl(run_dir / "temporal_shuffle_pairs.jsonl", [p.to_dict() for p in pairs])
    write_json(
        run_dir / "temporal_shuffle_metadata.json",
        {
            "diagnostic": "temporal_order",
            "integrity_note": INTEGRITY_NOTE,
            "shuffle_method": SHUFFLE_METHOD,
            "shuffle_seed": seed,
            "sampled_frames": [
                {
                    "example_id": p.example_id,
                    "clip_id": p.clip_id,
                    "ordered_indices": list(p.ordered_indices),
                    "shuffled_frame_indices": list(p.shuffled_frame_indices),
                    "same_frame_set": p.same_frame_set,
                    "order_changed": p.order_changed,
                }
                for p in pairs
            ],
            "model_id": config.model.model_id,
            "model_revision": config.model.revision,
            "checkpoint": config.checkpoint.to_dict(),
            "generation": config.generation.to_dict(),
            "video": config.video.to_dict(),
            "prompt_template": PROMPT_TEMPLATE_ID,
            "prompt_note": "identical wording for ordered and shuffled conditions",
            "split": summary.split,
            "training_use": False,
            "resampled": False,
            "evaluation": {
                "reward_id": "hidden_state_exact_match",
                "version": "1.0.0",
            },
        },
    )
    write_json(run_dir / "metrics.json", summary.to_dict())
