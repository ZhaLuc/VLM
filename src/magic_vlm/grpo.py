"""GRPO post-training on modular objective rewards (hidden-state exact match).

Uses TRL ``GRPOTrainer`` + optional PEFT/LoRA. Reward logic stays in
``magic_vlm.rewards`` - this module only adapts datasets and invokes rewards.

Scientific limits
-----------------
* Reward increase is **not** proof of reasoning improvement.
* Do not train on ``held_out`` examples or modify held-out membership.
* Do not select checkpoints using final held-out test scores.
* Real Qwen2.5-VL GRPO needs CUDA, local VLM weights, and a compatible TRL
  multimodal path; text smoke proves plumbing only.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from magic_vlm.dataset import filter_split, filter_task, load_manifest
from magic_vlm.inference import GenerationConfig, build_prompt, parse_answer
from magic_vlm.rewards import (
    HIDDEN_STATE_EXACT_MATCH_ID,
    HIDDEN_STATE_EXACT_MATCH_VERSION,
    HiddenStateExactMatchReward,
    ObjectiveReward,
    build_reward,
)
from magic_vlm.schemas import ExampleRecord, InferenceArtifact, Split, TaskType
from magic_vlm.utils import allocate_run_directory, utc_now_iso, write_json, write_jsonl

logger = logging.getLogger("magic_vlm")

INTEGRITY_DISCLAIMER = (
    "GRPO reward gains (including hidden-state exact-match) are not proof of "
    "reasoning improvement. Compare independently on the fixed held-out "
    "benchmark. Do not select checkpoints using final test performance. "
    "Failed rollouts must be scored (typically 0.0), never silently dropped."
)

PROTECTED_BASELINE_MARKERS = ("BASELINE_IMMUTABLE.json",)


class GRPOError(ValueError):
    """Invalid GRPO configuration, data, or environment."""


@dataclass(frozen=True)
class GRPOStackInfo:
    """Verified local GRPO / VLM compatibility snapshot."""

    torch_version: str | None
    cuda_available: bool
    transformers_version: str | None
    trl_version: str | None
    peft_version: str | None
    accelerate_version: str | None
    datasets_version: str | None
    vllm_available: bool
    trl_has_grpo_trainer: bool
    trl_mentions_mm_token_type_ids: bool
    trl_mentions_vision: bool
    trl_mentions_video: bool
    ready_for_text_grpo: bool
    ready_for_vlm_grpo: bool
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_grpo_stack() -> GRPOStackInfo:
    """Inspect installed packages; do not invent unsupported API behavior."""
    limitations: list[str] = []
    torch_version = None
    cuda_available = False
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        limitations.append("torch not installed")

    def _ver(name: str) -> str | None:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "unknown")
        except ImportError:
            limitations.append(f"{name} not installed")
            return None

    transformers_version = _ver("transformers")
    trl_version = _ver("trl")
    peft_version = _ver("peft")
    accelerate_version = _ver("accelerate")
    datasets_version = _ver("datasets")

    vllm_available = False
    try:
        import importlib.util

        vllm_available = importlib.util.find_spec("vllm") is not None
    except Exception:  # noqa: BLE001
        vllm_available = False
    if not vllm_available:
        limitations.append("vLLM not installed (optional for TRL generation acceleration)")

    trl_has_grpo = False
    mentions_mm = False
    mentions_vision = False
    mentions_video = False
    if trl_version is not None:
        try:
            import inspect
            from pathlib import Path as _Path

            from trl import GRPOConfig, GRPOTrainer

            trl_has_grpo = True
            try:
                src_path = _Path(inspect.getfile(GRPOTrainer))
                src = src_path.read_text(encoding="utf-8")
            except OSError:
                src = inspect.getsource(GRPOTrainer)
            mentions_mm = "mm_token_type_ids" in src
            mentions_vision = "vision" in src.lower() or "image" in src
            mentions_video = "video" in src.lower()
            _ = GRPOConfig  # noqa: F841
        except Exception as exc:  # noqa: BLE001
            limitations.append(f"trl GRPOTrainer import/probe failed: {exc}")

    if not cuda_available:
        limitations.append(
            "CUDA unavailable: full Qwen2.5-VL GRPO is not practical; text smoke may use CPU."
        )
    if trl_has_grpo and not mentions_mm:
        limitations.append(
            "Installed TRL GRPOTrainer source lacks mm_token_type_ids mentions; "
            "multimodal Qwen GRPO may be unsafe until TRL vision path is verified."
        )
    if peft_version is None:
        limitations.append("peft not installed: LoRA path unavailable")

    ready_text = (
        torch_version is not None
        and trl_has_grpo
        and transformers_version is not None
        and datasets_version is not None
    )
    ready_vlm = (
        ready_text
        and cuda_available
        and peft_version is not None
        and mentions_mm
        and mentions_vision
    )
    if ready_text and not ready_vlm:
        limitations.append(
            "Text-mode GRPO plumbing may run; multimodal Qwen2.5-VL GRPO is blocked "
            "until CUDA + PEFT + compatible TRL multimodal path are available."
        )

    return GRPOStackInfo(
        torch_version=torch_version,
        cuda_available=cuda_available,
        transformers_version=transformers_version,
        trl_version=trl_version,
        peft_version=peft_version,
        accelerate_version=accelerate_version,
        datasets_version=datasets_version,
        vllm_available=vllm_available,
        trl_has_grpo_trainer=trl_has_grpo,
        trl_mentions_mm_token_type_ids=mentions_mm,
        trl_mentions_vision=mentions_vision,
        trl_mentions_video=mentions_video,
        ready_for_text_grpo=ready_text,
        ready_for_vlm_grpo=ready_vlm,
        limitations=tuple(limitations),
    )


@dataclass(frozen=True)
class GRPOConfigSpec:
    """YAML-serializable GRPO experiment configuration."""

    manifest: str
    output_dir: str = "runs/grpo"
    run_id: str | None = None
    model_id: str = "sshleifer/tiny-gpt2"
    baseline_run_dir: str | None = None
    dataset_version: str | None = None
    reward_id: str = HIDDEN_STATE_EXACT_MATCH_ID
    reward_version: str = HIDDEN_STATE_EXACT_MATCH_VERSION
    num_generations: int = 2  # group size
    learning_rate: float = 1e-6
    max_steps: int = 10
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    logging_steps: int = 1
    save_steps: int = 10
    seed: int = 0
    max_completion_length: int = 64
    max_prompt_length: int | None = 256
    temperature: float = 0.7
    top_p: float = 1.0
    beta: float = 0.0
    use_peft: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("c_attn",)
    allow_download: bool = False
    modality: str = "text"  # text | vision
    require_vlm_ready: bool = False
    checkpoint_selection: str = "last_train_step"  # never final held_out test
    eval_held_out_after_train: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.num_generations < 2:
            raise GRPOError("num_generations (group size) must be >= 2 for GRPO")
        if "held_out" in self.checkpoint_selection or self.checkpoint_selection in {
            "test",
            "final_test",
            "heldout",
        }:
            raise GRPOError(
                "checkpoint_selection must not use final held-out/test performance"
            )
        if self.reward_id != HIDDEN_STATE_EXACT_MATCH_ID:
            # First GRPO experiment is intentionally scoped; other rewards stay callable
            # via build_reward but are not the default training target yet.
            pass

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lora_target_modules"] = list(self.lora_target_modules)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GRPOConfigSpec:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {k: v for k, v in dict(data).items() if k in known}
        if "lora_target_modules" in payload and payload["lora_target_modules"] is not None:
            payload["lora_target_modules"] = tuple(payload["lora_target_modules"])
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GRPOConfigSpec:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise GRPOError(f"GRPO config must be a mapping: {path}")
        return cls.from_dict(raw)


def assert_baseline_not_overwritten(baseline_run_dir: str | Path | None, output_dir: Path) -> None:
    """Refuse writing into an immutable baseline run directory."""
    if baseline_run_dir is None:
        return
    base = Path(baseline_run_dir).resolve()
    out = output_dir.resolve()
    if out == base or base in out.parents:
        raise GRPOError(
            f"Refusing to write GRPO outputs into baseline run dir {base}. "
            "Baseline checkpoints must remain untouched."
        )
    for marker in PROTECTED_BASELINE_MARKERS:
        if (out / marker).exists():
            raise GRPOError(
                f"Output dir {out} contains {marker}; refusing to overwrite baseline artifacts"
            )


def examples_to_grpo_records(
    examples: Sequence[ExampleRecord],
    *,
    allow_held_out: bool = False,
) -> list[dict[str, Any]]:
    """Adapt ExampleRecords to TRL GRPO rows (prompt + metadata columns).

    Does not rewrite questions or ground_truth. Refuses held_out by default.
    """
    rows: list[dict[str, Any]] = []
    for example in examples:
        if (not allow_held_out) and example.split is Split.HELD_OUT:
            raise GRPOError(
                f"Refusing held_out example {example.example_id!r} for GRPO training"
            )
        if example.task is not TaskType.HIDDEN_STATE:
            continue
        if example.ground_truth is None or not str(example.ground_truth).strip():
            raise GRPOError(
                f"hidden_state example {example.example_id!r} lacks ground_truth"
            )
        prompt = build_prompt(example)
        rows.append(
            {
                "prompt": prompt,
                "example_id": example.example_id,
                "clip_id": example.clip_id,
                "trick_id": example.trick_id,
                "question": example.question,
                "ground_truth": example.ground_truth,
                "task": example.task.value,
                "split": example.split.value,
            }
        )
    if not rows:
        raise GRPOError("No usable hidden_state examples for GRPO")
    return rows


def split_grpo_records(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [r for r in records if r["split"] == Split.TRAIN.value]
    val = [r for r in records if r["split"] == Split.VAL.value]
    if not train:
        raise GRPOError("No train-split GRPO records")
    return train, val


def build_hf_dataset(records: Sequence[dict[str, Any]]) -> Any:
    from datasets import Dataset

    return Dataset.from_list(list(records))


def completion_to_text(completion: Any) -> str:
    """Normalize TRL completion payloads to raw text (never mutate dataset gold)."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", item)
                parts.append(str(content))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def make_objective_reward_fn(
    examples_by_id: dict[str, ExampleRecord],
    reward: ObjectiveReward,
    *,
    completion_log: list[dict[str, Any]] | None = None,
) -> Callable[..., list[float]]:
    """Build a TRL-compatible reward callable that delegates to ``ObjectiveReward``.

    Failed / malformed completions score 0.0 (not dropped / not None).
    """

    def reward_fn(
        prompts: Sequence[Any],
        completions: Sequence[Any],
        example_id: Sequence[str] | None = None,
        ground_truth: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        n = len(completions)
        ids = list(example_id) if example_id is not None else [None] * n
        golds = list(ground_truth) if ground_truth is not None else [None] * n
        if len(ids) != n:
            # TRL repeats columns for num_generations; lengths must match completions.
            raise GRPOError(
                f"example_id length {len(ids)} != completions length {n}; "
                "failed rollouts must remain aligned, not dropped"
            )
        values: list[float] = []
        for i in range(n):
            raw = completion_to_text(completions[i])
            eid = ids[i]
            example = examples_by_id.get(str(eid)) if eid is not None else None
            if example is None:
                # Still score 0 rather than dropping the rollout.
                value = 0.0
                parse_failed = True
                matched = False
                notes = "missing_example_for_rollout"
                pred = parse_answer(raw) if raw.strip() else None
            else:
                artifact = InferenceArtifact(
                    example_id=example.example_id,
                    model_id="grpo_rollout",
                    prompt=completion_to_text(prompts[i]) if i < len(prompts) else build_prompt(example),
                    raw_text=raw,
                    parsed_answer=parse_answer(raw) if raw.strip() else None,
                    clip_id=example.clip_id,
                    task=example.task.value,
                    question=example.question,
                )
                result = reward.evaluate(artifact, example)
                value = float(result.value)
                parse_failed = bool(result.parse_failed)
                matched = bool(result.matched)
                notes = result.notes
                pred = result.prediction
            values.append(value)
            if completion_log is not None:
                completion_log.append(
                    {
                        "example_id": eid,
                        "ground_truth": None if example is None else example.ground_truth,
                        "dataset_ground_truth_column": golds[i] if i < len(golds) else None,
                        "raw_completion": raw,
                        "parsed": pred,
                        "reward": value,
                        "parse_failed": parse_failed,
                        "matched": matched,
                        "notes": notes,
                        "reward_id": reward.reward_id,
                        "reward_version": reward.version,
                    }
                )
        return values

    reward_fn.__name__ = f"objective_{reward.reward_id}"
    return reward_fn


@dataclass(frozen=True)
class GRPOTrainResult:
    run_dir: str
    checkpoint_dir: str
    status: str
    stack: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, Any]
    held_out_eval: dict[str, Any] | None = None
    disclaimer: str = INTEGRITY_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_run_dir(config: GRPOConfigSpec) -> Path:
    run_id = config.run_id or f"grpo_{utc_now_iso().replace(':', '').replace('+', '_')}"
    run_dir = allocate_run_directory(config.output_dir, run_id, overwrite=False)
    assert_baseline_not_overwritten(config.baseline_run_dir, run_dir)
    return run_dir


def _build_lora_config(config: GRPOConfigSpec) -> Any:
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def evaluate_hidden_state_split(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[ExampleRecord],
    reward: ObjectiveReward,
    generation: GenerationConfig | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Independent evaluation helper (not used for checkpoint selection)."""
    gen = generation or GenerationConfig(max_new_tokens=32, temperature=0.0, do_sample=False)
    rows: list[dict[str, Any]] = []
    n_correct = 0
    for example in examples:
        prompt = build_prompt(example)
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = model.generate(
            **encoded,
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.do_sample,
            temperature=gen.temperature if gen.do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = int(encoded["input_ids"].shape[-1])
        completion_ids = out[0][prompt_len:]
        raw = tokenizer.decode(completion_ids, skip_special_tokens=True)
        artifact = InferenceArtifact(
            example_id=example.example_id,
            model_id=getattr(model, "name_or_path", "grpo_eval"),
            prompt=prompt,
            raw_text=raw,
            parsed_answer=parse_answer(raw) if raw.strip() else None,
            clip_id=example.clip_id,
            task=example.task.value,
            question=example.question,
            generation=gen.to_dict(),
        )
        scored = reward.evaluate(artifact, example)
        if scored.matched:
            n_correct += 1
        rows.append(
            {
                "example_id": example.example_id,
                "split": example.split.value,
                "raw_text": raw,
                "parsed_answer": artifact.parsed_answer,
                "ground_truth": example.ground_truth,
                "reward": scored.value,
                "matched": scored.matched,
                "parse_failed": scored.parse_failed,
            }
        )
    n = len(examples)
    return {
        "n_examples": n,
        "n_correct": n_correct,
        "accuracy": (n_correct / n) if n else None,
        "reward_id": reward.reward_id,
        "reward_version": reward.version,
        "rows": rows,
        "note": (
            "Independent evaluation on a fixed split. Not used for checkpoint selection. "
            "Reward accuracy is not a reasoning-quality claim."
        ),
    }


def train_grpo(config: GRPOConfigSpec) -> GRPOTrainResult:
    """Run GRPO via TRL with an external objective reward."""
    stack = probe_grpo_stack()
    if config.modality == "vision" and config.require_vlm_ready and not stack.ready_for_vlm_grpo:
        raise GRPOError(
            "VLM GRPO requested but stack is not ready_for_vlm_grpo. "
            f"Limitations: {list(stack.limitations)}"
        )
    if not stack.ready_for_text_grpo:
        raise GRPOError(
            "Text GRPO stack incomplete. "
            f"Limitations: {list(stack.limitations)}"
        )
    if config.modality == "vision" and not stack.ready_for_vlm_grpo:
        raise GRPOError(
            "Vision GRPO is not runnable in this environment. "
            f"Limitations: {list(stack.limitations)}"
        )

    all_examples = load_manifest(config.manifest)
    train_examples = filter_task(
        filter_split(all_examples, Split.TRAIN), TaskType.HIDDEN_STATE
    )
    val_examples = filter_task(filter_split(all_examples, Split.VAL), TaskType.HIDDEN_STATE)
    held_out_examples = filter_task(
        filter_split(all_examples, Split.HELD_OUT), TaskType.HIDDEN_STATE
    )
    if any(ex.split is Split.HELD_OUT for ex in train_examples):
        raise GRPOError("held_out leaked into training examples")

    train_records = examples_to_grpo_records(train_examples, allow_held_out=False)
    val_records = (
        examples_to_grpo_records(val_examples, allow_held_out=False) if val_examples else []
    )

    run_dir = _resolve_run_dir(config)
    write_json(run_dir / "stack_probe.json", stack.to_dict())
    write_json(run_dir / "config.json", config.to_dict())
    write_jsonl(run_dir / "grpo_train_records.jsonl", train_records)
    write_jsonl(run_dir / "grpo_val_records.jsonl", val_records)
    write_json(
        run_dir / "DISCLAIMER.json",
        {
            "disclaimer": INTEGRITY_DISCLAIMER,
            "checkpoint_selection": config.checkpoint_selection,
            "reward_id": config.reward_id,
            "reward_version": config.reward_version,
            "num_generations": config.num_generations,
        },
    )
    if config.baseline_run_dir:
        base = Path(config.baseline_run_dir)
        write_json(
            run_dir / "baseline_protection.json",
            {
                "baseline_run_dir": str(base),
                "baseline_exists": base.exists(),
                "protected": True,
                "message": "GRPO outputs are isolated from the immutable baseline run.",
            },
        )

    reward = build_reward(config.reward_id)
    if config.reward_version and getattr(reward, "version", None) != config.reward_version:
        raise GRPOError(
            f"Requested reward version {config.reward_version!r} but "
            f"{config.reward_id} is {getattr(reward, 'version', None)!r}"
        )

    examples_by_id = {ex.example_id: ex for ex in train_examples + val_examples + held_out_examples}
    completion_log: list[dict[str, Any]] = []
    reward_fn = make_objective_reward_fn(
        examples_by_id, reward, completion_log=completion_log
    )

    train_ds = build_hf_dataset(train_records)
    eval_ds = build_hf_dataset(val_records) if val_records else None

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    model_kwargs: dict[str, Any] = {}
    if not config.allow_download:
        model_kwargs["local_files_only"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, **model_kwargs)
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    except Exception as exc:
        raise GRPOError(
            f"Failed to load model_id={config.model_id!r}. "
            "Provide a local checkpoint or allow_download for smoke. "
            f"Underlying error: {exc}"
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = None
    if config.use_peft:
        if stack.peft_version is None:
            raise GRPOError("use_peft=True but peft is not installed")
        peft_config = _build_lora_config(config)

    use_cpu = not stack.cuda_available
    grpo_args = GRPOConfig(
        output_dir=str(run_dir / "trainer_out"),
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        seed=config.seed,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        beta=config.beta,
        remove_unused_columns=False,
        report_to=[],
        bf16=False,
        fp16=False,
        use_cpu=use_cpu,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_output = trainer.train()
    ckpt_dir = run_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    # Preserve raw completions collected during reward calls.
    write_jsonl(run_dir / "raw_completions.jsonl", completion_log)

    log_history = list(getattr(trainer.state, "log_history", []) or [])
    write_jsonl(run_dir / "train_log.jsonl", log_history)

    # Per-step reward statistics from completion log (aligned to rollouts).
    rewards = [float(r["reward"]) for r in completion_log]
    reward_stats = {
        "n_rollouts_logged": len(rewards),
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "n_reward_one": sum(1 for r in rewards if r == 1.0),
        "n_reward_zero": sum(1 for r in rewards if r == 0.0),
        "n_parse_failed": sum(1 for r in completion_log if r.get("parse_failed")),
        "reward_id": config.reward_id,
        "reward_version": config.reward_version,
    }
    write_json(run_dir / "reward_stats.json", reward_stats)

    metrics: dict[str, Any] = {
        "train_runtime": getattr(train_output, "metrics", {}).get("train_runtime"),
        "train_loss": getattr(train_output, "metrics", {}).get("train_loss"),
        "global_step": getattr(trainer.state, "global_step", None),
        "num_generations": config.num_generations,
        "reward_stats": reward_stats,
        "checkpoint_selection": config.checkpoint_selection,
    }
    if hasattr(train_output, "metrics") and isinstance(train_output.metrics, dict):
        metrics.update({f"trainer_{k}": v for k, v in train_output.metrics.items()})

    hardware = {
        "cuda_available": stack.cuda_available,
        "torch_version": stack.torch_version,
        "device": "cuda" if stack.cuda_available else "cpu",
        "vllm_available": stack.vllm_available,
    }
    meta = {
        "base_checkpoint": config.model_id,
        "reward_id": config.reward_id,
        "reward_version": config.reward_version,
        "group_size_num_generations": config.num_generations,
        "generation": {
            "max_completion_length": config.max_completion_length,
            "temperature": config.temperature,
            "top_p": config.top_p,
        },
        "optimizer": "trl_grpo_default_adamw",
        "learning_rate": config.learning_rate,
        "batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "seed": config.seed,
        "peft": {
            "use_peft": config.use_peft,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_target_modules": list(config.lora_target_modules),
        },
        "hardware": hardware,
        "training_steps": config.max_steps,
        "checkpoint_selection": config.checkpoint_selection,
        "modality": config.modality,
        "dataset_version": config.dataset_version,
        "manifest": config.manifest,
        "n_train": len(train_records),
        "n_val": len(val_records),
        "n_held_out_available": len(held_out_examples),
        "created_at": utc_now_iso(),
        "disclaimer": INTEGRITY_DISCLAIMER,
    }
    write_json(run_dir / "train_metadata.json", meta)

    held_out_eval = None
    if config.eval_held_out_after_train and held_out_examples:
        # Independent held-out eval - never feeds checkpoint selection.
        import torch

        device = "cuda" if stack.cuda_available and torch.cuda.is_available() else "cpu"
        model_for_eval = trainer.model
        model_for_eval.to(device)
        model_for_eval.eval()
        held_out_eval = evaluate_hidden_state_split(
            model=model_for_eval,
            tokenizer=tokenizer,
            examples=held_out_examples,
            reward=reward,
            generation=GenerationConfig(
                max_new_tokens=min(32, config.max_completion_length),
                temperature=0.0,
                do_sample=False,
            ),
            device=device,
        )
        write_jsonl(run_dir / "held_out_eval_rows.jsonl", held_out_eval["rows"])
        write_json(
            run_dir / "held_out_eval.json",
            {k: v for k, v in held_out_eval.items() if k != "rows"},
        )

    if config.baseline_run_dir:
        base = Path(config.baseline_run_dir)
        if base.exists():
            marker = base / "BASELINE_IMMUTABLE.json"
            write_json(
                run_dir / "baseline_unchanged_check.json",
                {
                    "baseline_run_dir": str(base),
                    "immutable_marker_present": marker.exists(),
                    "checked_at": utc_now_iso(),
                },
            )

    write_json(
        run_dir / "comparison_plan.json",
        {
            "baseline_run_dir": config.baseline_run_dir,
            "grpo_run_dir": str(run_dir),
            "grpo_checkpoint": str(ckpt_dir),
            "held_out_eval_path": "held_out_eval.json",
            "note": (
                "Compare untouched base VLM vs GRPO-adapted VLM on the same held-out "
                "benchmark using an independent eval run. Do not cherry-pick "
                "checkpoints with final test scores."
            ),
        },
    )

    result = GRPOTrainResult(
        run_dir=str(run_dir),
        checkpoint_dir=str(ckpt_dir),
        status="completed",
        stack=stack.to_dict(),
        config=config.to_dict(),
        metrics=metrics,
        held_out_eval=(
            None
            if held_out_eval is None
            else {k: v for k, v in held_out_eval.items() if k != "rows"}
        ),
    )
    write_json(run_dir / "result.json", result.to_dict())
    write_json(run_dir / "metrics.json", metrics)
    logger.info("GRPO complete run_dir=%s checkpoint=%s", run_dir, ckpt_dir)
    return result


def load_grpo_checkpoint_dir(path: str | Path) -> Path:
    """Verify a GRPO checkpoint directory exists and is loadable as PEFT/causal LM."""
    ckpt = Path(path)
    if not ckpt.exists():
        raise GRPOError(f"GRPO checkpoint not found: {ckpt}")
    has_weights = any(ckpt.glob("*.safetensors")) or any(ckpt.glob("pytorch_model*.bin"))
    has_adapter = (ckpt / "adapter_config.json").exists() or (
        ckpt / "adapter_model.safetensors"
    ).exists()
    if not (has_weights or has_adapter or (ckpt / "config.json").exists()):
        raise GRPOError(f"GRPO checkpoint directory looks empty/unusable: {ckpt}")
    return ckpt
