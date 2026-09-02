"""DPO post-training for explanation preferences (base VLM vs DPO-adapted).

Uses TRL ``DPOTrainer`` + optional PEFT/LoRA. Never overwrites immutable
baseline checkpoints. Never trains on ``held_out`` preference rows.

Scientific limits
-----------------
* Loss reduction / lower rejected log-prob is **not** proof of reasoning gain.
* Preference alignment is not factual correctness on the hidden-state benchmark.
* Real Qwen2.5-VL DPO needs CUDA, local weights, and video/image columns when
  multimodal conditioning is required; text-only smoke proves plumbing only.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from magic_vlm.preferences import dpo_training_rows, load_preference_pairs
from magic_vlm.schemas import PreferencePair, Split
from magic_vlm.utils import utc_now_iso, write_json, write_jsonl

logger = logging.getLogger("magic_vlm")

INTEGRITY_DISCLAIMER = (
    "DPO loss reduction and lower likelihood of rejected responses do not equal "
    "reasoning improvement. Preference agreement is not ground-truth factual "
    "correctness on the held-out hidden-state benchmark. Checkpoints must not be "
    "selected using final test performance."
)

PROTECTED_BASELINE_MARKERS = ("BASELINE_IMMUTABLE.json",)


class DPOError(ValueError):
    """Invalid DPO configuration, data, or environment."""


@dataclass(frozen=True)
class DPOStackInfo:
    """Verified local framework compatibility snapshot."""

    torch_version: str | None
    cuda_available: bool
    transformers_version: str | None
    trl_version: str | None
    peft_version: str | None
    accelerate_version: str | None
    datasets_version: str | None
    trl_has_dpo_trainer: bool
    trl_mentions_mm_token_type_ids: bool
    trl_mentions_vision_collator: bool
    ready_for_text_dpo: bool
    ready_for_vlm_dpo: bool
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_dpo_stack() -> DPOStackInfo:
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

    trl_has_dpo = False
    mentions_mm = False
    mentions_vision = False
    if trl_version is not None:
        try:
            import inspect
            from pathlib import Path as _Path

            from trl import DPOConfig, DPOTrainer

            trl_has_dpo = True
            try:
                src_path = _Path(inspect.getfile(DPOTrainer))
                src = src_path.read_text(encoding="utf-8")
            except OSError:
                src = inspect.getsource(DPOTrainer)
            mentions_mm = "mm_token_type_ids" in src
            mentions_vision = "DataCollatorForVisionPreference" in src
            _ = DPOConfig  # noqa: F841
        except Exception as exc:  # noqa: BLE001
            limitations.append(f"trl DPOTrainer import/probe failed: {exc}")

    if not cuda_available:
        limitations.append(
            "CUDA unavailable: full Qwen2.5-VL DPO is not practical; text smoke may use CPU."
        )
    if trl_has_dpo and not mentions_mm:
        limitations.append(
            "Installed TRL DPOTrainer source lacks mm_token_type_ids handling; "
            "Qwen2.5-VL DPO may silently use wrong positional encodings "
            "(see huggingface/trl#5277). Upgrade TRL before real VLM runs."
        )
    if peft_version is None:
        limitations.append("peft not installed: LoRA path unavailable")

    ready_text = (
        torch_version is not None
        and trl_has_dpo
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
            "Text-mode DPO plumbing may run; multimodal Qwen2.5-VL DPO is blocked "
            "until CUDA + PEFT + compatible TRL vision path are available."
        )

    return DPOStackInfo(
        torch_version=torch_version,
        cuda_available=cuda_available,
        transformers_version=transformers_version,
        trl_version=trl_version,
        peft_version=peft_version,
        accelerate_version=accelerate_version,
        datasets_version=datasets_version,
        trl_has_dpo_trainer=trl_has_dpo,
        trl_mentions_mm_token_type_ids=mentions_mm,
        trl_mentions_vision_collator=mentions_vision,
        ready_for_text_dpo=ready_text,
        ready_for_vlm_dpo=ready_vlm,
        limitations=tuple(limitations),
    )


@dataclass(frozen=True)
class DPOConfigSpec:
    """Project DPO experiment configuration (YAML-serializable)."""

    prefs_path: str
    output_dir: str = "runs/dpo"
    run_id: str | None = None
    model_id: str = "sshleifer/tiny-gpt2"
    reference_model_id: str | None = None  # None => TRL creates frozen ref from model
    baseline_run_dir: str | None = None  # protected; never overwritten
    dataset_version: str | None = None
    beta: float = 0.1
    learning_rate: float = 1e-5
    max_steps: int = 10
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    logging_steps: int = 1
    save_steps: int = 10
    seed: int = 0
    max_length: int | None = 512
    use_peft: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("c_attn",)
    allow_download: bool = False
    modality: str = "text"  # text | vision
    require_vlm_ready: bool = False
    checkpoint_selection: str = "last_train_step"  # never final held_out test
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lora_target_modules"] = list(self.lora_target_modules)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DPOConfigSpec:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {k: v for k, v in dict(data).items() if k in known}
        if "lora_target_modules" in payload and payload["lora_target_modules"] is not None:
            payload["lora_target_modules"] = tuple(payload["lora_target_modules"])
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DPOConfigSpec:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise DPOError(f"DPO config must be a mapping: {path}")
        return cls.from_dict(raw)


def assert_baseline_not_overwritten(baseline_run_dir: str | Path | None, output_dir: Path) -> None:
    """Refuse writing into an immutable baseline run directory."""
    if baseline_run_dir is None:
        return
    base = Path(baseline_run_dir).resolve()
    out = output_dir.resolve()
    if out == base or base in out.parents:
        raise DPOError(
            f"Refusing to write DPO outputs into baseline run dir {base}. "
            "Baseline checkpoints must remain untouched."
        )
    for marker in PROTECTED_BASELINE_MARKERS:
        if (out / marker).exists():
            raise DPOError(
                f"Output dir {out} contains {marker}; refusing to overwrite baseline artifacts"
            )


def preferences_to_dpo_records(
    pairs: Sequence[PreferencePair],
    *,
    allow_held_out: bool = False,
) -> list[dict[str, Any]]:
    """Adapt PreferencePair rows to TRL DPO format without altering raw texts."""
    if not allow_held_out:
        held = [p for p in pairs if p.split is Split.HELD_OUT]
        if held:
            raise DPOError(
                f"Refusing {len(held)} held_out preference row(s) for DPO training"
            )
    rows = dpo_training_rows(pairs)
    # Drop held_out even if somehow present when allow_held_out True for eval-only adapters.
    adapted: list[dict[str, Any]] = []
    for row in rows:
        if (not allow_held_out) and row["split"] == Split.HELD_OUT.value:
            continue
        prompt = str(row["instruction"])
        adapted.append(
            {
                "prompt": prompt,
                "chosen": str(row["chosen"]),
                "rejected": str(row["rejected"]),
                "judgment_id": row["judgment_id"],
                "pair_id": row["pair_id"],
                "clip_id": row["clip_id"],
                "split": row["split"],
                "task": row["task"],
            }
        )
    if not adapted:
        raise DPOError("No usable non-tie preference rows for DPO")
    return adapted


def split_dpo_records(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [r for r in records if r["split"] == Split.TRAIN.value]
    val = [r for r in records if r["split"] == Split.VAL.value]
    if not train:
        # Fall back: all non-val as train if splits unlabeled as train
        other = [r for r in records if r["split"] != Split.VAL.value]
        if not other:
            raise DPOError("No train-split DPO records")
        train = list(other)
    return train, val


def build_hf_dataset(records: Sequence[dict[str, Any]]) -> Any:
    from datasets import Dataset

    # Keep only columns TRL needs for training; retain ids in separate export.
    core = [
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in records
    ]
    return Dataset.from_list(core)


@dataclass(frozen=True)
class DPOTrainResult:
    run_dir: str
    checkpoint_dir: str
    status: str
    stack: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, Any]
    disclaimer: str = INTEGRITY_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_run_dir(config: DPOConfigSpec) -> Path:
    from magic_vlm.utils import allocate_run_directory, utc_now_iso

    run_id = config.run_id or f"dpo_{utc_now_iso().replace(':', '').replace('+', '_')}"
    run_dir = allocate_run_directory(config.output_dir, run_id, overwrite=False)
    assert_baseline_not_overwritten(config.baseline_run_dir, run_dir)
    return run_dir


def _build_lora_config(config: DPOConfigSpec) -> Any:
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def train_dpo(config: DPOConfigSpec) -> DPOTrainResult:
    """Run DPO via TRL. Writes to a new run directory only."""
    stack = probe_dpo_stack()
    if config.modality == "vision" and config.require_vlm_ready and not stack.ready_for_vlm_dpo:
        raise DPOError(
            "VLM DPO requested but stack is not ready_for_vlm_dpo. "
            f"Limitations: {list(stack.limitations)}"
        )
    if not stack.ready_for_text_dpo:
        raise DPOError(
            "Text DPO stack incomplete. "
            f"Limitations: {list(stack.limitations)}"
        )
    if config.modality == "vision" and not stack.ready_for_vlm_dpo:
        logger.warning(
            "modality=vision but ready_for_vlm_dpo=False; refusing to invent APIs. "
            "Falling back is not performed automatically."
        )
        raise DPOError(
            "Vision DPO is not runnable in this environment. "
            f"Limitations: {list(stack.limitations)}"
        )

    pairs = load_preference_pairs(config.prefs_path)
    records = preferences_to_dpo_records(pairs, allow_held_out=False)
    train_records, val_records = split_dpo_records(records)

    run_dir = _resolve_run_dir(config)
    write_json(run_dir / "stack_probe.json", stack.to_dict())
    write_json(run_dir / "config.json", config.to_dict())
    write_jsonl(run_dir / "dpo_train_records.jsonl", train_records)
    write_jsonl(run_dir / "dpo_val_records.jsonl", val_records)
    write_json(
        run_dir / "DISCLAIMER.json",
        {"disclaimer": INTEGRITY_DISCLAIMER, "checkpoint_selection": config.checkpoint_selection},
    )

    # Ensure we never touch baseline marker locations
    if config.baseline_run_dir:
        base = Path(config.baseline_run_dir)
        write_json(
            run_dir / "baseline_protection.json",
            {
                "baseline_run_dir": str(base),
                "baseline_exists": base.exists(),
                "protected": True,
                "message": "DPO outputs are isolated from the immutable baseline run.",
            },
        )

    train_ds = build_hf_dataset(train_records)
    eval_ds = build_hf_dataset(val_records) if val_records else None

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    model_kwargs: dict[str, Any] = {}
    if not config.allow_download:
        model_kwargs["local_files_only"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, **model_kwargs)
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    except Exception as exc:
        raise DPOError(
            f"Failed to load model_id={config.model_id!r}. "
            "Provide a local checkpoint or pass allow_download=true for smoke. "
            f"Underlying error: {exc}"
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = None
    if config.use_peft:
        if stack.peft_version is None:
            raise DPOError("use_peft=True but peft is not installed")
        peft_config = _build_lora_config(config)

    # Reference: TRL freezes a copy when ref_model is None and peft is used.
    ref_model = None
    if config.reference_model_id and not config.use_peft:
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.reference_model_id, **model_kwargs
        )

    use_cpu = not stack.cuda_available
    dpo_args = DPOConfig(
        output_dir=str(run_dir / "trainer_out"),
        beta=config.beta,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        seed=config.seed,
        report_to=[],
        remove_unused_columns=False,
        use_cpu=use_cpu,
        max_length=config.max_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_output = trainer.train()
    # Checkpoint selection: last training step (never held_out test).
    ckpt_dir = run_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    metrics = {
        "train_runtime": getattr(train_output, "metrics", {}),
        "n_train_records": len(train_records),
        "n_val_records": len(val_records),
        "global_step": getattr(trainer.state, "global_step", None),
        "checkpoint_selection": config.checkpoint_selection,
    }
    # Flatten trainer log history if present
    if getattr(trainer, "state", None) is not None and trainer.state.log_history:
        metrics["log_history"] = list(trainer.state.log_history)
        write_jsonl(run_dir / "train_log.jsonl", list(trainer.state.log_history))

    meta = {
        "started_or_finished_at": utc_now_iso(),
        "base_model_id": config.model_id,
        "reference_model_id": config.reference_model_id,
        "reference_handling": (
            "peft_ref_from_base" if config.use_peft and ref_model is None else "explicit_or_none"
        ),
        "preference_dataset": config.prefs_path,
        "dataset_version": config.dataset_version,
        "splits": {"train": len(train_records), "val": len(val_records), "held_out_used": False},
        "beta": config.beta,
        "peft": {
            "enabled": config.use_peft,
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": list(config.lora_target_modules),
        },
        "optimizer": "adamw_torch_via_transformers_trainer",
        "scheduler": "transformers_default",
        "batch_size": config.per_device_train_batch_size,
        "grad_accum": config.gradient_accumulation_steps,
        "max_steps": config.max_steps,
        "seed": config.seed,
        "modality": config.modality,
        "hardware": {
            "cuda_available": stack.cuda_available,
            "torch": stack.torch_version,
            "use_cpu": use_cpu,
        },
        "stack": stack.to_dict(),
        "disclaimer": INTEGRITY_DISCLAIMER,
        "checkpoint_selection_rule": config.checkpoint_selection,
    }
    write_json(run_dir / "train_metadata.json", meta)
    write_json(run_dir / "train_result.json", {**metrics, "disclaimer": INTEGRITY_DISCLAIMER})

    # Verify baseline dir untouched if provided
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

    result = DPOTrainResult(
        run_dir=str(run_dir),
        checkpoint_dir=str(ckpt_dir),
        status="completed",
        stack=stack.to_dict(),
        config=config.to_dict(),
        metrics=metrics,
    )
    write_json(run_dir / "result.json", result.to_dict())
    logger.info("DPO complete run_dir=%s checkpoint=%s", run_dir, ckpt_dir)
    return result


def create_tiny_local_causal_lm(
    texts: Sequence[str],
    out_dir: str | Path,
    *,
    seed: int = 0,
) -> str:
    """Create a tiny randomly-initialized GPT-2-like LM + tokenizer on disk.

    Used for DPO plumbing smoke tests without Hub downloads or VLM weights.
    """
    torch = __import__("torch")
    torch.manual_seed(seed)

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer_backend = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
    tokenizer_backend.train_from_iterator(list(texts) + ["placeholder words for vocab"], trainer)
    tokenizer_backend.save(str(out / "tokenizer.json"))

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(out / "tokenizer.json"),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    hf_tok.save_pretrained(out)

    vocab_size = max(len(hf_tok), 32)
    # Resize embedding to tokenizer size
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=hf_tok.bos_token_id,
        eos_token_id=hf_tok.eos_token_id,
        pad_token_id=hf_tok.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(hf_tok))
    model.save_pretrained(out)
    return str(out)


def load_dpo_checkpoint_dir(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Validate a saved DPO checkpoint directory schema (does not run generation)."""
    path = Path(checkpoint_dir)
    if not path.is_dir():
        raise DPOError(f"Checkpoint dir missing: {path}")
    files = {p.name for p in path.iterdir()}
    ok_signals = (
        "config.json" in files
        or "adapter_config.json" in files
        or "adapter_model.safetensors" in files
        or "adapter_model.bin" in files
        or "model.safetensors" in files
        or "pytorch_model.bin" in files
    )
    if not ok_signals:
        raise DPOError(f"Checkpoint dir lacks recognized model files: {sorted(files)}")
    return {
        "checkpoint_dir": str(path),
        "files": sorted(files),
        "schema_ok": True,
        "disclaimer": INTEGRITY_DISCLAIMER,
    }


def compare_base_vs_dpo_metadata(
    *,
    baseline_run_dir: str | Path | None,
    dpo_run_dir: str | Path,
) -> dict[str, Any]:
    """Record comparison handles without selecting on held-out test scores."""
    payload = {
        "baseline_run_dir": None if baseline_run_dir is None else str(baseline_run_dir),
        "dpo_run_dir": str(dpo_run_dir),
        "protocol": (
            "Compare untouched base vs DPO-adapted model using the established "
            "held-out evaluation runner in a *separate* eval run_id. Do not pick "
            "checkpoints by final test performance."
        ),
        "disclaimer": INTEGRITY_DISCLAIMER,
        "recorded_at": utc_now_iso(),
    }
    write_json(Path(dpo_run_dir) / "comparison_plan.json", payload)
    return payload
