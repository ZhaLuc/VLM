"""Bradley-Terry preference reward model (small, text-conditioned).

Loss for preferred ``y_w`` and rejected ``y_l`` given context ``x``:

    L = -log(sigmoid(r(x, y_w) - r(x, y_l)))

Scientific limits
-----------------
* Preference-model agreement is **not** ground-truth reasoning accuracy.
* This stage uses a **small text encoder** over (task, clip_id, instruction,
  response). Clip identity + instruction are the task/video *representation*
  for small preference sets; pixel/video encoders are deferred.
* Training refuses ``held_out`` preference rows (final benchmark protection).
* Does **not** launch GRPO/DPO of the VLM.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from magic_vlm.preferences import dpo_training_rows, load_preference_pairs
from magic_vlm.schemas import PreferencePair, Split
from magic_vlm.utils import utc_now_iso, write_json, write_jsonl

logger = logging.getLogger("magic_vlm")

INTEGRITY_DISCLAIMER = (
    "Preference-model agreement measures how well the reward model predicts "
    "human pairwise choices. It is NOT equivalent to factual correctness or "
    "ground-truth reasoning accuracy on the hidden-state benchmark."
)


class RewardModelError(ValueError):
    """Invalid reward-model configuration or data."""


def bradley_terry_loss(reward_win: Any, reward_lose: Any) -> Any:
    """Numerically stable BT loss: -log(sigmoid(r_w - r_l))."""
    import torch
    import torch.nn.functional as F

    return -F.logsigmoid(reward_win - reward_lose)


def bradley_terry_loss_value(delta: float) -> float:
    """Scalar reference: -log(sigmoid(delta)) = softplus(-delta)."""
    x = -float(delta)
    if x > 50:
        return x
    if x < -50:
        return float(math.exp(x))
    return float(math.log1p(math.exp(x)))


@dataclass(frozen=True)
class PreferenceExample:
    """One chosen/rejected training row (ties excluded)."""

    judgment_id: str
    pair_id: str
    clip_id: str
    instruction: str
    chosen: str
    rejected: str
    task: str
    split: str
    example_id: str | None = None

    def context_text(self) -> str:
        """Task/video identity + instruction (x without response)."""
        return (
            f"[TASK] {self.task}\n"
            f"[CLIP] {self.clip_id}\n"
            f"[INSTRUCTION] {self.instruction}"
        )

    def pair_text(self, response: str) -> str:
        """Full (x, y) text fed to the reward encoder."""
        return f"{self.context_text()}\n[RESPONSE] {response}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preference_examples_from_pairs(
    pairs: Sequence[PreferencePair],
) -> list[PreferenceExample]:
    rows = dpo_training_rows(pairs)
    return [
        PreferenceExample(
            judgment_id=str(r["judgment_id"]),
            pair_id=str(r["pair_id"]),
            clip_id=str(r["clip_id"]),
            instruction=str(r["instruction"]),
            chosen=str(r["chosen"]),
            rejected=str(r["rejected"]),
            task=str(r["task"]),
            split=str(r["split"]),
            example_id=None if r.get("example_id") is None else str(r["example_id"]),
        )
        for r in rows
    ]


def assert_no_held_out_preferences(examples: Sequence[PreferenceExample]) -> None:
    held = [ex for ex in examples if ex.split == Split.HELD_OUT.value]
    if held:
        raise RewardModelError(
            f"Refusing {len(held)} held_out preference row(s) for reward-model "
            "training/validation fitting. Use train/val only."
        )


def split_preference_examples(
    examples: Sequence[PreferenceExample],
    *,
    seed: int = 0,
    val_fraction: float = 0.25,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Build train/val sets. Uses labeled splits when present; else seeded split.

    Never places ``held_out`` into either set.
    """
    assert_no_held_out_preferences(examples)
    usable = [ex for ex in examples if ex.split != Split.HELD_OUT.value]
    train = [ex for ex in usable if ex.split == Split.TRAIN.value]
    val = [ex for ex in usable if ex.split == Split.VAL.value]
    unlabeled = [ex for ex in usable if ex.split not in {Split.TRAIN.value, Split.VAL.value}]
    if train or val:
        if unlabeled:
            raise RewardModelError(
                "Mix of labeled train/val and unlabeled preference splits is unsupported"
            )
        if not train:
            raise RewardModelError("No train-split preferences available for reward-model fitting")
        if not val and val_fraction > 0:
            # Carve validation from train with a seeded shuffle (documented).
            rng = random.Random(seed)
            shuffled = list(train)
            rng.shuffle(shuffled)
            n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
            if n_val == 0:
                return shuffled, []
            return shuffled[n_val:], shuffled[:n_val]
        return train, val

    # All rows lack train/val labels (should be rare); seeded random split.
    rng = random.Random(seed)
    shuffled = list(usable)
    rng.shuffle(shuffled)
    if len(shuffled) < 2 or val_fraction <= 0:
        return shuffled, []
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    n_val = min(n_val, len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


@dataclass(frozen=True)
class RewardModelConfig:
    """Small BT reward-model hyperparameters (dataset-size appropriate)."""

    prefs_path: str
    output_dir: str = "runs/reward_model"
    run_id: str | None = None
    seed: int = 0
    embedding_dim: int = 64
    hidden_dim: int = 64
    max_length: int = 256
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 4
    val_fraction: float = 0.25
    device: str = "cpu"
    dataset_version: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardModelConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {k: v for k, v in dict(data).items() if k in known}
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RewardModelConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise RewardModelError(f"Config must be a mapping: {path}")
        return cls.from_dict(raw)


class SimpleTokenizer:
    """Whitespace tokenizer with a frozen vocab (small-data friendly)."""

    PAD = 0
    UNK = 1

    def __init__(self, word_to_id: dict[str, int]) -> None:
        self.word_to_id = dict(word_to_id)
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}

    @classmethod
    def build(cls, texts: Sequence[str], *, min_count: int = 1) -> SimpleTokenizer:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(text.lower().split())
        word_to_id = {"<pad>": cls.PAD, "<unk>": cls.UNK}
        for word, count in sorted(counts.items()):
            if count >= min_count and word not in word_to_id:
                word_to_id[word] = len(word_to_id)
        return cls(word_to_id)

    def encode(self, text: str, *, max_length: int) -> list[int]:
        ids = [self.word_to_id.get(tok, self.UNK) for tok in text.lower().split()]
        if not ids:
            ids = [self.UNK]
        ids = ids[:max_length]
        if len(ids) < max_length:
            ids = ids + [self.PAD] * (max_length - len(ids))
        return ids

    def to_dict(self) -> dict[str, Any]:
        return {"word_to_id": self.word_to_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimpleTokenizer:
        return cls(dict(data["word_to_id"]))


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RewardModelError(
            "PyTorch is required for the Bradley-Terry reward model. "
            "Install with: pip install -e '.[models]'"
        ) from exc
    return torch


class TextRewardModel:
    """Embedding-bag + MLP scalar reward r(x, y).

    Conditions on task/clip/instruction/response text. Not a VLM.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        pad_id: int = 0,
    ) -> None:
        torch = _require_torch()
        import torch.nn as nn

        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._torch = torch

    def parameters(self):  # noqa: ANN201
        yield from self.embedding.parameters()
        yield from self.mlp.parameters()

    def to(self, device: str) -> TextRewardModel:
        self.embedding.to(device)
        self.mlp.to(device)
        return self

    def train(self, mode: bool = True) -> TextRewardModel:
        self.embedding.train(mode)
        self.mlp.train(mode)
        return self

    def eval(self) -> TextRewardModel:
        return self.train(False)

    def state_dict(self) -> dict[str, Any]:
        return {
            "embedding": self.embedding.state_dict(),
            "mlp": self.mlp.state_dict(),
            "pad_id": self.pad_id,
            "vocab_size": int(self.embedding.num_embeddings),
            "embedding_dim": int(self.embedding.embedding_dim),
            "hidden_dim": int(self.mlp[0].out_features),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.embedding.load_state_dict(state["embedding"])
        self.mlp.load_state_dict(state["mlp"])

    def forward_tokens(self, token_ids: Any) -> Any:
        """token_ids: (batch, seq) long → (batch,) rewards."""
        emb = self.embedding(token_ids)
        mask = (token_ids != self.pad_id).unsqueeze(-1).float()
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        return self.mlp(pooled).squeeze(-1)

    def score_texts(
        self,
        texts: Sequence[str],
        *,
        tokenizer: SimpleTokenizer,
        max_length: int,
        device: str = "cpu",
    ) -> list[float]:
        torch = self._torch
        self.eval()
        ids = [tokenizer.encode(t, max_length=max_length) for t in texts]
        tensor = torch.tensor(ids, dtype=torch.long, device=device)
        with torch.no_grad():
            scores = self.forward_tokens(tensor)
        return [float(x) for x in scores.detach().cpu().tolist()]


def build_model_from_tokenizer(
    tokenizer: SimpleTokenizer,
    *,
    embedding_dim: int,
    hidden_dim: int,
) -> TextRewardModel:
    return TextRewardModel(
        vocab_size=len(tokenizer.word_to_id),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        pad_id=SimpleTokenizer.PAD,
    )


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_preference_accuracy: float
    val_loss: float | None
    val_preference_accuracy: float | None
    n_train: int
    n_val: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewardTrainResult:
    run_dir: str
    checkpoint_path: str
    config: dict[str, Any]
    history: tuple[EpochMetrics, ...]
    best_val_preference_accuracy: float | None
    best_epoch: int | None
    n_train: int
    n_val: int
    disclaimer: str = INTEGRITY_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "checkpoint_path": self.checkpoint_path,
            "config": dict(self.config),
            "history": [h.to_dict() for h in self.history],
            "best_val_preference_accuracy": self.best_val_preference_accuracy,
            "best_epoch": self.best_epoch,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "disclaimer": self.disclaimer,
        }


def _batch_indices(n: int, batch_size: int, *, rng: random.Random) -> list[list[int]]:
    order = list(range(n))
    rng.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def _encode_pair_batch(
    examples: Sequence[PreferenceExample],
    indices: Sequence[int],
    *,
    tokenizer: SimpleTokenizer,
    max_length: int,
    device: str,
) -> tuple[Any, Any]:
    torch = _require_torch()
    win_ids = [
        tokenizer.encode(examples[i].pair_text(examples[i].chosen), max_length=max_length)
        for i in indices
    ]
    lose_ids = [
        tokenizer.encode(examples[i].pair_text(examples[i].rejected), max_length=max_length)
        for i in indices
    ]
    return (
        torch.tensor(win_ids, dtype=torch.long, device=device),
        torch.tensor(lose_ids, dtype=torch.long, device=device),
    )


def evaluate_preference_accuracy(
    model: TextRewardModel,
    examples: Sequence[PreferenceExample],
    *,
    tokenizer: SimpleTokenizer,
    max_length: int,
    device: str = "cpu",
    batch_size: int = 16,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Return (preference_accuracy, mean_bt_loss, per-example scores)."""
    if not examples:
        return float("nan"), float("nan"), []
    torch = _require_torch()
    model.eval()
    correct = 0
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        batch = list(range(start, min(start + batch_size, len(examples))))
        win_t, lose_t = _encode_pair_batch(
            examples,
            batch,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
        )
        with torch.no_grad():
            r_w = model.forward_tokens(win_t)
            r_l = model.forward_tokens(lose_t)
            loss = bradley_terry_loss(r_w, r_l)
        for j, idx in enumerate(batch):
            rw = float(r_w[j].item())
            rl = float(r_l[j].item())
            ok = rw > rl
            correct += int(ok)
            losses.append(float(loss[j].item()) if loss.ndim else float(loss.item()))
            ex = examples[idx]
            rows.append(
                {
                    "judgment_id": ex.judgment_id,
                    "pair_id": ex.pair_id,
                    "clip_id": ex.clip_id,
                    "split": ex.split,
                    "reward_chosen": rw,
                    "reward_rejected": rl,
                    "correct_order": ok,
                    "delta": rw - rl,
                    "bt_loss": losses[-1],
                    "disclaimer": INTEGRITY_DISCLAIMER,
                }
            )
    acc = correct / len(examples)
    mean_loss = sum(losses) / len(losses)
    return acc, mean_loss, rows


def save_checkpoint(
    path: str | Path,
    *,
    model: TextRewardModel,
    tokenizer: SimpleTokenizer,
    config: RewardModelConfig,
    metrics: dict[str, Any] | None = None,
) -> None:
    torch = _require_torch()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "config": config.to_dict(),
        "metrics": metrics or {},
        "disclaimer": INTEGRITY_DISCLAIMER,
        "architecture": {
            "type": "text_embedding_mlp_bradley_terry",
            "conditions_on": ["task", "clip_id", "instruction", "response"],
            "video_pixels": False,
            "note": (
                "Clip_id + instruction provide task/video identity conditioning "
                "for small preference sets; pixel encoders are not used here."
            ),
        },
        "saved_at": utc_now_iso(),
    }
    torch.save(payload, out)


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[TextRewardModel, SimpleTokenizer, dict[str, Any]]:
    torch = _require_torch()
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(Path(path), map_location=device)
    tokenizer = SimpleTokenizer.from_dict(payload["tokenizer"])
    state = payload["model_state"]
    model = TextRewardModel(
        vocab_size=int(state["vocab_size"]),
        embedding_dim=int(state["embedding_dim"]),
        hidden_dim=int(state["hidden_dim"]),
        pad_id=int(state.get("pad_id", 0)),
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer, payload


def score_preference_file(
    prefs_path: str | Path,
    checkpoint_path: str | Path,
    *,
    out_path: str | Path | None = None,
    device: str = "cpu",
    max_length: int | None = None,
) -> list[dict[str, Any]]:
    """Reward inference only (no training)."""
    model, tokenizer, payload = load_checkpoint(checkpoint_path, device=device)
    cfg = payload.get("config") or {}
    max_len = int(max_length or cfg.get("max_length") or 256)
    pairs = load_preference_pairs(prefs_path)
    examples = preference_examples_from_pairs(pairs)
    _, _, rows = evaluate_preference_accuracy(
        model,
        examples,
        tokenizer=tokenizer,
        max_length=max_len,
        device=device,
    )
    if out_path is not None:
        write_jsonl(out_path, rows)
    return rows


def train_bradley_terry_reward_model(
    config: RewardModelConfig,
    *,
    pairs: Sequence[PreferencePair] | None = None,
) -> RewardTrainResult:
    """Fit a small BT reward model; validate with held-out preference agreement."""
    torch = _require_torch()

    if pairs is None:
        pairs = load_preference_pairs(config.prefs_path)
    examples = preference_examples_from_pairs(pairs)
    if len(examples) < 2:
        raise RewardModelError("Need at least 2 non-tie preference judgments to train")

    train_ex, val_ex = split_preference_examples(
        examples,
        seed=config.seed,
        val_fraction=config.val_fraction,
    )
    assert_no_held_out_preferences(train_ex)
    assert_no_held_out_preferences(val_ex)
    if not train_ex:
        raise RewardModelError("Empty training split after filtering")

    run_id = config.run_id or f"bt_rm_{utc_now_iso().replace(':', '').replace('+', '_')}"
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    corpus = []
    for ex in train_ex:
        corpus.append(ex.pair_text(ex.chosen))
        corpus.append(ex.pair_text(ex.rejected))
    tokenizer = SimpleTokenizer.build(corpus)
    model = build_model_from_tokenizer(
        tokenizer,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(config.device)

    # Deterministic init seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Re-init after seeding
    model = build_model_from_tokenizer(
        tokenizer,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(config.device)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=config.learning_rate)
    history: list[EpochMetrics] = []
    best_val_acc: float | None = None
    best_epoch: int | None = None
    best_path = run_dir / "checkpoint_best.pt"
    last_path = run_dir / "checkpoint_last.pt"

    meta = {
        "config": config.to_dict(),
        "dataset": {
            "prefs_path": config.prefs_path,
            "dataset_version": config.dataset_version,
            "n_pairs_loaded": len(pairs),
            "n_train": len(train_ex),
            "n_val": len(val_ex),
            "train_judgment_ids": [e.judgment_id for e in train_ex],
            "val_judgment_ids": [e.judgment_id for e in val_ex],
        },
        "architecture": {
            "type": "text_embedding_mlp_bradley_terry",
            "vocab_size": len(tokenizer.word_to_id),
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "conditions_on": ["task", "clip_id", "instruction", "response"],
            "video_pixels": False,
        },
        "optimization": {
            "optimizer": "Adam",
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "seed": config.seed,
        },
        "hardware": {"device": config.device, "torch": torch.__version__},
        "disclaimer": INTEGRITY_DISCLAIMER,
        "started_at": utc_now_iso(),
    }
    write_json(run_dir / "train_metadata.json", meta)

    rng = random.Random(config.seed)
    for epoch in range(1, config.epochs + 1):
        model.train(True)
        total_loss = 0.0
        n_batches = 0
        for batch in _batch_indices(len(train_ex), config.batch_size, rng=rng):
            win_t, lose_t = _encode_pair_batch(
                train_ex,
                batch,
                tokenizer=tokenizer,
                max_length=config.max_length,
                device=config.device,
            )
            r_w = model.forward_tokens(win_t)
            r_l = model.forward_tokens(lose_t)
            loss_vec = bradley_terry_loss(r_w, r_l)
            loss = loss_vec.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        train_acc, train_loss_eval, train_rows = evaluate_preference_accuracy(
            model,
            train_ex,
            tokenizer=tokenizer,
            max_length=config.max_length,
            device=config.device,
            batch_size=config.batch_size,
        )
        val_acc: float | None = None
        val_loss: float | None = None
        val_rows: list[dict[str, Any]] = []
        if val_ex:
            val_acc, val_loss, val_rows = evaluate_preference_accuracy(
                model,
                val_ex,
                tokenizer=tokenizer,
                max_length=config.max_length,
                device=config.device,
                batch_size=config.batch_size,
            )

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=total_loss / max(n_batches, 1),
            train_preference_accuracy=train_acc,
            val_loss=val_loss,
            val_preference_accuracy=val_acc,
            n_train=len(train_ex),
            n_val=len(val_ex),
        )
        history.append(metrics)
        logger.info(
            "BT-RM epoch=%s train_loss=%.4f train_acc=%.3f val_loss=%s val_acc=%s",
            epoch,
            metrics.train_loss,
            metrics.train_preference_accuracy,
            None if val_loss is None else round(val_loss, 4),
            val_acc,
        )
        write_jsonl(run_dir / "metrics.jsonl", [h.to_dict() for h in history])

        save_checkpoint(
            last_path,
            model=model,
            tokenizer=tokenizer,
            config=config,
            metrics=metrics.to_dict(),
        )
        if val_acc is not None and (best_val_acc is None or val_acc >= best_val_acc):
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                best_path,
                model=model,
                tokenizer=tokenizer,
                config=config,
                metrics=metrics.to_dict(),
            )
            write_jsonl(run_dir / "val_scores.jsonl", val_rows)
        write_jsonl(run_dir / "train_scores.jsonl", train_rows)

    if best_val_acc is None:
        # No val set: keep last as best.
        save_checkpoint(
            best_path,
            model=model,
            tokenizer=tokenizer,
            config=config,
            metrics=history[-1].to_dict() if history else {},
        )
        best_epoch = history[-1].epoch if history else None

    # Overfitting snapshot
    overfit = None
    if history and history[-1].val_preference_accuracy is not None:
        overfit = {
            "final_train_acc": history[-1].train_preference_accuracy,
            "final_val_acc": history[-1].val_preference_accuracy,
            "gap_train_minus_val": history[-1].train_preference_accuracy
            - float(history[-1].val_preference_accuracy),
            "note": (
                "A large positive gap suggests overfitting to preference labels; "
                "still not a measure of reasoning correctness."
            ),
        }
    result = RewardTrainResult(
        run_dir=str(run_dir),
        checkpoint_path=str(best_path),
        config=config.to_dict(),
        history=tuple(history),
        best_val_preference_accuracy=best_val_acc,
        best_epoch=best_epoch,
        n_train=len(train_ex),
        n_val=len(val_ex),
    )
    write_json(
        run_dir / "train_result.json",
        {**result.to_dict(), "overfitting": overfit, "finished_at": utc_now_iso()},
    )
    write_json(run_dir / "DISCLAIMER.json", {"disclaimer": INTEGRITY_DISCLAIMER})
    return result
