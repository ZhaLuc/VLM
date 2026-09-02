"""Modular objective rewards for future GRPO (trainer not implemented here).

Design
------
Parsing, canonicalization, and reward calculation are **separate**. A future
GRPO trainer should depend only on ``ObjectiveReward`` / ``RewardResult`` and
must not embed task-specific logic.

Initial reward
--------------
``hidden_state_exact_match`` @ version ``1.0.0``:
returns ``1.0`` if the hidden-state answer matches gold after parse+canonicalize,
else ``0.0``. Malformed / unparsable responses are **never** treated as correct.

This reward is an **objective correctness signal for a short label**, not a
reasoning-quality metric.

Known shortcut risks (documented, not mitigated by this module alone):
answer-frequency exploitation, parser exploitation, camera/leakage cues.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from magic_vlm.evaluation import exact_match, is_parse_failure, normalize_label
from magic_vlm.inference import parse_answer
from magic_vlm.schemas import ExampleRecord, InferenceArtifact, TaskType

# ---------------------------------------------------------------------------
# Versioned reward identity
# ---------------------------------------------------------------------------

HIDDEN_STATE_EXACT_MATCH_ID = "hidden_state_exact_match"
HIDDEN_STATE_EXACT_MATCH_VERSION = "1.0.0"

# Reserved IDs for future extensions (not implemented).
TEMPORAL_LOCALIZATION_ID = "temporal_localization_correctness"
TEMPORAL_IOU_ID = "temporal_iou"
EXPLANATION_REWARD_ID = "explanation_reward"
HYBRID_REWARD_ID = "hybrid_reward"

SHORTCUT_RISKS = (
    "answer_frequency_exploitation: model may emit majority labels without video reasoning",
    "parser_exploitation: model may format text to satisfy parse_answer heuristics",
    "camera_leakage: cues correlated with camera_id/performer may substitute for mechanism understanding",
)


class RewardError(ValueError):
    """Invalid reward configuration or unsupported reward id."""


@dataclass(frozen=True)
class RewardResult:
    """Structured reward output for logging and future GRPO consumption."""

    value: float
    reward_id: str
    version: str
    parse_failed: bool = False
    prediction: str | None = None
    gold: str | None = None
    matched: bool = False
    notes: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.value) is not float:
            object.__setattr__(self, "value", float(self.value))
        if self.value not in (0.0, 1.0) and self.reward_id == HIDDEN_STATE_EXACT_MATCH_ID:
            # Exact-match objective is strictly binary; other future rewards may differ.
            pass

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class ObjectiveReward(Protocol):
    """Trainer-facing objective reward. No GRPO/optimizer knowledge allowed."""

    reward_id: str
    version: str

    def evaluate(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> RewardResult:
        """Return a structured reward (preferred GRPO entrypoint)."""
        ...

    def score(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> float:
        """Return only the scalar reward value."""
        ...


# Backward-compatible alias used by earlier stages.
@runtime_checkable
class RewardFunction(Protocol):
    """Legacy score-only protocol. Prefer ``ObjectiveReward`` for GRPO."""

    name: str

    def score(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> float:
        ...


# ---------------------------------------------------------------------------
# Separated concerns: parse → canonicalize → match
# ---------------------------------------------------------------------------

ParseFn = Callable[[str], str | None]


def extract_prediction(
    artifact: InferenceArtifact,
    *,
    parse: ParseFn | None = None,
) -> tuple[str | None, bool]:
    """Extract a candidate label without mutating ``artifact.raw_text``.

    Returns ``(prediction, parse_failed)``.
    Prefer stored ``parsed_answer`` when present; otherwise run ``parse``.
    """
    raw = artifact.raw_text if artifact.raw_text is not None else ""
    if artifact.parsed_answer is not None:
        pred: str | None = artifact.parsed_answer
    elif parse is not None:
        pred = parse(raw)
    else:
        pred = parse_answer(raw)
    failed = is_parse_failure(raw, pred)
    if failed:
        return pred, True
    return pred, False


def canonicalize_label(value: str | None) -> str:
    """Compare-time canonicalization only (never writes back to the dataset)."""
    return normalize_label(value)


def labels_exact_match(prediction: str | None, gold: str | None) -> bool:
    """Canonical exact-match comparison for hidden-state labels."""
    return exact_match(prediction, gold)


def compute_hidden_state_exact_match_value(
    *,
    prediction: str | None,
    gold: str | None,
    parse_failed: bool,
) -> float:
    """Binary objective: 1.0 if usable parse and exact match, else 0.0.

    Malformed / failed parses are never correct.
    """
    if parse_failed:
        return 0.0
    if prediction is None or not str(prediction).strip():
        return 0.0
    if gold is None or not str(gold).strip():
        return 0.0
    return 1.0 if labels_exact_match(prediction, gold) else 0.0


# ---------------------------------------------------------------------------
# Initial implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HiddenStateExactMatchReward:
    """Versioned 0/1 reward for Task-B hidden-state answers.

    Not a reasoning metric. Suitable as the first GRPO objective signal.
    """

    reward_id: str = HIDDEN_STATE_EXACT_MATCH_ID
    version: str = HIDDEN_STATE_EXACT_MATCH_VERSION
    name: str = HIDDEN_STATE_EXACT_MATCH_ID  # legacy alias
    parse: ParseFn | None = None

    def evaluate(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> RewardResult:
        if example.task is not TaskType.HIDDEN_STATE:
            return RewardResult(
                value=0.0,
                reward_id=self.reward_id,
                version=self.version,
                parse_failed=False,
                prediction=None,
                gold=example.ground_truth,
                matched=False,
                notes=f"reward applies to hidden_state only (got {example.task.value})",
                extras={"shortcut_risks": list(SHORTCUT_RISKS)},
            )
        pred, parse_failed = extract_prediction(artifact, parse=self.parse)
        gold = example.ground_truth
        value = compute_hidden_state_exact_match_value(
            prediction=pred,
            gold=gold,
            parse_failed=parse_failed,
        )
        matched = value == 1.0
        return RewardResult(
            value=float(value),
            reward_id=self.reward_id,
            version=self.version,
            parse_failed=parse_failed,
            prediction=pred,
            gold=gold,
            matched=matched,
            notes=None if matched else ("parse_failed" if parse_failed else "label_mismatch"),
            extras={
                "canonical_prediction": canonicalize_label(pred),
                "canonical_gold": canonicalize_label(gold),
                "shortcut_risks": list(SHORTCUT_RISKS),
                "is_reasoning_metric": False,
            },
        )

    def score(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> float:
        return float(self.evaluate(artifact, example, **kwargs).value)


@dataclass(frozen=True)
class ExactMatchReward:
    """Legacy name for ``HiddenStateExactMatchReward`` (same semantics)."""

    name: str = "exact_match"
    _inner: HiddenStateExactMatchReward = field(default_factory=HiddenStateExactMatchReward)

    @property
    def reward_id(self) -> str:
        return self._inner.reward_id

    @property
    def version(self) -> str:
        return self._inner.version

    def evaluate(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> RewardResult:
        return self._inner.evaluate(artifact, example, **kwargs)

    def score(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> float:
        return self._inner.score(artifact, example, **kwargs)


@dataclass(frozen=True)
class LengthPenaltyReward:
    """Diagnostic reward-hacking probe only — **not** for GRPO composition.

    Intentionally multiplies a base score by a length factor. Do not register
    this as a production GRPO objective; the modular interface forbids combining
    rewards at the trainer layer in this stage.
    """

    name: str = "length_penalty_diagnostic"
    max_chars: int = 200
    base: RewardFunction = field(default_factory=ExactMatchReward)

    def score(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> float:
        base_score = self.base.score(artifact, example, **kwargs)
        penalty = min(len(artifact.raw_text) / float(self.max_chars), 1.0)
        return float(base_score) * (1.0 - 0.5 * penalty)


# ---------------------------------------------------------------------------
# Future extension points (stubs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalLocalizationRewardStub:
    """Extension point for temporal/causal localization correctness (unimplemented)."""

    reward_id: str = TEMPORAL_LOCALIZATION_ID
    version: str = "0.0.0-stub"
    name: str = TEMPORAL_LOCALIZATION_ID

    def evaluate(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> RewardResult:
        raise NotImplementedError(
            "temporal_localization_correctness is a reserved extension point; "
            "not implemented (GRPO trainer also not implemented)."
        )

    def score(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> float:
        raise NotImplementedError(self.evaluate.__doc__)


@dataclass(frozen=True)
class TemporalIoURewardStub:
    """Extension point for temporal IoU reward (unimplemented)."""

    reward_id: str = TEMPORAL_IOU_ID
    version: str = "0.0.0-stub"
    name: str = TEMPORAL_IOU_ID

    def evaluate(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> RewardResult:
        raise NotImplementedError(
            "temporal_iou is a reserved extension point; not implemented."
        )

    def score(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> float:
        raise NotImplementedError("temporal_iou is not implemented")


# ---------------------------------------------------------------------------
# Registry / factory (single reward; no hybrid mixing)
# ---------------------------------------------------------------------------

_REWARD_BUILDERS: dict[str, Callable[[], ObjectiveReward]] = {
    HIDDEN_STATE_EXACT_MATCH_ID: HiddenStateExactMatchReward,
    "exact_match": ExactMatchReward,  # legacy config alias
}


def list_registered_rewards() -> dict[str, str]:
    """Return reward_id -> status for discoverability."""
    return {
        HIDDEN_STATE_EXACT_MATCH_ID: f"implemented@{HIDDEN_STATE_EXACT_MATCH_VERSION}",
        "exact_match": f"legacy_alias->{HIDDEN_STATE_EXACT_MATCH_ID}",
        TEMPORAL_LOCALIZATION_ID: "stub",
        TEMPORAL_IOU_ID: "stub",
        EXPLANATION_REWARD_ID: "reserved",
        HYBRID_REWARD_ID: "reserved_not_combined_in_this_stage",
    }


def build_reward(reward_id: str) -> ObjectiveReward:
    """Construct one registered objective reward (no hybrids)."""
    key = str(reward_id).strip()
    if key in {EXPLANATION_REWARD_ID, HYBRID_REWARD_ID}:
        raise RewardError(
            f"Reward {key!r} is reserved and not implemented. "
            "Do not combine rewards in this stage."
        )
    if key == TEMPORAL_LOCALIZATION_ID:
        return TemporalLocalizationRewardStub()
    if key == TEMPORAL_IOU_ID:
        return TemporalIoURewardStub()
    builder = _REWARD_BUILDERS.get(key)
    if builder is None:
        raise RewardError(
            f"Unknown reward_id {key!r}. Registered: {sorted(_REWARD_BUILDERS)}"
        )
    return builder()


@dataclass(frozen=True)
class RewardConfig:
    """YAML-friendly reward selection for experiments."""

    reward_id: str = HIDDEN_STATE_EXACT_MATCH_ID
    version: str | None = None  # if set, must match implementation version

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RewardConfig:
        raw = dict(data or {})
        return cls(
            reward_id=str(raw.get("reward_id", HIDDEN_STATE_EXACT_MATCH_ID)),
            version=None if raw.get("version") is None else str(raw.get("version")),
        )

    @classmethod
    def from_yaml(cls, path: str) -> RewardConfig:
        from pathlib import Path

        import yaml

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise RewardError(f"Reward config must be a mapping: {path}")
        # Allow top-level or nested "reward:" block
        if "reward_id" not in payload and isinstance(payload.get("reward"), dict):
            payload = payload["reward"]
        return cls.from_dict(payload)

    def build(self) -> ObjectiveReward:
        reward = build_reward(self.reward_id)
        if self.version is not None and getattr(reward, "version", None) != self.version:
            raise RewardError(
                f"Requested version {self.version!r} but {self.reward_id} is "
                f"{getattr(reward, 'version', None)!r}"
            )
        return reward


def score_batch(
    reward: RewardFunction | ObjectiveReward,
    artifacts: list[InferenceArtifact],
    examples: list[ExampleRecord],
) -> list[float]:
    by_id = {example.example_id: example for example in examples}
    scores: list[float] = []
    for artifact in artifacts:
        example = by_id[artifact.example_id]
        scores.append(float(reward.score(artifact, example)))
    return scores


def evaluate_batch(
    reward: ObjectiveReward,
    artifacts: list[InferenceArtifact],
    examples: list[ExampleRecord],
) -> list[RewardResult]:
    by_id = {example.example_id: example for example in examples}
    return [reward.evaluate(artifact, by_id[artifact.example_id]) for artifact in artifacts]
