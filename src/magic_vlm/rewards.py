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
from magic_vlm.temporal_parse import (
    interval_iou,
    parse_interval_text,
    resolve_gold_interval_object,
)

# ---------------------------------------------------------------------------
# Versioned reward identity
# ---------------------------------------------------------------------------

HIDDEN_STATE_EXACT_MATCH_ID = "hidden_state_exact_match"
HIDDEN_STATE_EXACT_MATCH_VERSION = "1.0.0"

TEMPORAL_LOCALIZATION_ID = "temporal_localization_correctness"
TEMPORAL_IOU_ID = "temporal_iou"
TEMPORAL_CAUSAL_VERSION = "1.0.0"
EXPLANATION_REWARD_ID = "explanation_reward"
HYBRID_REWARD_ID = "hybrid_reward"

SHORTCUT_RISKS = (
    "answer_frequency_exploitation: model may emit majority labels without video reasoning",
    "parser_exploitation: model may format text to satisfy parse_answer heuristics",
    "camera_leakage: cues correlated with camera_id/performer may substitute for mechanism understanding",
)

TEMPORAL_SHORTCUT_RISKS = (
    "salient_motion_exploitation: model may point at the most visually salient action "
    "rather than a causally responsible one",
    "interval_parser_exploitation: model may emit tagged intervals to satisfy parse heuristics",
    "ambiguous_cause_collapse: scoring a single span when multiple simultaneous actions exist",
)

INTEGRITY_NOTE_TEMPORAL = (
    "Temporal/causal IoU reward requires defensible causal annotations with explicit "
    "status and provenance. Clip-level temporal spans are not causal gold. "
    "A salient action is not a proven causal action. Ambiguous labels are exposed "
    "and not scored as gold. This reward is not a hybrid with hidden-state exact match."
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
# Temporal / causal IoU reward (independently callable; no hybrid weighting)
# ---------------------------------------------------------------------------


def compute_temporal_causal_value(
    *,
    iou: float | None,
    mode: str,
    iou_threshold: float,
    eligible: bool,
    parse_failed: bool,
) -> float:
    """Map IoU + mode to a scalar. Invalid / ineligible cases score 0.0."""
    if not eligible or parse_failed or iou is None:
        return 0.0
    mode_norm = str(mode).strip().lower()
    if mode_norm == "partial":
        return float(max(0.0, min(1.0, iou)))
    if mode_norm == "binary":
        return 1.0 if float(iou) >= float(iou_threshold) else 0.0
    raise RewardError(f"Unsupported temporal reward mode: {mode!r} (use binary|partial)")


@dataclass(frozen=True)
class TemporalCausalReward:
    """IoU-based temporal/causal localization reward (Dataset C style).

    Independently callable. Comparable with ``HiddenStateExactMatchReward`` but
    never combined/weighted here. Does not invent causal ground truth.
    """

    reward_id: str = TEMPORAL_IOU_ID
    version: str = TEMPORAL_CAUSAL_VERSION
    name: str = TEMPORAL_IOU_ID
    mode: str = "binary"  # binary | partial
    iou_threshold: float = 0.5
    unit: str = "auto"  # auto | seconds | frames

    def __post_init__(self) -> None:
        if self.mode not in {"binary", "partial"}:
            raise RewardError(f"mode must be binary|partial, got {self.mode!r}")
        if self.unit not in {"auto", "seconds", "frames"}:
            raise RewardError(f"unit must be auto|seconds|frames, got {self.unit!r}")
        if not (0.0 <= float(self.iou_threshold) <= 1.0):
            raise RewardError("iou_threshold must be in [0, 1]")

    def evaluate(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> RewardResult:
        preferred = self.unit  # type: ignore[assignment]
        gold_interval, gold_meta = resolve_gold_interval_object(
            example, preferred_unit=preferred
        )
        pred_unit = gold_interval.unit if gold_interval is not None else preferred
        pred_interval, parse_failed, parse_reason = parse_interval_text(
            artifact.raw_text,
            preferred_unit="auto" if pred_unit == "auto" else pred_unit,  # type: ignore[arg-type]
        )

        eligible = bool(gold_meta.get("eligible"))
        iou_value: float | None = None
        notes: str | None = None

        if not eligible:
            notes = str(gold_meta.get("reason") or "ineligible_annotation")
            value = 0.0
            matched = False
        elif parse_failed or pred_interval is None:
            notes = parse_reason or "parse_failed"
            value = 0.0
            matched = False
            parse_failed = True
        elif gold_interval is None:
            notes = "invalid_gold_interval"
            value = 0.0
            matched = False
        elif pred_interval.unit != gold_interval.unit:
            notes = "unit_mismatch"
            value = 0.0
            matched = False
            parse_failed = True
        else:
            iou_value = interval_iou(pred_interval, gold_interval)
            value = compute_temporal_causal_value(
                iou=iou_value,
                mode=self.mode,
                iou_threshold=self.iou_threshold,
                eligible=True,
                parse_failed=False,
            )
            matched = iou_value >= float(self.iou_threshold)
            if self.mode == "binary" and not matched:
                notes = "below_iou_threshold"
            elif self.mode == "partial" and iou_value == 0.0:
                notes = "no_overlap"

        pred_repr = None if pred_interval is None else (
            f"{pred_interval.start}-{pred_interval.end}{pred_interval.unit[0]}"
        )
        gold_repr = None if gold_interval is None else (
            f"{gold_interval.start}-{gold_interval.end}{gold_interval.unit[0]}"
        )

        return RewardResult(
            value=float(value),
            reward_id=self.reward_id,
            version=self.version,
            parse_failed=bool(parse_failed),
            prediction=pred_repr,
            gold=gold_repr,
            matched=matched,
            notes=notes,
            extras={
                "mode": self.mode,
                "iou_threshold": float(self.iou_threshold),
                "iou": iou_value,
                "predicted_interval": None if pred_interval is None else pred_interval.to_dict(),
                "gold_interval": gold_meta.get("interval"),
                "annotation_status": gold_meta.get("annotation_status"),
                "status_label": gold_meta.get("status_label"),
                "unique_cause": gold_meta.get("unique_cause"),
                "causal_provenance": gold_meta.get("provenance"),
                "eligible": eligible,
                "used_clip_temporal_as_gold": False,
                "salient_action_is_not_causal_proof": True,
                "shortcut_risks": list(TEMPORAL_SHORTCUT_RISKS),
                "integrity_note": INTEGRITY_NOTE_TEMPORAL,
                "is_reasoning_metric": False,
                "is_hybrid": False,
                "parse_reason": parse_reason,
            },
        )

    def score(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> float:
        return float(self.evaluate(artifact, example, **kwargs).value)


def compare_hidden_state_and_temporal(
    artifact: InferenceArtifact,
    example: ExampleRecord,
    *,
    hidden_state_reward: HiddenStateExactMatchReward | None = None,
    temporal_reward: TemporalCausalReward | None = None,
) -> dict[str, Any]:
    """Score both rewards independently (no weighting / no hybrid)."""
    hs = hidden_state_reward or HiddenStateExactMatchReward()
    tc = temporal_reward or TemporalCausalReward()
    hs_result = hs.evaluate(artifact, example)
    tc_result = tc.evaluate(artifact, example)
    return {
        "example_id": example.example_id,
        "clip_id": example.clip_id,
        "hidden_state": hs_result.to_dict(),
        "temporal_iou": tc_result.to_dict(),
        "combined": False,
        "weighted": False,
        "note": (
            "Independent objective scores for direct comparison. "
            "Not a hybrid reward; no weighting applied."
        ),
    }


# ---------------------------------------------------------------------------
# Future extension points (stubs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplanationRewardStub:
    """Reserved explanation preference reward (unimplemented)."""

    reward_id: str = EXPLANATION_REWARD_ID
    version: str = "0.0.0-stub"
    name: str = EXPLANATION_REWARD_ID

    def evaluate(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> RewardResult:
        raise NotImplementedError("explanation_reward is reserved; not implemented.")

    def score(
        self,
        artifact: InferenceArtifact,
        example: ExampleRecord,
        **kwargs: Any,
    ) -> float:
        raise NotImplementedError("explanation_reward is reserved; not implemented.")


# ---------------------------------------------------------------------------
# Registry / factory (single reward; no hybrid mixing)
# ---------------------------------------------------------------------------


def _build_temporal(**kwargs: Any) -> TemporalCausalReward:
    allowed = {
        "mode": kwargs.get("mode", "binary"),
        "iou_threshold": float(kwargs.get("iou_threshold", 0.5)),
        "unit": kwargs.get("unit", "auto"),
        "reward_id": kwargs.get("reward_id", TEMPORAL_IOU_ID),
    }
    return TemporalCausalReward(**allowed)  # type: ignore[arg-type]


_REWARD_BUILDERS: dict[str, Callable[..., ObjectiveReward]] = {
    HIDDEN_STATE_EXACT_MATCH_ID: HiddenStateExactMatchReward,
    "exact_match": ExactMatchReward,  # legacy config alias
    TEMPORAL_IOU_ID: lambda **kw: _build_temporal(reward_id=TEMPORAL_IOU_ID, **kw),
    TEMPORAL_LOCALIZATION_ID: lambda **kw: _build_temporal(
        reward_id=TEMPORAL_LOCALIZATION_ID,
        mode=kw.get("mode", "binary"),
        iou_threshold=kw.get("iou_threshold", 0.5),
        unit=kw.get("unit", "auto"),
    ),
}


def list_registered_rewards() -> dict[str, str]:
    """Return reward_id -> status for discoverability."""
    return {
        HIDDEN_STATE_EXACT_MATCH_ID: f"implemented@{HIDDEN_STATE_EXACT_MATCH_VERSION}",
        "exact_match": f"legacy_alias->{HIDDEN_STATE_EXACT_MATCH_ID}",
        TEMPORAL_IOU_ID: f"implemented@{TEMPORAL_CAUSAL_VERSION}",
        TEMPORAL_LOCALIZATION_ID: f"alias->{TEMPORAL_IOU_ID}@binary_default",
        EXPLANATION_REWARD_ID: "reserved",
        HYBRID_REWARD_ID: "reserved_not_combined_in_this_stage",
    }


def build_reward(reward_id: str, **kwargs: Any) -> ObjectiveReward:
    """Construct one registered objective reward (no hybrids)."""
    key = str(reward_id).strip()
    if key in {EXPLANATION_REWARD_ID, HYBRID_REWARD_ID}:
        raise RewardError(
            f"Reward {key!r} is reserved and not implemented. "
            "Do not combine rewards in this stage."
        )
    builder = _REWARD_BUILDERS.get(key)
    if builder is None:
        raise RewardError(
            f"Unknown reward_id {key!r}. Registered: {sorted(_REWARD_BUILDERS)}"
        )
    if key in {TEMPORAL_IOU_ID, TEMPORAL_LOCALIZATION_ID}:
        return builder(**kwargs)
    return builder()


@dataclass(frozen=True)
class RewardConfig:
    """YAML-friendly reward selection for experiments."""

    reward_id: str = HIDDEN_STATE_EXACT_MATCH_ID
    version: str | None = None  # if set, must match implementation version
    mode: str | None = None  # temporal: binary | partial
    iou_threshold: float = 0.5
    unit: str = "auto"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RewardConfig:
        raw = dict(data or {})
        extras = dict(raw.pop("extras", {}) or {})
        version_raw = raw.pop("version", None)
        mode_raw = raw.pop("mode", None)
        return cls(
            reward_id=str(raw.pop("reward_id", HIDDEN_STATE_EXACT_MATCH_ID)),
            version=None if version_raw is None else str(version_raw),
            mode=None if mode_raw is None else str(mode_raw),
            iou_threshold=float(raw.pop("iou_threshold", 0.5)),
            unit=str(raw.pop("unit", "auto")),
            extras={**extras, **raw},
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
        kwargs: dict[str, Any] = {}
        if self.reward_id in {TEMPORAL_IOU_ID, TEMPORAL_LOCALIZATION_ID}:
            kwargs["mode"] = self.mode or "binary"
            kwargs["iou_threshold"] = self.iou_threshold
            kwargs["unit"] = self.unit
        reward = build_reward(self.reward_id, **kwargs)
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
