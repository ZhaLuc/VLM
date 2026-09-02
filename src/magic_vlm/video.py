"""Minimal reproducible video preprocessing for baseline and temporal-shuffle.

Design rules
------------
- Sampling is deterministic and independent of temporal shuffle.
- Ordered and shuffled conditions must use the **same** sampled frame indices;
  shuffle only changes presentation order.
- Source videos on disk are never overwritten.
- Prompt construction stays outside this module.

Research integrity
------------------
Temporal-shuffle sensitivity is a diagnostic of temporal-order dependence, not
proof of causal reasoning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ResizeConfig:
    """Optional resize applied after decode (CPU). ``None`` means keep native size."""

    width: int | None = None
    height: int | None = None
    keep_aspect: bool = False

    def __post_init__(self) -> None:
        if (self.width is None) ^ (self.height is None):
            if not self.keep_aspect:
                raise ValueError(
                    "Provide both width and height, or set keep_aspect=True with one side"
                )
        if self.width is not None and self.width < 1:
            raise ValueError("resize.width must be >= 1")
        if self.height is not None and self.height < 1:
            raise ValueError("resize.height must be >= 1")

    @property
    def enabled(self) -> bool:
        return self.width is not None or self.height is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ResizeConfig | None:
        if not data:
            return None
        return cls(**dict(data))


@dataclass(frozen=True)
class VideoPreprocessConfig:
    """Controls frame sampling and optional presentation shuffle/resize.

    ``temporal_shuffle`` affects **presentation order only**. Sampling always
    uses chronological selection via :func:`select_frame_indices`.
    """

    max_frames: int = 8
    sample_strategy: str = "uniform"  # uniform | first_n
    temporal_shuffle: bool = False
    shuffle_seed: int = 0
    resize: ResizeConfig | None = None

    def __post_init__(self) -> None:
        if self.max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        if self.sample_strategy not in {"uniform", "first_n"}:
            raise ValueError(f"Unsupported sample_strategy: {self.sample_strategy}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_frames": self.max_frames,
            "sample_strategy": self.sample_strategy,
            "temporal_shuffle": self.temporal_shuffle,
            "shuffle_seed": self.shuffle_seed,
            "resize": None if self.resize is None else self.resize.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> VideoPreprocessConfig:
        raw = dict(data or {})
        resize_raw = raw.pop("resize", None)
        resize = ResizeConfig.from_dict(resize_raw) if isinstance(resize_raw, dict) else None
        if resize is None and resize_raw is None:
            # Allow flat width/height keys in YAML for convenience.
            w = raw.pop("resize_width", None)
            h = raw.pop("resize_height", None)
            if w is not None or h is not None:
                resize = ResizeConfig(width=w, height=h)
        return cls(resize=resize, **raw)


@dataclass(frozen=True)
class FrameSamplePlan:
    """Reproducible record of which frames were selected (before presentation).

    ``ordered_indices`` are always in source timeline order of the sample set.
    """

    source_path: str
    source_num_frames: int
    ordered_indices: tuple[int, ...]
    sample_strategy: str
    max_frames: int
    source_fps: float | None = None
    source_content_hash: str | None = None
    resize: ResizeConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_num_frames": self.source_num_frames,
            "source_fps": self.source_fps,
            "source_content_hash": self.source_content_hash,
            "ordered_indices": list(self.ordered_indices),
            "sample_strategy": self.sample_strategy,
            "max_frames": self.max_frames,
            "resize": None if self.resize is None else self.resize.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SampledClip:
    """Sampled frames ready for a VLM, with explicit ordering metadata.

    ``frame_indices`` is the presentation order (ordered or shuffled).
    ``ordered_indices`` is the underlying sample set in timeline order.
    ``frames`` holds decoded arrays in **presentation** order when loaded.
    """

    source_path: str
    ordered_indices: tuple[int, ...]
    frame_indices: tuple[int, ...]
    temporal_shuffled: bool
    shuffle_seed: int | None
    sample_strategy: str
    max_frames: int
    source_num_frames: int
    source_fps: float | None = None
    source_content_hash: str | None = None
    resize: ResizeConfig | None = None
    frames: tuple[Any, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.frame_indices)

    def plan(self) -> FrameSamplePlan:
        return FrameSamplePlan(
            source_path=self.source_path,
            source_num_frames=self.source_num_frames,
            ordered_indices=self.ordered_indices,
            sample_strategy=self.sample_strategy,
            max_frames=self.max_frames,
            source_fps=self.source_fps,
            source_content_hash=self.source_content_hash,
            resize=self.resize,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_num_frames": self.source_num_frames,
            "source_fps": self.source_fps,
            "source_content_hash": self.source_content_hash,
            "ordered_indices": list(self.ordered_indices),
            "frame_indices": list(self.frame_indices),
            "temporal_shuffled": self.temporal_shuffled,
            "shuffle_seed": self.shuffle_seed,
            "sample_strategy": self.sample_strategy,
            "max_frames": self.max_frames,
            "resize": None if self.resize is None else self.resize.to_dict(),
            "n_frames": self.n_frames,
            "frames_loaded": self.frames is not None,
            "metadata": dict(self.metadata),
        }


# Backward-compatible alias used by earlier stages.
PreprocessedVideo = SampledClip


def select_frame_indices(
    num_frames: int,
    config: VideoPreprocessConfig,
) -> tuple[int, ...]:
    """Return deterministic **chronologically ordered** sample indices.

    Does not apply temporal shuffle. Use :func:`apply_temporal_shuffle` so
    ordered/shuffled conditions share one sample set.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")

    k = min(config.max_frames, num_frames)
    if config.sample_strategy == "first_n":
        indices = list(range(k))
    else:
        if k == 1:
            indices = [0]
        else:
            # Inclusive endpoints; integer rounding is stable across platforms.
            indices = [int(round(i * (num_frames - 1) / (k - 1))) for i in range(k)]
    return tuple(indices)


def _shuffle_indices(indices: Sequence[int], *, seed: int) -> tuple[int, ...]:
    """Fisher-Yates with a tiny LCG so we do not depend on Python's hash seed."""
    values = list(indices)
    state = seed & 0xFFFFFFFF
    for i in range(len(values) - 1, 0, -1):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        j = state % (i + 1)
        values[i], values[j] = values[j], values[i]
    return tuple(values)


def build_sample_plan(
    source_path: str | Path,
    *,
    num_frames: int,
    config: VideoPreprocessConfig | None = None,
    source_fps: float | None = None,
    source_content_hash: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> FrameSamplePlan:
    """Create a reproducible sample plan (indices only; no decode)."""
    cfg = config or VideoPreprocessConfig()
    path = Path(source_path)
    if num_frames < 1:
        raise ValueError(f"Malformed input: num_frames must be >= 1 (got {num_frames})")
    ordered = select_frame_indices(num_frames, cfg)
    return FrameSamplePlan(
        source_path=str(path),
        source_num_frames=num_frames,
        ordered_indices=ordered,
        sample_strategy=cfg.sample_strategy,
        max_frames=cfg.max_frames,
        source_fps=source_fps,
        source_content_hash=source_content_hash,
        resize=cfg.resize,
        metadata={
            "sampling_policy": cfg.sample_strategy,
            "max_frames_requested": cfg.max_frames,
            **(metadata or {}),
        },
    )


def apply_temporal_shuffle(
    plan: FrameSamplePlan,
    *,
    seed: int,
    frames: Sequence[Any] | None = None,
) -> SampledClip:
    """Return a shuffled presentation of the **same** sampled indices/frames.

    If ``frames`` is provided it must already be aligned to ``plan.ordered_indices``.
    """
    if frames is not None and len(frames) != len(plan.ordered_indices):
        raise ValueError("frames length must match plan.ordered_indices")
    presentation = _shuffle_indices(plan.ordered_indices, seed=seed)
    # Map ordered frame payloads into presentation order by index identity.
    presented_frames: tuple[Any, ...] | None = None
    if frames is not None:
        by_index = {idx: frame for idx, frame in zip(plan.ordered_indices, frames)}
        presented_frames = tuple(by_index[idx] for idx in presentation)
    return SampledClip(
        source_path=plan.source_path,
        ordered_indices=plan.ordered_indices,
        frame_indices=presentation,
        temporal_shuffled=True,
        shuffle_seed=seed,
        sample_strategy=plan.sample_strategy,
        max_frames=plan.max_frames,
        source_num_frames=plan.source_num_frames,
        source_fps=plan.source_fps,
        source_content_hash=plan.source_content_hash,
        resize=plan.resize,
        frames=presented_frames,
        metadata={**plan.metadata, "ordering": "shuffled"},
    )


def as_ordered_clip(
    plan: FrameSamplePlan,
    *,
    frames: Sequence[Any] | None = None,
) -> SampledClip:
    """Presentation in chronological sample order (no shuffle)."""
    if frames is not None and len(frames) != len(plan.ordered_indices):
        raise ValueError("frames length must match plan.ordered_indices")
    return SampledClip(
        source_path=plan.source_path,
        ordered_indices=plan.ordered_indices,
        frame_indices=plan.ordered_indices,
        temporal_shuffled=False,
        shuffle_seed=None,
        sample_strategy=plan.sample_strategy,
        max_frames=plan.max_frames,
        source_num_frames=plan.source_num_frames,
        source_fps=plan.source_fps,
        source_content_hash=plan.source_content_hash,
        resize=plan.resize,
        frames=None if frames is None else tuple(frames),
        metadata={**plan.metadata, "ordering": "temporal"},
    )


def preprocess_video_meta(
    source_path: str | Path,
    *,
    num_frames: int,
    config: VideoPreprocessConfig | None = None,
) -> SampledClip:
    """Index-only preprocessing (compatible with earlier stages).

    Sampling never depends on ``temporal_shuffle``. When the config requests
    shuffle, presentation indices are a permutation of the same sample set.
    """
    cfg = config or VideoPreprocessConfig()
    plan = build_sample_plan(source_path, num_frames=num_frames, config=cfg)
    if cfg.temporal_shuffle:
        return apply_temporal_shuffle(plan, seed=cfg.shuffle_seed)
    return as_ordered_clip(plan)


def probe_video(source_path: str | Path) -> dict[str, Any]:
    """Read basic media properties without mutating the file."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenCV is required for probe_video. Install magic-vlm[video]."
        ) from exc

    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {path}")
    try:
        n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if n < 1:
        raise OSError(f"Malformed video (no frames): {path}")
    return {
        "path": str(path),
        "num_frames": n,
        "fps": fps,
        "width": width,
        "height": height,
    }


def decode_frames_bgr(
    source_path: str | Path,
    frame_indices: Sequence[int],
    *,
    resize: ResizeConfig | None = None,
) -> list[Any]:
    """Decode selected frames as BGR arrays in the given index order (CPU)."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenCV is required for frame decoding. Install magic-vlm[video]."
        ) from exc

    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if not frame_indices:
        raise ValueError("frame_indices must be non-empty")
    if any(int(i) < 0 for i in frame_indices):
        raise ValueError("frame_indices must be >= 0")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {path}")

    wanted = set(int(i) for i in frame_indices)
    max_index = max(wanted)
    decoded: dict[int, Any] = {}
    index = 0
    try:
        while index <= max_index:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                decoded[index] = _maybe_resize(frame, resize)
            index += 1
    finally:
        capture.release()

    missing = sorted(wanted - set(decoded))
    if missing:
        raise RuntimeError(f"Failed to decode frame indices {missing} from {path}")
    return [decoded[int(i)] for i in frame_indices]


def _maybe_resize(frame: Any, resize: ResizeConfig | None) -> Any:
    if resize is None or not resize.enabled:
        return frame
    import cv2  # type: ignore

    h, w = frame.shape[:2]
    if resize.keep_aspect:
        if resize.width is not None and resize.height is None:
            scale = resize.width / float(w)
            new_w, new_h = resize.width, max(1, int(round(h * scale)))
        elif resize.height is not None and resize.width is None:
            scale = resize.height / float(h)
            new_w, new_h = max(1, int(round(w * scale))), resize.height
        else:
            # Both set with keep_aspect: fit inside box.
            assert resize.width is not None and resize.height is not None
            scale = min(resize.width / float(w), resize.height / float(h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
    else:
        new_w = int(resize.width if resize.width is not None else w)
        new_h = int(resize.height if resize.height is not None else h)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_sampled_clip(
    plan: FrameSamplePlan,
    *,
    shuffle: bool = False,
    shuffle_seed: int = 0,
) -> SampledClip:
    """Decode the plan's ordered sample set, then optionally shuffle presentation."""
    ordered_frames = decode_frames_bgr(
        plan.source_path,
        plan.ordered_indices,
        resize=plan.resize,
    )
    if shuffle:
        return apply_temporal_shuffle(plan, seed=shuffle_seed, frames=ordered_frames)
    return as_ordered_clip(plan, frames=ordered_frames)


def preprocess_video(
    source_path: str | Path,
    *,
    config: VideoPreprocessConfig | None = None,
    num_frames: int | None = None,
    source_fps: float | None = None,
    source_content_hash: str | None = None,
    load_frames: bool = True,
) -> SampledClip:
    """Full CPU preprocessing: probe (optional), sample, decode, optional shuffle.

    Does not write to disk or alter the source file.
    """
    cfg = config or VideoPreprocessConfig()
    path = Path(source_path)
    fps = source_fps
    n_frames = num_frames
    if n_frames is None or fps is None:
        props = probe_video(path)
        n_frames = int(n_frames if n_frames is not None else props["num_frames"])
        fps = fps if fps is not None else props.get("fps")
    plan = build_sample_plan(
        path,
        num_frames=n_frames,
        config=cfg,
        source_fps=fps,
        source_content_hash=source_content_hash,
    )
    if not load_frames:
        return (
            apply_temporal_shuffle(plan, seed=cfg.shuffle_seed)
            if cfg.temporal_shuffle
            else as_ordered_clip(plan)
        )
    return load_sampled_clip(
        plan,
        shuffle=cfg.temporal_shuffle,
        shuffle_seed=cfg.shuffle_seed,
    )


def ordered_and_shuffled_pair(
    source_path: str | Path,
    *,
    config: VideoPreprocessConfig | None = None,
    num_frames: int | None = None,
    shuffle_seed: int | None = None,
    load_frames: bool = True,
) -> tuple[SampledClip, SampledClip]:
    """Build ordered and shuffled views from one shared sample plan.

    Guarantees identical ``ordered_indices`` (same sampled frames).
    """
    cfg = config or VideoPreprocessConfig()
    # Force sampling without baking shuffle into config for the plan.
    sample_cfg = replace(cfg, temporal_shuffle=False)
    seed = cfg.shuffle_seed if shuffle_seed is None else shuffle_seed
    path = Path(source_path)
    n_frames = num_frames
    fps = None
    if n_frames is None:
        props = probe_video(path)
        n_frames = int(props["num_frames"])
        fps = props.get("fps")
    plan = build_sample_plan(path, num_frames=n_frames, config=sample_cfg, source_fps=fps)
    if load_frames:
        ordered = load_sampled_clip(plan, shuffle=False)
        shuffled = apply_temporal_shuffle(plan, seed=seed, frames=ordered.frames)
    else:
        ordered = as_ordered_clip(plan)
        shuffled = apply_temporal_shuffle(plan, seed=seed)
    return ordered, shuffled
