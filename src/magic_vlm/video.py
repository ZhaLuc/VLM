"""Deterministic video preprocessing interfaces.

Frame selection is index-based and seed-stable. Optional OpenCV decoding is
used only when the ``video`` extra is installed; architecture tests do not
require media decoding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class VideoPreprocessConfig:
    """Controls frame sampling for VLM input.

    ``temporal_shuffle`` is reserved for future diagnostic experiments. When
    False (default), frame order matches the source timeline. Enabling it must
    not mutate raw media on disk.
    """

    max_frames: int = 8
    sample_strategy: str = "uniform"  # uniform | first_n
    temporal_shuffle: bool = False
    shuffle_seed: int = 0

    def __post_init__(self) -> None:
        if self.max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        if self.sample_strategy not in {"uniform", "first_n"}:
            raise ValueError(f"Unsupported sample_strategy: {self.sample_strategy}")


@dataclass(frozen=True)
class PreprocessedVideo:
    """Deterministic preprocessing result (indices are the contract)."""

    source_path: str
    frame_indices: tuple[int, ...]
    temporal_shuffled: bool = False
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frame_indices"] = list(self.frame_indices)
        return payload


def select_frame_indices(
    num_frames: int,
    config: VideoPreprocessConfig,
) -> tuple[int, ...]:
    """Return deterministic frame indices for a clip with ``num_frames`` frames."""
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

    if config.temporal_shuffle:
        indices = _shuffle_indices(indices, seed=config.shuffle_seed)

    return tuple(indices)


def _shuffle_indices(indices: Sequence[int], *, seed: int) -> list[int]:
    """Fisher-Yates with a tiny LCG so we do not depend on Python's hash seed."""
    values = list(indices)
    state = seed & 0xFFFFFFFF
    for i in range(len(values) - 1, 0, -1):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        j = state % (i + 1)
        values[i], values[j] = values[j], values[i]
    return values


def preprocess_video_meta(
    source_path: str | Path,
    *,
    num_frames: int,
    config: VideoPreprocessConfig | None = None,
) -> PreprocessedVideo:
    """Compute preprocessing indices without decoding pixels."""
    cfg = config or VideoPreprocessConfig()
    path = Path(source_path)
    indices = select_frame_indices(num_frames, cfg)
    return PreprocessedVideo(
        source_path=str(path),
        frame_indices=indices,
        temporal_shuffled=cfg.temporal_shuffle,
        metadata={"num_frames": num_frames, "sample_strategy": cfg.sample_strategy},
    )


def decode_frames_bgr(
    source_path: str | Path,
    frame_indices: Sequence[int],
) -> list[Any]:
    """Decode selected frames as BGR arrays.

    Requires the optional ``magic-vlm[video]`` extra (OpenCV).
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised when extra missing
        raise ImportError(
            "OpenCV is required for frame decoding. Install magic-vlm[video]."
        ) from exc

    path = Path(source_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    wanted = set(int(i) for i in frame_indices)
    max_index = max(wanted) if wanted else -1
    decoded: dict[int, Any] = {}
    index = 0
    try:
        while index <= max_index:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                decoded[index] = frame
            index += 1
    finally:
        capture.release()

    missing = sorted(wanted - set(decoded))
    if missing:
        raise RuntimeError(f"Failed to decode frame indices {missing} from {path}")
    return [decoded[i] for i in frame_indices]
