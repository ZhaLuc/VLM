from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from magic_vlm.video import (
    ResizeConfig,
    VideoPreprocessConfig,
    apply_temporal_shuffle,
    as_ordered_clip,
    build_sample_plan,
    decode_frames_bgr,
    load_sampled_clip,
    ordered_and_shuffled_pair,
    preprocess_video,
    preprocess_video_meta,
    probe_video,
    select_frame_indices,
)


def test_uniform_indices_deterministic() -> None:
    cfg = VideoPreprocessConfig(max_frames=4, sample_strategy="uniform")
    a = select_frame_indices(16, cfg)
    b = select_frame_indices(16, cfg)
    assert a == b == (0, 5, 10, 15)


def test_first_n() -> None:
    cfg = VideoPreprocessConfig(max_frames=3, sample_strategy="first_n")
    assert select_frame_indices(10, cfg) == (0, 1, 2)


def test_sampling_ignores_shuffle_flag() -> None:
    ordered_cfg = VideoPreprocessConfig(max_frames=4, temporal_shuffle=False)
    shuffled_cfg = VideoPreprocessConfig(max_frames=4, temporal_shuffle=True, shuffle_seed=7)
    assert select_frame_indices(16, ordered_cfg) == select_frame_indices(16, shuffled_cfg)


def test_temporal_shuffle_preserves_frame_identity_set() -> None:
    plan = build_sample_plan("clip.mp4", num_frames=16, config=VideoPreprocessConfig(max_frames=4))
    ordered = as_ordered_clip(plan)
    shuffled = apply_temporal_shuffle(plan, seed=7)
    assert ordered.ordered_indices == shuffled.ordered_indices == (0, 5, 10, 15)
    assert set(shuffled.frame_indices) == set(ordered.frame_indices)
    assert shuffled.frame_indices != ordered.frame_indices
    assert apply_temporal_shuffle(plan, seed=7).frame_indices == shuffled.frame_indices
    assert apply_temporal_shuffle(plan, seed=8).frame_indices != shuffled.frame_indices


def test_preprocess_meta_repeatable() -> None:
    cfg = VideoPreprocessConfig(max_frames=3, temporal_shuffle=True, shuffle_seed=3)
    a = preprocess_video_meta("clip.mp4", num_frames=9, config=cfg)
    b = preprocess_video_meta("clip.mp4", num_frames=9, config=cfg)
    assert a.to_dict() == b.to_dict()
    assert a.ordered_indices == (0, 4, 8)
    assert a.temporal_shuffled is True
    assert set(a.frame_indices) == set(a.ordered_indices)


def test_malformed_num_frames() -> None:
    with pytest.raises(ValueError):
        build_sample_plan("x.mp4", num_frames=0)


def _write_synthetic_video(path: Path, n_frames: int = 16, size: tuple[int, int] = (40, 30)) -> None:
    cv2 = pytest.importorskip("cv2")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 8.0, size)
    assert writer.isOpened()
    for i in range(n_frames):
        # Unique intensity per frame for identity checks.
        frame = np.full((size[1], size[0], 3), i, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_cpu_decode_order_shuffle_and_resize(tmp_path: Path) -> None:
    video_path = tmp_path / "synth.mp4"
    _write_synthetic_video(video_path, n_frames=16)
    props = probe_video(video_path)
    assert props["num_frames"] >= 8

    cfg = VideoPreprocessConfig(
        max_frames=4,
        sample_strategy="uniform",
        resize=ResizeConfig(width=20, height=10),
    )
    plan = build_sample_plan(
        video_path,
        num_frames=int(props["num_frames"]),
        config=cfg,
        source_fps=props.get("fps"),
    )
    ordered = load_sampled_clip(plan, shuffle=False)
    shuffled = load_sampled_clip(plan, shuffle=True, shuffle_seed=11)

    assert ordered.n_frames == 4
    assert ordered.frames is not None and shuffled.frames is not None
    assert ordered.frame_indices == ordered.ordered_indices
    assert shuffled.ordered_indices == ordered.ordered_indices
    assert set(shuffled.frame_indices) == set(ordered.frame_indices)
    assert shuffled.frame_indices != ordered.frame_indices

    # Frame identity under shuffle: same index => identical array payload.
    by_index = {idx: frame for idx, frame in zip(ordered.frame_indices, ordered.frames)}
    for idx, frame in zip(shuffled.frame_indices, shuffled.frames):
        assert frame.shape[0] == 10 and frame.shape[1] == 20
        np.testing.assert_array_equal(frame, by_index[idx])

    # Source file unchanged after preprocess.
    before = video_path.read_bytes()
    _ = preprocess_video(video_path, config=cfg, load_frames=True)
    assert video_path.read_bytes() == before


def test_ordered_and_shuffled_pair_same_sample(tmp_path: Path) -> None:
    video_path = tmp_path / "pair.mp4"
    _write_synthetic_video(video_path, n_frames=12)
    cfg = VideoPreprocessConfig(max_frames=4, shuffle_seed=5)
    ordered, shuffled = ordered_and_shuffled_pair(
        video_path, config=cfg, load_frames=True
    )
    assert ordered.ordered_indices == shuffled.ordered_indices
    assert ordered.temporal_shuffled is False
    assert shuffled.temporal_shuffled is True
    assert ordered.metadata.get("ordering") == "temporal"
    assert shuffled.metadata.get("ordering") == "shuffled"


def test_decode_malformed_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mp4"
    with pytest.raises(FileNotFoundError):
        decode_frames_bgr(missing, [0, 1])


def test_config_roundtrip_resize() -> None:
    cfg = VideoPreprocessConfig.from_dict(
        {
            "max_frames": 6,
            "sample_strategy": "first_n",
            "temporal_shuffle": False,
            "shuffle_seed": 2,
            "resize": {"width": 64, "height": 64, "keep_aspect": False},
        }
    )
    assert cfg.resize is not None
    assert cfg.resize.width == 64
    assert VideoPreprocessConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
