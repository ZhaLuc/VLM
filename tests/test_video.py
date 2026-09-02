from magic_vlm.video import VideoPreprocessConfig, preprocess_video_meta, select_frame_indices


def test_uniform_indices_deterministic() -> None:
    cfg = VideoPreprocessConfig(max_frames=4, sample_strategy="uniform")
    a = select_frame_indices(16, cfg)
    b = select_frame_indices(16, cfg)
    assert a == b == (0, 5, 10, 15)


def test_first_n() -> None:
    cfg = VideoPreprocessConfig(max_frames=3, sample_strategy="first_n")
    assert select_frame_indices(10, cfg) == (0, 1, 2)


def test_temporal_shuffle_is_seed_stable() -> None:
    cfg = VideoPreprocessConfig(max_frames=4, temporal_shuffle=True, shuffle_seed=7)
    assert select_frame_indices(16, cfg) == select_frame_indices(16, cfg)
    other = VideoPreprocessConfig(max_frames=4, temporal_shuffle=True, shuffle_seed=8)
    assert select_frame_indices(16, cfg) != select_frame_indices(16, other)


def test_preprocess_meta() -> None:
    result = preprocess_video_meta("clip.mp4", num_frames=9, config=VideoPreprocessConfig(max_frames=3))
    assert result.source_path.endswith("clip.mp4")
    assert result.frame_indices == (0, 4, 8)
    assert result.temporal_shuffled is False
