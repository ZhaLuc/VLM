from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_convert_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "convert_ogv_to_mp4.py"
    spec = importlib.util.spec_from_file_location("convert_ogv_to_mp4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ffmpeg_command_is_reproducible() -> None:
    mod = _load_convert_module()
    cmd = mod.build_command(
        "ffmpeg",
        Path("in.ogv"),
        Path("out.mp4"),
    )
    assert cmd[:4] == ["ffmpeg", "-hide_banner", "-y", "-i"]
    assert cmd[4:] == [
        "in.ogv",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        "out.mp4",
    ]
