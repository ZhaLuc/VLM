#!/usr/bin/env python
"""Convert local source OGV clips to MP4 without overwriting originals.

This is data-prep only. It does not invent labels or run a model.

Reproducible encoder settings (video-only H.264):

    ffmpeg -hide_banner -y -i INPUT.ogv -an -c:v libx264 -pix_fmt yuv420p
           -preset medium -crf 18 -movflags +faststart OUTPUT.mp4

FFmpeg is resolved from PATH, then optional ``imageio_ffmpeg.get_ffmpeg_exe()``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

FFMPEG_ARGS = (
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
)


def resolve_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "ffmpeg not found on PATH and imageio_ffmpeg is unavailable. "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def build_command(ffmpeg: str, source: Path, dest: Path) -> list[str]:
    return [ffmpeg, "-hide_banner", "-y", "-i", str(source), *FFMPEG_ARGS, str(dest)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/videos"),
        help="Directory containing source .ogv files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .mp4 outputs. Never overwrites .ogv sources.",
    )
    args = parser.parse_args(argv)

    ffmpeg = resolve_ffmpeg()
    videos_dir = args.videos_dir
    sources = sorted(videos_dir.glob("*.ogv"))
    if not sources:
        print(f"No .ogv files in {videos_dir}", file=sys.stderr)
        return 1

    converted = 0
    skipped = 0
    for source in sources:
        dest = source.with_suffix(".mp4")
        if dest.exists() and not args.force:
            print(f"skip existing {dest.name}")
            skipped += 1
            continue
        cmd = build_command(ffmpeg, source, dest)
        print(" ".join(cmd))
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            print(f"ffmpeg failed for {source.name} (exit {completed.returncode})", file=sys.stderr)
            return completed.returncode
        converted += 1
    print(f"converted={converted} skipped_existing={skipped} ffmpeg={ffmpeg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
