#!/usr/bin/env python
"""Probe Mac King Movie*.MP4 files and dump review stills. Does not modify sources."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2  # type: ignore

from magic_vlm.video import VideoPreprocessConfig, preprocess_video, probe_video

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "data" / "videos"
STILLS = ROOT / "reports" / "mac_king_clip_review" / "stills"
FRAMES = ROOT / "reports" / "mac_king_clip_review" / "frames"
OUT_JSON = ROOT / "reports" / "mac_king_clip_review" / "probe.json"

FILES = [
    "Movie1.MP4",
    "Movie2.MP4",
    "Movie3.MP4",
    "Movie4.MP4",
    "Movie5.MP4",
    "Movie6.MP4",
    "Movie7.MP4",
]

# Extra review stills (do not replace start/mid/end). Indices from timeline inspection.
CURATED_STILLS = {
    "movie6": {"coin_in_right": 80},  # ~2.67 s; coin shown in the right hand
    "movie7": {"coin_in_right": 115},  # ~3.84 s; coin shown in the right hand
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fourcc_name(capture: cv2.VideoCapture) -> str | None:
    raw = int(capture.get(cv2.CAP_PROP_FOURCC))
    if raw <= 0:
        return None
    chars = "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4))
    return chars.strip() or None


def ffprobe_codec(path: Path) -> dict[str, str | None]:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,codec_long_name,pix_fmt,profile",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {"codec_name": None, "error": "ffprobe_not_on_path"}
    if proc.returncode != 0:
        return {"codec_name": None, "error": proc.stderr.strip()[:400] or "ffprobe_failed"}
    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        return {"codec_name": None, "error": "no_video_stream"}
    stream = streams[0]
    return {
        "codec_name": stream.get("codec_name"),
        "codec_long_name": stream.get("codec_long_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "profile": stream.get("profile"),
    }


def pick_indices(n: int) -> list[int]:
    if n < 1:
        return []
    # Start, 8 uniform interior, last; unique and sorted.
    uniform = [int(round(i * (n - 1) / 11)) for i in range(12)]
    extra = [max(0, n // 3), max(0, n // 2), min(n - 1, (2 * n) // 3)]
    return sorted(set(uniform + extra + [0, n - 1]))


def decode_ok(path: Path, n: int) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"opened": False, "decoded_frames": 0, "fourcc": None}
    fourcc = fourcc_name(capture)
    decoded = 0
    last_ok = False
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        last_ok = frame is not None and getattr(frame, "size", 0) > 0
        if last_ok:
            decoded += 1
    capture.release()
    return {
        "opened": True,
        "fourcc": fourcc,
        "decoded_frames": decoded,
        "declared_frames": n,
        "decode_matches_declared": decoded == n,
        "last_frame_nonempty": last_ok,
    }


def write_stills(
    path: Path,
    clip_id: str,
    indices: list[int],
    fps: float | None,
    extra_labels: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    still_dir = STILLS
    dense = FRAMES / clip_id
    still_dir.mkdir(parents=True, exist_ok=True)
    dense.mkdir(parents=True, exist_ok=True)
    frames = []
    decoded = {}
    capture = cv2.VideoCapture(str(path))
    wanted = set(indices)
    index = 0
    max_index = max(wanted)
    while index <= max_index:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            decoded[index] = frame
        index += 1
    capture.release()
    for idx in indices:
        frame = decoded.get(idx)
        if frame is None:
            continue
        t = (idx / fps) if fps else None
        name = f"{clip_id}_f{idx:04d}" + (f"_t{t:05.2f}s" if t is not None else "") + ".jpg"
        dense_path = dense / name
        cv2.imwrite(str(dense_path), frame)
        frames.append(
            {
                "index": idx,
                "t_s": t,
                "dense_relpath": str(dense_path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    # Curated stills: start, mid, last
    n = indices[-1] if indices else 0
    picks = {
        "start": 0,
        "mid": indices[len(indices) // 2] if indices else 0,
        "end": n,
    }
    for label, idx in picks.items():
        frame = decoded.get(idx)
        if frame is None:
            continue
        out = still_dir / f"{clip_id}_{label}.jpg"
        cv2.imwrite(str(out), frame)
    for label, idx in (extra_labels or {}).items():
        frame = decoded.get(idx)
        if frame is None:
            continue
        out = still_dir / f"{clip_id}_{label}.jpg"
        cv2.imwrite(str(out), frame)
    return frames


def main() -> int:
    STILLS.mkdir(parents=True, exist_ok=True)
    rows = []
    mtimes = {}
    sizes = {}
    for name in FILES:
        path = VIDEOS / name
        if not path.is_file():
            rows.append({"filename": name, "present": False})
            continue
        mtimes[name] = path.stat().st_mtime
        sizes[name] = path.stat().st_size
        info = probe_video(path)
        info["path"] = str(path.relative_to(ROOT)).replace("\\", "/")
        fps = info.get("fps")
        duration = None
        if fps and info["num_frames"]:
            duration = info["num_frames"] / float(fps)
        decode = decode_ok(path, info["num_frames"])
        sampled = preprocess_video(
            path,
            config=VideoPreprocessConfig(max_frames=8, sample_strategy="uniform"),
            load_frames=True,
        )
        clip_id = name.replace(".MP4", "").replace(".mp4", "").lower()
        extra = CURATED_STILLS.get(clip_id, {})
        indices = sorted(set(pick_indices(info["num_frames"]) + list(extra.values())))
        still_meta = write_stills(path, clip_id, indices, fps, extra_labels=extra)
        rows.append(
            {
                "filename": name,
                "present": True,
                "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "probe": info,
                "duration_s": duration,
                "ffprobe": ffprobe_codec(path),
                "opencv_decode": decode,
                "preprocess": {
                    "frames_loaded": sampled.frames is not None,
                    "n_frames": sampled.n_frames,
                    "ordered_indices": list(sampled.ordered_indices),
                },
                "review_stills": still_meta,
            }
        )
        after = path.stat()
        if after.st_mtime != mtimes[name] or after.st_size != sizes[name]:
            raise RuntimeError(f"Source file was modified: {path}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"clips": rows}, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    for row in rows:
        if not row.get("present"):
            print(f"MISSING {row['filename']}")
            continue
        print(
            f"{row['filename']} frames={row['probe']['num_frames']} "
            f"{row['probe']['width']}x{row['probe']['height']} "
            f"fps={row['probe']['fps']} dur={row['duration_s']:.3f}s "
            f"decoded={row['opencv_decode']['decoded_frames']} "
            f"pre={row['preprocess']['frames_loaded']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
