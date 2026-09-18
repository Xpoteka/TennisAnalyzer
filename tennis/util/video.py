"""Thin wrappers around ffprobe and ffmpeg.

Timing rule (spec section 2): frame times always come from presentation timestamps (PTS),
never from ``frame_index / fps``, because the footage may have a variable frame rate.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tennis.errors import UserError

# A stream counts as VFR when frame intervals spread by more than this fraction of the median
# (between the 1st and 99th percentile). Millisecond timebase rounding at 120 fps stays well below.
VFR_SPREAD_THRESHOLD = 0.5
# Nominal and average frame rates further apart than this also mark the stream as VFR.
VFR_RATE_TOLERANCE = 0.01


class ToolMissingError(UserError):
    pass


class ProbeError(UserError):
    """ffprobe or ffmpeg could not read or convert the file."""


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ToolMissingError(f"'{name}' was not found on PATH; install ffmpeg first")
    return path


def _run(args: list[str]) -> str:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ProbeError(f"could not run {args[0]}: {exc}") from exc
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-5:]
        raise ProbeError(f"{Path(args[0]).name} exited with {proc.returncode}: " + " | ".join(tail))
    return proc.stdout


def parse_rate(value: str | None) -> float | None:
    """Parse an ffprobe rational such as ``120000/1001``; ``0/0`` and junk give ``None``."""
    if not value:
        return None
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return float(rate) if rate > 0 else None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_creation_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    # Cameras without a clock write the epoch; treat that as unknown.
    return None if parsed.year < 2000 else parsed


@dataclass(frozen=True)
class VideoStream:
    index: int
    codec: str
    width: int
    height: int
    rotation: int
    fps_nominal: float | None
    fps_avg: float | None
    start_time: float
    duration: float | None
    nb_frames: int | None
    pix_fmt: str | None


@dataclass(frozen=True)
class AudioStream:
    index: int
    codec: str
    sample_rate: int
    channels: int
    start_time: float
    duration: float | None


@dataclass(frozen=True)
class ProbeResult:
    path: Path
    format_name: str
    duration: float | None
    size_bytes: int
    video: VideoStream | None
    audio: AudioStream | None
    creation_time: datetime | None


def _rotation(stream: dict[str, Any]) -> int:
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            return int(float(side["rotation"])) % 360
    rotate = stream.get("tags", {}).get("rotate")
    return int(float(rotate)) % 360 if rotate is not None else 0


def probe(path: Path) -> ProbeResult:
    """Read container and stream metadata. Uses the first video and first audio stream."""
    exe = require_tool("ffprobe")
    out = _run(
        [exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    )
    data = json.loads(out)
    fmt = data.get("format", {})
    video: VideoStream | None = None
    audio: AudioStream | None = None
    creation: datetime | None = parse_creation_time(fmt.get("tags", {}).get("creation_time"))

    for s in data.get("streams", []):
        kind = s.get("codec_type")
        # Cover art is stored as a single-frame video stream; skip it.
        if kind == "video" and video is None and not s.get("disposition", {}).get("attached_pic"):
            nb = s.get("nb_frames")
            video = VideoStream(
                index=int(s["index"]),
                codec=s.get("codec_name", "unknown"),
                width=int(s.get("width", 0)),
                height=int(s.get("height", 0)),
                rotation=_rotation(s),
                fps_nominal=parse_rate(s.get("r_frame_rate")),
                fps_avg=parse_rate(s.get("avg_frame_rate")),
                start_time=_float(s.get("start_time")) or 0.0,
                duration=_float(s.get("duration")),
                nb_frames=int(nb) if nb and str(nb).isdigit() else None,
                pix_fmt=s.get("pix_fmt"),
            )
            creation = creation or parse_creation_time(s.get("tags", {}).get("creation_time"))
        elif kind == "audio" and audio is None:
            audio = AudioStream(
                index=int(s["index"]),
                codec=s.get("codec_name", "unknown"),
                sample_rate=int(s.get("sample_rate", 0)),
                channels=int(s.get("channels", 0)),
                start_time=_float(s.get("start_time")) or 0.0,
                duration=_float(s.get("duration")),
            )

    return ProbeResult(
        path=path,
        format_name=fmt.get("format_name", "unknown"),
        duration=_float(fmt.get("duration")),
        size_bytes=int(fmt.get("size", 0)),
        video=video,
        audio=audio,
        creation_time=creation,
    )


def file_creation_time(path: Path) -> datetime:
    """Filesystem fallback when the container has no creation timestamp."""
    st = os.stat(path)
    ts = getattr(st, "st_birthtime", None) or st.st_mtime
    return datetime.fromtimestamp(ts, UTC)


@dataclass(frozen=True)
class FrameIntervals:
    frames: int
    median_ms: float
    p01_ms: float
    p99_ms: float

    @property
    def spread(self) -> float:
        return (self.p99_ms - self.p01_ms) / self.median_ms if self.median_ms > 0 else 0.0


def read_frame_pts(path: Path, stream_index: int) -> npt.NDArray[np.float64]:
    """Presentation timestamps (seconds) of every frame of a video stream, sorted.

    Reads packet headers only, so it is fast even for multi-GB files. Packets flagged as
    discarded (``D``, e.g. before an edit-list start) never become frames and are skipped.
    Frame index ``i`` everywhere in the pipeline means the ``i``-th element of this array.
    """
    exe = require_tool("ffprobe")
    out = _run(
        [
            exe, "-v", "error",
            "-select_streams", str(stream_index),
            "-show_entries", "packet=pts_time,flags",
            "-of", "csv=p=0",
            str(path),
        ]
    )  # fmt: skip
    values = []
    for line in out.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2 or "D" in parts[1]:
            continue
        pts = _float(parts[0])
        if pts is not None:
            values.append(pts)
    return np.unique(np.asarray(values, dtype=np.float64))


def frame_intervals(pts: npt.NDArray[np.float64]) -> FrameIntervals:
    """Spacing statistics of sorted frame timestamps."""
    deltas = np.diff(pts)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return FrameIntervals(frames=int(pts.size), median_ms=0.0, p01_ms=0.0, p99_ms=0.0)
    p01, med, p99 = np.percentile(deltas, [1, 50, 99], method="nearest") * 1000
    return FrameIntervals(
        frames=int(pts.size), median_ms=float(med), p01_ms=float(p01), p99_ms=float(p99)
    )


def nearest_frames(
    pts: npt.NDArray[np.float64], times: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """Index of the frame whose PTS is closest to each time (ties go to the earlier frame)."""
    if pts.size == 0:
        raise ValueError("no frame timestamps")
    if pts.size == 1:
        return np.zeros(times.shape, dtype=np.int64)
    right = np.clip(np.searchsorted(pts, times, side="left"), 1, pts.size - 1)
    left = right - 1
    choose_right = np.abs(pts[right] - times) < np.abs(times - pts[left])
    return np.where(choose_right, right, left).astype(np.int64)


def is_variable_frame_rate(stream: VideoStream, intervals: FrameIntervals) -> bool:
    if stream.fps_nominal and stream.fps_avg:
        rel = abs(stream.fps_nominal - stream.fps_avg) / stream.fps_nominal
        if rel > VFR_RATE_TOLERANCE:
            return True
    return intervals.spread > VFR_SPREAD_THRESHOLD


def extract_audio(src: Path, dst: Path, stream_index: int, sample_rate: int) -> None:
    """Decode one audio stream to mono 16-bit PCM WAV at ``sample_rate``."""
    exe = require_tool("ffmpeg")
    _run(
        [
            exe, "-nostdin", "-hide_banner", "-v", "error", "-y",
            "-i", str(src),
            "-map", f"0:{stream_index}",
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            "-map_metadata", "-1", "-bitexact",
            str(dst),
        ]
    )  # fmt: skip


def grab_frame(path: Path, t: float) -> npt.NDArray[np.uint8] | None:
    """One decoded frame (BGR) at about ``t`` seconds from the start, or None past the end."""
    import cv2

    exe = require_tool("ffmpeg")
    proc = subprocess.run(
        [
            exe, "-nostdin", "-hide_banner", "-v", "error",
            "-ss", f"{max(0.0, t):.3f}", "-i", str(path),
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
        ],
        capture_output=True,
        check=False,
    )  # fmt: skip
    if proc.returncode != 0 or not proc.stdout:
        return None
    image = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
    return None if image is None else np.asarray(image, np.uint8)
