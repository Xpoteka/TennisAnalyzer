"""Thin wrappers around ffprobe and ffmpeg.

Timing rule (spec section 2): frame times always come from presentation timestamps (PTS),
never from ``frame_index / fps``, because the footage may have a variable frame rate.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

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
    sampled_frames: int
    median_ms: float
    p01_ms: float
    p99_ms: float

    @property
    def spread(self) -> float:
        return (self.p99_ms - self.p01_ms) / self.median_ms if self.median_ms > 0 else 0.0


def _percentile(sorted_values: list[float], q: float) -> float:
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def sample_frame_intervals(path: Path, stream_index: int, seconds: float = 30.0) -> FrameIntervals:
    """Measure PTS spacing over the first ``seconds`` of a video stream (packet-level, fast)."""
    exe = require_tool("ffprobe")
    out = _run(
        [
            exe, "-v", "error",
            "-select_streams", str(stream_index),
            "-read_intervals", f"%+{seconds}",
            "-show_entries", "packet=pts_time",
            "-of", "csv=p=0",
            str(path),
        ]
    )  # fmt: skip
    pts = sorted(
        v for v in (_float(line.strip().rstrip(",")) for line in out.splitlines()) if v is not None
    )
    deltas = sorted(b - a for a, b in itertools.pairwise(pts) if b > a)
    if not deltas:
        return FrameIntervals(sampled_frames=len(pts), median_ms=0.0, p01_ms=0.0, p99_ms=0.0)
    return FrameIntervals(
        sampled_frames=len(pts),
        median_ms=statistics.median(deltas) * 1000,
        p01_ms=_percentile(deltas, 0.01) * 1000,
        p99_ms=_percentile(deltas, 0.99) * 1000,
    )


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
