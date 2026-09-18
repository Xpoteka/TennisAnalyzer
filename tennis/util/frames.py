"""Decode selected time ranges of a video as BGR frames, with exact PTS.

Frames come from an ``ffmpeg`` subprocess (raw BGR on stdout). The ``showinfo`` filter
reports each frame's PTS on stderr, which is matched to the frame index from
``frame_times.parquet``. Only one frame is held in memory at a time. ffmpeg applies the
container's rotation, so frames arrive upright.

PyAV is not used because on macOS it bundles an FFmpeg that clashes with OpenCV's.
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tennis.util.video import ProbeError, require_tool

_SHOWINFO_RE = re.compile(r"\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:\s*(-?[0-9.eE+-]+)")
PTS_TOLERANCE_S = 0.001
_STDERR_TAIL = 20


@dataclass(frozen=True)
class Frame:
    index: int
    pts: float
    image: npt.NDArray[np.uint8]  # H x W x 3, BGR
    window_id: int


@dataclass(frozen=True)
class Window:
    id: int
    start: float  # PTS, inclusive
    end: float  # PTS, inclusive


def merge_windows(
    centers: Sequence[float], pre_s: float, post_s: float, lo: float, hi: float
) -> list[Window]:
    """``[t - pre, t + post]`` around each time, clipped to ``[lo, hi]``; overlaps merged."""
    spans = sorted((max(lo, t - pre_s), min(hi, t + post_s)) for t in centers)
    merged: list[list[float]] = []
    for start, end in spans:
        if end < start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [Window(i, a, b) for i, (a, b) in enumerate(merged)]


def group_windows(windows: Sequence[Window], max_gap_s: float) -> list[list[Window]]:
    """Consecutive windows closer than ``max_gap_s`` are decoded in one pass."""
    groups: list[list[Window]] = []
    for w in windows:
        if groups and w.start - groups[-1][-1].end <= max_gap_s:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


class FrameReader:
    def __init__(
        self,
        path: Path,
        frame_pts: npt.NDArray[np.float64],
        width: int,
        height: int,
        video_start_s: float = 0.0,
        hwaccel: str | None = None,
        *,
        every: int = 1,
        out_width: int | None = None,
    ) -> None:
        """``every``: keep one frame in this many. ``out_width``: scale frames to this width
        (the height follows, rounded to an even number)."""
        self.path = path
        self.frame_pts = frame_pts
        self.src_width = width
        self.src_height = height
        self.every = max(1, every)
        if out_width is not None and out_width < width:
            self.width = out_width - out_width % 2
            self.height = round(height * self.width / width / 2) * 2
        else:
            self.width, self.height = width, height
        self.video_start_s = video_start_s
        self.hwaccel = hwaccel
        self._ffmpeg = require_tool("ffmpeg")

    @property
    def scale(self) -> float:
        """Output pixels per source pixel."""
        return self.width / self.src_width

    def _command(self, start_pts: float) -> list[str]:
        cmd = [self._ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info"]
        if self.hwaccel:
            cmd += ["-hwaccel", self.hwaccel]
        seek = max(0.0, start_pts - self.video_start_s)
        cmd += [
            "-copyts",
            "-ss", f"{seek:.6f}",
            "-i", str(self.path),
            "-map", "0:v:0",
            "-an", "-sn", "-dn",
            "-vf", self._filters(),
            "-fps_mode", "passthrough",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-",
        ]  # fmt: skip
        return cmd

    def _filters(self) -> str:
        chain = []
        if self.every > 1:
            chain.append(f"select='not(mod(n\\,{self.every}))'")
        if (self.width, self.height) != (self.src_width, self.src_height):
            chain.append(f"scale={self.width}:{self.height}:flags=area")
        chain.append("showinfo")
        return ",".join(chain)

    def read(self, windows: Sequence[Window]) -> Iterator[Frame]:
        """Frames inside ``windows`` (sorted, non-overlapping), decoded in a single pass."""
        if not windows:
            return
        first, last = windows[0].start, windows[-1].end
        proc = subprocess.Popen(
            self._command(first),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert proc.stdout is not None and proc.stderr is not None
        pts_queue: queue.Queue[float | None] = queue.Queue()
        tail: list[str] = []

        def pump(stream: object) -> None:
            for raw in iter(stream.readline, b""):  # type: ignore[attr-defined]
                line = raw.decode("utf-8", "replace")
                match = _SHOWINFO_RE.search(line)
                if match:
                    pts_queue.put(float(match.group(1)))
                elif line.strip():
                    tail.append(line.strip())
                    del tail[:-_STDERR_TAIL]
            pts_queue.put(None)

        thread = threading.Thread(target=pump, args=(proc.stderr,), daemon=True)
        thread.start()
        size = self.width * self.height * 3
        w = 0
        try:
            while True:
                buf = _read_exact(proc.stdout, size)
                if buf is None:
                    break
                pts = pts_queue.get(timeout=60)
                if pts is None:
                    raise ProbeError("ffmpeg produced a frame without a timestamp")
                if pts > last + PTS_TOLERANCE_S:
                    break
                while w < len(windows) and pts > windows[w].end + PTS_TOLERANCE_S:
                    w += 1
                if w == len(windows) or pts < windows[w].start - PTS_TOLERANCE_S:
                    continue
                index = self._index(pts)
                image = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
                yield Frame(index=index, pts=float(self.frame_pts[index]), image=image,
                            window_id=windows[w].id)  # fmt: skip
        finally:
            proc.kill()
            proc.wait()
            thread.join(timeout=5)
        if proc.returncode not in (0, -9) and w < len(windows):
            raise ProbeError("ffmpeg failed while decoding: " + " | ".join(tail[-5:]))

    def _index(self, pts: float) -> int:
        i = int(np.searchsorted(self.frame_pts, pts - PTS_TOLERANCE_S, side="left"))
        if i >= self.frame_pts.size or abs(self.frame_pts[i] - pts) > PTS_TOLERANCE_S:
            raise ProbeError(f"decoded frame at {pts:.6f}s is not in frame_times.parquet")
        return i


def _read_exact(stream: object, size: int) -> bytes | None:
    buf = bytearray(size)
    view = memoryview(buf)
    got = 0
    while got < size:
        n = stream.readinto(view[got:])  # type: ignore[attr-defined]
        if not n:
            return None
        got += n
    return bytes(buf)
