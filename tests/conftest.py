from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")

MakeVideo = Callable[..., Path]


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def make_video(tmp_path_factory: pytest.TempPathFactory) -> MakeVideo:
    """Build small synthetic clips with ffmpeg; cached per parameter set."""
    root = tmp_path_factory.mktemp("videos")
    cache: dict[tuple[object, ...], Path] = {}

    def make(
        name: str = "clip.mp4",
        *,
        seconds: float = 2.0,
        fps: int = 120,
        audio: bool = True,
        vfr: bool = False,
        creation_time: str | None = "2026-09-20T18:30:00Z",
    ) -> Path:
        key = (name, seconds, fps, audio, vfr, creation_time)
        if key in cache:
            return cache[key]
        out = root / name
        args = ["-f", "lavfi", "-i", f"testsrc2=size=320x240:rate={fps}:duration={seconds}"]
        if audio:
            args += [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=1000:sample_rate=44100:duration={seconds}",
            ]
            args += ["-ac", "2", "-c:a", "aac"]
        if vfr:
            # Drop every third frame and keep the original timestamps: uneven PTS spacing.
            args += ["-vf", "select='not(eq(mod(n\\,3)\\,2))'", "-fps_mode", "vfr"]
        args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
        if creation_time:
            args += ["-metadata", f"creation_time={creation_time}"]
        args += [str(out)]
        _ffmpeg(*args)
        cache[key] = out
        return out

    return make


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"
