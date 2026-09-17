from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")

MakeVideo = Callable[..., Path]
COUNTER_PERIOD = 110  # keeps the counter pattern's luma within 16..235


def implemented_stages(start: str = "ingest", *, labels: bool = True) -> list[str]:
    """Names of the stages that run today, from ``start`` on, in the order the runner uses.

    The runner stops at the first unimplemented stage. Optional stages are included when
    enabled (the labels stage is enabled by default).
    """
    from tennis.stages import STAGES

    names: list[str] = []
    for s in STAGES:
        if s.optional and not labels:
            continue
        if not s.implemented:
            break
        names.append(s.name)
    return names[names.index(start) :]


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
        audio_wav: Path | None = None,
        counter: bool = False,
    ) -> Path:
        """Synthetic clip.

        ``audio_wav`` replaces the default 1 kHz tone with the given WAV file. ``counter``
        makes a small grey video whose brightness encodes the source frame number n
        modulo ``COUNTER_PERIOD`` (luma 16 + 2 (n mod 110), see ``frame_number``).
        """
        key = (name, seconds, fps, audio, vfr, creation_time, audio_wav, counter)
        if key in cache:
            return cache[key]
        out = root / name
        if counter:
            lum = f"16+2*mod(N\\,{COUNTER_PERIOD})"
            source = f"nullsrc=s=64x48:r={fps}:d={seconds},geq=lum='{lum}':cb=128:cr=128"
        else:
            source = f"testsrc2=size=320x240:rate={fps}:duration={seconds}"
        args = ["-f", "lavfi", "-i", source]
        if audio_wav is not None:
            args += ["-i", str(audio_wav), "-map", "0:v", "-map", "1:a", "-shortest"]
            args += ["-ac", "2", "-c:a", "aac", "-b:a", "256k"]
        elif audio:
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


def frame_number(image: object) -> int:
    """Inverse of the ``counter`` video pattern: source frame number mod COUNTER_PERIOD."""
    import numpy as np

    mean = float(np.asarray(image).mean())
    return round((mean * 219 / 255) / 2)


class _NoPeopleBackend:
    name = "yolo"

    def infer(self, frames: list[object]) -> list[list[object]]:
        return [[] for _ in frames]


@pytest.fixture(autouse=True)
def _no_real_pose_model(request: pytest.FixtureRequest) -> Iterator[None]:
    """Tests never load or download YOLO unless marked ``real_yolo``."""
    from tennis import pose_backends

    if request.node.get_closest_marker("real_yolo"):
        yield
        return
    original = pose_backends._REGISTRY["yolo"]
    pose_backends.register_backend("yolo", lambda cfg, path, device: _NoPeopleBackend())  # type: ignore[arg-type,return-value]
    try:
        yield
    finally:
        pose_backends.register_backend("yolo", original)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"
