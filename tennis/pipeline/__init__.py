"""The analysis pipeline: per-video stages, then per-session stages.

Each **video stage** reads the source file and earlier outputs in ``<data_root>/videos/<id>/``
and writes files there. Each **session stage** combines every video of a session and writes
its results to the database. Stages never share in-memory state, so any stage can be rerun
on its own.

A video stage is skipped when its stamp (``.stamps/<stage>.json``) matches: same stage
version, same hash of the config keys it uses, same source file, and every output present.
Session stages are cheap and always run.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from tennis import __version__
from tennis.config import Config
from tennis.util.frames import FrameReader, Window
from tennis.util.io import read_json, write_json
from tennis.util.log import log

ProgressFn = Callable[[str, float, str | None], None]  # stage, fraction of stage, message


def video_dir(data_root: Path, video_id: int) -> Path:
    return data_root / "videos" / str(video_id)


def session_dir(data_root: Path, session_id: int) -> Path:
    return data_root / "sessions" / str(session_id)


def store_video_path(data_root: Path, path: Path) -> str:
    """How a video's path is kept in the database: relative when it is inside the data folder.

    A relative path keeps working when the whole data folder moves, for example from a Mac to
    a server.
    """
    path = path.resolve()
    try:
        return str(path.relative_to(data_root.resolve()))
    except ValueError:
        return str(path)


def video_source(data_root: Path, stored: str) -> Path:
    """The file a stored video path points to (see :func:`store_video_path`)."""
    p = Path(stored)
    return p if p.is_absolute() else data_root / p


@dataclass
class VideoContext:
    """What a video stage gets: where to read and write, the config, and a progress hook."""

    config: Config
    data_root: Path
    video_id: int
    source: Path
    dir: Path
    stage: str
    logger: logging.Logger
    on_progress: ProgressFn | None = None
    _last_progress: float = field(default=0.0, repr=False)

    def path(self, name: str) -> Path:
        return self.dir / name

    def log(self, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
        log(self.logger, event, level=level, stage=self.stage, video=self.video_id, **fields)

    def progress(self, fraction: float, message: str | None = None) -> None:
        """Report progress within this stage; throttled to a few updates per second."""
        now = time.monotonic()
        if self.on_progress is None or (now - self._last_progress < 0.5 and fraction < 1):
            return
        self._last_progress = now
        self.on_progress(self.stage, min(max(fraction, 0.0), 1.0), message)

    def metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = read_json(self.path("metadata.json"))
        return data

    def frame_pts(self) -> npt.NDArray[np.float64]:
        table = pq.read_table(self.path("frame_times.parquet"), columns=["pts"])
        return np.asarray(table.column("pts").to_numpy(), np.float64)

    def frame_size(self) -> tuple[int, int]:
        """Width and height of frames as decoded (ffmpeg applies the rotation)."""
        meta = self.metadata()
        w, h = int(meta["width"]), int(meta["height"])
        if int(meta["video"].get("rotation_deg") or 0) % 180 == 90:
            w, h = h, w
        return w, h

    def frame_reader(self, *, every: int = 1, out_width: int | None = None) -> FrameReader:
        meta = self.metadata()
        w, h = self.frame_size()
        return FrameReader(
            self.source,
            self.frame_pts(),
            w,
            h,
            float(meta.get("video_start_s") or 0.0),
            hwaccel="videotoolbox" if sys.platform == "darwin" else None,
            every=every,
            out_width=out_width,
        )

    def whole_video(self) -> list[Window]:
        pts = self.frame_pts()
        return [Window(0, float(pts[0]), float(pts[-1]))]


@dataclass(frozen=True)
class VideoStage:
    name: str
    run: Callable[[VideoContext], None]
    outputs: tuple[str, ...]
    config_keys: tuple[str, ...] = ()
    # Bump when the stage's code changes what it writes, to invalidate old results.
    version: int = 1
    # Relative cost, for the overall progress bar.
    weight: float = 1.0
    title: str = ""


def _stamp_path(directory: Path, stage: str) -> Path:
    return directory / ".stamps" / f"{stage}.json"


def _source_fingerprint(source: Path) -> dict[str, Any]:
    st = source.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def stage_is_current(stage: VideoStage, config: Config, source: Path, directory: Path) -> bool:
    stamp_file = _stamp_path(directory, stage.name)
    if not stamp_file.is_file():
        return False
    try:
        stamp = read_json(stamp_file)
    except ValueError:
        return False
    expected = {
        "pipeline_version": __version__,
        "stage_version": stage.version,
        "config_hash": config.section_hash(*stage.config_keys) if stage.config_keys else "",
        "source": _source_fingerprint(source),
    }
    if any(stamp.get(k) != v for k, v in expected.items()):
        return False
    return all((directory / name).exists() for name in stage.outputs)


def run_video_stage(stage: VideoStage, ctx: VideoContext, *, force: bool = False) -> bool:
    """Run one stage unless it is current. Returns whether it ran."""
    if not force and stage_is_current(stage, ctx.config, ctx.source, ctx.dir):
        ctx.log("up to date")
        return False
    stamp_file = _stamp_path(ctx.dir, stage.name)
    stamp_file.unlink(missing_ok=True)
    ctx.dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    ctx.progress(0.0, stage.title or stage.name)
    stage.run(ctx)
    elapsed = time.monotonic() - started
    write_json(
        stamp_file,
        {
            "pipeline_version": __version__,
            "stage_version": stage.version,
            "config_hash": (
                ctx.config.section_hash(*stage.config_keys) if stage.config_keys else ""
            ),
            "source": _source_fingerprint(ctx.source),
            "elapsed_s": round(elapsed, 3),
        },
    )
    ctx.log("done", elapsed_s=round(elapsed, 2))
    return True
