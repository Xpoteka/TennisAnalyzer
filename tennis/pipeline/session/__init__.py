"""Per-session stages: they combine every video of a session and write to the database."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tennis.config import Config
from tennis.pipeline import ProgressFn, session_dir
from tennis.util.io import read_json
from tennis.util.log import log


@dataclass
class VideoInfo:
    """A finished video as the session stages see it."""

    id: int
    source: Path
    dir: Path
    metadata: dict[str, Any]
    recorded_at: datetime | None
    offset_s: float = 0.0  # set by the timeline stage

    def path(self, name: str) -> Path:
        return self.dir / name

    def read_json(self, name: str) -> Any:
        return read_json(self.path(name))


@dataclass
class SessionContext:
    config: Config
    data_root: Path
    session_id: int
    videos: list[VideoInfo]
    stage: str
    logger: logging.Logger
    on_progress: ProgressFn | None = None
    _last_progress: float = field(default=0.0, repr=False)

    @property
    def dir(self) -> Path:
        return session_dir(self.data_root, self.session_id)

    def log(self, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
        log(self.logger, event, level=level, stage=self.stage, session=self.session_id, **fields)

    def progress(self, fraction: float, message: str | None = None) -> None:
        now = time.monotonic()
        if self.on_progress is None or (now - self._last_progress < 0.5 and fraction < 1):
            return
        self._last_progress = now
        self.on_progress(self.stage, min(max(fraction, 0.0), 1.0), message)


@dataclass(frozen=True)
class SessionStage:
    name: str
    run: Callable[[SessionContext], None]
    weight: float = 0.1
    title: str = ""


def _stages() -> tuple[SessionStage, ...]:
    from tennis.pipeline.session import timeline

    return (SessionStage("timeline", timeline.run, title="Lining up the videos"),)


SESSION_STAGES: tuple[SessionStage, ...] = _stages()
