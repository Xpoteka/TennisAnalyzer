"""Run the whole analysis of one session: every video's stages, then the session stages."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from sqlmodel import select

from tennis.config import Config
from tennis.db import session_scope
from tennis.db.models import Session, Video
from tennis.errors import TennisError, UserError
from tennis.pipeline import VideoContext, run_video_stage, video_dir, video_source
from tennis.pipeline.session import SESSION_STAGES, SessionContext, VideoInfo
from tennis.pipeline.video import VIDEO_STAGES
from tennis.util.io import read_json
from tennis.util.log import log, session_log

# overall fraction 0..1, current stage title, optional message
OverallProgress = Callable[[float, str, str | None], None]


def analyze_session(
    data_root: Path,
    config: Config,
    session_id: int,
    logger: logging.Logger,
    *,
    force: bool = False,
    on_progress: OverallProgress | None = None,
) -> None:
    """Analyse a session end to end. Raises when no video could be analysed."""
    with session_scope(data_root) as db:
        session = db.get(Session, session_id)
        if session is None:
            raise UserError(f"no session {session_id}")
        videos = list(db.exec(select(Video).where(Video.session_id == session_id)))
        session.status = "processing"
        session.error = None
        db.add(session)
    if not videos:
        _finish(data_root, session_id, "failed", "the session has no videos")
        raise UserError(f"session {session_id} has no videos")

    video_weight = sum(s.weight for s in VIDEO_STAGES)
    total = video_weight * len(videos) + sum(s.weight for s in SESSION_STAGES)
    done = 0.0

    def reporter(offset: float, weight: float, title: str) -> Callable[..., None]:
        def report(_stage: str, fraction: float, message: str | None) -> None:
            if on_progress is not None:
                on_progress((offset + weight * fraction) / total, title, message)

        return report

    finished: list[VideoInfo] = []
    for n, row in enumerate(videos, start=1):
        assert row.id is not None
        directory = video_dir(data_root, row.id)
        with session_log(logger, directory / "pipeline.log"):
            try:
                _set_video(data_root, row.id, status="processing", error=None)
                source = video_source(data_root, row.path)
                if not source.is_file():
                    raise UserError(f"the video file is missing: {source}")
                for stage in VIDEO_STAGES:
                    label = stage.title or stage.name
                    if len(videos) > 1:
                        label = f"{label} (video {n} of {len(videos)})"
                    ctx = VideoContext(
                        config=config,
                        data_root=data_root,
                        video_id=row.id,
                        source=source,
                        dir=directory,
                        stage=stage.name,
                        logger=logger,
                        on_progress=reporter(done, stage.weight, label),
                    )
                    run_video_stage(stage, ctx, force=force)
                    done += stage.weight
                    if stage.name == "ingest":
                        _store_metadata(data_root, row.id, read_json(ctx.path("metadata.json")))
                    elif stage.name == "court":
                        _store_court(data_root, row.id, read_json(ctx.path("court.json")))
                meta = read_json(directory / "metadata.json")
                finished.append(
                    VideoInfo(
                        id=row.id,
                        source=source,
                        dir=directory,
                        metadata=meta,
                        recorded_at=datetime.fromisoformat(meta["creation_time"]),
                    )
                )
                _set_video(data_root, row.id, status="ready")
            except Exception as exc:
                log(logger, "video failed", level=logging.ERROR, video=row.id, error=str(exc))
                if not isinstance(exc, TennisError):
                    log(logger, traceback.format_exc(), level=logging.DEBUG)
                _set_video(data_root, row.id, status="failed", error=_message(exc))
                done = video_weight * n

    if not finished:
        message = "no video could be analysed"
        _finish(data_root, session_id, "failed", message)
        raise UserError(message)

    for sstage in SESSION_STAGES:
        sctx = SessionContext(
            config=config,
            data_root=data_root,
            session_id=session_id,
            videos=finished,
            stage=sstage.name,
            logger=logger,
            on_progress=reporter(done, sstage.weight, sstage.title or sstage.name),
        )
        try:
            sstage.run(sctx)
        except Exception as exc:
            _finish(data_root, session_id, "failed", f"{sstage.name}: {_message(exc)}")
            raise
        done += sstage.weight

    _finish(data_root, session_id, "ready", None)
    if on_progress is not None:
        on_progress(1.0, "Done", None)


def _message(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


def _set_video(data_root: Path, video_id: int, **fields: object) -> None:
    with session_scope(data_root) as db:
        row = db.get(Video, video_id)
        if row is not None:
            for key, value in fields.items():
                setattr(row, key, value)
            db.add(row)


def _store_metadata(data_root: Path, video_id: int, meta: dict[str, object]) -> None:
    created = meta.get("creation_time")
    _set_video(
        data_root,
        video_id,
        duration_s=meta.get("duration_s"),
        fps=meta.get("fps"),
        width=meta.get("width"),
        height=meta.get("height"),
        recorded_at=datetime.fromisoformat(created) if isinstance(created, str) else None,
        warnings=list(meta.get("warnings") or []),  # type: ignore[call-overload]
    )


def _store_court(data_root: Path, video_id: int, court: dict[str, object]) -> None:
    quality = court.get("quality")
    _set_video(
        data_root,
        video_id,
        court=court if court.get("found") else None,
        court_quality=float(quality) if isinstance(quality, int | float) else 0.0,
    )
    if not court.get("found"):
        with session_scope(data_root) as db:
            row = db.get(Video, video_id)
            if row is not None:
                note = (
                    "the court is not fully visible, so ball speed, height and placement "
                    f"are not measured ({court.get('reason')})"
                )
                row.warnings = [w for w in row.warnings if not w.startswith("the court")] + [note]
                db.add(row)


def _finish(data_root: Path, session_id: int, status: str, error: str | None) -> None:
    with session_scope(data_root) as db:
        session = db.get(Session, session_id)
        if session is not None:
            session.status = status
            session.error = error
            db.add(session)
