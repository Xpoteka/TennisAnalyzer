"""JSON routes for sessions, videos, jobs, players, the video library and the config."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import Session as DbSession
from sqlmodel import col, delete, select

from tennis import worker
from tennis.api.app import AppState, get_state
from tennis.api.uploads import VIDEO_SUFFIXES, delete_video, list_videos
from tennis.config import DEFAULT_CONFIG_NAME, Config
from tennis.db import get_engine
from tennis.db.models import (
    Job,
    Player,
    Rally,
    Session,
    SessionPlayer,
    Shot,
    ShotMetric,
    Video,
)
from tennis.errors import ConfigError
from tennis.pipeline import session_dir, store_video_path, video_dir, video_source
from tennis.util.io import read_json

router = APIRouter()


def utc(d: datetime | None) -> str | None:
    """SQLite stores naive datetimes; every one of ours is UTC, so say so to the browser."""
    if d is None:
        return None
    return (d if d.tzinfo else d.replace(tzinfo=UTC)).isoformat()


def db_dep(request: Request) -> Any:
    with DbSession(get_engine(get_state(request).data_root), expire_on_commit=False) as db:
        yield db


Db = Depends(db_dep)
State = Depends(get_state)


# --- serializers -----------------------------------------------------------------------------


def _video_json(v: Video, data_root: Path) -> dict[str, Any]:
    proxy_info = video_dir(data_root, v.id or 0) / "proxy.json"
    start_pts = read_json(proxy_info).get("start_pts", 0.0) if proxy_info.is_file() else None
    return {
        "id": v.id,
        "filename": v.filename,
        "size_bytes": v.size_bytes,
        "recorded_at": utc(v.recorded_at),
        "duration_s": v.duration_s,
        "fps": v.fps,
        "width": v.width,
        "height": v.height,
        "offset_s": v.offset_s,
        "sync_method": v.sync_method,
        "court_quality": v.court_quality,
        "has_court": v.court is not None,
        "status": v.status,
        "error": v.error,
        "warnings": v.warnings,
        "proxy_url": f"/api/videos/{v.id}/proxy" if start_pts is not None else None,
        "proxy_start_pts": start_pts,
    }


def _job_json(job: Job | None) -> dict[str, Any] | None:
    if job is None:
        return None
    return {
        "id": job.id,
        "session_id": job.session_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
        "created_at": utc(job.created_at),
        "started_at": utc(job.started_at),
        "finished_at": utc(job.finished_at),
    }


def _latest_job(db: DbSession, session_id: int) -> Job | None:
    return db.exec(
        select(Job).where(Job.session_id == session_id).order_by(col(Job.id).desc())
    ).first()


def _session_players(db: DbSession, session_id: int) -> list[dict[str, Any]]:
    rows = db.exec(
        select(SessionPlayer, Player)
        .join(Player, col(Player.id) == SessionPlayer.player_id)
        .where(SessionPlayer.session_id == session_id)
        .order_by(col(SessionPlayer.label))
    ).all()
    return [
        {
            "player_id": p.id,
            "name": p.name,
            "label": sp.label,
            "handedness": p.handedness,
            "new_profile": sp.match_distance is None,
            "stats": sp.stats,
            "thumbnail_url": f"/api/media/{sp.thumbnail}" if sp.thumbnail else None,
        }
        for sp, p in rows
    ]


def _session_json(db: DbSession, s: Session, data_root: Path, *, full: bool) -> dict[str, Any]:
    assert s.id is not None
    videos = list(db.exec(select(Video).where(Video.session_id == s.id).order_by(col(Video.id))))
    out: dict[str, Any] = {
        "id": s.id,
        "name": s.name,
        "created_at": utc(s.created_at),
        "recorded_at": utc(s.recorded_at),
        "kind": s.kind,
        "kind_confidence": s.kind_confidence,
        "kind_source": s.kind_source,
        "status": s.status,
        "error": s.error,
        "duration_s": s.duration_s,
        "video_count": len(videos),
        "players": _session_players(db, s.id),
        "job": _job_json(_latest_job(db, s.id)),
        "headline": s.summary.get("headline", {}),
    }
    if full:
        out["videos"] = [_video_json(v, data_root) for v in videos]
        out["summary"] = s.summary
    return out


# --- sessions --------------------------------------------------------------------------------


class NewSession(BaseModel):
    paths: list[str] = Field(min_length=1)
    name: str | None = None


class AddVideos(BaseModel):
    paths: list[str] = Field(min_length=1)


class SessionPatch(BaseModel):
    name: str | None = None
    kind: str | None = None  # training | match | auto


def _check_video_paths(paths: list[str]) -> list[Path]:
    out = []
    for raw in paths:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            raise HTTPException(400, f"not an absolute path: {raw}")
        if not p.is_file():
            raise HTTPException(404, f"no such file: {raw}")
        if p.suffix.lower() not in VIDEO_SUFFIXES:
            raise HTTPException(415, f"not a video file: {p.name}")
        out.append(p.resolve())
    return out


def _add_videos(db: DbSession, data_root: Path, session_id: int, paths: list[Path]) -> None:
    known = {v.path for v in db.exec(select(Video).where(Video.session_id == session_id))}
    for p in paths:
        stored = store_video_path(data_root, p)
        if stored in known:
            continue
        db.add(
            Video(session_id=session_id, filename=p.name, path=stored, size_bytes=p.stat().st_size)
        )
        known.add(stored)


@router.get("/sessions")
def list_sessions(db: DbSession = Db, state: AppState = State) -> list[dict[str, Any]]:
    rows = db.exec(select(Session)).all()
    rows = sorted(rows, key=lambda s: (s.recorded_at or s.created_at).timestamp(), reverse=True)
    return [_session_json(db, s, state.data_root, full=False) for s in rows]


@router.post("/sessions", status_code=201)
def create_session(body: NewSession, db: DbSession = Db, state: AppState = State) -> dict[str, Any]:
    paths = _check_video_paths(body.paths)
    session = Session(name=(body.name or "").strip() or None)
    db.add(session)
    db.flush()
    assert session.id is not None
    _add_videos(db, state.data_root, session.id, paths)
    db.commit()
    worker.enqueue(state.data_root, session.id)
    db.refresh(session)
    return _session_json(db, session, state.data_root, full=True)


def _get_session(db: DbSession, session_id: int) -> Session:
    s = db.get(Session, session_id)
    if s is None:
        raise HTTPException(404, f"no session {session_id}")
    return s


@router.get("/sessions/{session_id}")
def get_session(session_id: int, db: DbSession = Db, state: AppState = State) -> dict[str, Any]:
    return _session_json(db, _get_session(db, session_id), state.data_root, full=True)


@router.patch("/sessions/{session_id}")
def patch_session(
    session_id: int, body: SessionPatch, db: DbSession = Db, state: AppState = State
) -> dict[str, Any]:
    s = _get_session(db, session_id)
    if body.name is not None:
        s.name = body.name.strip() or None
    rerun = False
    if body.kind is not None:
        if body.kind not in ("training", "match", "auto"):
            raise HTTPException(400, "kind must be training, match or auto")
        s.kind_source = "auto" if body.kind == "auto" else "manual"
        if body.kind != "auto":
            s.kind = body.kind
        rerun = True  # scoring depends on the kind
    db.add(s)
    db.commit()
    if rerun and s.status in ("ready", "failed"):
        worker.enqueue(state.data_root, session_id)
    db.refresh(s)
    return _session_json(db, s, state.data_root, full=True)


@router.post("/sessions/{session_id}/videos")
def add_videos(
    session_id: int, body: AddVideos, db: DbSession = Db, state: AppState = State
) -> dict[str, Any]:
    s = _get_session(db, session_id)
    _add_videos(db, state.data_root, session_id, _check_video_paths(body.paths))
    db.commit()
    worker.enqueue(state.data_root, session_id)
    return _session_json(db, s, state.data_root, full=True)


@router.delete("/sessions/{session_id}/videos/{video_id}")
def remove_video(
    session_id: int, video_id: int, db: DbSession = Db, state: AppState = State
) -> dict[str, Any]:
    s = _get_session(db, session_id)
    v = db.get(Video, video_id)
    if v is None or v.session_id != session_id:
        raise HTTPException(404, f"no video {video_id} in session {session_id}")
    _delete_results(db, session_id, video_ids=[video_id])
    db.delete(v)
    db.commit()
    shutil.rmtree(video_dir(state.data_root, video_id), ignore_errors=True)
    if db.exec(select(Video).where(Video.session_id == session_id)).first() is not None:
        worker.enqueue(state.data_root, session_id)
    return _session_json(db, s, state.data_root, full=True)


@router.post("/sessions/{session_id}/analyze")
def reanalyze(
    session_id: int, force: bool = False, db: DbSession = Db, state: AppState = State
) -> dict[str, Any]:
    _get_session(db, session_id)
    return _job_json(worker.enqueue(state.data_root, session_id, force=force)) or {}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: int, db: DbSession = Db, state: AppState = State) -> dict[str, Any]:
    """Delete a session and everything derived from it. The source videos stay."""
    _get_session(db, session_id)
    for job in db.exec(select(Job).where(Job.session_id == session_id)):
        if job.status in ("queued", "running") and job.id is not None:
            worker.cancel(state.data_root, job.id)
    db.expire_all()
    video_ids = [v.id for v in db.exec(select(Video).where(Video.session_id == session_id))]
    _delete_results(db, session_id)
    db.exec(delete(Job).where(col(Job.session_id) == session_id))
    db.exec(delete(Video).where(col(Video.session_id) == session_id))
    db.exec(delete(Session).where(col(Session.id) == session_id))
    db.commit()
    for vid in video_ids:
        if vid is not None:
            shutil.rmtree(video_dir(state.data_root, vid), ignore_errors=True)
    shutil.rmtree(session_dir(state.data_root, session_id), ignore_errors=True)
    return {"deleted": session_id}


def _delete_results(db: DbSession, session_id: int, video_ids: list[int] | None = None) -> None:
    """Remove the analysis results of a session (all of them, or those of some videos)."""
    shots = select(Shot.id).where(Shot.session_id == session_id)
    if video_ids is not None:
        shots = shots.where(col(Shot.video_id).in_(video_ids))
    db.exec(delete(ShotMetric).where(col(ShotMetric.shot_id).in_(shots)))
    if video_ids is None:
        db.exec(delete(Shot).where(col(Shot.session_id) == session_id))
        db.exec(delete(Rally).where(col(Rally.session_id) == session_id))
        db.exec(delete(SessionPlayer).where(col(SessionPlayer.session_id) == session_id))
    else:
        db.exec(
            delete(Shot).where(
                col(Shot.session_id) == session_id, col(Shot.video_id).in_(video_ids)
            )
        )


@router.get("/sessions/{session_id}/shots")
def list_shots(session_id: int, db: DbSession = Db) -> list[dict[str, Any]]:
    _get_session(db, session_id)
    shots = db.exec(select(Shot).where(Shot.session_id == session_id).order_by(col(Shot.t))).all()
    metrics: dict[int, dict[str, float]] = {}
    ids = [s.id for s in shots]
    for m in db.exec(select(ShotMetric).where(col(ShotMetric.shot_id).in_(ids))):
        metrics.setdefault(m.shot_id, {})[m.name] = m.value
    return [{**s.model_dump(), "metrics": metrics.get(s.id or 0, {})} for s in shots]


@router.get("/sessions/{session_id}/rallies")
def list_rallies(session_id: int, db: DbSession = Db) -> list[dict[str, Any]]:
    _get_session(db, session_id)
    rows = db.exec(
        select(Rally).where(Rally.session_id == session_id).order_by(col(Rally.index))
    ).all()
    return [r.model_dump() for r in rows]


# --- videos, media, jobs -------------------------------------------------------------------


@router.get("/videos/{video_id}/proxy")
def video_proxy(video_id: int, state: AppState = State) -> FileResponse:
    path = video_dir(state.data_root, video_id) / "proxy.mp4"
    if not path.exists():
        raise HTTPException(404, "this video has no playable copy yet")
    return FileResponse(path, media_type="video/mp4")


MEDIA_ROOTS = ("videos", "sessions", "players")


@router.get("/media/{rel:path}")
def media(rel: str, state: AppState = State) -> FileResponse:
    """Images and small files the pipeline writes (thumbnails, debug frames)."""
    root = state.data_root.resolve()
    path = (root / rel).resolve()
    if root not in path.parents or path.relative_to(root).parts[0] not in MEDIA_ROOTS:
        raise HTTPException(404)
    if path.name.startswith(".") or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@router.get("/jobs")
def list_jobs(active: bool = False, db: DbSession = Db) -> list[dict[str, Any]]:
    q = select(Job).order_by(col(Job.id).desc())
    if active:
        q = q.where(col(Job.status).in_(("queued", "running")))
    return [j for j in (_job_json(job) for job in db.exec(q.limit(50))) if j is not None]


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, state: AppState = State) -> dict[str, Any]:
    return {"cancelled": worker.cancel(state.data_root, job_id)}


# --- players -------------------------------------------------------------------------------


class PlayerPatch(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class PlayerMerge(BaseModel):
    into: int


def _player_json(db: DbSession, p: Player, *, full: bool) -> dict[str, Any]:
    assert p.id is not None
    appearances = db.exec(
        select(SessionPlayer, Session)
        .join(Session, col(Session.id) == SessionPlayer.session_id)
        .where(SessionPlayer.player_id == p.id)
    ).all()
    appearances = sorted(
        appearances, key=lambda r: (r[1].recorded_at or r[1].created_at).timestamp()
    )
    shots = db.exec(select(Shot).where(Shot.player_id == p.id)).all()
    by_stroke: dict[str, int] = {}
    for s in shots:
        by_stroke[s.stroke] = by_stroke.get(s.stroke, 0) + 1
    out: dict[str, Any] = {
        "id": p.id,
        "name": p.name,
        "handedness": p.handedness,
        "height_m": p.height_m,
        "created_at": utc(p.created_at),
        "thumbnail_url": f"/api/media/{p.thumbnail}" if p.thumbnail else None,
        "session_count": len(appearances),
        "shot_count": len(shots),
        "strokes": by_stroke,
        "last_seen": utc(appearances[-1][1].recorded_at) if appearances else None,
    }
    if full:
        out["sessions"] = [
            {
                "session_id": s.id,
                "name": s.name,
                "recorded_at": utc(s.recorded_at),
                "kind": s.kind,
                "label": sp.label,
                "stats": sp.stats,
            }
            for sp, s in appearances
        ]
    return out


@router.get("/players")
def list_players(db: DbSession = Db) -> list[dict[str, Any]]:
    players = db.exec(select(Player).where(col(Player.merged_into).is_(None))).all()
    return [_player_json(db, p, full=False) for p in players]


def _get_player(db: DbSession, player_id: int) -> Player:
    p = db.get(Player, player_id)
    while p is not None and p.merged_into is not None:
        p = db.get(Player, p.merged_into)
    if p is None:
        raise HTTPException(404, f"no player {player_id}")
    return p


@router.get("/players/{player_id}")
def get_player(player_id: int, db: DbSession = Db) -> dict[str, Any]:
    return _player_json(db, _get_player(db, player_id), full=True)


@router.patch("/players/{player_id}")
def rename_player(player_id: int, body: PlayerPatch, db: DbSession = Db) -> dict[str, Any]:
    p = _get_player(db, player_id)
    p.name = body.name.strip()
    db.add(p)
    db.commit()
    return _player_json(db, p, full=True)


@router.post("/players/{player_id}/merge")
def merge_player(player_id: int, body: PlayerMerge, db: DbSession = Db) -> dict[str, Any]:
    """Fold this player into another: their sessions and shots move over."""
    src, dst = _get_player(db, player_id), _get_player(db, body.into)
    if src.id == dst.id:
        raise HTTPException(400, "cannot merge a player into itself")
    for sp in db.exec(select(SessionPlayer).where(SessionPlayer.player_id == src.id)):
        sp.player_id = dst.id  # type: ignore[assignment]
        db.add(sp)
    for shot in db.exec(select(Shot).where(Shot.player_id == src.id)):
        shot.player_id = dst.id
        db.add(shot)
    for rally in db.exec(select(Rally).where(Rally.server_id == src.id)):
        rally.server_id = dst.id
        db.add(rally)
    for rally in db.exec(select(Rally).where(Rally.winner_id == src.id)):
        rally.winner_id = dst.id
        db.add(rally)
    src.merged_into = dst.id
    db.add(src)
    db.commit()
    return _player_json(db, dst, full=True)


# --- library and config ----------------------------------------------------------------------


@router.get("/library")
def library(db: DbSession = Db, state: AppState = State) -> dict[str, Any]:
    used_by: dict[Path, list[int]] = {}
    for v in db.exec(select(Video)):
        used_by.setdefault(video_source(state.data_root, v.path).resolve(), []).append(v.session_id)
    return list_videos(state.data_root, used_by)


@router.delete("/library/{name}")
def library_delete(name: str, state: AppState = State) -> dict[str, Any]:
    path = delete_video(state.data_root, name)
    return {"deleted": path.name}


class ConfigText(BaseModel):
    text: str


def _config_file(state: AppState) -> Path:
    return state.config_path or state.data_root.parent / DEFAULT_CONFIG_NAME


@router.get("/config")
def get_config(state: AppState = State) -> dict[str, Any]:
    path = _config_file(state)
    text = path.read_text() if path.is_file() else ""
    return {"path": str(path), "text": text, "defaults": Config().model_dump(mode="json")}


@router.put("/config")
def put_config(body: ConfigText, state: AppState = State) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(body.text) or {}
        if not isinstance(raw, dict):
            raise ConfigError("the top level must be a mapping")
        Config.model_validate(raw)
    except Exception as exc:
        raise HTTPException(400, f"not saved: {exc}") from exc
    path = _config_file(state)
    path.write_text(body.text)
    return {"saved": str(path)}
