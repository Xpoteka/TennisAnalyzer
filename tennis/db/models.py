"""Database tables.

The database holds what the app shows: sessions, their videos, players, rallies, shots and
the job queue. Bulky per-frame data (keypoints, ball positions) stays in Parquet files under
``<data_root>/videos/<id>/`` and is read by the pipeline only.

Times in the database are **session seconds**: a video's time ``t`` is at session time
``t + Video.offset_s``. With a single video the two clocks are the same.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


def _json(default: Any = None) -> Any:
    factory = (lambda: default.copy()) if isinstance(default, dict | list) else (lambda: default)
    return Field(default_factory=factory, sa_column=Column(JSON))


class Session(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str | None = None  # None: the UI names it after its date
    created_at: datetime = Field(default_factory=utcnow)
    recorded_at: datetime | None = None  # the earliest video's creation time
    # What the session is: unknown until analysed, then training or match.
    kind: str = "unknown"
    kind_confidence: float | None = None
    kind_source: str = "auto"  # auto | manual
    # pending | queued | processing | ready | failed
    status: str = "pending"
    error: str | None = None
    duration_s: float | None = None
    # Session-level results: stats per player, the match score, warnings. Written by the
    # pipeline; the API passes it through.
    summary: dict[str, Any] = _json({})


class Video(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="session.id", index=True)
    filename: str
    path: str  # absolute path to the source file
    size_bytes: int = 0
    added_at: datetime = Field(default_factory=utcnow)
    recorded_at: datetime | None = None
    duration_s: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    offset_s: float = 0.0  # where this video starts on the session clock
    sync_method: str | None = None  # audio | creation_time | single
    # Court calibration: homography, camera model and a 0..1 quality. None: not found.
    court: dict[str, Any] | None = _json(None)
    court_quality: float | None = None
    status: str = "pending"  # pending | processing | ready | failed
    error: str | None = None
    warnings: list[str] = _json([])


class Player(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=utcnow)
    handedness: str | None = None  # left | right
    height_m: float | None = None
    # Re-identification signature, averaged over every session the player appeared in.
    signature: dict[str, Any] = _json({})
    thumbnail: str | None = None  # path relative to the data root
    merged_into: int | None = Field(default=None, foreign_key="player.id")


class SessionPlayer(SQLModel, table=True):
    """A player as seen in one session."""

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="session.id", index=True)
    player_id: int = Field(foreign_key="player.id", index=True)
    label: str  # the session-local track label: A, B, ...
    match_distance: float | None = None  # re-ID distance to the profile; None: new profile
    signature: dict[str, Any] = _json({})
    stats: dict[str, Any] = _json({})
    thumbnail: str | None = None


class Rally(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="session.id", index=True)
    index: int
    start_s: float
    end_s: float
    shot_count: int = 0
    server_id: int | None = Field(default=None, foreign_key="player.id")
    winner_id: int | None = Field(default=None, foreign_key="player.id")
    # winner | error | out | net | double_fault | ace | unknown
    end_reason: str | None = None
    score_before: dict[str, Any] | None = _json(None)


class Shot(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="session.id", index=True)
    video_id: int = Field(foreign_key="video.id")
    rally_id: int | None = Field(default=None, foreign_key="rally.id", index=True)
    player_id: int | None = Field(default=None, foreign_key="player.id", index=True)
    t: float  # session seconds
    index_in_rally: int | None = None
    # serve | forehand | backhand | volley_forehand | volley_backhand | overhead | unknown
    stroke: str = "unknown"
    spin: str | None = None  # topspin | slice | flat
    stroke_confidence: float | None = None
    speed_kmh: float | None = None  # off the racket
    avg_speed_kmh: float | None = None
    net_clearance_m: float | None = None
    apex_m: float | None = None
    contact_height_m: float | None = None
    # Hitter's position and the bounce, in court metres: origin at the centre of the net,
    # x across (positive right from the near baseline's view), y along (positive far).
    hit_x: float | None = None
    hit_y: float | None = None
    bounce_x: float | None = None
    bounce_y: float | None = None
    in_court: bool | None = None
    depth: str | None = None  # short | mid | deep
    direction: str | None = None  # cross | middle | line
    outcome: str | None = None  # in | out | net | winner | error
    sources: list[str] = _json([])  # which cues found the hit: audio, ball, pose
    quality: dict[str, Any] = _json({})  # per-value confidence flags


class ShotMetric(SQLModel, table=True):
    """One technique measurement of one shot."""

    id: int | None = Field(default=None, primary_key=True)
    shot_id: int = Field(foreign_key="shot.id", index=True)
    name: str = Field(index=True)
    value: float


class Job(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="session.id", index=True)
    kind: str = "analyze"
    # queued | running | done | failed | cancelled
    status: str = Field(default="queued", index=True)
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    stage: str | None = None
    progress: float = 0.0  # 0..1 over the whole job
    message: str | None = None
    error: str | None = None
    force: bool = False
    pid: int | None = None
