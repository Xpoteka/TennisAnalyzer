"""Session stage ``timeline``: place every video on one session clock.

Videos are taken in order of their creation time; the first starts at 0. Each further video
is lined up by its sound against the videos already placed (:mod:`tennis.analysis.sync`):
two cameras filming at once hear the same ball impacts, and that fixes the offset to a few
milliseconds whatever the cameras' clocks say. A video that shares no sound with the others
(a recording split into parts) is placed by its creation time instead, or right after the
previous one when that is unknown.

A video's PTS ``t`` is at session time ``t + offset_s``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
from sqlmodel import select

from tennis.analysis.sync import find_offset
from tennis.db import session_scope
from tennis.db.models import Session, Video

if TYPE_CHECKING:
    from tennis.pipeline.session import SessionContext, VideoInfo


def _envelope(v: VideoInfo) -> tuple[np.ndarray, float] | None:
    path = v.path("envelope.npy")
    if not path.is_file():
        return None
    env = np.load(path)
    meta = pq.read_metadata(v.path("onsets.parquet")).metadata or {}
    rate = float(meta.get(b"tennis.envelope_rate", b"0") or 0)
    if rate <= 0 or len(env) == 0:
        return None
    return env.astype(np.float64), rate


def run(ctx: SessionContext) -> None:
    videos = sorted(ctx.videos, key=lambda v: (v.recorded_at is None, v.recorded_at or 0, v.id))
    first = next((v.recorded_at for v in videos if v.recorded_at is not None), None)
    methods: dict[int, str] = {}
    placed: list[VideoInfo] = []
    end = 0.0
    for v in videos:
        if not placed:
            v.offset_s = 0.0
            methods[v.id] = "single" if len(videos) == 1 else "first"
        else:
            by_clock = (
                (v.recorded_at - first).total_seconds()
                if first is not None and v.recorded_at is not None
                else end
            )
            best = None
            mine = _envelope(v)
            if mine is not None:
                for other in placed:
                    theirs = _envelope(other)
                    if theirs is None or abs(theirs[1] - mine[1]) > 1e-6:
                        continue
                    result = find_offset(theirs[0], mine[0], mine[1])
                    if result.confidence > 0 and (best is None or result.confidence > best[0]):
                        a0 = float(other.metadata.get("audio_start_s") or 0.0)
                        b0 = float(v.metadata.get("audio_start_s") or 0.0)
                        best = (result.confidence, other.offset_s + result.offset_s + a0 - b0)
            if best is not None:
                v.offset_s = best[1]
                methods[v.id] = "audio"
                ctx.log(
                    "synced by sound",
                    video=v.id,
                    offset_s=round(best[1], 3),
                    confidence=round(best[0], 2),
                )
            else:
                v.offset_s = by_clock
                methods[v.id] = "creation_time"
        placed.append(v)
        end = max(end, v.offset_s + float(v.metadata.get("duration_s") or 0.0))

    start = min(v.offset_s for v in videos)
    for v in videos:  # the session clock starts with the earliest video
        v.offset_s -= start
    end -= start
    with session_scope(ctx.data_root) as db:
        rows = {r.id: r for r in db.exec(select(Video).where(Video.session_id == ctx.session_id))}
        for v in videos:
            row = rows.get(v.id)
            if row is not None:
                row.offset_s = round(v.offset_s, 4)
                row.sync_method = (
                    "single"
                    if len(videos) == 1
                    else methods[v.id].replace("first", "creation_time")
                )
                db.add(row)
        session = db.get(Session, ctx.session_id)
        if session is not None:
            session.recorded_at = first
            session.duration_s = end
            db.add(session)
    ctx.log("videos placed", methods=methods, duration_s=round(end, 1))
