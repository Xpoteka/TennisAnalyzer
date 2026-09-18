"""Session stage ``timeline``: place every video on one session clock.

For now, videos are ordered by their creation time and each starts at its creation time
relative to the earliest one. Audio-based sync of overlapping videos replaces this for
videos that overlap (milestone 6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlmodel import select

from tennis.db import session_scope
from tennis.db.models import Session, Video

if TYPE_CHECKING:
    from tennis.pipeline.session import SessionContext


def run(ctx: SessionContext) -> None:
    videos = sorted(ctx.videos, key=lambda v: (v.recorded_at is None, v.recorded_at or 0, v.id))
    first = next((v.recorded_at for v in videos if v.recorded_at is not None), None)
    end = 0.0
    for v in videos:
        if first is not None and v.recorded_at is not None:
            v.offset_s = (v.recorded_at - first).total_seconds()
        else:
            v.offset_s = end  # unknown time: after the previous video
        end = max(end, v.offset_s + float(v.metadata.get("duration_s") or 0.0))

    method = "single" if len(videos) == 1 else "creation_time"
    with session_scope(ctx.data_root) as db:
        rows = {r.id: r for r in db.exec(select(Video).where(Video.session_id == ctx.session_id))}
        for v in videos:
            row = rows.get(v.id)
            if row is not None:
                row.offset_s = v.offset_s
                row.sync_method = method
                db.add(row)
        session = db.get(Session, ctx.session_id)
        if session is not None:
            session.recorded_at = first
            session.duration_s = end
            db.add(session)
    ctx.log("videos placed", method=method, duration_s=round(end, 1))
