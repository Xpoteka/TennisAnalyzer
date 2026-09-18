"""Video stage ``hits``: when the ball was hit, from which side, by which tracked person.

See :mod:`tennis.vision.hits`. The ball track only counts when the court was found: without
it, the "ball" is often texture flicker (clay filmed from the ground), and the sound and the
players' swings have to carry the decision.

Writes ``hits.parquet``: ``t`` (PTS), ``side`` (-1 near, +1 far), ``track`` (people track id),
``score`` and the cue values behind it, and ``sources``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tennis.util.io import read_json, write_parquet
from tennis.vision.hits import Ball, Track, detect_hits

if TYPE_CHECKING:
    from tennis.pipeline import VideoContext

OUTPUTS = ("hits.parquet",)
SCHEMA_VERSION = 1


def load_tracks(ctx: VideoContext, has_court: bool) -> list[Track]:
    """Players at the full frame rate, each frame tagged with the side of the net."""
    people = pq.read_table(
        ctx.path("people.parquet"), columns=["t", "track_id", "court_y", "foot_py"]
    ).to_pydict()
    if not people["t"]:
        return []
    p_track = np.asarray(people["track_id"])
    p_t = np.asarray(people["t"], np.float64)
    if has_court:
        court_y = np.asarray([np.nan if v is None else v for v in people["court_y"]], np.float64)
        p_side = np.where(court_y > 0, 1, -1).astype(np.int8)
    else:
        # Without a court, the near player's feet are lower in the picture.
        foot_y = np.asarray(people["foot_py"], np.float64)
        lo, hi = np.percentile(foot_y, [20, 80])
        p_side = np.where(foot_y > (lo + hi) / 2, -1, 1).astype(np.int8)

    m = pq.read_table(ctx.path("motion.parquet")).to_pydict()
    if not m["t"]:
        return []
    track_ids = np.asarray(m["track"])
    times = np.asarray(m["t"], np.float64)
    kp = np.stack(
        [np.asarray(m["kp_x"]), np.asarray(m["kp_y"]), np.asarray(m["kp_c"])], axis=2
    ).astype(np.float64)
    height = np.asarray(m["height"], np.float64)
    out = []
    for tid in np.unique(track_ids):
        sel = track_ids == tid
        order = np.argsort(times[sel])
        t = times[sel][order]
        ps = p_track == tid
        if ps.sum() == 0 or len(t) < 5:
            continue
        # Each frame takes the side of the nearest tracker sample.
        st, ss = p_t[ps], p_side[ps]
        o = np.argsort(st)
        st, ss = st[o], ss[o]
        idx = np.clip(np.searchsorted(st, t), 0, len(st) - 1)
        out.append(
            Track(id=int(tid), t=t, kp=kp[sel][order], height=height[sel][order], side=ss[idx])
        )
    return out


def run(ctx: VideoContext) -> None:
    court = read_json(ctx.path("court.json"))
    has_court = bool(court.get("found"))
    tracks = load_tracks(ctx, has_court)
    b = pq.read_table(ctx.path("ball.parquet"), columns=["t", "x", "y"])
    ball = Ball(
        t=np.asarray(b.column("t").to_numpy(), np.float64),
        xy=np.stack([b.column("x").to_numpy(), b.column("y").to_numpy()], axis=1).astype(
            np.float64
        ),
        reliable=has_court,
    )
    o = pq.read_table(ctx.path("onsets.parquet"), columns=["t", "peak_db"])
    hits = detect_hits(
        np.asarray(o.column("t").to_numpy(), np.float64),
        np.asarray(o.column("peak_db").to_numpy(), np.float64),
        ball,
        tracks,
    )
    feats = ("loudness", "sound", "swing", "timing", "near", "turn", "ball_seen")
    table = pa.table(
        {
            "t": pa.array([h.t for h in hits], pa.float64()),
            "side": pa.array([h.side for h in hits], pa.int8()),
            "track": pa.array([h.track for h in hits], pa.int32()),
            "score": pa.array([h.score for h in hits], pa.float32()),
            **{f: pa.array([h.features.get(f, 0.0) for h in hits], pa.float32()) for f in feats},
            "sources": pa.array([h.sources for h in hits], pa.list_(pa.string())),
        }
    )
    write_parquet(
        table,
        ctx.path("hits.parquet"),
        stage=ctx.stage,
        config_hash="",
        schema_version=SCHEMA_VERSION,
    )
    ctx.log("hits", count=len(hits), near=sum(h.side < 0 for h in hits), tracks=len(tracks))
