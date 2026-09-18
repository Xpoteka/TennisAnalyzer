"""Session stage ``summary``: the numbers the session page shows, per player and overall.

Per player (``SessionPlayer.stats``): shots by stroke, ball speed per stroke (average and
top), height over the net, the share of shots that landed in, where they landed, and how
far the player ran. The session gets a one-line headline (``Session.summary``).
Speeds and placement only count shots whose ball flight was measured.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
from sqlmodel import col, select

from tennis.db import session_scope
from tennis.db.models import Rally, Session, SessionPlayer, Shot
from tennis.util.io import read_json

if TYPE_CHECKING:
    from tennis.pipeline.session import SessionContext

SCORING_KEYS = ("points_won", "service_points", "first_serve_in_pct", "aces", "double_faults")
MAX_STEP_M = 1.5  # between 1 s samples; longer steps are tracking jumps, not running


def distance_run(ctx: SessionContext, label: str, identities: dict[str, Any]) -> float | None:
    """Metres covered on court, from the player's feet sampled once a second."""
    total = 0.0
    found = False
    for v in ctx.videos:
        tracks = {
            int(k) for k, lab in identities["tracks"].get(str(v.id), {}).items() if lab == label
        }
        if not tracks:
            continue
        d = pq.read_table(
            v.path("people.parquet"), columns=["t", "track_id", "court_x", "court_y"]
        ).to_pydict()
        pts = [
            (t, x, y)
            for t, k, x, y in zip(d["t"], d["track_id"], d["court_x"], d["court_y"], strict=True)
            if k in tracks and x is not None and y is not None
        ]
        if len(pts) < 2:
            continue
        found = True
        pts.sort()
        arr = np.array(pts)
        seconds = np.floor(arr[:, 0])
        _, first = np.unique(seconds, return_index=True)
        per_s = arr[first]
        steps = np.hypot(np.diff(per_s[:, 1]), np.diff(per_s[:, 2]))
        gaps = np.diff(per_s[:, 0])
        total += float(steps[(steps < MAX_STEP_M * gaps) & (gaps <= 2.0)].sum())
    return round(total, 1) if found else None


def player_stats(shots: list[Shot]) -> dict[str, Any]:
    by_stroke = Counter(s.stroke for s in shots)
    speeds: dict[str, list[float]] = defaultdict(list)
    for s in shots:
        if s.speed_kmh is not None:
            speeds[s.stroke].append(s.speed_kmh)
    clear = [
        s.net_clearance_m for s in shots if s.net_clearance_m is not None and s.stroke != "serve"
    ]
    landed = [s.in_court for s in shots if s.in_court is not None]
    depth = Counter(s.depth for s in shots if s.depth)
    direction = Counter(s.direction for s in shots if s.direction)
    spin = Counter(s.spin for s in shots if s.spin)
    return {
        "shots": len(shots),
        "by_stroke": dict(by_stroke),
        "speed": {
            k: {"avg": round(float(np.mean(v)), 1), "max": round(float(np.max(v)), 1), "n": len(v)}
            for k, v in speeds.items()
        },
        "net_clearance_avg_m": round(float(np.mean(clear)), 2) if clear else None,
        "in_pct": round(sum(landed) / len(landed), 3) if landed else None,
        "depth": dict(depth),
        "direction": dict(direction),
        "spin": dict(spin),
    }


def run(ctx: SessionContext) -> None:
    identities = read_json(ctx.dir / "identities.json")
    with session_scope(ctx.data_root) as db:
        shots = list(
            db.exec(select(Shot).where(Shot.session_id == ctx.session_id).order_by(col(Shot.t)))
        )
        rallies = list(db.exec(select(Rally).where(Rally.session_id == ctx.session_id)))
        players = list(
            db.exec(select(SessionPlayer).where(SessionPlayer.session_id == ctx.session_id))
        )
        session_row = db.get(Session, ctx.session_id)
        if session_row is not None and session_row.kind == "match":
            # In a match only points count, not balls hit back between them.
            playing = {r.id for r in rallies if r.end_reason not in ("not_a_point", None)}
            if playing:
                shots = [s for s in shots if s.rally_id in playing]
                rallies = [r for r in rallies if r.id in playing]
        for sp in players:
            mine = [s for s in shots if s.player_id == sp.player_id]
            stats = player_stats(mine)
            stats["distance_m"] = distance_run(ctx, sp.label, identities)
            # Keep what the scoring stage added.
            kept = {k: v for k, v in sp.stats.items() if k in SCORING_KEYS}
            sp.stats = {**kept, **stats}
            db.add(sp)
        session = db.get(Session, ctx.session_id)
        if session is not None:
            no_court = not any(v.read_json("court.json").get("found") for v in ctx.videos)
            warnings = []
            if no_court:
                warnings.append(
                    "The court was not visible from this camera: ball speed, height and "
                    "placement could not be measured."
                )
            headline = {
                **session.summary.get("headline", {}),
                "shots": len(shots),
                "rallies": len(rallies),
            }
            session.summary = {**session.summary, "headline": headline, "warnings": warnings}
            db.add(session)
    ctx.log("summary", shots=len(shots), rallies=len(rallies), players=len(players))
