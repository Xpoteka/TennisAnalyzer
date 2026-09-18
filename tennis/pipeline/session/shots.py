"""Session stage ``shots``: every shot with its player, stroke and ball flight; and rallies.

For each video: the hits become shots. The hitter's track gives the player (through
``identities.json``); the swing gives the stroke (:mod:`tennis.vision.strokes`); the ball
detections between this hit and the next give the flight (:mod:`tennis.vision.physics`)
when the court and camera are known. Hits less than ``RALLY_GAP_S`` apart form a rally, and
a serve always starts a new one.

Rows in the ``shot`` and ``rally`` tables for the session are replaced.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
from sqlmodel import col, delete, select

from tennis.db import session_scope
from tennis.db.models import Player, Rally, SessionPlayer, Shot, ShotMetric
from tennis.util.io import read_json
from tennis.vision import court as court_mod
from tennis.vision.physics import ShotFlight, fit_flight
from tennis.vision.strokes import KP, Swing, classify, racket_hand

if TYPE_CHECKING:
    from tennis.pipeline.session import SessionContext, VideoInfo

MAX_FLIGHT_S = 2.2
RALLY_GAP_S = 3.0  # longer between two hits: the rally is over
MIN_SPEED_KMH, MAX_SPEED_KMH = 15.0, 260.0
# Plausible speed off the racket per stroke; outside, the fit latched onto the wrong ball.
SPEED_RANGE_KMH = {
    "serve": (60.0, 230.0),
    "overhead": (40.0, 200.0),
    "forehand": (30.0, 180.0),
    "backhand": (30.0, 170.0),
    "volley_forehand": (20.0, 140.0),
    "volley_backhand": (20.0, 140.0),
    "unknown": (20.0, 200.0),
}
MAX_RMS_PX = 8.0  # fit error, in pixels at 1000 px image size


@dataclass
class VideoData:
    info: VideoInfo
    hits: dict[str, list[Any]]
    swings: dict[int, Swing]
    tracks: dict[int, dict[str, np.ndarray]]  # track -> t, court_x, court_y
    ball_t: np.ndarray
    ball_xy: np.ndarray
    cal: court_mod.Calibration | None
    labels: dict[int, str]  # track -> player label


def load(v: VideoInfo, identities: dict[str, Any]) -> VideoData:
    hits = pq.read_table(v.path("hits.parquet")).to_pydict()
    swing_objs = swings_around_hits(v, hits)
    ppl = pq.read_table(
        v.path("people.parquet"),
        columns=["t", "track_id", "court_x", "court_y", "foot_px", "foot_py"],
    ).to_pydict()
    ids = np.asarray(ppl["track_id"])
    tracks = {}
    for tid in np.unique(ids):
        m = ids == tid
        tracks[int(tid)] = {
            "t": np.asarray(ppl["t"], np.float64)[m],
            "x": np.asarray(
                [np.nan if c is None else c for c in np.asarray(ppl["court_x"], object)[m]],
                np.float64,
            ),
            "y": np.asarray(
                [np.nan if c is None else c for c in np.asarray(ppl["court_y"], object)[m]],
                np.float64,
            ),
            "px": np.asarray(ppl["foot_px"], np.float64)[m],
            "py": np.asarray(ppl["foot_py"], np.float64)[m],
        }
    ball = pq.read_table(v.path("ball.parquet"), columns=["t", "x", "y"]).to_pydict()
    court = v.read_json("court.json")
    cal = court_mod.Calibration.from_json(court) if court.get("found") else None
    labels = {int(k): lab for k, lab in identities.get("tracks", {}).get(str(v.id), {}).items()}
    return VideoData(
        info=v,
        hits=hits,
        swings=swing_objs,
        tracks=tracks,
        ball_t=np.asarray(ball["t"], np.float64),
        ball_xy=np.stack([ball["x"], ball["y"]], axis=1).astype(np.float64)
        if ball["t"]
        else np.zeros((0, 2)),
        cal=cal,
        labels=labels,
    )


SWING_BEFORE_S, SWING_AFTER_S = 0.7, 0.35


def swings_around_hits(v: VideoInfo, hits: dict[str, list[Any]]) -> dict[int, Swing]:
    """The hitter's full-rate keypoints around each hit (from the ``motion`` stage)."""
    m = pq.read_table(v.path("motion.parquet")).to_pydict()
    if not m["t"] or not hits["t"]:
        return {}
    track = np.asarray(m["track"])
    t = np.asarray(m["t"], np.float64)
    kp = np.stack([np.asarray(m["kp_x"]), np.asarray(m["kp_y"]), np.asarray(m["kp_c"])], axis=2)
    by_track = {}
    for tid in np.unique(track):
        sel = np.nonzero(track == tid)[0]
        sel = sel[np.argsort(t[sel])]
        by_track[int(tid)] = (t[sel], kp[sel].astype(np.float64))
    out = {}
    for i, (t_hit, tid) in enumerate(zip(hits["t"], hits["track"], strict=True)):
        if int(tid) not in by_track:
            continue
        tt, kk = by_track[int(tid)]
        a = int(np.searchsorted(tt, t_hit - SWING_BEFORE_S))
        b = int(np.searchsorted(tt, t_hit + SWING_AFTER_S, side="right"))
        if b - a >= 5:
            out[i] = Swing(tt[a:b] - t_hit, kk[a:b])
    return out


def feet_at(vd: VideoData, track: int, t: float) -> tuple[float, float] | None:
    tr = vd.tracks.get(track)
    if tr is None or not len(tr["t"]):
        return None
    i = int(np.argmin(np.abs(tr["t"] - t)))
    if abs(tr["t"][i] - t) > 0.5 or np.isnan(tr["x"][i]):
        return None
    return float(tr["x"][i]), float(tr["y"][i])


def right_direction(vd: VideoData, feet: tuple[float, float] | None, side: int) -> np.ndarray:
    """Unit image vector to the hitter's right as they face the net."""
    toward = 1.0 if side < 0 else -1.0  # near players face +y, far players -y
    if vd.cal is not None and feet is not None:
        p = np.array([feet, (feet[0] + 0.5 * toward, feet[1])])
        a, b = vd.cal.court_to_image(p)
        d = b - a
        n = float(np.linalg.norm(d))
        if np.isfinite(n) and n > 0:
            return np.asarray(d / n)
    # A camera behind the near baseline: the near player's right is the picture's right.
    return np.array([toward, 0.0])


def contact_point(
    vd: VideoData, sw: Swing | None, feet: tuple[float, float], overhead: bool
) -> np.ndarray:
    """Where the ball was hit, in court metres: the racket wrist's ray at the player's depth."""
    default_z = 2.6 if overhead else 1.0
    guess = np.array([feet[0], feet[1], default_z])
    if vd.cal is None or not vd.cal.has_camera or sw is None:
        return guess
    k = sw.at(0.0, 0.04)
    if k is None:
        return guess
    wrists = [k[KP["l_wrist"]], k[KP["r_wrist"]]]
    wrists = [w for w in wrists if np.isfinite(w).all()]
    if not wrists:
        return guess
    w = min(wrists, key=lambda p: p[1]) if overhead else wrists[0]
    centre, d = vd.cal.image_rays(np.asarray([w]))
    if abs(d[0, 1]) < 1e-6:
        return guess
    s = (feet[1] - centre[1]) / d[0, 1]
    x = centre + s * d[0]
    if not (0.1 < x[2] < 3.4) or abs(x[0] - feet[0]) > 2.5:
        return guess
    # The racket extends the arm: the ball is a little further out and higher.
    return np.array([x[0], x[1], min(3.3, x[2] + (0.35 if overhead else 0.1))])


def flight_for(
    vd: VideoData, t_hit: float, t_next: float | None, p0: np.ndarray, toward_y: float
) -> ShotFlight | None:
    if vd.cal is None or not vd.cal.has_camera or not len(vd.ball_t):
        return None
    end = min(t_next - 0.03 if t_next is not None else t_hit + MAX_FLIGHT_S, t_hit + MAX_FLIGHT_S)
    m = (vd.ball_t > t_hit + 0.02) & (vd.ball_t < end)
    if m.sum() < 5:
        return None
    fit = fit_flight(
        vd.cal,
        p0,
        vd.ball_t[m] - t_hit,
        vd.ball_xy[m],
        toward_y=toward_y,
        duration=end - t_hit,
    )
    if fit is None or fit.rms_px > MAX_RMS_PX:
        return None
    if not (MIN_SPEED_KMH <= fit.speed_kmh <= MAX_SPEED_KMH):
        return None
    return fit


def placement(
    flight: ShotFlight, hitter_x: float, toward_y: float, stroke: str, doubles: bool
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if flight.bounce is None or not flight.bounce_seen:
        return out
    bx, by = flight.bounce
    out["bounce_x"], out["bounce_y"] = bx, by
    half_w = court_mod.HALF_DW if doubles else court_mod.HALF_SW
    other_half = by * toward_y > 0
    if stroke == "serve":
        inside = (
            other_half and abs(by) <= court_mod.SERVICE_FROM_NET and abs(bx) <= court_mod.HALF_SW
        )
    else:
        inside = other_half and abs(by) <= court_mod.HALF_L and abs(bx) <= half_w
    out["in_court"] = bool(inside)
    depth = abs(by)
    out["depth"] = (
        "short"
        if depth < court_mod.SERVICE_FROM_NET
        else "deep"
        if depth > court_mod.HALF_L - 3.0
        else "mid"
    )
    if abs(bx) < 1.4:
        out["direction"] = "middle"
    elif np.sign(bx) == np.sign(hitter_x) or abs(hitter_x) < 0.5:
        out["direction"] = "line" if abs(hitter_x) >= 0.5 else "middle"
    else:
        out["direction"] = "cross"
    return out


def run(ctx: SessionContext) -> None:
    identities = read_json(ctx.dir / "identities.json")
    doubles = len(identities.get("players", {})) >= 4
    with session_scope(ctx.data_root) as db:
        player_of = {
            sp.label: sp.player_id
            for sp in db.exec(
                select(SessionPlayer).where(SessionPlayer.session_id == ctx.session_id)
            )
        }
    data = [load(v, identities) for v in ctx.videos]

    # Racket hand per player label, voted over every swing of theirs.
    swings_by_label: dict[str, list[Swing]] = defaultdict(list)
    for vd in data:
        for h, track in enumerate(vd.hits["track"]):
            label = vd.labels.get(int(track))
            if label and h in vd.swings:
                swings_by_label[label].append(vd.swings[h])
    hands = {label: racket_hand(sws) for label, sws in swings_by_label.items()}

    shots: list[dict[str, Any]] = []
    for vd in data:
        times = vd.hits["t"]
        n = len(times)
        rally_start = 0
        for i in range(n):
            t = float(times[i])
            if i > 0 and t - float(times[i - 1]) > RALLY_GAP_S:
                rally_start = i
            first = i == rally_start
            t_next = (
                float(times[i + 1])
                if i + 1 < n and float(times[i + 1]) - t <= RALLY_GAP_S
                else None
            )
            track = int(vd.hits["track"][i])
            side = int(vd.hits["side"][i])
            label = vd.labels.get(track)
            feet = feet_at(vd, track, t)
            sw = vd.swings.get(i)
            hand = hands.get(label or "") or "right"
            call = classify(
                sw,
                hand=hand,
                right_dir=right_direction(vd, feet, side),
                first_in_rally=first,
                distance_from_net_m=abs(feet[1]) if feet is not None else None,
            )
            if call.stroke == "serve" and not first:
                # A serve always starts a rally (the hits before it were not part of it).
                rally_start = i
            toward_y = 1.0 if side < 0 else -1.0
            row: dict[str, Any] = {
                "video": vd.info.id,
                "t": t + vd.info.offset_s,
                "rally_key": (vd.info.id, rally_start),
                "label": label,
                "stroke": call.stroke,
                "spin": call.spin,
                "stroke_confidence": round(call.confidence, 3),
                "sources": list(vd.hits["sources"][i]),
                "quality": {"hit_score": round(float(vd.hits["score"][i]), 3)},
                "metrics": {},
            }
            if feet is not None:
                row["hit_x"], row["hit_y"] = feet
                overhead = call.stroke in ("serve", "overhead")
                p0 = contact_point(vd, sw, feet, overhead)
                row["contact_height_m"] = round(float(p0[2]), 2)
                flight = flight_for(vd, t, t_next, p0, toward_y)
                lo, hi = SPEED_RANGE_KMH.get(call.stroke, (MIN_SPEED_KMH, MAX_SPEED_KMH))
                if flight is not None and not lo <= flight.speed_kmh <= hi:
                    flight = None
                if flight is not None:
                    row["speed_kmh"] = round(flight.speed_kmh, 1)
                    row["avg_speed_kmh"] = (
                        round(flight.avg_speed_kmh, 1) if flight.avg_speed_kmh else None
                    )
                    row["net_clearance_m"] = (
                        round(flight.net_clearance_m, 2)
                        if flight.net_clearance_m is not None
                        else None
                    )
                    row["apex_m"] = round(flight.apex_m, 2)
                    row.update(placement(flight, feet[0], toward_y, call.stroke, doubles))
                    row["quality"]["flight_rms_px"] = round(flight.rms_px, 2)
                    row["quality"]["flight_points"] = flight.n_obs
                    row["crosses_net"] = flight.crosses_net
            if sw is not None:
                lw_speed = _racket_speed(sw)
                if lw_speed is not None:
                    row["metrics"]["racket_arm_speed"] = lw_speed
            shots.append(row)

    # Rallies, in time order across videos.
    shots.sort(key=lambda r: r["t"])
    rally_keys: list[tuple[int, int]] = []
    for r in shots:
        if r["rally_key"] not in rally_keys:
            rally_keys.append(r["rally_key"])
    with session_scope(ctx.data_root) as db:
        old = select(Shot.id).where(Shot.session_id == ctx.session_id)
        db.exec(delete(ShotMetric).where(col(ShotMetric.shot_id).in_(old)))
        db.exec(delete(Shot).where(col(Shot.session_id) == ctx.session_id))
        db.exec(delete(Rally).where(col(Rally.session_id) == ctx.session_id))
        rally_ids: dict[tuple[int, int], int] = {}
        for idx, key in enumerate(rally_keys):
            members = [r for r in shots if r["rally_key"] == key]
            rally = Rally(
                session_id=ctx.session_id,
                index=idx,
                start_s=members[0]["t"] - 0.5,
                end_s=members[-1]["t"] + 1.5,
                shot_count=len(members),
                server_id=player_of.get(members[0]["label"] or "")
                if members[0]["stroke"] == "serve"
                else None,
            )
            db.add(rally)
            db.flush()
            assert rally.id is not None
            rally_ids[key] = rally.id
        position: dict[tuple[int, int], int] = defaultdict(int)
        for r in shots:
            key = r["rally_key"]
            shot = Shot(
                session_id=ctx.session_id,
                video_id=r["video"],
                rally_id=rally_ids[key],
                player_id=player_of.get(r["label"] or ""),
                t=r["t"],
                index_in_rally=position[key],
                stroke=r["stroke"],
                spin=r["spin"],
                stroke_confidence=r["stroke_confidence"],
                speed_kmh=r.get("speed_kmh"),
                avg_speed_kmh=r.get("avg_speed_kmh"),
                net_clearance_m=r.get("net_clearance_m"),
                apex_m=r.get("apex_m"),
                contact_height_m=r.get("contact_height_m"),
                hit_x=r.get("hit_x"),
                hit_y=r.get("hit_y"),
                bounce_x=r.get("bounce_x"),
                bounce_y=r.get("bounce_y"),
                in_court=r.get("in_court"),
                depth=r.get("depth"),
                direction=r.get("direction"),
                sources=r["sources"],
                quality={**r["quality"], "crosses_net": r.get("crosses_net")},
            )
            position[key] += 1
            db.add(shot)
            db.flush()
            for name, value in r["metrics"].items():
                db.add(ShotMetric(shot_id=shot.id or 0, name=name, value=float(value)))
        for label, voted in hands.items():
            pid = player_of.get(label)
            player = db.get(Player, pid) if pid else None
            if player is not None and voted and player.handedness is None:
                player.handedness = voted
                db.add(player)
    ctx.log(
        "shots",
        shots=len(shots),
        rallies=len(rally_keys),
        with_flight=sum(1 for r in shots if "speed_kmh" in r),
        hands=hands,
    )


def _racket_speed(sw: Swing) -> float | None:
    """Peak racket-wrist speed into contact, in body heights per second."""
    from tennis.vision.strokes import wrist_peak_speeds

    h = sw.body_height()
    if not np.isfinite(h) or h <= 0:
        return None
    return round(max(wrist_peak_speeds(sw)) / h, 3)
