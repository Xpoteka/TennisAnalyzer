"""Session stage ``identities``: who the players are, as opposed to the tracks.

The tracker's tracks break whenever a player is lost for a moment, and players change ends.
Tracks are joined into players by clothing colour, under one hard rule: two tracks seen at
the same time are different people. Of the joined groups, the ones that hit the ball or
spend a good share of the session on court are the players; the rest (ball kids, people
walking past, a coach) are ignored.

This stage creates the session's players (A, B, ...). Matching them to profiles from other
sessions happens in the ``players`` stage; until it has run, each session gets its own.

Writes ``sessions/<id>/identities.json``: per video, which track belongs to which label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pyarrow.parquet as pq
from sqlmodel import select

from tennis.db import session_scope
from tennis.db.models import Player, SessionPlayer
from tennis.util.appearance import distance
from tennis.util.io import atomic_path, write_json
from tennis.util.video import grab_frame

if TYPE_CHECKING:
    from tennis.pipeline.session import SessionContext, VideoInfo

MERGE_MAX_DISTANCE = 0.38  # clothing-colour distance above which tracks are not joined
OVERLAP_S = 0.5  # tracks seen together for longer than this are different people
MIN_TRACK_SAMPLES = 5
MIN_HIT_SHARE = 0.05
MIN_TIME_SHARE = 0.15
MAX_PLAYERS = 4
LABELS = "ABCD"


@dataclass
class TrackInfo:
    video: int
    id: int
    start: float  # session seconds
    end: float
    samples: int
    appearance: np.ndarray
    hits: int = 0
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # court metres, NaN if unknown


@dataclass
class Group:
    tracks: list[TrackInfo]

    @property
    def appearance(self) -> np.ndarray:
        w = np.array([t.samples for t in self.tracks], np.float64)
        return np.asarray(np.average([t.appearance for t in self.tracks], axis=0, weights=w))

    @property
    def duration(self) -> float:
        return sum(t.end - t.start for t in self.tracks)

    @property
    def hits(self) -> int:
        return sum(t.hits for t in self.tracks)

    def overlaps(self, other: Group) -> bool:
        for a in self.tracks:
            for b in other.tracks:
                if a.video != b.video:
                    continue
                if min(a.end, b.end) - max(a.start, b.start) <= OVERLAP_S:
                    continue
                # Their spans overlap: were they really seen at the same moments?
                both = np.intersect1d(np.round(a.times, 1), np.round(b.times, 1))
                if len(both) * 0.1 > OVERLAP_S:
                    return True
        return False


def load_tracks(v: VideoInfo) -> list[TrackInfo]:
    d = pq.read_table(
        v.path("people.parquet"), columns=["t", "track_id", "appearance", "court_x", "court_y"]
    ).to_pydict()
    xy_all = np.array(
        [
            [np.nan if x is None else x, np.nan if y is None else y]
            for x, y in zip(d["court_x"], d["court_y"], strict=True)
        ],
        np.float64,
    ).reshape(-1, 2)
    hits = pq.read_table(v.path("hits.parquet"), columns=["track"]).column("track").to_pylist()
    ids = np.asarray(d["track_id"])
    times = np.asarray(d["t"], np.float64) + v.offset_s
    looks = np.asarray(d["appearance"], np.float64) if d["appearance"] else np.zeros((0, 1))
    out = []
    for tid in np.unique(ids):
        m = ids == tid
        if m.sum() < MIN_TRACK_SAMPLES:
            continue
        order = np.argsort(times[m])
        out.append(
            TrackInfo(
                video=v.id,
                id=int(tid),
                start=float(times[m].min()),
                end=float(times[m].max()),
                samples=int(m.sum()),
                appearance=np.median(looks[m], axis=0),
                hits=sum(1 for h in hits if h == tid),
                times=times[m][order],
                xy=xy_all[m][order],
            )
        )
    return out


SAME_PLACE_M = 1.5
OTHER_PLACE_M = 4.0
CROSS_VIEW_MIN_S = 3.0


def same_place(a: TrackInfo, b: TrackInfo) -> tuple[str, float] | None:
    """Two tracks from different videos, compared where both were seen at the same time.

    ("same", seconds) when they stood in the same spot of the court: the same person seen
    by two cameras. ("different", seconds) when they stood far apart. None otherwise.
    """
    if a.video == b.video:
        return None
    lo, hi = max(a.start, b.start), min(a.end, b.end)
    if hi - lo < CROSS_VIEW_MIN_S:
        return None
    m = (a.times >= lo) & (a.times <= hi) & np.isfinite(a.xy).all(axis=1)
    ok_b = np.isfinite(b.xy).all(axis=1)
    if m.sum() < 10 or ok_b.sum() < 10:
        return None
    bx = np.interp(a.times[m], b.times[ok_b], b.xy[ok_b, 0])
    by = np.interp(a.times[m], b.times[ok_b], b.xy[ok_b, 1])
    d = float(np.median(np.hypot(a.xy[m, 0] - bx, a.xy[m, 1] - by)))
    seconds = float(hi - lo)
    if d < SAME_PLACE_M:
        return "same", seconds
    if d > OTHER_PLACE_M:
        return "different", seconds
    return None


def cluster(tracks: list[TrackInfo]) -> list[Group]:
    """Greedy agglomeration: join the most similar pair that never appears together."""
    groups = [Group([t]) for t in tracks]
    while True:
        best: tuple[float, int, int] | None = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = distance(groups[i].appearance, groups[j].appearance)
                if d > MERGE_MAX_DISTANCE or (best is not None and d >= best[0]):
                    continue
                if groups[i].overlaps(groups[j]):
                    continue
                best = (d, i, j)
        if best is None:
            return groups
        _, i, j = best
        groups[i] = Group(groups[i].tracks + groups[j].tracks)
        del groups[j]


SIGNIFICANT_S = 8.0  # groups on court for less than this are not considered players


def concurrent_players(groups: list[Group]) -> int:
    """How many significant groups are usually seen at the same time (2 singles, 4 doubles)."""
    sig = [g for g in groups if g.duration >= SIGNIFICANT_S or g.hits >= 3]
    if not sig:
        return 0
    stamps: dict[tuple[int, float], int] = {}
    for g in sig:
        seen = set()
        for tr in g.tracks:
            for t in np.round(tr.times):
                seen.add((tr.video, float(t)))
        for key in seen:
            stamps[key] = stamps.get(key, 0) + 1
    counts = np.array(list(stamps.values()))
    return int(np.clip(np.percentile(counts, 90), 1, MAX_PLAYERS))


MIN_SINGLES_TRACK_S = 2.0


def seen_together(a: TrackInfo, b: TrackInfo) -> float:
    """Seconds during which both tracks were seen (same video, same samples)."""
    if a.video != b.video or min(a.end, b.end) <= max(a.start, b.start):
        return 0.0
    both = np.intersect1d(np.round(a.times, 1), np.round(b.times, 1))
    return len(both) * 0.1


def two_players(tracks: list[TrackInfo]) -> list[Group]:
    """Singles: split the tracks into two players.

    Two tracks seen at the same time are different people. With two players on court that
    relation chains through the whole session: near track 1 is not far track 2, which is not
    near track 3, so near tracks 1 and 3 are the same person. With several cameras, tracks
    from two videos that stand on the same spot of the court at the same time are the same
    person. The chain is taken along a maximum spanning tree of these relations (weighted by
    seconds), so a short spell with a third person (a ball kid) or a tracker mix-up cannot
    flip it. Parts that never meet are matched to each other by clothing colour.
    """
    tr = [t for t in tracks if t.end - t.start >= MIN_SINGLES_TRACK_S or t.hits >= 2]
    n = len(tr)
    # (weight, i, j, parity): parity 1 = different people, 0 = the same person.
    edges: list[tuple[float, int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            w = seen_together(tr[i], tr[j])
            if w > OVERLAP_S:
                edges.append((w, i, j, 1))
                continue
            relation = same_place(tr[i], tr[j])
            if relation is not None:
                edges.append((relation[1], i, j, 0 if relation[0] == "same" else 1))
    edges.sort(reverse=True)
    parent = list(range(n))
    parity = [0] * n  # colour relative to the parent

    def root(i: int) -> tuple[int, int]:
        p = 0
        while parent[i] != i:
            p ^= parity[i]
            i = parent[i]
        return i, p

    for _, i, j, rel in edges:
        (ri, pi), (rj, pj) = root(i), root(j)
        if ri != rj:
            parent[ri] = rj
            parity[ri] = pi ^ pj ^ rel
    groups_of: dict[int, list[int]] = {}
    colour = [0] * n
    for i in range(n):
        r, p = root(i)
        colour[i] = p
        groups_of.setdefault(r, []).append(i)
    parts = list(groups_of.values())
    parts.sort(key=lambda p: -sum(tr[i].end - tr[i].start for i in p))
    players: list[list[TrackInfo]] = [[], []]
    for part in parts:
        costs = []
        for flip in (0, 1):
            cost = 0.0
            for side in (0, 1):
                mine = [tr[i] for i in part if colour[i] ^ flip == side]
                if mine and players[side]:
                    cost += distance(Group(players[side]).appearance, Group(mine).appearance) * sum(
                        t.end - t.start for t in mine
                    )
            costs.append(cost)
        flip = int(np.argmin(costs))
        for i in part:
            players[colour[i] ^ flip].append(tr[i])
    return [Group(p) for p in players if p]


def pick_players(groups: list[Group], session_s: float) -> list[Group]:
    total_hits = sum(g.hits for g in groups)
    players = [
        g
        for g in groups
        if (total_hits and g.hits >= MIN_HIT_SHARE * total_hits)
        or g.duration >= MIN_TIME_SHARE * session_s
    ]
    players.sort(key=lambda g: (-g.hits, -g.duration))
    players = players[:MAX_PLAYERS]
    players.sort(key=lambda g: min(t.start for t in g.tracks))  # A is seen first
    return players


def save_thumbnail(ctx: SessionContext, label: str, group: Group) -> str | None:
    """A crop of the player from the moment their box was largest."""
    best: tuple[float, VideoInfo, float, list[float]] | None = None
    videos = {v.id: v for v in ctx.videos}
    for tr in group.tracks:
        v = videos[tr.video]
        d = pq.read_table(
            v.path("people.parquet"), columns=["t", "track_id", "x1", "y1", "x2", "y2", "conf"]
        ).to_pydict()
        for t, k, x1, y1, x2, y2, c in zip(*(d[c] for c in d), strict=True):
            if k != tr.id or c < 0.5:
                continue
            area = (x2 - x1) * (y2 - y1)
            if best is None or area > best[0]:
                best = (area, v, t, [x1, y1, x2, y2])
    if best is None:
        return None
    _, v, t, (x1, y1, x2, y2) = best
    start = float(v.metadata.get("video_start_s") or 0.0)
    frame = grab_frame(v.source, t - start)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    bh = y2 - y1
    side = max(x2 - x1, bh * 0.75)
    cx = (x1 + x2) / 2
    box = (
        int(max(0, cx - side / 2)),
        int(max(0, y1 - bh * 0.05)),
        int(min(w, cx + side / 2)),
        int(min(h, y1 + bh * 0.75)),
    )
    crop = frame[box[1] : box[3], box[0] : box[2]]
    if crop.size == 0:
        return None
    crop = np.asarray(
        cv2.resize(crop, (160, round(160 * crop.shape[0] / max(1, crop.shape[1])))), np.uint8
    )
    rel = f"sessions/{ctx.session_id}/players/{label}.jpg"
    path = ctx.data_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        cv2.imwrite(str(tmp), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return rel


def run(ctx: SessionContext) -> None:
    tracks = [t for v in ctx.videos for t in load_tracks(v)]
    session_s = max((t.end for t in tracks), default=0.0) - min(
        (t.start for t in tracks), default=0.0
    )
    groups = cluster(tracks)
    if concurrent_players(groups) <= 2:
        players = two_players(tracks)
        players.sort(key=lambda g: min(t.start for t in g.tracks))  # A is seen first
    else:
        players = pick_players(groups, max(session_s, 1.0))
    mapping: dict[str, dict[str, str]] = {}
    labels: dict[str, dict[str, Any]] = {}
    for label, g in zip(LABELS, players, strict=False):
        for t in g.tracks:
            mapping.setdefault(str(t.video), {})[str(t.id)] = label
        labels[label] = {
            "appearance": g.appearance.round(4).tolist(),
            "hits": g.hits,
            "on_court_s": round(g.duration, 1),
            "tracks": len(g.tracks),
        }
    write_json(ctx.dir / "identities.json", {"tracks": mapping, "players": labels})

    with session_scope(ctx.data_root) as db:
        existing = {
            sp.label: sp
            for sp in db.exec(
                select(SessionPlayer).where(SessionPlayer.session_id == ctx.session_id)
            )
        }
        for label, g in zip(LABELS, players, strict=False):
            thumb = save_thumbnail(ctx, label, g)
            sp = existing.pop(label, None)
            if sp is None:
                player = Player(name="")
                db.add(player)
                db.flush()
                player.name = f"Player {player.id}"
                sp = SessionPlayer(session_id=ctx.session_id, player_id=player.id or 0, label=label)
            sp.signature = {"appearance": labels[label]["appearance"]}
            sp.thumbnail = thumb
            db.add(sp)
            player_row = db.get(Player, sp.player_id)
            if player_row is not None and thumb and not player_row.thumbnail:
                player_row.thumbnail = thumb
                db.add(player_row)
        for sp in existing.values():  # a label that no longer exists
            db.delete(sp)
    ctx.log("players", count=len(players), groups=len(groups), tracks=len(tracks))
