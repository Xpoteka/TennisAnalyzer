"""Session stage ``identities``: who the players are, as opposed to the tracks.

The tracker's tracks break whenever a player is lost for a moment, and players change ends.
Tracks are joined into players under one hard rule: two tracks seen at the same time are
different people. People off the court (on the bench, on the next court, walking past) are
not players and are left out first.

For singles, that rule alone chains through the session: the near track is not the far
track, which is not the next near track, so the near tracks are one person. For doubles,
tracks are joined by clothing colour instead, and the groups that hit the ball or spend a
good share of the session on court are the players.

This stage creates the session's players (A, B, ...). Matching them to profiles from other
sessions happens in the ``players`` stage; until it has run, each session gets its own.

Writes ``sessions/<id>/identities.json``: per video, which track belongs to which label.
"""

from __future__ import annotations

from collections import defaultdict
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
from tennis.vision import court as court_mod

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

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def side(self) -> int:
        """-1 near, +1 far, 0 unknown (no court)."""
        y = self.xy[:, 1]
        y = y[np.isfinite(y)]
        if len(y) == 0:
            return 0
        return 1 if float(np.median(y)) > 0 else -1

    def is_bystander(self) -> bool:
        x = self.xy[:, 0]
        x = x[np.isfinite(x)]
        return len(x) > 0 and float(np.median(np.abs(x))) > court_mod.HALF_PLAY_W


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
    hit_tracks = pq.read_table(v.path("hits.parquet"), columns=["track"]).column("track")
    hits_of: dict[int, int] = defaultdict(int)
    for k in hit_tracks.to_pylist():
        hits_of[int(k)] += 1
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
                hits=hits_of.get(int(tid), 0),
                times=times[m][order],
                xy=xy_all[m][order],
            )
        )
    return out


def seen_together(tracks: list[TrackInfo]) -> np.ndarray:
    """Seconds during which each pair of tracks was seen in the same frames (same video).

    Built frame by frame: a frame holds a handful of people, so this is linear in the number
    of samples rather than quadratic in the number of tracks.
    """
    n = len(tracks)
    together = np.zeros((n, n))
    by_video: dict[int, list[int]] = defaultdict(list)
    for i, tr in enumerate(tracks):
        by_video[tr.video].append(i)
    for members in by_video.values():
        at: dict[float, list[int]] = defaultdict(list)
        for i in members:
            for t in tracks[i].times:
                at[round(float(t), 3)].append(i)
        stamps = np.array(sorted(at))
        step = float(np.median(np.diff(stamps))) if len(stamps) > 1 else 0.1
        step = min(max(step, 0.01), 1.0)
        for present in at.values():
            for a in range(len(present)):
                for b in range(a + 1, len(present)):
                    together[present[a], present[b]] += step
                    together[present[b], present[a]] += step
    return together


def cluster(tracks: list[TrackInfo], together: np.ndarray) -> list[Group]:
    """Greedy agglomeration: join the most similar pair that never appears together."""
    n = len(tracks)
    if n == 0:
        return []
    looks = np.array([t.appearance for t in tracks], np.float64)
    weights = np.array([t.samples for t in tracks], np.float64)
    conflict = together > OVERLAP_S
    alive = np.ones(n, bool)
    members: list[list[int]] = [[i] for i in range(n)]

    def dist_row(i: int) -> np.ndarray:
        a = np.clip(looks[i], 0, None)
        b = np.clip(looks, 0, None)
        sa, sb = a.sum(), b.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            bc = np.sqrt(a / sa * b / sb[:, None]).sum(axis=1)
        d = np.sqrt(np.clip(1.0 - bc, 0.0, None))
        d[(sa <= 0) | (sb <= 0)] = 1.0
        return np.asarray(d)

    dist = np.stack([dist_row(i) for i in range(n)]) if n else np.zeros((0, 0))
    np.fill_diagonal(dist, np.inf)
    while True:
        cand = dist.copy()
        cand[conflict] = np.inf
        cand[~alive] = np.inf
        cand[:, ~alive] = np.inf
        k = int(np.argmin(cand))
        i, j = divmod(k, n)
        if not np.isfinite(cand[i, j]) or cand[i, j] > MERGE_MAX_DISTANCE:
            break
        members[i].extend(members[j])
        alive[j] = False
        looks[i] = np.average(looks[members[i]], axis=0, weights=weights[members[i]])
        conflict[i] |= conflict[j]
        conflict[:, i] |= conflict[:, j]
        row = dist_row(i)
        dist[i] = row
        dist[:, i] = row
        dist[i, i] = np.inf
    return [Group([tracks[i] for i in members[k]]) for k in range(n) if alive[k]]


MIN_SINGLES_TRACK_S = 2.0


def concurrent_players(tracks: list[TrackInfo]) -> int:
    """How many people are usually on court at the same time (2 singles, 4 doubles).

    Counted per analysed frame, so a track ending as its replacement starts is not two
    people. The 90th percentile ignores the occasional ball retrieved by someone else.
    """
    counts: list[int] = []
    by_video: dict[int, list[TrackInfo]] = defaultdict(list)
    for tr in tracks:
        if tr.duration >= MIN_SINGLES_TRACK_S or tr.hits >= 2:
            by_video[tr.video].append(tr)
    for members in by_video.values():
        at: dict[float, int] = defaultdict(int)
        for tr in members:
            for t in tr.times:
                at[round(float(t), 3)] += 1
        counts.extend(at.values())
    if not counts:
        return 0
    return int(np.clip(np.percentile(counts, 90), 1, MAX_PLAYERS))


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


def two_players(tracks: list[TrackInfo], together: np.ndarray | None = None) -> list[Group]:
    """Singles: split the tracks into two players.

    Two tracks seen at the same time are different people. With two players on court that
    relation chains through the whole session: near track 1 is not far track 2, which is not
    near track 3, so near tracks 1 and 3 are the same person. With several cameras, tracks
    from two videos that stand on the same spot of the court at the same time are the same
    person. The chain is taken along a maximum spanning tree of these relations (weighted by
    seconds), so a short spell with a third person (a ball kid) or a tracker mix-up cannot
    flip it. Parts that never meet are matched to each other by clothing colour.

    Tracks too short to take part are then given to the player who was on their side of the
    court at the time (:func:`assign_by_side`).
    """
    if together is None:
        together = seen_together(tracks)
    keep = [i for i, t in enumerate(tracks) if t.duration >= MIN_SINGLES_TRACK_S or t.hits >= 2]
    tr = [tracks[i] for i in keep]
    n = len(tr)
    # (weight, i, j, parity): parity 1 = different people, 0 = the same person.
    edges: list[tuple[float, int, int, int]] = []
    sub = together[np.ix_(keep, keep)]
    for i in range(n):
        for j in range(i + 1, n):
            w = float(sub[i, j])
            if w > OVERLAP_S:
                edges.append((w, i, j, 1))
                continue
            if tr[i].video == tr[j].video:
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
    groups = [Group(p) for p in players if p]
    rest = [tracks[i] for i in range(len(tracks)) if i not in set(keep)]
    assign_by_side(groups, rest)
    return groups


SIDE_WINDOW_S = 90.0  # players keep their end of the court for at least a couple of games


def assign_by_side(players: list[Group], rest: list[TrackInfo]) -> None:
    """Give each leftover track to the player who was on its side of the court then.

    In singles, a player keeps one end between changeovers, so the side of the court at a
    given moment identifies the player. A track without a court position stays unassigned.
    """
    if len(players) != 2:
        return
    timelines: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for g in players:
        stamps: list[float] = []
        sides: list[int] = []
        videos: list[int] = []
        for tr in g.tracks:
            s = tr.side
            if s == 0:
                continue
            sample = tr.times[::5]
            stamps.extend(sample.tolist())
            sides.extend([s] * len(sample))
            videos.extend([tr.video] * len(sample))
        timelines.append((np.array(stamps), np.array(sides), np.array(videos)))
    for tr in rest:
        s = tr.side
        if s == 0:
            continue
        mid = (tr.start + tr.end) / 2
        votes = []
        for stamps_a, sides_a, videos_a in timelines:
            m = (np.abs(stamps_a - mid) <= SIDE_WINDOW_S) & (videos_a == tr.video)
            votes.append(float(np.mean(sides_a[m] == s)) if m.any() else 0.0)
        best = int(np.argmax(votes))
        if votes[best] >= 0.6 and votes[best] > votes[1 - best]:
            players[best].tracks.append(tr)


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


def find_players(tracks: list[TrackInfo]) -> tuple[list[Group], int]:
    """The session's players from every track, and how many people were usually on court."""
    playing = [t for t in tracks if not t.is_bystander()]
    concurrent = concurrent_players(playing)
    if concurrent <= 2:
        players = two_players(playing)
        players.sort(key=lambda g: min(t.start for t in g.tracks))  # A is seen first
        return players, concurrent
    session_s = max((t.end for t in playing), default=0.0) - min(
        (t.start for t in playing), default=0.0
    )
    groups = cluster(playing, seen_together(playing))
    return pick_players(groups, max(session_s, 1.0)), concurrent


def save_thumbnail(ctx: SessionContext, label: str, group: Group) -> str | None:
    """A crop of the player from the moment their box was largest."""
    best: tuple[float, VideoInfo, float, list[float]] | None = None
    videos = {v.id: v for v in ctx.videos}
    wanted = {(tr.video, tr.id) for tr in group.tracks}
    for v in videos.values():
        d = pq.read_table(
            v.path("people.parquet"), columns=["t", "track_id", "x1", "y1", "x2", "y2", "conf"]
        ).to_pydict()
        for t, k, x1, y1, x2, y2, c in zip(*(d[c] for c in d), strict=True):
            if (v.id, int(k)) not in wanted or c < 0.5:
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
    players, concurrent = find_players(tracks)
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
    write_json(
        ctx.dir / "identities.json",
        {"tracks": mapping, "players": labels, "concurrent": concurrent},
    )

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
    ctx.log(
        "players",
        count=len(players),
        concurrent=concurrent,
        tracks=len(tracks),
        hits={label: info["hits"] for label, info in labels.items()},
    )
