"""Find the moments the ball is hit, and on which side of the net.

Three cues point at a hit, each unreliable on its own:

- **Sound:** a sharp onset. Also produced by bounces, the next court, and footsteps.
- **The ball** changes direction sharply next to a player's racket hand. Bounces bend its
  path too, but away from the players' hands, and less sharply; the ball is often lost.
- **A wrist** reaches its peak speed at that very moment. Players move their arms all the
  time (running, bouncing the ball before a serve, practice swings), but a swing's peak
  is sharp and lands on the hit.

Every onset, sharp ball turn and wrist-speed peak is a candidate. Each candidate is scored
for each side of the net from the cues around it. Then the best sequence is chosen by
dynamic programming, under the rules of a rally: shots alternate sides, and they are at
least half a second apart. Bounces between two hits and noise are left out that way,
without a model trained on labelled matches.

Sides are "near" (-1, lower in the picture) and "far" (+1). With a known court they come
from the players' feet in court metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from tennis.pose_backends.base import KEYPOINT_NAMES

FloatArray = npt.NDArray[np.float64]
KP = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

MIN_GAP_S = 0.55  # two hits by alternate players are never closer than this
RALLY_BREAK_S = 4.0  # longer without a hit: the rally is over, the alternation restarts
MERGE_S = 0.08  # candidates this close are the same moment
THRESHOLD = 0.5  # minimum score for a hit
PEAK_WINDOW = (-0.12, 0.08)  # the wrist peaks at the hit (onsets can lag a frame or two)
KP_MIN = 0.25
NEAR_S = 0.1  # a track has to have a frame this close to a candidate to be its hitter


@dataclass
class Track:
    """One tracked person at the full frame rate (from the ``motion`` stage)."""

    id: int
    t: FloatArray
    kp: FloatArray  # (N, 17, 3) pixels
    height: FloatArray  # box height, pixels
    side: npt.NDArray[np.int8]  # -1 near, +1 far, per frame
    speed: FloatArray = field(default_factory=lambda: np.zeros(0))  # fastest wrist, heights/s
    typical: float = 0.0  # this player's usual wrist speed
    strong: float = 1.0  # a fast swing, for this player

    def __post_init__(self) -> None:
        self.speed = wrist_speed(self.t, self.kp, self.height)
        if len(self.speed):
            self.typical = float(np.percentile(self.speed, 50))
            self.strong = max(float(np.percentile(self.speed, 97)), self.typical + 1e-3)

    def index_near(self, t: float, max_dt: float = NEAR_S) -> int | None:
        if len(self.t) < 2:
            return None
        i = int(np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1))
        j = i if abs(self.t[i] - t) < abs(self.t[i - 1] - t) else i - 1
        return j if abs(self.t[j] - t) <= max_dt else None

    def peak(self, t: float, lo: float, hi: float) -> tuple[float, float] | None:
        """The fastest wrist speed within ``[t+lo, t+hi]`` and when it happened."""
        a = int(np.searchsorted(self.t, t + lo))
        b = int(np.searchsorted(self.t, t + hi, side="right"))
        if b - a < 2:
            return None
        k = a + int(np.argmax(self.speed[a:b]))
        return float(self.speed[k]), float(self.t[k] - t)

    def wrists(self, i: int) -> FloatArray:
        k = self.kp[i]
        out = [k[KP[n], :2] for n in ("l_wrist", "r_wrist") if k[KP[n], 2] >= KP_MIN]
        return np.array(out) if out else np.zeros((0, 2))


def wrist_speed(t: FloatArray, kp: FloatArray, height: FloatArray) -> FloatArray:
    """The faster wrist's speed per frame, in body heights per second (lightly smoothed)."""
    n = len(t)
    if n < 3:
        return np.zeros(n)
    h = np.maximum(height, 1.0)
    speeds = []
    for name in ("l_wrist", "r_wrist"):
        w = kp[:, KP[name], :2]
        ok = kp[:, KP[name], 2] >= KP_MIN
        dt = np.maximum(np.diff(t), 1e-3)
        d = np.linalg.norm(np.diff(w, axis=0), axis=1) / dt
        good = ok[1:] & ok[:-1] & (np.diff(t) < 0.1)
        d = np.where(good, d, 0.0) / h[1:]
        speeds.append(np.concatenate([[0.0], d]))
    s = np.max(speeds, axis=0)
    # A three-frame median removes single-frame keypoint jumps.
    padded = np.concatenate([[s[0]], s, [s[-1]]])
    return np.asarray(np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0))


@dataclass
class Ball:
    t: FloatArray
    xy: FloatArray  # (N, 2) pixels
    reliable: bool

    def near(self, t: float, max_dt: float = 0.05) -> FloatArray | None:
        if not self.reliable or len(self.t) < 2:
            return None
        i = int(np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1))
        j = i if abs(self.t[i] - t) < abs(self.t[i - 1] - t) else i - 1
        return self.xy[j] if abs(self.t[j] - t) <= max_dt else None

    def turn(self, t: float) -> float | None:
        """How sharply the ball's image path turns at ``t``: 0 (straight) to 1 (reverses)."""
        if not self.reliable:
            return None
        a = int(np.searchsorted(self.t, t - 0.15))
        m = int(np.searchsorted(self.t, t))
        b = int(np.searchsorted(self.t, t + 0.15))
        if m - a < 2 or b - m < 2:
            return None
        before = self.xy[m - 1] - self.xy[a]
        after = self.xy[b - 1] - self.xy[m]
        nb, na = np.linalg.norm(before), np.linalg.norm(after)
        if nb < 3 or na < 3:
            return None
        cos = float(before @ after / (nb * na))
        return (1 - cos) / 2

    def turn_events(self) -> list[float]:
        """Times the ball's path turns by more than 60 degrees."""
        if not self.reliable or len(self.t) < 8:
            return []
        out: list[float] = []
        for i in range(3, len(self.t) - 3):
            if self.t[i + 3] - self.t[i - 3] > 0.3:
                continue
            before = self.xy[i] - self.xy[i - 3]
            after = self.xy[i + 3] - self.xy[i]
            nb, na = np.linalg.norm(before), np.linalg.norm(after)
            if nb < 4 or na < 4:
                continue
            if before @ after / (nb * na) < 0.5 and not (out and self.t[i] - out[-1] < 0.1):
                out.append(float(self.t[i]))
        return out


@dataclass
class Candidate:
    t: float
    loudness: float = 0.0  # 0..1: the onset's loudness rank in the video, 0 without one
    sources: set[str] = field(default_factory=set)


@dataclass
class Option:
    """A candidate hit by a given side."""

    t: float
    side: int
    score: float
    track: int
    features: dict[str, float]
    sources: set[str]


@dataclass
class Hit:
    t: float
    side: int
    track: int
    score: float
    features: dict[str, float]
    sources: list[str]


def gather_candidates(
    onset_t: FloatArray, onset_db: FloatArray, ball: Ball, tracks: list[Track]
) -> list[Candidate]:
    cands: list[Candidate] = []
    if len(onset_t):
        ranks = np.argsort(np.argsort(onset_db)) / max(1, len(onset_db) - 1)
        for t, r in zip(onset_t, ranks, strict=True):
            cands.append(Candidate(float(t), loudness=float(r), sources={"audio"}))
    for t in ball.turn_events():
        cands.append(Candidate(t, sources={"ball"}))
    for tr in tracks:
        s = tr.speed
        if len(s) < 3:
            continue
        peaks = np.nonzero((s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:]) & (s[1:-1] > 0.6 * tr.strong))[
            0
        ]
        for i in peaks + 1:
            cands.append(Candidate(float(tr.t[i]), sources={"pose"}))
    cands.sort(key=lambda c: c.t)
    merged: list[Candidate] = []
    for c in cands:
        if merged and c.t - merged[-1].t <= MERGE_S:
            m = merged[-1]
            # The sound gives the most precise time; then the ball.
            sound = "audio" in c.sources and "audio" not in m.sources
            ball_only = "ball" in c.sources and not ({"audio", "ball"} & m.sources)
            if sound or ball_only:
                m.t = c.t
            m.loudness = max(m.loudness, c.loudness)
            m.sources |= c.sources
        else:
            merged.append(c)
    return merged


def features_for(cand: Candidate, tr: Track, i: int, ball: Ball) -> dict[str, float]:
    """The cues for "``tr`` hit the ball at ``cand.t``", each 0..1."""
    h = max(float(tr.height[i]), 1.0)
    swing = 0.0
    timing = 0.0
    peak = tr.peak(cand.t, *PEAK_WINDOW)
    if peak is not None:
        value, offset = peak
        swing = float(np.clip((value - tr.typical) / (tr.strong - tr.typical), 0.0, 1.0))
        # A swing that peaks well before or after the moment is not this hit.
        wider = tr.peak(cand.t, -0.4, 0.3)
        if wider is not None and wider[0] > value * 1.15:
            swing *= 0.5
        timing = float(np.clip(1 - abs(offset) / 0.1, 0.0, 1.0))
    near = 0.0
    ball_px = ball.near(cand.t)
    wrists = tr.wrists(i)
    if ball_px is not None and len(wrists):
        # The racket reaches about 0.45 body heights past the wrist.
        d = float(np.min(np.linalg.norm(wrists - ball_px, axis=1))) / h
        near = float(np.clip(1.0 - (d - 0.45) / 0.5, 0.0, 1.0))
    turn = ball.turn(cand.t)
    return {
        "loudness": cand.loudness,
        "sound": 1.0 if "audio" in cand.sources else 0.0,
        "swing": swing,
        "timing": timing,
        "near": near,
        "turn": turn if turn is not None else 0.0,
        "ball_seen": 1.0 if ball_px is not None else 0.0,
    }


def score(f: dict[str, float], ball_reliable: bool) -> float:
    """Combine the cues. The swing must be there; sound and ball confirm it."""
    swing = f["swing"] * (0.5 + 0.5 * f["timing"])
    evidence = 0.35 * f["sound"] * (0.6 + 0.4 * f["loudness"])
    if ball_reliable and f["ball_seen"]:
        evidence += 0.35 * f["near"] + 0.1 * f["turn"]
    elif ball_reliable:
        evidence += 0.12  # the ball is often lost around a hit: neutral
    else:
        evidence *= 1.6  # no ball at all: the sound carries more weight
    return 0.55 * swing + evidence


def score_options(cand: Candidate, ball: Ball, tracks: list[Track]) -> list[Option]:
    """Score the candidate for each side: the best-placed player there."""
    options: dict[int, Option] = {}
    for tr in tracks:
        i = tr.index_near(cand.t)
        if i is None:
            continue
        side = int(tr.side[i])
        f = features_for(cand, tr, i, ball)
        s = score(f, ball.reliable)
        if side not in options or s > options[side].score:
            options[side] = Option(cand.t, side, s, tr.id, f, set(cand.sources))
    return list(options.values())


def choose_hits(options: list[Option], threshold: float = THRESHOLD) -> list[Hit]:
    """The best sequence of hits under the rally rules (dynamic programming).

    Each option gains its score minus ``threshold``. A chain may continue from any earlier
    option more than ``RALLY_BREAK_S`` before (a new rally); within a rally, consecutive
    hits must come from alternate sides and be ``MIN_GAP_S`` apart. A weak option can end up
    in the chain when it keeps a rally consistent: that is how a quiet, half-hidden shot is
    still counted.
    """
    opts = sorted((o for o in options if o.score > threshold * 0.6), key=lambda o: o.t)
    n = len(opts)
    if n == 0:
        return []
    gain = np.array([o.score - threshold for o in opts])
    times = np.array([o.t for o in opts])
    best = np.zeros(n)
    prev = np.full(n, -1, dtype=np.int64)
    free_best, free_idx, k = 0.0, -1, 0  # best chain ending before the current rally window
    for j in range(n):
        while k < j and times[k] <= times[j] - RALLY_BREAK_S:
            if best[k] > free_best:
                free_best, free_idx = float(best[k]), k
            k += 1
        best[j] = free_best + gain[j]
        prev[j] = free_idx
        for i in range(k, j):
            dt = times[j] - times[i]
            if dt < MIN_GAP_S or opts[i].side == opts[j].side:
                continue
            if best[i] + gain[j] > best[j]:
                best[j] = best[i] + gain[j]
                prev[j] = i
    j = int(np.argmax(best))
    if best[j] <= 0:
        return []
    chain = []
    while j >= 0:
        chain.append(j)
        j = int(prev[j])
    chain.reverse()
    return [
        Hit(
            t=opts[c].t,
            side=opts[c].side,
            track=opts[c].track,
            score=opts[c].score,
            features=opts[c].features,
            sources=sorted(opts[c].sources),
        )
        for c in chain
    ]


def detect_hits(
    onset_t: FloatArray, onset_db: FloatArray, ball: Ball, tracks: list[Track]
) -> list[Hit]:
    cands = gather_candidates(onset_t, onset_db, ball, tracks)
    # Only the tracks on court at a candidate's time can have hit it. Asking every track of a
    # long video about every candidate made the time grow with the square of its length.
    tracks = [tr for tr in tracks if len(tr.t)]
    first = np.array([tr.t[0] for tr in tracks])
    last = np.array([tr.t[-1] for tr in tracks])
    options = []
    for c in cands:
        present = np.nonzero((first <= c.t + NEAR_S) & (last >= c.t - NEAR_S))[0]
        options += score_options(c, ball, [tracks[i] for i in present])
    return choose_hits(options)
