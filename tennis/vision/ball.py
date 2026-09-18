"""Find the ball in every frame and link the detections into trajectories.

**Candidates.** A tennis ball is small, fast and yellow-green. A pixel that differs from both
the previous and the next frame belongs to something moving *now* (three-frame
differencing, which leaves no ghost where the ball was a frame ago). Connected blobs of
such pixels whose size fits a ball are candidates, scored by their colour, shape and size.
Motion blur turns a fast ball into a streak, so elongated blobs are allowed.

**Trajectories.** Candidates are linked frame to frame into tracklets. Each one predicts the
next position from its velocity. A tracklet that barely moves (a flickering light), or
that stays inside a player (arms, racket, shoes), is dropped. Hits and bounces break a
path into pieces; the shot stage joins them using the audio onsets and the players.

Positions are in source-video pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Gray = npt.NDArray[np.uint8]

# OpenCV hue runs 0..180; a tennis ball is about 25..45 (yellow to yellow-green).
BALL_HUE = (22, 48)
MAX_CANDIDATES = 24


@dataclass(frozen=True)
class Candidate:
    x: float
    y: float
    area: float
    score: float  # 0..1: how ball-like
    in_person: bool


@dataclass
class DetectorParams:
    min_area: float = 2.0
    max_area: float = 900.0
    diff_threshold: int = 14
    max_elongation: float = 5.0


def prepare(frame: npt.NDArray[np.uint8]) -> tuple[Gray, npt.NDArray[np.uint8]]:
    """Blurred grey image for differencing, and the HSV image for colour."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    return np.asarray(gray, np.uint8), np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), np.uint8)


def candidates(
    prev: Gray,
    cur: Gray,
    nxt: Gray,
    hsv: npt.NDArray[np.uint8],
    people: list[tuple[float, float, float, float]],
    params: DetectorParams,
) -> list[Candidate]:
    d1 = cv2.absdiff(cur, prev)
    d2 = cv2.absdiff(cur, nxt)
    moving = cv2.min(d1, d2)
    # Brightness changes of the whole frame (clouds, flicker) shift the threshold.
    thr = max(params.diff_threshold, int(np.percentile(moving[::4, ::4], 99.5)) // 2)
    mask = cv2.morphologyEx(
        (moving > thr).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: list[Candidate] = []
    boxes = np.array(people, np.float64).reshape(-1, 4)
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if not (params.min_area <= area <= params.max_area):
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        elong = max(w, h) / max(1, min(w, h))
        if elong > params.max_elongation:
            continue
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        blob = labels[y0 : y0 + h, x0 : x0 + w] == i
        pix = hsv[y0 : y0 + h, x0 : x0 + w][blob]
        hue, sat, val = pix[:, 0], pix[:, 1], pix[:, 2]
        yellow = float(
            np.mean((hue >= BALL_HUE[0]) & (hue <= BALL_HUE[1]) & (sat > 50) & (val > 90))
        )
        bright = float(np.mean(val > 120))
        fill = area / float(w * h)
        cx, cy = float(cents[i, 0]), float(cents[i, 1])
        inside = bool(
            len(boxes)
            and np.any(
                (cx >= boxes[:, 0])
                & (cx <= boxes[:, 2])
                & (cy >= boxes[:, 1])
                & (cy <= boxes[:, 3])
            )
        )
        score = 0.45 * yellow + 0.2 * bright + 0.15 * fill + 0.2 * (1.0 / elong)
        if inside:
            score *= 0.6
        out.append(Candidate(cx, cy, area, score, inside))
    out.sort(key=lambda c: -c.score)
    return out[:MAX_CANDIDATES]


@dataclass
class Tracklet:
    id: int
    frames: list[int] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    in_person: list[bool] = field(default_factory=list)
    misses: int = 0

    def add(self, f: int, c: Candidate) -> None:
        self.frames.append(f)
        self.xs.append(c.x)
        self.ys.append(c.y)
        self.scores.append(c.score)
        self.in_person.append(c.in_person)
        self.misses = 0

    def predict(self, f: int) -> tuple[float, float, float]:
        """Predicted position at frame ``f`` and a search radius."""
        if len(self.frames) == 1:
            return self.xs[-1], self.ys[-1], 60.0
        df = self.frames[-1] - self.frames[-2]
        vx = (self.xs[-1] - self.xs[-2]) / df
        vy = (self.ys[-1] - self.ys[-2]) / df
        gap = f - self.frames[-1]
        speed = float(np.hypot(vx, vy))
        return self.xs[-1] + vx * gap, self.ys[-1] + vy * gap, 8.0 + 0.6 * speed * gap

    def usable(self) -> bool:
        if len(self.frames) < 4:
            return False
        steps = np.hypot(np.diff(self.xs), np.diff(self.ys)) / np.diff(self.frames)
        if float(np.median(steps)) < 1.0:
            return False  # something that flickers in place
        return float(np.mean(self.in_person)) < 0.7 and float(np.mean(self.scores)) > 0.15


class Linker:
    """Online linking of per-frame candidates into tracklets."""

    def __init__(self, max_gap: int) -> None:
        self.max_gap = max_gap
        self.active: list[Tracklet] = []
        self.finished: list[Tracklet] = []
        self.next_id = 0

    def update(self, f: int, cands: list[Candidate]) -> None:
        free = list(range(len(cands)))
        # Longer tracklets choose first: they predict better.
        for tr in sorted(self.active, key=lambda t: -len(t.frames)):
            px, py, radius = tr.predict(f)
            best, best_d = None, radius
            for j in free:
                d = float(np.hypot(cands[j].x - px, cands[j].y - py))
                if d < best_d:
                    best, best_d = j, d
            if best is not None:
                tr.add(f, cands[best])
                free.remove(best)
            else:
                tr.misses += 1
        keep = []
        for tr in self.active:
            if tr.misses > self.max_gap:
                self.finished.append(tr)
            else:
                keep.append(tr)
        self.active = keep
        for j in free[:8]:
            tr = Tracklet(self.next_id)
            self.next_id += 1
            tr.add(f, cands[j])
            self.active.append(tr)

    def close(self) -> list[Tracklet]:
        self.finished.extend(self.active)
        self.active = []
        return [t for t in self.finished if t.usable()]


def choose_ball(tracklets: list[Tracklet]) -> dict[int, tuple[float, float, int, float]]:
    """One ball position per frame: where tracklets overlap in time, the stronger one wins."""
    best: dict[int, tuple[float, float, int, float]] = {}
    for tr in tracklets:
        strength = len(tr.frames) * float(np.mean(tr.scores))
        for f, x, y in zip(tr.frames, tr.xs, tr.ys, strict=True):
            if f not in best or best[f][3] < strength:
                best[f] = (x, y, tr.id, strength)
    return best
