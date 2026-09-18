"""Link per-frame person detections into tracks.

Detections are matched frame to frame with the Hungarian algorithm. The cost mixes where
the feet are (in court metres when the court is known, else in image heights), box overlap
and clothing colour. A track survives short gaps (the detector misses a frame, a player is
hidden by the net post). Tracks are cut when someone is lost for longer; joining the pieces
into the same player happens later, per session, with appearance and court side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from tennis.util.appearance import distance as appearance_distance

FloatArray = npt.NDArray[np.float64]

MAX_SPEED_M_S = 9.0  # faster than any tennis player; bounds how far a foot can move
MAX_SPEED_IMG_S = 1.2  # image heights per second, when the court is unknown
# Measurement noise on the foot point: far away, one pixel is several centimetres of court.
SLACK_M = 1.2
SLACK_IMG = 0.05
MAX_GAP_S = 1.0


@dataclass
class Detection:
    t: float
    foot: FloatArray  # court metres (x, y) or image position in image heights
    bbox: FloatArray  # x1, y1, x2, y2
    appearance: FloatArray
    track_id: int = -1


@dataclass
class _Track:
    id: int
    last: Detection
    velocity: FloatArray = field(default_factory=lambda: np.zeros(2))
    appearance: FloatArray | None = None


def _iou(a: FloatArray, b: FloatArray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    def __init__(self, *, in_metres: bool) -> None:
        self.max_speed = MAX_SPEED_M_S if in_metres else MAX_SPEED_IMG_S
        self.slack = SLACK_M if in_metres else SLACK_IMG
        self.tracks: list[_Track] = []
        self.next_id = 0

    def update(self, dets: list[Detection]) -> None:
        """Assign ``track_id`` to each detection of one frame (all with the same ``t``)."""
        if not dets:
            return
        t = dets[0].t
        self.tracks = [tr for tr in self.tracks if t - tr.last.t <= MAX_GAP_S]
        cost = np.full((len(self.tracks), len(dets)), 1e6)
        for i, tr in enumerate(self.tracks):
            dt = max(t - tr.last.t, 1e-3)
            predicted = tr.last.foot + tr.velocity * min(dt, 0.3)
            reach = self.max_speed * dt + self.slack
            for j, d in enumerate(dets):
                dist = float(np.linalg.norm(d.foot - predicted))
                if dist > reach:
                    continue
                look = (
                    appearance_distance(tr.appearance, d.appearance)
                    if tr.appearance is not None
                    else 0.5
                )
                cost[i, j] = dist / reach + 0.7 * look + 0.5 * (1 - _iou(tr.last.bbox, d.bbox))
        matched: set[int] = set()
        if self.tracks:
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols, strict=True):
                if cost[i, j] >= 1e6:
                    continue
                tr, d = self.tracks[i], dets[j]
                dt = max(t - tr.last.t, 1e-3)
                v = (d.foot - tr.last.foot) / dt
                tr.velocity = 0.6 * tr.velocity + 0.4 * v
                tr.appearance = (
                    d.appearance
                    if tr.appearance is None
                    else 0.9 * tr.appearance + 0.1 * d.appearance
                )
                tr.last = d
                d.track_id = tr.id
                matched.add(j)
        for j, d in enumerate(dets):
            if j not in matched:
                d.track_id = self.next_id
                self.tracks.append(_Track(self.next_id, d, appearance=d.appearance))
                self.next_id += 1
