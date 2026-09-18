"""What kind of shot a swing was, from the hitter's pose around the hit.

- **Serve / overhead:** a wrist clearly above the head around contact. It is a serve when it
  starts the rally or is hit from behind the baseline; otherwise a smash.
- **Forehand / backhand:** which way the hands travel across the body during the swing,
  measured along the player's left-to-right *as they face the net* (from the court
  geometry). From the racket-hand side across the body: forehand; the other way: backhand.
  The position at contact alone is a much weaker cue: it depends on the exact hit time.
- **Volley:** a ground stroke played close to the net (inside the service line).
- **Spin:** a wrist path rising into contact is topspin, a falling one is slice.

The racket hand is voted per player over their serves and smashes: it is the hand above the
head at contact.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tennis.pose_backends.base import KEYPOINT_NAMES

FloatArray = npt.NDArray[np.float64]
KP = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
KP_MIN = 0.3
VOLLEY_MAX_DEPTH_M = 6.0  # distance from the net inside which a stroke counts as a volley


@dataclass
class Swing:
    """The hitter's keypoints around one hit: t_rel (N,), kp (N, 17, 3) in pixels."""

    t_rel: FloatArray
    kp: FloatArray

    def at(self, t: float, window: float = 0.05) -> FloatArray | None:
        """Keypoints averaged over ``t ± window`` seconds, NaN where not seen."""
        m = np.abs(self.t_rel - t) <= window
        if not m.any():
            return None
        kp = self.kp[m]
        seen = kp[..., 2] >= KP_MIN
        xy = np.where(seen[..., None], kp[..., :2], np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.asarray(np.nanmean(xy, axis=0))

    def body_height(self) -> float:
        k = self.at(0.0, 0.2)
        if k is None:
            return float("nan")
        top = np.nanmin(k[[KP["nose"], KP["l_shoulder"], KP["r_shoulder"]], 1])
        bottom = np.nanmax(k[[KP["l_ankle"], KP["r_ankle"], KP["l_knee"], KP["r_knee"]], 1])
        return float(bottom - top) * 1.15


def wrist_peak_speeds(sw: Swing) -> tuple[float, float]:
    """Peak speed of the left and right wrist in the 0.35 s before contact (pixels/s)."""
    m = (sw.t_rel >= -0.35) & (sw.t_rel <= 0.05)
    out = []
    for name in ("l_wrist", "r_wrist"):
        k = sw.kp[m][:, KP[name]]
        t = sw.t_rel[m]
        ok = k[:, 2] >= KP_MIN
        if ok.sum() < 3:
            out.append(0.0)
            continue
        d = np.linalg.norm(np.diff(k[ok, :2], axis=0), axis=1) / np.maximum(np.diff(t[ok]), 1e-3)
        out.append(float(np.percentile(d, 90)))
    return out[0], out[1]


def overhead_hand(sw: Swing) -> str | None:
    """On a serve or smash, the racket hand is the one above the head at contact.

    The other arm tossed the ball and has dropped by then. This is far more telling than
    which wrist moves faster: at 30 frames a second both arms move about as fast.
    """
    for t in (0.0, -0.03, -0.07):
        k = sw.at(t, 0.02)
        if k is None:
            continue
        head = np.nanmin(k[[KP["nose"], KP["l_eye"], KP["r_eye"]], 1])
        feet = np.nanmax(k[[KP["l_ankle"], KP["r_ankle"]], 1])
        left, right = k[KP["l_wrist"], 1], k[KP["r_wrist"], 1]
        if not np.isfinite([head, feet, left, right]).all():
            continue
        height = feet - head
        if min(left, right) < head - 0.05 * height and abs(left - right) > 0.2 * height:
            return "left" if left < right else "right"
    return None


def racket_hand(swings: list[Swing]) -> str | None:
    """ "left" or "right" by vote over a player's overhead swings; None if too few."""
    votes = [h for h in (overhead_hand(sw) for sw in swings) if h is not None]
    if len(votes) < 3:
        return None
    left = votes.count("left")
    right = len(votes) - left
    if max(left, right) < 0.65 * len(votes):
        return None
    return "left" if left > right else "right"


@dataclass
class StrokeCall:
    stroke: (
        str  # serve | forehand | backhand | volley_forehand | volley_backhand | overhead | unknown
    )
    spin: str | None  # topspin | slice | flat
    confidence: float
    lateral: float  # racket wrist offset along the player's left-right, in body heights


OVERHEAD_WINDOW = (-0.3, 0.15)  # the hit time can be off by a frame or two
OVERHEAD_MARGIN = 0.15  # wrist above the head by this share of the body height
SWING_WINDOW = (-0.25, 0.25)


def is_overhead(sw: Swing) -> bool:
    """A wrist clearly above the head around contact: a serve or a smash."""
    h = sw.body_height()
    if not np.isfinite(h) or h <= 0:
        return False
    m = (sw.t_rel >= OVERHEAD_WINDOW[0]) & (sw.t_rel <= OVERHEAD_WINDOW[1])
    for k in sw.kp[m]:
        seen = k[:, 2] >= KP_MIN
        heads = [k[i, 1] for i in (KP["nose"], KP["l_eye"], KP["r_eye"]) if seen[i]]
        wrists = [k[i, 1] for i in (KP["l_wrist"], KP["r_wrist"]) if seen[i]]
        if heads and wrists and min(wrists) < min(heads) - OVERHEAD_MARGIN * h:
            return True
    return False


def swing_direction(sw: Swing, right_dir: FloatArray, hand: str) -> float | None:
    """How far the hands travel across the body during the swing, in body heights.

    Positive: from the racket-hand side across to the other side, as on a forehand.
    Negative: the other way, as on a backhand. Both wrists are averaged, which works for
    one- and two-handed strokes and does not depend on which wrist is the racket wrist.
    """
    h = sw.body_height()
    a, b = sw.at(SWING_WINDOW[0], 0.05), sw.at(SWING_WINDOW[1], 0.05)
    if a is None or b is None or not np.isfinite(h) or h <= 0:
        return None
    hips = [KP["l_hip"], KP["r_hip"]]
    wrists = [KP["l_wrist"], KP["r_wrist"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        before = np.nanmean(a[wrists], axis=0) - np.nanmean(a[hips], axis=0)
        after = np.nanmean(b[wrists], axis=0) - np.nanmean(b[hips], axis=0)
    v = float((before - after) @ right_dir) / h
    if not np.isfinite(v):
        return None
    return v if hand == "right" else -v


def classify(
    sw: Swing | None,
    *,
    hand: str,
    right_dir: FloatArray,
    first_in_rally: bool,
    distance_from_net_m: float | None,
) -> StrokeCall:
    """``right_dir``: unit image vector pointing to the player's right as they face the net."""
    if sw is None or sw.at(0.0, 0.1) is None:
        return StrokeCall("unknown", None, 0.0, 0.0)
    h = sw.body_height()
    if not np.isfinite(h) or h <= 0:
        return StrokeCall("unknown", None, 0.0, 0.0)
    if is_overhead(sw):
        behind = distance_from_net_m is not None and distance_from_net_m > 10.5
        return StrokeCall("serve" if first_in_rally or behind else "overhead", None, 0.8, 0.0)
    lw, rw = wrist_peak_speeds(sw)
    spin = _spin(sw, "r" if rw >= lw else "l", h)
    across = swing_direction(sw, right_dir, hand)
    if across is None:
        return StrokeCall("unknown", spin, 0.2, 0.0)
    base = "forehand" if across > 0 else "backhand"
    conf = float(np.clip(abs(across) / 0.3, 0.3, 1.0))
    if distance_from_net_m is not None and distance_from_net_m < VOLLEY_MAX_DEPTH_M:
        return StrokeCall(f"volley_{base}", None, conf, across)
    return StrokeCall(base, spin, conf, across)


def _spin(sw: Swing, side: str, h: float) -> str | None:
    """Wrist rising into contact: topspin; falling: slice (image y grows downwards)."""
    before = sw.at(-0.2, 0.05)
    at = sw.at(0.0, 0.03)
    if before is None or at is None:
        return None
    w0, w1 = before[KP[f"{side}_wrist"]], at[KP[f"{side}_wrist"]]
    if np.isnan(w0).any() or np.isnan(w1).any():
        return None
    rise = float(w0[1] - w1[1]) / h  # positive: the wrist went up
    if rise > 0.12:
        return "topspin"
    if rise < -0.08:
        return "slice"
    return "flat"
