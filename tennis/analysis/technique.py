"""Technique measurements per shot, and plain-language insights per player.

Measurements come from the full-rate pose (angles in the image plane, lengths in body
heights). A single camera sees the body at an angle, so the absolute numbers are
approximate; they are consistent for one player and camera, which is what comparisons
between strokes and sessions need.

Per shot:

- ``knee_bend_deg``: the smallest knee angle while preparing (180 = straight legs);
- ``elbow_contact_deg``: the racket arm's elbow angle at contact;
- ``shoulder_turn``: how much narrower the shoulders look at the turn than after contact
  (0 = no turn, towards 1 = fully side-on);
- ``racket_arm_speed``: peak wrist speed into contact, in body heights per second;
- ``split_step``: 1 when the player hopped (split step) as the opponent hit, 0 if not;
- ``recovery_m``: how far from the centre line the player is 1.2 s after their hit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from tennis.vision.strokes import KP, KP_MIN, Swing, wrist_peak_speeds

FloatArray = npt.NDArray[np.float64]

METRIC_LABELS: dict[str, tuple[str, str]] = {
    "knee_bend_deg": ("Knee angle at the load", "°"),
    "elbow_contact_deg": ("Elbow angle at contact", "°"),
    "shoulder_turn": ("Shoulder turn", ""),
    "racket_arm_speed": ("Racket arm speed", "heights/s"),
    "split_step": ("Split step before the opponent's shot", ""),
    "recovery_m": ("Distance from the centre after the shot", "m"),
}


def _angle(a: FloatArray, b: FloatArray, c: FloatArray) -> float:
    """Angle ABC in degrees."""
    ba, bc = a - b, c - b
    n = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if n <= 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(ba @ bc / n, -1.0, 1.0))))


def _frames(sw: Swing, lo: float, hi: float) -> FloatArray:
    return sw.kp[(sw.t_rel >= lo) & (sw.t_rel <= hi)]


def knee_bend(sw: Swing) -> float | None:
    angles = []
    for k in _frames(sw, -0.45, 0.0):
        for side in ("l", "r"):
            idx = [KP[f"{side}_hip"], KP[f"{side}_knee"], KP[f"{side}_ankle"]]
            if (k[idx, 2] >= KP_MIN).all():
                angles.append(_angle(k[idx[0], :2], k[idx[1], :2], k[idx[2], :2]))
    angles = [a for a in angles if np.isfinite(a)]
    if len(angles) < 4:
        return None
    return round(float(np.percentile(angles, 10)), 1)  # the deepest, ignoring one odd frame


def elbow_at_contact(sw: Swing) -> float | None:
    lw, rw = wrist_peak_speeds(sw)
    side = "r" if rw >= lw else "l"
    k = sw.at(0.0, 0.04)
    if k is None:
        return None
    pts = [k[KP[f"{side}_shoulder"]], k[KP[f"{side}_elbow"]], k[KP[f"{side}_wrist"]]]
    if any(np.isnan(p).any() for p in pts):
        return None
    a = _angle(*pts)
    return round(a, 1) if np.isfinite(a) else None


def shoulder_turn(sw: Swing) -> float | None:
    widths = []
    for t in np.arange(-0.6, 0.31, 0.05):
        k = sw.at(float(t), 0.03)
        if k is None:
            continue
        w = k[KP["r_shoulder"]] - k[KP["l_shoulder"]]
        widths.append((t, float(np.linalg.norm(w)) if np.isfinite(w).all() else np.nan))
    before = [w for t, w in widths if t < -0.05 and np.isfinite(w)]
    after = [w for t, w in widths if t > 0.1 and np.isfinite(w)]
    if len(before) < 2 or not after:
        return None
    ref = max(after)
    if ref <= 0:
        return None
    return round(float(np.clip(1 - min(before) / ref, 0.0, 1.0)), 3)


def split_step(ankles_y: FloatArray, t: FloatArray, height_px: float) -> float | None:
    """1 if both ankles leave the ground and land again around ``t = 0``, else 0.

    ``ankles_y``: the mean ankle height in the image (pixels, down is positive) per frame
    at times ``t`` relative to the opponent's hit.
    """
    m = (t >= -0.45) & (t <= 0.25)
    if m.sum() < 6 or height_px <= 0:
        return None
    y = ankles_y[m]
    base = (
        float(np.median(ankles_y[(t >= -0.8) & (t <= -0.45)]))
        if ((t >= -0.8) & (t <= -0.45)).any()
        else float(np.max(y))
    )
    hop = (base - float(np.min(y))) / height_px
    return 1.0 if hop > 0.025 else 0.0


def swing_metrics(sw: Swing) -> dict[str, float]:
    out: dict[str, float] = {}
    h = sw.body_height()
    for name, value in (
        ("knee_bend_deg", knee_bend(sw)),
        ("elbow_contact_deg", elbow_at_contact(sw)),
        ("shoulder_turn", shoulder_turn(sw)),
        (
            "racket_arm_speed",
            round(max(wrist_peak_speeds(sw)) / h, 3) if np.isfinite(h) and h > 0 else None,
        ),
    ):
        if value is not None and np.isfinite(value):
            out[name] = float(value)
    return out


# --- insights ----------------------------------------------------------------------------


@dataclass
class Insight:
    text: str
    level: str  # good | info | work


def _median(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if len(vals) < 5:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.median(vals))


def insights(
    by_stroke: dict[str, dict[str, Any]], metrics: dict[str, list[float]]
) -> list[Insight]:
    """Rules of thumb over a player's aggregated numbers (see ``player_profile``)."""
    out: list[Insight] = []
    fh, bh = by_stroke.get("forehand", {}), by_stroke.get("backhand", {})
    fs, bs = fh.get("speed_avg"), bh.get("speed_avg")
    if fs and bs:
        if fs > bs * 1.15:
            out.append(
                Insight(
                    f"The forehand is the weapon: {fs:.0f} km/h on average against {bs:.0f} km/h "
                    "on the backhand.",
                    "info",
                )
            )
        elif bs > fs * 1.1:
            out.append(
                Insight(
                    f"The backhand ({bs:.0f} km/h) is faster than the forehand ({fs:.0f} km/h): "
                    "the forehand has room to grow.",
                    "work",
                )
            )
        else:
            out.append(
                Insight(f"Forehand and backhand are balanced ({fs:.0f} and {bs:.0f} km/h).", "good")
            )
    for name, stroke in (("forehand", fh), ("backhand", bh)):
        rate = stroke.get("in_pct")
        if rate is not None and stroke.get("n_in", 0) >= 10:
            if rate < 0.6:
                out.append(
                    Insight(
                        f"Only {rate:.0%} of measured {name}s landed in: consistency first.", "work"
                    )
                )
            elif rate > 0.8:
                out.append(Insight(f"{rate:.0%} of measured {name}s landed in.", "good"))
        clear = stroke.get("net_clearance_avg")
        if clear is not None and stroke.get("n_clear", 0) >= 10:
            if clear < 0.4:
                out.append(
                    Insight(
                        f"The {name} passes the net only {clear:.2f} m high on average: little "
                        "margin for error.",
                        "work",
                    )
                )
            elif clear > 1.3:
                out.append(
                    Insight(
                        f"The {name} clears the net by {clear:.1f} m on average: safe, but gives "
                        "the opponent time.",
                        "info",
                    )
                )
    knee = _median(metrics.get("knee_bend_deg", []))
    if knee is not None:
        if knee > 155:
            out.append(
                Insight(
                    f"Legs stay fairly straight while loading (knee angle about {knee:.0f}°): "
                    "bending more adds power and balance.",
                    "work",
                )
            )
        elif knee < 135:
            out.append(
                Insight(f"Good leg bend while loading (knee angle about {knee:.0f}°).", "good")
            )
    turn = _median(metrics.get("shoulder_turn", []))
    if turn is not None:
        if turn < 0.25:
            out.append(
                Insight(
                    "Little shoulder turn before the stroke: turning the upper body earlier adds "
                    "pace without effort.",
                    "work",
                )
            )
        elif turn > 0.45:
            out.append(Insight("A full shoulder turn before the stroke.", "good"))
    split = [v for v in metrics.get("split_step", []) if v is not None]
    if len(split) >= 10:
        rate = float(np.mean(split))
        if rate < 0.5:
            out.append(
                Insight(
                    f"Split step before only {rate:.0%} of the opponent's shots: a small hop as "
                    "they hit makes the first step faster.",
                    "work",
                )
            )
        else:
            out.append(Insight(f"Split step before {rate:.0%} of the opponent's shots.", "good"))
    rec = _median(metrics.get("recovery_m", []))
    if rec is not None and rec > 1.8:
        out.append(
            Insight(
                f"After a shot the player stays {rec:.1f} m off the centre on average: recovering "
                "towards the middle covers the court better.",
                "work",
            )
        )
    serve = by_stroke.get("serve", {})
    if serve.get("speed_avg"):
        out.append(
            Insight(
                f"Serve: {serve['speed_avg']:.0f} km/h on average, {serve['speed_max']:.0f} at "
                "best.",
                "info",
            )
        )
    return out
