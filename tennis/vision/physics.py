"""The ball's 3-D flight from one hit to the next, fitted to where the camera saw it.

The model is simple physics: gravity and air drag (quadratic in speed), then one bounce
that keeps the ball's horizontal direction, loses some horizontal speed and sends it up
with part of its vertical speed. Spin is left out: a topspin ball dips faster, so its
speed off the racket comes out a little low.

The unknowns are the ball's position and velocity at the hit. The position starts at the
contact point estimated from the hitter's pose and is only allowed to move a little; the
velocity is free. They are fitted by least squares on the pixel distance between the
projected flight and every ball detection between the two hits, with a robust loss so
that a few wrong detections do not matter.

From the fitted flight: speed off the racket, average speed to the bounce, height over the
net, highest point, and where it bounced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from tennis.vision.court import NET_HEIGHT_CENTER, NET_HEIGHT_POST, Calibration

FloatArray = npt.NDArray[np.float64]

G = 9.81
# 0.5 * air density * drag coefficient * cross-section / mass, for a tennis ball.
DRAG_K = 0.5 * 1.21 * 0.55 * np.pi * 0.0335**2 / 0.057
RESTITUTION = 0.75  # vertical speed kept by a bounce
BOUNCE_FRICTION = 0.7  # horizontal speed kept by a bounce
DT = 0.005


@dataclass
class Flight:
    t: FloatArray  # seconds after the hit
    pos: FloatArray  # (N, 3) metres
    vel: FloatArray  # (N, 3) metres per second
    bounce_index: int | None


def simulate(p0: FloatArray, v0: FloatArray, duration: float, *, bounces: bool = True) -> Flight:
    """Integrate the flight for ``duration`` seconds (semi-implicit Euler, 5 ms steps)."""
    n = max(2, int(np.ceil(duration / DT)) + 1)
    pos = np.empty((n, 3))
    vel = np.empty((n, 3))
    p = np.asarray(p0, np.float64).copy()
    v = np.asarray(v0, np.float64).copy()
    bounce: int | None = None
    for i in range(n):
        pos[i], vel[i] = p, v
        speed = float(np.sqrt(v @ v))
        a = -DRAG_K * speed * v
        a[2] -= G
        v = v + a * DT
        p = p + v * DT
        if bounces and bounce is None and p[2] < 0 and v[2] < 0 and i + 1 < n:
            p[2] = -p[2] * RESTITUTION
            v = np.array([v[0] * BOUNCE_FRICTION, v[1] * BOUNCE_FRICTION, -v[2] * RESTITUTION])
            bounce = i + 1
    return Flight(np.arange(n, dtype=np.float64) * DT, pos, vel, bounce)


def _at(flight: Flight, times: FloatArray) -> FloatArray:
    idx = np.clip(np.round(times / DT).astype(np.int64), 0, len(flight.t) - 1)
    return flight.pos[idx]


@dataclass
class ShotFlight:
    speed_kmh: float  # off the racket
    avg_speed_kmh: float | None  # hit to bounce
    net_clearance_m: float | None  # height over the net tape where it crossed (negative: into it)
    apex_m: float
    bounce: tuple[float, float] | None  # court metres
    bounce_seen: bool  # the ball was seen before and after the bounce, so it is measured
    crosses_net: bool
    rms_px: float
    n_obs: int
    p0: FloatArray
    v0: FloatArray


def net_height(x: float) -> float:
    return float(np.interp(abs(x), [0.0, 6.4], [NET_HEIGHT_CENTER, NET_HEIGHT_POST]))


def fit_flight(
    cal: Calibration,
    p0_guess: FloatArray,
    obs_t: FloatArray,
    obs_px: FloatArray,
    *,
    toward_y: float,
    duration: float | None = None,
    contact_slack_m: float = 0.6,
) -> ShotFlight | None:
    """Fit a flight to ball detections ``obs_px`` (N, 2) at ``obs_t`` seconds after the hit.

    ``toward_y`` is the sign of the direction of play (+1 towards the far end). ``duration``
    bounds the simulation (the time until the next hit). Returns None with too few points
    or when nothing fits.
    """
    if len(obs_t) < 4 or not cal.has_camera:
        return None
    obs_t = np.asarray(obs_t, np.float64)
    obs_px = np.asarray(obs_px, np.float64)
    horizon = float(duration if duration is not None else obs_t.max() + 0.2)
    p0_guess = np.asarray(p0_guess, np.float64)
    scale_px = max(cal.width, cal.height) / 1000  # residuals in about "pixels at 1000 px"

    def model(params: FloatArray) -> FloatArray:
        p0 = p0_guess + params[:3]
        f = simulate(p0, params[3:6], horizon)
        return cal.world_to_image(_at(f, obs_t))

    def residuals(params: FloatArray) -> FloatArray:
        px = model(params)
        r = (px - obs_px) / scale_px
        r = np.where(np.isfinite(r), r, 200.0).ravel()
        prior = params[:3] / contact_slack_m * 3.0  # keep the contact point near the guess
        return np.concatenate([r, prior])

    best = None
    # Start from a few speeds; the direction of play is known.
    for speed in (15.0, 25.0, 38.0):
        v_guess = np.array([0.0, toward_y * speed, 3.0])
        x0 = np.concatenate([np.zeros(3), v_guess])
        try:
            sol = least_squares(residuals, x0, loss="soft_l1", f_scale=4.0, max_nfev=200)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return None
    p0 = p0_guess + best.x[:3]
    v0 = best.x[3:6]
    flight = simulate(p0, v0, max(horizon, 0.05))
    px = cal.world_to_image(_at(flight, obs_t))
    err = np.linalg.norm(px - obs_px, axis=1)
    inliers = err[np.isfinite(err)]
    rms = float(np.sqrt(np.median(inliers**2))) if inliers.size else np.inf

    bounce = None
    avg = None
    bounce_seen = False
    if flight.bounce_index is not None:
        b = flight.pos[flight.bounce_index]
        bounce = (float(b[0]), float(b[1]))
        t_b = flight.t[flight.bounce_index]
        ok = err < 3 * max(rms, 2.0)
        bounce_seen = bool(
            np.sum(ok & (obs_t < t_b)) >= 3 and np.sum(ok & (obs_t > t_b + 0.02)) >= 2
        )
        dist = float(np.linalg.norm(b[:2] - p0[:2]))
        avg = dist / t_b * 3.6 if t_b > 0 else None
    before = flight.pos[: flight.bounce_index] if flight.bounce_index else flight.pos
    clearance = None
    crosses = False
    ys = before[:, 1]
    sign = np.sign(ys - 0.0)
    cross = np.nonzero(np.diff(sign) != 0)[0]
    if cross.size:
        i = int(cross[0])
        crosses = True
        frac = ys[i] / (ys[i] - ys[i + 1]) if ys[i] != ys[i + 1] else 0.0
        at = before[i] + (before[i + 1] - before[i]) * frac
        clearance = float(at[2] - net_height(float(at[0])))
    return ShotFlight(
        speed_kmh=float(np.linalg.norm(v0) * 3.6),
        avg_speed_kmh=avg,
        net_clearance_m=clearance,
        apex_m=float(before[:, 2].max()),
        bounce=bounce,
        bounce_seen=bounce_seen,
        crosses_net=crosses,
        rms_px=rms,
        n_obs=len(obs_t),
        p0=p0,
        v0=v0,
    )
