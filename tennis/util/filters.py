"""Smoothing filters for keypoint trajectories (spec section 6.4, step 4).

Both work on a (T, K) array of K coordinate series sampled at timestamps ``t`` and leave
NaN where the input is NaN. A NaN gap restarts the filter.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

FloatArray = npt.NDArray[np.float64]


def _alpha(dt: float, cutoff: FloatArray | float) -> FloatArray:
    tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff, dtype=np.float64))
    return np.asarray(1.0 / (1.0 + tau / dt))


def _one_euro_pass(
    x: FloatArray, t: FloatArray, min_cutoff: float, beta: float, d_cutoff: float
) -> FloatArray:
    n, k = x.shape
    out = np.full((n, k), np.nan)
    xh = np.full(k, np.nan)
    dxh = np.zeros(k)
    for i in range(n):
        row = x[i]
        valid = np.isfinite(row)
        fresh = valid & ~np.isfinite(xh)
        xh[fresh] = row[fresh]
        dxh[fresh] = 0.0
        cont = valid & ~fresh
        if i > 0 and cont.any():
            dt = t[i] - t[i - 1]
            if dt > 0:
                dx = (row[cont] - xh[cont]) / dt
                a_d = _alpha(dt, d_cutoff)
                dxh[cont] = a_d * dx + (1.0 - a_d) * dxh[cont]
                a = _alpha(dt, min_cutoff + beta * np.abs(dxh[cont]))
                xh[cont] = a * row[cont] + (1.0 - a) * xh[cont]
        out[i, valid] = xh[valid]
        xh[~valid] = np.nan  # a gap restarts the filter
    return out


def one_euro(
    x: npt.ArrayLike,
    t: npt.ArrayLike,
    min_cutoff: float = 1.0,
    beta: float = 0.05,
    d_cutoff: float = 1.0,
    zero_phase: bool = False,
) -> FloatArray:
    """One Euro filter (Casiez et al., 2012) using real timestamps.

    ``beta`` multiplies the speed in the units of ``x`` per second. With
    ``zero_phase=True`` the result is the mean of a forward and a time-reversed pass, which
    cancels the filter's lag (useful offline, where timing metrics matter).
    """
    arr = np.asarray(x, dtype=np.float64)
    ts = np.asarray(t, dtype=np.float64)
    squeeze = arr.ndim == 1
    data = arr[:, None] if squeeze else arr
    forward = _one_euro_pass(data, ts, min_cutoff, beta, d_cutoff)
    if zero_phase:
        backward = _one_euro_pass(data[::-1], -ts[::-1], min_cutoff, beta, d_cutoff)[::-1]
        forward = (forward + backward) / 2.0
    return forward[:, 0] if squeeze else forward


def _segments(valid: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], valid.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def savgol(x: npt.ArrayLike, window: int, order: int) -> FloatArray:
    """Savitzky-Golay filter per finite segment; segments shorter than ``window`` are kept.

    Assumes roughly even frame spacing, which holds for constant-frame-rate footage.
    """
    arr = np.asarray(x, dtype=np.float64)
    squeeze = arr.ndim == 1
    data = arr[:, None] if squeeze else arr
    out = data.copy()
    for col in range(data.shape[1]):
        for a, b in _segments(np.isfinite(data[:, col])):
            if b - a >= window:
                out[a:b, col] = savgol_filter(data[a:b, col], window, order)
    return out[:, 0] if squeeze else out


def fill_gaps(x: npt.ArrayLike, t: npt.ArrayLike, max_gap: int) -> FloatArray:
    """Linearly interpolate interior NaN runs of at most ``max_gap`` samples (by timestamp).

    Gaps touching the start or end, and longer gaps, stay NaN.
    """
    arr = np.asarray(x, dtype=np.float64)
    ts = np.asarray(t, dtype=np.float64)
    squeeze = arr.ndim == 1
    data = arr[:, None] if squeeze else arr
    out = data.copy()
    if max_gap <= 0:
        return out[:, 0] if squeeze else out
    for col in range(data.shape[1]):
        column = data[:, col]
        valid = np.isfinite(column)
        for a, b in _segments(~valid):
            if a == 0 or b == len(column) or b - a > max_gap:
                continue
            out[a:b, col] = np.interp(ts[a:b], [ts[a - 1], ts[b]], [column[a - 1], column[b]])
    return out[:, 0] if squeeze else out
