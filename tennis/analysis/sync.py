"""Line up two recordings of the same session by their sound.

Every ball impact is a sharp click that both microphones hear. The onset-strength envelopes
of the two soundtracks (200 values per second) are compressed, their slow trend removed,
and cross-correlated with a phase transform (GCC-PHAT), which sharpens the peak to a few
milliseconds regardless of how loud each microphone is. The offset is accepted only if its
peak clearly beats every other lag: two videos that never overlap have no such peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter1d

FloatArray = npt.NDArray[np.float64]

MIN_OVERLAP_S = 20.0
MIN_PEAK_RATIO = 1.6  # the best lag must beat the best lag elsewhere by this much


@dataclass
class SyncResult:
    offset_s: float  # add to a time in video B to get the time in video A
    confidence: float  # peak ratio; 0 when not found


def _prepare(env: FloatArray, rate: float) -> FloatArray:
    x = np.log1p(np.maximum(env, 0.0) * 10)
    x = x - uniform_filter1d(x, size=max(3, int(rate)))  # remove the trend over one second
    return np.asarray(np.clip(x, 0.0, None), np.float64)


def find_offset(
    env_a: FloatArray, env_b: FloatArray, rate: float, *, max_offset_s: float | None = None
) -> SyncResult:
    """The lag of B relative to A: B's time t is A's time t + offset."""
    if min(len(env_a), len(env_b)) < MIN_OVERLAP_S * rate:
        return SyncResult(0.0, 0.0)
    a, b = _prepare(env_a, rate), _prepare(env_b, rate)
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    fa, fb = np.fft.rfft(a, n), np.fft.rfft(b, n)
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-12)  # phase transform
    corr = np.fft.irfft(cross, n)
    # Lags from -(len(b)-1) to len(a)-1.
    lags = np.concatenate([np.arange(0, len(a)), np.arange(-(len(b) - 1), 0)])
    values = np.concatenate([corr[: len(a)], corr[n - (len(b) - 1) :]])
    # Require some overlap for a lag to count.
    overlap = np.minimum(len(a), lags + len(b)) - np.maximum(0, lags)
    values = np.where(overlap >= MIN_OVERLAP_S * rate, values, -np.inf)
    if max_offset_s is not None:
        values = np.where(np.abs(lags) <= max_offset_s * rate, values, -np.inf)
    best = int(np.argmax(values))
    peak = float(values[best])
    if not np.isfinite(peak) or peak <= 0:
        return SyncResult(0.0, 0.0)
    elsewhere = values.copy()
    guard = int(rate)  # one second around the peak
    lo, hi = max(0, best - guard), min(len(values), best + guard + 1)
    elsewhere[lo:hi] = -np.inf
    second = float(np.max(elsewhere)) if np.isfinite(elsewhere).any() else 0.0
    ratio = peak / second if second > 0 else np.inf
    if ratio < MIN_PEAK_RATIO:
        return SyncResult(0.0, float(ratio))
    # Refine to a fraction of a sample with a parabola through the peak and its neighbours.
    frac = 0.0
    if (
        0 < best < len(values) - 1
        and np.isfinite(values[best - 1])
        and np.isfinite(values[best + 1])
    ):
        y0, y1, y2 = values[best - 1], values[best], values[best + 1]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    return SyncResult(float((lags[best] + frac) / rate), float(min(ratio, 99.0)))
