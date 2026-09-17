"""2-D geometry on keypoint arrays. Points are arrays whose last axis is (x, y)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def midpoint(a: npt.ArrayLike, b: npt.ArrayLike) -> FloatArray:
    return (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)) / 2.0


def distance(a: npt.ArrayLike, b: npt.ArrayLike) -> FloatArray:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.asarray(np.hypot(diff[..., 0], diff[..., 1]))


def angle_deg(a: npt.ArrayLike, b: npt.ArrayLike, c: npt.ArrayLike) -> FloatArray:
    """Angle ABC at vertex ``b``, in degrees (0..180). NaN if a side has zero length."""
    ba = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    bc = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    norms = np.hypot(ba[..., 0], ba[..., 1]) * np.hypot(bc[..., 0], bc[..., 1])
    dot = ba[..., 0] * bc[..., 0] + ba[..., 1] * bc[..., 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(norms > 0, dot / norms, np.nan)
    return np.asarray(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
