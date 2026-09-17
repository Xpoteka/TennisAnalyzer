"""Keypoint cleaning steps (spec section 6.4): confidence mask and left/right swap fix."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tennis.pose_backends.base import KEYPOINT_NAMES

KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
PAIRED_GROUPS: tuple[tuple[int, int], ...] = tuple(
    (KP[f"l_{part}"], KP[f"r_{part}"])
    for part in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
)


def mask_low_confidence(
    xy: npt.NDArray[np.float64], conf: npt.NDArray[np.float64], minimum: float
) -> npt.NDArray[np.float64]:
    """(T, 17, 2) positions with every keypoint below ``minimum`` confidence set to NaN."""
    out = xy.astype(np.float64, copy=True)
    out[~(conf >= minimum)] = np.nan
    return out


def fix_swaps(
    xy: npt.NDArray[np.float64], improvement_ratio: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Undo left/right label swaps, group by group, frame by frame.

    For each paired group (shoulders, elbows, ...) the summed displacement from the
    previous (already corrected) frame is compared with and without swapping the pair.
    The pair is swapped when that cuts the displacement by more than
    ``improvement_ratio``. Frames where any of the four points is missing are left alone.
    Returns the corrected positions and a (T, 6) mask of the pairs swapped in each frame,
    in ``PAIRED_GROUPS`` order.
    """
    out = xy.astype(np.float64, copy=True)
    swaps = np.zeros((out.shape[0], len(PAIRED_GROUPS)), dtype=bool)
    for i in range(1, out.shape[0]):
        for g, (left, right) in enumerate(PAIRED_GROUPS):
            lp, rp = out[i - 1, left], out[i - 1, right]
            lc, rc = out[i, left], out[i, right]
            if not (np.isfinite(lp).all() and np.isfinite(rp).all()
                    and np.isfinite(lc).all() and np.isfinite(rc).all()):  # fmt: skip
                continue
            keep = np.hypot(*(lc - lp)) + np.hypot(*(rc - rp))
            swapped = np.hypot(*(rc - lp)) + np.hypot(*(lc - rp))
            if swapped < (1.0 - improvement_ratio) * keep:
                out[i, left], out[i, right] = rc.copy(), lc.copy()
                swaps[i, g] = True
    return out, swaps


def apply_swaps(
    values: npt.NDArray[np.float64], swaps: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Apply the swaps found by ``fix_swaps`` to a (T, 17, ...) array such as confidences."""
    out = values.copy()
    for g, (left, right) in enumerate(PAIRED_GROUPS):
        rows = swaps[:, g]
        out[rows, left], out[rows, right] = values[rows, right], values[rows, left]
    return out
