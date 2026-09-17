"""Clothing-color descriptors to tell the two players apart."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from tennis.pose_backends.base import KEYPOINT_NAMES

HUE_BINS = 12
SAT_BINS = 4
SIZE = HUE_BINS * SAT_BINS + 2  # hue x saturation histogram, plus dark and light fractions
_KP = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
_TORSO = [_KP[n] for n in ("l_shoulder", "r_shoulder", "l_hip", "r_hip")]


def torso_box(
    bbox: tuple[float, float, float, float],
    keypoints: npt.NDArray[np.float32] | None,
    kp_conf_min: float = 0.3,
) -> tuple[int, int, int, int]:
    """The shirt area: shoulders to hips if they were found, else the upper body of the box."""
    x1, y1, x2, y2 = bbox
    if keypoints is not None:
        pts = keypoints[_TORSO]
        ok = pts[:, 2] >= kp_conf_min
        if ok.sum() >= 3:
            xs, ys = pts[ok, 0], pts[ok, 1]
            return (int(xs.min()), int(ys.min()), int(np.ceil(xs.max())), int(np.ceil(ys.max())))
    w, h = x2 - x1, y2 - y1
    return (int(x1 + 0.25 * w), int(y1 + 0.2 * h), int(x2 - 0.25 * w), int(y1 + 0.55 * h))


def descriptor(
    image: npt.NDArray[np.uint8], box: tuple[int, int, int, int]
) -> npt.NDArray[np.float32]:
    """Normalized color histogram of ``box`` in a BGR image; all zeros if the box is empty."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    out = np.zeros(SIZE, np.float32)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return out
    hsv = cv2.cvtColor(np.ascontiguousarray(image[y1:y2, x1:x2]), cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dark = val < 50
    light = (sat < 40) & ~dark
    colored = ~dark & ~light
    n = hue.size
    hist, _, _ = np.histogram2d(
        hue[colored].ravel(), sat[colored].ravel(),
        bins=(HUE_BINS, SAT_BINS), range=((0, 180), (40, 256)),
    )  # fmt: skip
    out[: HUE_BINS * SAT_BINS] = hist.ravel() / n
    out[-2] = dark.sum() / n
    out[-1] = light.sum() / n
    return out


def distance(a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]) -> float:
    """Hellinger distance between two descriptors (0 = identical, 1 = disjoint)."""
    pa = np.clip(np.asarray(a, np.float64), 0, None)
    pb = np.clip(np.asarray(b, np.float64), 0, None)
    sa, sb = pa.sum(), pb.sum()
    if sa <= 0 or sb <= 0:
        return 1.0
    bc = float(np.sum(np.sqrt(pa / sa * pb / sb)))
    return float(np.sqrt(max(0.0, 1.0 - bc)))
