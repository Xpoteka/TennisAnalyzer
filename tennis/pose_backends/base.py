"""Pose backend interface (spec section 6.3).

A backend turns a batch of BGR images into the people found in each. Everything else in
the pipeline depends only on this interface, so the model (and its license) can be swapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]  # H x W x 3, BGR

# COCO-17 order, with the short names used for keypoints.parquet columns.
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
)
NUM_KEYPOINTS = len(KEYPOINT_NAMES)

# Limb segments for drawing, as index pairs into KEYPOINT_NAMES.
SKELETON: tuple[tuple[int, int], ...] = (
    (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 6), (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (0, 5), (0, 6),  # head to shoulders
)  # fmt: skip


@dataclass(frozen=True)
class PersonPose:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    bbox_conf: float
    keypoints: npt.NDArray[np.float32]  # (17, 3): x, y, conf in pixels, COCO-17 order

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class PoseBackend(Protocol):
    name: str

    def infer(self, frames: list[Image]) -> list[list[PersonPose]]:
        """People found in each frame, in the same order as ``frames``."""
        ...
