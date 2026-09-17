"""Follow one player per court half through the frames of a window (spec section 6.3).

* First frame of a window: the largest person in the tracker's half. The near half is
  where the box bottom is below ``near_court_min_y`` of the image height; the far half is
  everything above it.
* Later frames: the person with the highest IoU to the previous selection.
* If nobody overlaps enough, fall back to the first-frame rule and flag a track reset.
* A frame with no usable detection keeps the previous box, so the track can pick the
  player up again after a missed frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tennis.pose_backends.base import PersonPose

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class Selection:
    index: int | None  # into the frame's person list; None when nobody qualifies
    track_reset: bool


class PlayerTracker:
    def __init__(
        self, image_height: int, near_court_min_y: float, iou_min: float, region: str = "near"
    ) -> None:
        if region not in ("near", "far"):
            raise ValueError(f"unknown region {region!r}")
        self.split = near_court_min_y * image_height
        self.region = region
        self.iou_min = iou_min
        self.previous: Box | None = None

    def in_region(self, box: Box) -> bool:
        return (box[3] >= self.split) == (self.region == "near")

    def reset(self) -> None:
        self.previous = None

    def _largest_in_region(self, people: Sequence[PersonPose]) -> int | None:
        best: int | None = None
        for i, p in enumerate(people):
            if not self.in_region(p.bbox):
                continue
            if best is None or p.area > people[best].area:
                best = i
        return best

    def update(self, people: Sequence[PersonPose]) -> Selection:
        if self.previous is None:
            index = self._largest_in_region(people)
            reset = False
        else:
            prev = self.previous
            scores = [iou(prev, p.bbox) for p in people]
            best = max(range(len(people)), key=scores.__getitem__, default=None)
            if best is not None and scores[best] >= self.iou_min:
                index, reset = best, False
            else:
                index = self._largest_in_region(people)
                reset = index is not None
        if index is not None:
            self.previous = people[index].bbox
        return Selection(index=index, track_reset=reset)


def crop_box(box: Box, pad: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Integer crop around ``box``, padded by ``pad`` of its size on each side."""
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(width, round(x2 + px)),
        min(height, round(y2 + py)),
    )
