from __future__ import annotations

import numpy as np
import pytest

from tennis.pose_backends.base import PersonPose
from tennis.util.frames import Window, group_windows, merge_windows
from tennis.util.tracking import PlayerTracker, crop_box, iou

H = 1000


def person(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> PersonPose:
    return PersonPose((x1, y1, x2, y2), conf, np.zeros((17, 3), np.float32))


NEAR = person(400, 500, 500, 800)  # bottom at 0.8 H
FAR = person(450, 100, 470, 150)  # bottom at 0.15 H: the partner
BIG_FAR = person(0, 0, 300, 350)  # large but not in the near court


def test_iou() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_first_frame_picks_largest_near_court_person() -> None:
    t = PlayerTracker(H, 0.4, 0.3)
    sel = t.update([FAR, BIG_FAR, NEAR])
    assert sel.index == 2 and not sel.track_reset


def test_nobody_in_near_court() -> None:
    t = PlayerTracker(H, 0.4, 0.3)
    assert t.update([FAR, BIG_FAR]).index is None
    assert t.update([]).index is None


def test_follows_by_iou_even_when_another_person_is_larger() -> None:
    t = PlayerTracker(H, 0.4, 0.3)
    t.update([NEAR])
    moved = person(420, 500, 520, 800)
    bigger = person(700, 400, 900, 900)
    sel = t.update([bigger, moved])
    assert sel.index == 1 and not sel.track_reset


def test_falls_back_and_flags_reset() -> None:
    t = PlayerTracker(H, 0.4, 0.3)
    t.update([NEAR])
    jumped = person(100, 500, 200, 800)
    sel = t.update([FAR, jumped])
    assert sel.index == 1 and sel.track_reset


def test_missed_frame_keeps_previous_box() -> None:
    t = PlayerTracker(H, 0.4, 0.3)
    t.update([NEAR])
    assert t.update([]).index is None
    sel = t.update([FAR, NEAR])
    assert sel.index == 1 and not sel.track_reset


def test_reset_starts_over() -> None:
    t = PlayerTracker(H, 0.4, 0.3)
    t.update([NEAR])
    t.reset()
    sel = t.update([person(100, 500, 200, 800)])
    assert sel.index == 0 and not sel.track_reset


def test_crop_box_is_padded_and_clipped() -> None:
    assert crop_box((100, 100, 200, 300), 0.2, 1000, 1000) == (80, 60, 220, 340)
    assert crop_box((0, 0, 100, 100), 0.2, 110, 110) == (0, 0, 110, 110)


def test_merge_windows() -> None:
    windows = merge_windows([5.0, 1.0, 5.5, 20.0], 1.0, 0.5, 0.0, 20.2)
    assert [(w.start, w.end) for w in windows] == [(0.0, 1.5), (4.0, 6.0), (19.0, 20.2)]
    assert [w.id for w in windows] == [0, 1, 2]
    assert merge_windows([], 1.0, 0.5, 0.0, 10.0) == []


def test_group_windows() -> None:
    ws = [Window(0, 0, 1), Window(1, 3, 4), Window(2, 10, 11)]
    assert [[w.id for w in g] for g in group_windows(ws, 3.0)] == [[0, 1], [2]]
    assert [[w.id for w in g] for g in group_windows(ws, 0.0)] == [[0], [1], [2]]
