from __future__ import annotations

import numpy as np
import pytest

from tennis.pose_backends.base import PersonPose
from tennis.util import appearance
from tennis.util.identity import A, B, WindowLooks, assign_identities, side_segments, summarize
from tennis.util.tracking import PlayerTracker

ORANGE = (0, 140, 255)  # BGR
BLUE = (200, 60, 20)


def _look(color: tuple[int, int, int], noise: int = 0) -> np.ndarray:
    rng = np.random.default_rng(noise)
    img = np.zeros((60, 40, 3), np.uint8)
    img[:] = color
    img = np.clip(img.astype(int) + rng.integers(-10, 10, img.shape), 0, 255).astype(np.uint8)
    return appearance.descriptor(img, (0, 0, 40, 60))


def test_descriptor_separates_colors() -> None:
    orange, orange2, blue = _look(ORANGE, 1), _look(ORANGE, 2), _look(BLUE, 3)
    assert orange.shape == (appearance.SIZE,)
    assert orange.sum() == pytest.approx(1.0)
    assert appearance.distance(orange, orange2) < 0.2
    assert appearance.distance(orange, blue) > 0.8
    empty = appearance.descriptor(np.zeros((10, 10, 3), np.uint8), (5, 5, 5, 5))
    assert empty.sum() == 0 and appearance.distance(empty, orange) == 1.0


def test_torso_box_prefers_keypoints() -> None:
    kp = np.zeros((17, 3), np.float32)
    for k, (x, y) in {5: (10, 20), 6: (30, 20), 11: (12, 50), 12: (28, 50)}.items():
        kp[k] = (x, y, 0.9)
    assert appearance.torso_box((0, 0, 40, 100), kp) == (10, 20, 30, 50)
    assert appearance.torso_box((0, 0, 40, 100), None) == (10, 20, 30, 55)


def _windows(pattern: str, missing_far: tuple[int, ...] = ()) -> list[WindowLooks]:
    """pattern[i] is who is near in window i: 'A' (orange) or 'B' (blue)."""
    looks = []
    for i, near in enumerate(pattern):
        n, f = (ORANGE, BLUE) if near == "A" else (BLUE, ORANGE)
        looks.append(WindowLooks(i, float(i * 10), _look(n, i),
                                 None if i in missing_far else _look(f, 100 + i)))  # fmt: skip
    return looks


def test_assignment_follows_end_changes() -> None:
    result = assign_identities(_windows("AAAABBBBBAAA", missing_far=(2, 6)))
    assert result.resolved
    assert [result.near_identity[i] for i in range(12)] == [A] * 4 + [B] * 5 + [A] * 3


def test_assignment_smooths_a_single_confused_window() -> None:
    looks = _windows("AAAAAAA")
    # Window 3: both descriptors look like a mix of the two players.
    mixed = (_look(ORANGE, 50) + _look(BLUE, 51)) / 2
    looks[3] = WindowLooks(3, 30.0, mixed, mixed)
    result = assign_identities(looks)
    assert set(result.near_identity.values()) == {A}


def test_assignment_without_far_player() -> None:
    looks = [WindowLooks(i, float(i), _look(ORANGE, i), None) for i in range(3)]
    result = assign_identities(looks)
    assert not result.resolved
    assert set(result.near_identity.values()) == {A}


def test_summarize_and_segments() -> None:
    rows = np.stack([np.zeros(appearance.SIZE, np.float32), _look(ORANGE)])
    assert summarize(rows) is not None
    assert summarize(np.zeros((2, appearance.SIZE), np.float32)) is None
    segs = side_segments({0: A, 1: A, 2: B}, {0: 0.0, 1: 5.0, 2: 9.0},
                         {0: 1.0, 1: 6.0, 2: 10.0}, me=A)  # fmt: skip
    assert segs == [
        {"start": 0.0, "end": 6.0, "side": "near", "windows": 2},
        {"start": 9.0, "end": 10.0, "side": "far", "windows": 1},
    ]


def _person(x1: float, y1: float, x2: float, y2: float) -> PersonPose:
    return PersonPose((x1, y1, x2, y2), 0.9, np.zeros((17, 3), np.float32))


def test_far_tracker_uses_the_far_half() -> None:
    near, far_small, far_big = (
        _person(0, 500, 100, 900),
        _person(10, 50, 20, 80),
        _person(200, 100, 260, 300),
    )
    tracker = PlayerTracker(1000, 0.4, 0.3, region="far")
    sel = tracker.update([near, far_small, far_big])
    assert sel.index == 2
    assert PlayerTracker(1000, 0.4, 0.3).update([near, far_big]).index == 0
    with pytest.raises(ValueError):
        PlayerTracker(1000, 0.4, 0.3, region="middle")


def test_short_side_stretches_are_merged() -> None:
    looks = _windows("AAAAAABBAAAAAABBBBBB")  # windows 10 s apart; a 20 s "B near" blip
    result = assign_identities(looks, min_side_duration_s=60.0)
    assert [result.near_identity[i] for i in range(20)] == [A] * 14 + [B] * 6
    kept = assign_identities(looks, min_side_duration_s=0.0, smooth=0)
    assert kept.near_identity[6] == B


def test_far_candidates_skip_the_near_player() -> None:
    from tennis.stages.pose import far_candidates

    tracker = PlayerTracker(1000, 0.4, 0.3, region="far")
    at_net = _person(400, 150, 500, 390)  # the near player, feet just above the line
    partner = _person(700, 100, 740, 200)
    assert far_candidates([at_net, partner], [], tracker) == [at_net, partner]
    assert far_candidates([at_net, partner], [], tracker, near_player=at_net) == [partner]
