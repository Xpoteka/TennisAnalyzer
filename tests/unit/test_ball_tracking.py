"""Ball candidates and trajectories, and person tracking, on synthetic data."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from tennis.vision.ball import DetectorParams, Linker, candidates, choose_ball, prepare
from tennis.vision.tracking import Detection, Tracker

W, H = 640, 360
BALL_BGR = (60, 230, 210)  # yellow-green


def frames_with_ball(
    n: int, seed: int = 0
) -> tuple[list[npt.NDArray[np.uint8]], npt.NDArray[np.float64]]:
    """A static noisy court with a ball flying a parabola (pixels), plus its true path."""
    rng = np.random.default_rng(seed)
    base = np.full((H, W, 3), (110, 90, 70), np.uint8)
    cv2.line(base, (0, 300), (W, 300), (235, 235, 235), 3)
    t = np.arange(n, dtype=np.float64)
    path = np.stack([40 + 14 * t, 80 + 9 * t - 0.25 * t**2 + 0.004 * t**3], axis=1)
    frames = []
    for x, y in path:
        img = base.copy()
        cv2.circle(img, (round(x), round(y)), 4, BALL_BGR, -1, cv2.LINE_AA)
        noise = rng.normal(0, 3, img.shape)
        frames.append(np.clip(img + noise, 0, 255).astype(np.uint8))
    return frames, path


def track(
    frames: list[npt.NDArray[np.uint8]], people: list[tuple[float, ...]] | None = None
) -> dict[int, tuple[float, float, int, float]]:
    prepared = [prepare(f) for f in frames]
    linker = Linker(max_gap=3)
    for i in range(1, len(frames) - 1):
        cands = candidates(
            prepared[i - 1][0], prepared[i][0], prepared[i + 1][0], prepared[i][1],
            people or [], DetectorParams(),  # type: ignore[arg-type]
        )  # fmt: skip
        linker.update(i, cands)
    return choose_ball(linker.close())


def test_follows_a_flying_ball() -> None:
    frames, path = frames_with_ball(40)
    ball = track(frames)
    inner = range(1, len(frames) - 1)
    found = [f for f in inner if f in ball]
    assert len(found) >= 0.9 * len(inner)
    err = [np.hypot(ball[f][0] - path[f, 0], ball[f][1] - path[f, 1]) for f in found]
    assert max(err) < 2.0
    assert len({ball[f][2] for f in found}) <= 2  # one trajectory, maybe split once


def test_static_scene_gives_no_ball() -> None:
    rng = np.random.default_rng(1)
    base = np.full((H, W, 3), (110, 90, 70), np.uint8)
    frames = [
        np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8) for _ in range(30)
    ]
    assert track(frames) == {}


def test_blob_inside_a_player_counts_less() -> None:
    frames, _ = frames_with_ball(20)
    g = [prepare(f) for f in frames]
    params = DetectorParams()
    free = candidates(g[4][0], g[5][0], g[6][0], g[5][1], [], params)  # type: ignore[arg-type]
    boxed = candidates(g[4][0], g[5][0], g[6][0], g[5][1], [(0, 0, W, H)], params)  # type: ignore[arg-type]
    assert free and boxed
    assert boxed[0].in_person and boxed[0].score < free[0].score


def _det(t: float, x: float, y: float, look: float) -> Detection:
    return Detection(
        t=t,
        foot=np.array([x, y]),
        bbox=np.array([x * 10, y * 10, x * 10 + 5, y * 10 + 20]),
        appearance=np.array([look, 1 - look, 0.0], np.float32),
    )


def test_tracker_keeps_two_players_apart() -> None:
    tracker = Tracker(in_metres=True)
    ids_a, ids_b = set(), set()
    for k in range(50):
        t = k * 0.1
        a = _det(t, -2 + 0.3 * np.sin(k / 5), -12, 0.9)
        b = _det(t, 1 + 0.4 * np.cos(k / 4), 12, 0.1)
        tracker.update([b, a] if k % 2 else [a, b])
        ids_a.add(a.track_id)
        ids_b.add(b.track_id)
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_tracker_bridges_a_short_gap_but_not_a_long_one() -> None:
    tracker = Tracker(in_metres=True)
    first = _det(0.0, 0, -10, 0.5)
    tracker.update([first])
    after_gap = _det(0.6, 0.5, -10, 0.5)
    tracker.update([after_gap])
    assert after_gap.track_id == first.track_id
    much_later = _det(3.0, 0.5, -10, 0.5)
    tracker.update([much_later])
    assert much_later.track_id != first.track_id
