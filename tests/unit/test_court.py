"""Court calibration on synthetic images drawn by a known camera."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from tennis.vision import court
from tennis.vision.court import CORNERS, COURT_LINES, Calibration, with_camera

W, H = 1280, 720


def look_at(eye: npt.ArrayLike, target: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """World-to-camera rotation for a camera at ``eye`` looking at ``target`` (z up)."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return np.stack([right, down, f])  # rows: camera x (right), y (down), z (forward)


def camera(
    eye: tuple[float, float, float] = (0.5, -17.0, 5.0),
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    focal: float = 850.0,
    k1: float = 0.0,
) -> Calibration:
    R = look_at(eye, target)
    t = -R @ np.asarray(eye, float)
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1.0]])
    Hm = K @ np.stack([R[:, 0], R[:, 1], t], axis=1)
    cal = Calibration(H=Hm / Hm[2, 2], k1=k1, width=W, height=H, quality=1.0)
    return with_camera(cal)


def render(cal: Calibration, seed: int = 0) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (70, 110, 60)  # green surround
    court_px = cal.court_to_image(np.array([[-8, 16], [8, 16], [8, -16], [-8, -16]]))
    cv2.fillPoly(img, [court_px.astype(np.int32)], (120, 90, 70))  # blue-grey court
    for x1, y1, x2, y2 in COURT_LINES.values():
        t = np.linspace(0, 1, 200)
        p = cal.court_to_image(np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], 1))
        cv2.polylines(img, [np.round(p).astype(np.int32)], False, (235, 235, 235), 3, cv2.LINE_AA)
    net = cal.world_to_image(court.net_samples(0.05))
    cv2.polylines(img, [np.round(net).astype(np.int32)], False, (230, 230, 230), 3, cv2.LINE_AA)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    ("eye", "k1"),
    [
        ((0.5, -17.0, 5.0), 0.0),  # behind the baseline, raised
        ((-2.0, -20.0, 7.0), 0.0),  # off-centre, higher
        ((0.5, -16.5, 4.6), -0.16),  # a wide lens that bends the lines
    ],
)
def test_detects_court_and_camera(eye: tuple[float, float, float], k1: float) -> None:
    truth = camera(eye=eye, k1=k1)
    found = court.detect(render(truth))
    assert found is not None
    assert found.quality > 0.8
    # Every court corner that is in view lands within a few pixels.
    true_px = truth.court_to_image(CORNERS)
    in_view = (
        (true_px[:, 0] >= 0) & (true_px[:, 0] < W) & (true_px[:, 1] >= 0) & (true_px[:, 1] < H)
    )
    err = np.linalg.norm(found.court_to_image(CORNERS) - true_px, axis=1)[in_view]
    assert err.max() < 6.0, err
    # The camera is found to within 4 % of its distance, which keeps heights right.
    assert found.has_camera
    pos = found.camera_position
    assert pos is not None
    assert np.linalg.norm(pos - np.array(eye)) < 0.04 * np.linalg.norm(eye), pos
    # A point one metre above the service line projects where it should.
    probe = np.array([[1.0, -6.4, 1.0]])
    assert np.linalg.norm(found.world_to_image(probe) - truth.world_to_image(probe)) < 6.0


def test_round_trip_ground_points() -> None:
    cal = camera(k1=-0.1)
    pts = np.array([[0.0, 0.0], [3.0, -8.0], [-4.1, 11.0]])
    back = cal.image_to_court(cal.court_to_image(pts))
    np.testing.assert_allclose(back, pts, atol=1e-4)


def test_world_to_image_matches_ground_mapping() -> None:
    cal = camera()
    pts = np.array([[1.0, 2.0], [-3.0, -9.0]])
    a = cal.court_to_image(pts)
    b = cal.world_to_image(np.c_[pts, np.zeros(2)])
    np.testing.assert_allclose(a, b, atol=0.5)


def test_rays_pass_through_3d_points() -> None:
    cal = camera()
    X = np.array([[0.0, 5.0, 1.0], [2.0, -10.0, 0.3]])
    centre, d = cal.image_rays(cal.world_to_image(X))
    for x, di in zip(X, d, strict=True):
        v = x - centre
        closest = centre + di * (v @ di)
        assert np.linalg.norm(closest - x) < 0.05


def test_empty_image_has_no_court() -> None:
    img = np.full((H, W, 3), (70, 110, 60), np.uint8)
    found = court.detect(img)
    assert found is None or found.quality < 0.6


def test_json_round_trip() -> None:
    cal = camera(k1=-0.1)
    back = Calibration.from_json(cal.to_json())
    np.testing.assert_allclose(back.court_to_image(CORNERS), cal.court_to_image(CORNERS))
    assert back.has_camera
