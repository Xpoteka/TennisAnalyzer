"""Ball flight fitting on simulated shots seen by a known camera."""

from __future__ import annotations

import numpy as np
import pytest

from tennis.vision.physics import fit_flight, simulate
from tests.unit.test_court import camera


def test_simulated_ball_falls_and_bounces_once() -> None:
    f = simulate(np.array([0.0, -11.0, 1.0]), np.array([0.0, 25.0, 3.0]), 1.5)
    assert f.bounce_index is not None
    assert f.pos[:, 2].min() > -0.05
    # Drag slows the ball down in flight.
    assert np.linalg.norm(f.vel[f.bounce_index - 1]) < 25.0


@pytest.mark.parametrize(("speed", "vz"), [(20.0, 4.0), (30.0, 2.0), (40.0, 0.5)])
def test_fit_recovers_speed_and_bounce(speed: float, vz: float) -> None:
    cal = camera(eye=(0.5, -17.0, 5.0))
    rng = np.random.default_rng(int(speed))
    p0 = np.array([1.0, -11.5, 1.0])
    v0 = np.array([-1.5, speed, vz])
    truth = simulate(p0, v0, 1.6)
    t = np.arange(2, int(1.2 / 0.0333)) * 0.0333
    idx = np.round(t / 0.005).astype(int)
    px = cal.world_to_image(truth.pos[idx]) + rng.normal(0, 1.5, (len(t), 2))
    # Two wrong detections, as the ball tracker sometimes produces.
    px[5] += 80
    px[11] -= 60
    fit = fit_flight(cal, p0 + np.array([0.2, 0.2, 0.15]), t, px, toward_y=1.0, duration=1.3)
    assert fit is not None
    assert fit.speed_kmh == pytest.approx(np.linalg.norm(v0) * 3.6, rel=0.08)
    assert truth.bounce_index is not None
    tb = truth.pos[truth.bounce_index]
    if fit.bounce is not None and truth.t[truth.bounce_index] < 1.2:
        assert np.hypot(fit.bounce[0] - tb[0], fit.bounce[1] - tb[1]) < 1.0
    assert fit.crosses_net
    assert fit.rms_px < 4.0
