from __future__ import annotations

import numpy as np
import pytest

from tennis.util.cleaning import KP, PAIRED_GROUPS, apply_swaps, fix_swaps, mask_low_confidence
from tennis.util.filters import fill_gaps, one_euro, savgol
from tennis.util.geometry import angle_deg, distance, midpoint

# --- geometry ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "c", "expected"),
    [
        ((1, 0), (0, 0), (0, 1), 90.0),
        ((1, 0), (0, 0), (-1, 0), 180.0),
        ((1, 0), (0, 0), (1, 0), 0.0),
        ((1, 0), (0, 0), (1, 1), 45.0),
        ((0, 0), (1, 0), (0.5, np.sqrt(3) / 2), 60.0),  # equilateral triangle
        ((3, 0), (0, 0), (0, 4), 90.0),  # 3-4-5 triangle
    ],
)
def test_angle_known_triangles(a: tuple, b: tuple, c: tuple, expected: float) -> None:
    assert angle_deg(a, b, c) == pytest.approx(expected, abs=1e-9)


def test_angle_vectorized_and_degenerate() -> None:
    a = np.array([[1.0, 0.0], [np.nan, 0.0], [0.0, 0.0]])
    b = np.zeros((3, 2))
    c = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    out = angle_deg(a, b, c)
    assert out[0] == pytest.approx(90.0)
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_midpoint_and_distance() -> None:
    assert midpoint([0, 0], [2, 4]).tolist() == [1.0, 2.0]
    assert distance([0, 0], [3, 4]) == pytest.approx(5.0)


# --- One Euro --------------------------------------------------------------------------------


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _jitter(x: np.ndarray) -> float:
    return float(np.std(np.diff(x, n=2)))


def test_one_euro_reduces_noise_on_sine() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(0, 20, 1 / 120)
    clean = np.sin(2 * np.pi * 0.1 * t)  # well below the cutoff, so barely attenuated
    noisy = clean + rng.normal(0, 0.05, t.size)
    # Causal: lags behind the sine, but the frame-to-frame noise is gone.
    causal = one_euro(noisy, t, min_cutoff=1.0, beta=0.05)
    assert _jitter(causal) < 0.1 * _jitter(noisy)
    # Zero-phase (the pipeline default): close to the clean signal.
    symmetric = one_euro(noisy, t, min_cutoff=1.0, beta=0.05, zero_phase=True)
    assert _rmse(symmetric[60:-60], clean[60:-60]) < 0.5 * _rmse(noisy, clean)


def test_one_euro_handles_irregular_timestamps() -> None:
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(1 / 200, 1 / 30, 2000))
    clean = np.sin(2 * np.pi * 0.1 * t)
    noisy = clean + rng.normal(0, 0.05, t.size)
    out = one_euro(noisy, t, min_cutoff=1.0, beta=0.05, zero_phase=True)
    assert np.isfinite(out).all()
    assert _rmse(out[50:-50], clean[50:-50]) < 0.6 * _rmse(noisy, clean)


def test_one_euro_zero_phase_removes_lag() -> None:
    t = np.arange(0, 4, 1 / 60)
    ramp = np.where(t < 2, 0.0, 1.0)  # a step at t = 2
    causal = one_euro(ramp, t, min_cutoff=2.0, beta=0.0)
    symmetric = one_euro(ramp, t, min_cutoff=2.0, beta=0.0, zero_phase=True)
    half = np.flatnonzero(t >= 2)[0]
    assert causal[half] < 0.25  # the causal filter lags behind the step
    assert symmetric[half] == pytest.approx(0.5, abs=0.15)  # centred on it


def test_one_euro_restarts_after_nan_and_keeps_shape() -> None:
    t = np.arange(10) / 30
    x = np.array([[0.0, 5.0]] * 10)
    x[4, 0] = np.nan
    x[5:, 0] = 10.0
    out = one_euro(x, t)
    assert np.isnan(out[4, 0])
    assert out[5, 0] == 10.0  # restarted: no memory of the old level
    assert np.allclose(out[:, 1], 5.0)
    assert out.shape == x.shape


def test_one_euro_constant_input_is_unchanged() -> None:
    t = np.arange(20) / 30
    assert np.allclose(one_euro(np.full(20, 3.0), t, zero_phase=True), 3.0)


def test_savgol_per_segment() -> None:
    x = np.concatenate([np.linspace(0, 1, 20), [np.nan], np.linspace(0, 1, 5)])
    out = savgol(x, 9, 3)
    assert np.allclose(out[:20], x[:20])  # a polynomial is preserved
    assert np.isnan(out[20])
    assert np.array_equal(out[21:], x[21:])  # too short to filter


# --- gap filling ------------------------------------------------------------------------------


def test_fill_interior_gap_by_time() -> None:
    t = np.array([0.0, 1.0, 1.5, 4.0])
    x = np.array([0.0, np.nan, np.nan, 8.0])
    assert fill_gaps(x, t, 5).tolist() == [0.0, 2.0, 3.0, 8.0]


def test_gap_at_start_and_end_stay_nan() -> None:
    t = np.arange(6.0)
    x = np.array([np.nan, 1.0, 2.0, 3.0, np.nan, np.nan])
    out = fill_gaps(x, t, 5)
    assert np.isnan(out[0]) and np.isnan(out[4:]).all()
    assert out[1:4].tolist() == [1.0, 2.0, 3.0]


def test_gap_longer_than_limit_stays_nan() -> None:
    t = np.arange(9.0)
    x = np.array([0.0] + [np.nan] * 6 + [7.0, 8.0])
    assert np.isnan(fill_gaps(x, t, 5)[1:7]).all()
    filled = fill_gaps(x, t, 6)
    assert filled[1:7].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_fill_gaps_2d_and_disabled() -> None:
    t = np.arange(3.0)
    x = np.array([[0.0, 0.0], [np.nan, 1.0], [2.0, 2.0]])
    assert fill_gaps(x, t, 1)[1].tolist() == [1.0, 1.0]
    assert np.isnan(fill_gaps(x, t, 0)[1, 0])


# --- confidence mask and swap fix ------------------------------------------------------------


def _walker(n: int = 30) -> np.ndarray:
    """A person moving right; left joints at x - 20, right joints at x + 20."""
    xy = np.zeros((n, 17, 2))
    for i in range(n):
        base = 100.0 + 3.0 * i
        for k in range(17):
            xy[i, k] = (base, 50.0 + k * 10.0)
        for left, right in PAIRED_GROUPS:
            xy[i, left, 0] = base - 20
            xy[i, right, 0] = base + 20
    return xy


def test_swap_fix_repairs_injected_swaps() -> None:
    truth = _walker()
    corrupted = truth.copy()
    wl, wr = KP["l_wrist"], KP["r_wrist"]
    hl, hr = KP["l_hip"], KP["r_hip"]
    for i in (5, 6, 7):  # a three-frame wrist swap
        corrupted[i, [wl, wr]] = corrupted[i, [wr, wl]]
    corrupted[20, [hl, hr]] = corrupted[20, [hr, hl]]  # a single-frame hip swap
    fixed, swaps = fix_swaps(corrupted, 0.3)
    assert np.allclose(fixed, truth)
    assert swaps.sum() == 4
    assert swaps[:, 2].sum() == 3  # wrists are the third group


def test_swap_fix_ignores_real_motion_and_missing_points() -> None:
    truth = _walker()
    truth[10, KP["l_knee"]] = np.nan
    fixed, swaps = fix_swaps(truth, 0.3)
    assert swaps.sum() == 0
    assert np.array_equal(np.isnan(fixed), np.isnan(truth))


def test_apply_swaps_moves_confidences() -> None:
    conf = np.tile(np.arange(17, dtype=float), (2, 1))
    swaps = np.zeros((2, 6), bool)
    swaps[1, 0] = True  # shoulders in frame 1
    out = apply_swaps(conf, swaps)
    assert out[1, KP["l_shoulder"]] == KP["r_shoulder"]
    assert out[1, KP["r_shoulder"]] == KP["l_shoulder"]
    assert np.array_equal(out[0], conf[0])


def test_mask_low_confidence() -> None:
    xy = np.ones((1, 17, 2))
    conf = np.full((1, 17), 0.9)
    conf[0, 3] = 0.1
    conf[0, 4] = np.nan
    out = mask_low_confidence(xy, conf, 0.3)
    assert np.isnan(out[0, 3]).all() and np.isnan(out[0, 4]).all()
    assert np.isfinite(out[0, 5]).all()
