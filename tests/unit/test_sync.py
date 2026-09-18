"""Lining up two soundtracks by their impact onsets."""

from __future__ import annotations

import numpy as np
import pytest

from tennis.analysis.sync import find_offset

RATE = 200.0


def envelope(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Onset strength with a click every second or so, over noise."""
    n = int(seconds * RATE)
    env = rng.gamma(1.0, 0.05, n)
    t = 0.0
    while True:
        t += rng.uniform(0.6, 2.5)
        if t * RATE >= n:
            return env
        env[int(t * RATE)] += rng.uniform(1.0, 4.0)


@pytest.mark.parametrize("start_s", [0.0, 37.215, 400.5])
def test_finds_where_the_second_camera_started(start_s: float) -> None:
    rng = np.random.default_rng(1)
    a = envelope(900, rng)
    first = int(start_s * RATE)
    b = a[first : first + int(600 * RATE)].copy()
    b = b * 0.3 + rng.gamma(1.0, 0.05, len(b))  # another microphone: quieter, its own noise
    result = find_offset(a, b, RATE)
    assert result.confidence > 1.6
    assert result.offset_s == pytest.approx(round(start_s * RATE) / RATE, abs=0.01)


def test_unrelated_recordings_are_not_lined_up() -> None:
    rng = np.random.default_rng(2)
    result = find_offset(envelope(600, rng), envelope(600, rng), RATE)
    assert result.confidence < 1.6
