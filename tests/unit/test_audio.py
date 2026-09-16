from __future__ import annotations

import numpy as np
import pytest

from tennis.errors import UserError
from tennis.util.audio import (
    OnsetParams,
    adaptive_threshold,
    classify_self,
    compute_envelope,
    detect_onsets,
    pick_onsets,
)
from tests.synth import render

SR = 48_000
LOUD = -12.0
QUIET = -30.0


def _times(n: int, start: float = 1.3, gap: float = 0.87) -> list[float]:
    return [start + i * gap + 0.0137 * (i % 3) for i in range(n)]


def test_finds_clicks_within_5_ms() -> None:
    times = _times(20)
    x = render(20.0, times, [LOUD] * len(times))
    onsets = detect_onsets(x, SR, OnsetParams())
    assert len(onsets) == len(times)
    errors = np.abs(onsets.time_s - np.array(times))
    assert errors.max() < 0.005


def test_quiet_clicks_in_noise_are_found() -> None:
    times = _times(15)
    x = render(15.0, times, [-40.0] * len(times), noise_db=-60.0)
    onsets = detect_onsets(x, SR, OnsetParams())
    assert len(onsets) == len(times)


def test_noise_alone_gives_few_detections() -> None:
    x = render(30.0, [], [], noise_db=-50.0)
    assert len(detect_onsets(x, SR, OnsetParams())) <= 1


def test_digital_silence_gives_no_detections() -> None:
    x = np.zeros(10 * SR, dtype=np.int16)
    assert len(detect_onsets(x, SR, OnsetParams())) == 0


def test_min_separation_keeps_one_of_two_close_clicks() -> None:
    x = render(5.0, [2.0, 2.1], [LOUD, LOUD - 10])
    onsets = detect_onsets(x, SR, OnsetParams(min_separation_s=0.25))
    assert len(onsets) == 1
    assert abs(onsets.time_s[0] - 2.0) < 0.005


def test_self_classification_uses_session_median() -> None:
    times = _times(21)
    levels = [LOUD if i % 3 == 0 else QUIET for i in range(len(times))]
    x = render(21.0, times, levels)
    onsets = detect_onsets(x, SR, OnsetParams())
    assert len(onsets) == len(times)
    is_self = classify_self(onsets.peak_db, 6.0)
    assert is_self.tolist() == [lvl == LOUD for lvl in levels]


def test_peak_db_tracks_click_level() -> None:
    x = render(6.0, [1.0, 3.0, 5.0], [-10.0, -20.0, -30.0])
    db = detect_onsets(x, SR, OnsetParams()).peak_db
    assert np.diff(db) == pytest.approx([-10.0, -10.0], abs=1.5)


def test_chunking_does_not_change_results() -> None:
    # Clicks right at the 60 s chunk boundary and near it.
    times = [10.0, 59.99, 60.0 + 0.3, 119.9, 130.0]
    x = render(135.0, times, [LOUD] * len(times))
    whole = compute_envelope(x, SR, 800, 4, chunk_s=1000.0)
    chunked = compute_envelope(x, SR, 800, 4, chunk_s=60.0)
    assert np.allclose(whole.strength, chunked.strength, atol=1e-3)
    a = pick_onsets(whole, OnsetParams())
    b = pick_onsets(chunked, OnsetParams())
    assert np.array_equal(a.sample, b.sample)
    assert len(a) == len(times)


def test_detection_is_deterministic() -> None:
    x = render(10.0, _times(8), [LOUD] * 8)
    a = detect_onsets(x, SR, OnsetParams())
    b = detect_onsets(x, SR, OnsetParams())
    assert np.array_equal(a.sample, b.sample)
    assert np.array_equal(a.peak_db, b.peak_db)


def test_other_sample_rate() -> None:
    times = _times(6)
    x = render(7.0, times, [LOUD] * 6, sr=44_100)
    onsets = detect_onsets(x, 44_100, OnsetParams())
    assert np.abs(onsets.time_s - np.array(times)).max() < 0.005


def test_invalid_cutoff() -> None:
    with pytest.raises(UserError, match="highpass_hz"):
        compute_envelope(np.zeros(1000, dtype=np.int16), 8000, 5000, 4)


def test_adaptive_threshold_follows_local_level() -> None:
    fr = 200.0
    rng = np.random.default_rng(1)
    quiet = rng.normal(1.0, 0.1, 2000)
    loud = rng.normal(5.0, 1.0, 2000)
    thr = adaptive_threshold(np.concatenate([quiet, loud]).astype(np.float32), fr, 5.0, 6.0)
    assert thr[500] < 2.0
    assert thr[3500] > 6.0


def test_classify_self_empty() -> None:
    assert classify_self(np.zeros(0), 6.0).size == 0


def test_prominence_filter() -> None:
    x = render(10.0, [3.0, 6.0], [-50.0, -20.0], noise_db=-55.0)
    loose = detect_onsets(x, SR, OnsetParams(min_prominence_db=0.0))
    strict = detect_onsets(x, SR, OnsetParams(min_prominence_db=20.0))
    assert len(strict) == 1 and abs(strict.time_s[0] - 6.0) < 0.005
    assert len(loose) >= len(strict)
    assert (strict.prominence_db >= 20.0).all()
