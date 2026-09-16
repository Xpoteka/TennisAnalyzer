"""Synthetic audio for detector tests: impact-like clicks at known times in noise."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def click(sr: int, rng: np.random.Generator, length_s: float = 0.03) -> npt.NDArray[np.float64]:
    """A racket-impact-like transient: instant rise, ~4 ms exponential decay, broadband."""
    n = int(length_s * sr)
    t = np.arange(n) / sr
    decay = np.exp(-t / 0.004)
    tone = np.sin(2 * np.pi * 2500 * t) + 0.5 * np.sin(2 * np.pi * 4200 * t)
    return decay * (0.6 * tone + 0.4 * rng.standard_normal(n))


def render(
    duration_s: float,
    times_s: Sequence[float],
    levels_db: Sequence[float],
    *,
    sr: int = 48_000,
    noise_db: float = -60.0,
    hum_db: float | None = -30.0,
    seed: int = 0,
) -> npt.NDArray[np.int16]:
    """Mono int16 signal with clicks at ``times_s`` (peak level ``levels_db`` dBFS).

    Adds white noise and optional 150 Hz hum (which the high-pass filter should remove).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    x = rng.standard_normal(n) * 10 ** (noise_db / 20)
    if hum_db is not None:
        x += 10 ** (hum_db / 20) * np.sin(2 * np.pi * 150 * np.arange(n) / sr)
    for t, level in zip(times_s, levels_db, strict=True):
        c = click(sr, rng)
        c *= 10 ** (level / 20) / np.max(np.abs(c))
        i = round(t * sr)
        seg = c[: max(0, n - i)]
        x[i : i + seg.size] += seg
    return np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
