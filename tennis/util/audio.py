"""Ball-impact onset detection on the session audio (spec section 6.2).

Pipeline: high-pass Butterworth -> log-mel spectral flux (``librosa.onset.onset_strength``)
-> adaptive threshold (sliding median + k * MAD) -> peak picking with a minimum separation
-> per-onset refinement to the transient's rise on the filtered waveform -> peak RMS level
-> prominence check against the local background level.

The prominence check is an addition to the spec: log-spectral flux has a heavy tail on
plain noise, so without it steady background noise produces several detections a minute.

The audio is processed in fixed chunks with overlapping context, so memory stays bounded
for long sessions and results do not depend on anything but the input and parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt
from scipy.io import wavfile
from scipy.signal import butter, find_peaks, sosfiltfilt

from tennis.errors import UserError

FloatArray = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

FRAME_S = 0.005  # onset envelope resolution
WINDOW_S = 0.02  # STFT window (rounded up to a power of two)
N_MELS = 64
CHUNK_S = 60.0
CONTEXT_S = 1.0
THRESHOLD_STEP_S = 0.25  # the sliding median/MAD is evaluated on this grid, then interpolated
REFINE_SEARCH_S = 0.025  # look this far around the envelope peak for the waveform peak
REFINE_BACK_S = 0.010  # and at most this far back from it for the rise
REFINE_RISE_FRACTION = 0.3
RMS_WINDOW_S = 0.005
DB_FLOOR = -120.0


@dataclass(frozen=True)
class OnsetParams:
    highpass_hz: float = 800.0
    highpass_order: int = 4
    onset_k: float = 6.0
    min_separation_s: float = 0.25
    threshold_window_s: float = 5.0
    amplitude_window_s: float = 0.02
    min_prominence_db: float = 6.0


@dataclass(frozen=True)
class Envelope:
    """High-passed signal and its onset-strength envelope (one value per ``hop`` samples)."""

    sr: int
    hop: int
    highpassed: FloatArray
    strength: FloatArray
    level_db: FloatArray  # short-time RMS level (dBFS) of ``highpassed``, per frame

    @property
    def frame_rate(self) -> float:
        return self.sr / self.hop


@dataclass(frozen=True)
class Onsets:
    sample: IntArray  # refined onset position in the WAV
    time_s: Float64Array  # sample / sr (WAV timeline)
    strength: FloatArray  # onset envelope value at the detected peak
    threshold: FloatArray  # adaptive threshold at that peak
    peak_db: Float64Array  # peak short-time RMS (dBFS) of the filtered signal around the onset
    prominence_db: Float64Array  # peak_db minus the local background level

    def __len__(self) -> int:
        return int(self.sample.size)


def read_wav_mono(path: Path) -> tuple[int, npt.NDArray[np.int16]]:
    """Memory-map a mono 16-bit WAV (as written by the ingest stage)."""
    try:
        sr, data = wavfile.read(path, mmap=True)
    except (OSError, ValueError) as exc:
        raise UserError(f"cannot read {path}: {exc}") from exc
    if data.dtype != np.int16 or data.ndim != 1:
        raise UserError(f"{path}: expected mono 16-bit PCM, got {data.dtype} x {data.shape}")
    return int(sr), data


def _frame_sizes(sr: int) -> tuple[int, int]:
    hop = max(1, round(sr * FRAME_S))
    n_fft = 1 << int(np.ceil(np.log2(sr * WINDOW_S)))
    return hop, n_fft


def _to_float(x: npt.NDArray[np.generic]) -> Float64Array:
    if x.dtype == np.int16:
        return x.astype(np.float64) / 32768.0
    return x.astype(np.float64)


def compute_envelope(
    samples: npt.NDArray[np.generic],
    sr: int,
    highpass_hz: float,
    highpass_order: int,
    chunk_s: float = CHUNK_S,
) -> Envelope:
    """High-pass the signal and compute its spectral-flux onset envelope, chunk by chunk."""
    if not 0 < highpass_hz < sr / 2:
        raise UserError(f"highpass_hz must be between 0 and {sr / 2:g} Hz, got {highpass_hz:g}")
    hop, n_fft = _frame_sizes(sr)
    n = int(samples.shape[0])
    n_frames = 1 + n // hop
    chunk = max(hop, int(chunk_s * sr) // hop * hop)
    context = max(n_fft * 2, int(CONTEXT_S * sr) // hop * hop)
    sos = butter(highpass_order, highpass_hz, btype="highpass", fs=sr, output="sos")
    fmin = min(highpass_hz, 0.9 * sr / 2)

    highpassed = np.zeros(n, dtype=np.float32)
    strength = np.zeros(n_frames, dtype=np.float32)
    level_db = np.zeros(n_frames, dtype=np.float32)
    for s0 in range(0, n, chunk):
        s1 = min(n, s0 + chunk)
        a = max(0, s0 - context)
        b = min(n, s1 + context)
        seg = _to_float(samples[a:b])
        padlen = min(3 * (2 * len(sos) + 1), seg.size - 1)
        filtered = sosfiltfilt(sos, seg, padlen=max(padlen, 0)) if seg.size > 1 else seg
        highpassed[s0:s1] = filtered[s0 - a : s1 - a]

        mel = librosa.feature.melspectrogram(
            y=filtered.astype(np.float32),
            sr=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=N_MELS,
            fmin=fmin,
            center=True,
            pad_mode="reflect",  # zero padding would create a fake onset at the file start
        )
        # Fixed reference and no top_db clipping, so every chunk is scaled identically.
        mel_db = librosa.power_to_db(mel, ref=1.0, top_db=None)
        flux = librosa.onset.onset_strength(
            S=mel_db, sr=sr, hop_length=hop, n_fft=n_fft, center=True
        )
        f0 = s0 // hop
        f1 = n_frames if s1 == n else s1 // hop
        base = a // hop  # a is a multiple of hop
        strength[f0:f1] = flux[f0 - base : f1 - base]

        blocks = np.zeros((f1 - f0) * hop)
        own = filtered[s0 - a : s1 - a]
        used = min(own.size, blocks.size)
        blocks[:used] = own[:used] ** 2
        power = blocks.reshape(f1 - f0, hop).mean(axis=1)
        level_db[f0:f1] = 10.0 * np.log10(np.maximum(power, 10 ** (DB_FLOOR / 10)))
    return Envelope(sr=sr, hop=hop, highpassed=highpassed, strength=strength, level_db=level_db)


def _sliding_median(
    values: FloatArray, frame_rate: float, window_s: float, with_mad: bool
) -> tuple[Float64Array, Float64Array]:
    """Centred sliding median (and MAD), evaluated on a coarse grid and interpolated."""
    n = values.size
    half = max(1, round(window_s * frame_rate / 2))
    step = max(1, round(THRESHOLD_STEP_S * frame_rate))
    centers = np.arange(0, n, step)
    if centers[-1] != n - 1:
        centers = np.append(centers, n - 1)
    med = np.empty(centers.size)
    mad = np.zeros(centers.size)
    for i, c in enumerate(centers):
        window = values[max(0, c - half) : c + half + 1]
        m = float(np.median(window))
        med[i] = m
        if with_mad:
            mad[i] = float(np.median(np.abs(window - m)))
    frames = np.arange(n)
    return np.interp(frames, centers, med), np.interp(frames, centers, mad)


def adaptive_threshold(
    strength: FloatArray, frame_rate: float, window_s: float, k: float
) -> FloatArray:
    """``median + k * MAD`` over a centred sliding window."""
    if strength.size == 0:
        return strength.copy()
    med, mad = _sliding_median(strength, frame_rate, window_s, with_mad=True)
    # Long stretches of digital silence have MAD 0; keep the threshold above the noise floor.
    global_mad = float(np.median(np.abs(strength - np.median(strength))))
    mad = np.maximum(mad, max(0.1 * global_mad, 1e-6))
    return (med + k * mad).astype(np.float32)


def _refine(highpassed: FloatArray, sr: int, center: int) -> int:
    n = highpassed.size
    search = int(REFINE_SEARCH_S * sr)
    a, b = max(0, center - search), min(n, center + search + 1)
    if b - a < 3:
        return center
    env = np.abs(highpassed[a:b].astype(np.float64))
    smooth = max(1, int(0.0005 * sr))
    if smooth > 1:
        env = np.convolve(env, np.ones(smooth) / smooth, mode="same")
    peak = int(np.argmax(env))
    lo = max(0, peak - int(REFINE_BACK_S * sr))
    below = np.flatnonzero(env[lo : peak + 1] < REFINE_RISE_FRACTION * env[peak])
    rise = lo + int(below[-1]) + 1 if below.size else lo
    return a + rise


def _peak_rms_db(highpassed: FloatArray, sr: int, sample: int, half_window_s: float) -> float:
    half = int(half_window_s * sr)
    seg = highpassed[max(0, sample - half) : sample + half + 1].astype(np.float64)
    w = max(1, int(RMS_WINDOW_S * sr))
    if seg.size == 0:
        return DB_FLOOR
    if seg.size < w:
        power = float(np.mean(seg**2))
    else:
        csum = np.concatenate(([0.0], np.cumsum(seg**2)))
        power = float(np.max(csum[w:] - csum[:-w]) / w)
    return max(DB_FLOOR, 10.0 * np.log10(power)) if power > 0 else DB_FLOOR


def pick_onsets(envelope: Envelope, params: OnsetParams) -> Onsets:
    """Peaks of the onset envelope above the adaptive threshold, refined and measured.

    Only ``onset_k``, ``min_separation_s``, ``threshold_window_s``, ``amplitude_window_s``
    and ``min_prominence_db`` are used; the filter settings are baked into ``envelope``.
    """
    fr = envelope.frame_rate
    sr = envelope.sr
    threshold = adaptive_threshold(envelope.strength, fr, params.threshold_window_s, params.onset_k)
    distance = max(1, round(params.min_separation_s * fr))
    peaks, _ = find_peaks(envelope.strength, height=threshold, distance=distance)

    samples = np.array(
        [_refine(envelope.highpassed, sr, int(p) * envelope.hop) for p in peaks], dtype=np.int64
    )
    strength = envelope.strength[peaks]
    thr = threshold[peaks]
    # Refinement can pull two neighbouring detections together; keep the stronger one.
    if samples.size > 1:
        keep = np.ones(samples.size, dtype=bool)
        min_gap = int(params.min_separation_s * sr)
        last = 0
        for i in range(1, samples.size):
            if samples[i] - samples[last] < min_gap:
                if strength[i] > strength[last]:
                    keep[last] = False
                    last = i
                else:
                    keep[i] = False
            else:
                last = i
        samples, strength, thr = samples[keep], strength[keep], thr[keep]

    peak_db = np.array(
        [_peak_rms_db(envelope.highpassed, sr, int(s), params.amplitude_window_s) for s in samples],
        dtype=np.float64,
    )
    background, _ = _sliding_median(
        envelope.level_db, fr, params.threshold_window_s, with_mad=False
    )
    frames = np.minimum(samples // envelope.hop, envelope.level_db.size - 1)
    prominence = peak_db - background[frames] if samples.size else np.zeros(0)
    loud = prominence >= params.min_prominence_db
    return Onsets(
        sample=samples[loud],
        time_s=samples[loud].astype(np.float64) / sr,
        strength=strength[loud].astype(np.float32),
        threshold=thr[loud].astype(np.float32),
        peak_db=peak_db[loud],
        prominence_db=prominence[loud].astype(np.float64),
    )


def detect_onsets(samples: npt.NDArray[np.generic], sr: int, params: OnsetParams) -> Onsets:
    envelope = compute_envelope(samples, sr, params.highpass_hz, params.highpass_order)
    return pick_onsets(envelope, params)


def classify_self(peak_db: Float64Array, own_hit_db_threshold: float) -> npt.NDArray[np.bool_]:
    """First pass: an onset is the player's own hit if it is loud relative to the session.

    Loud means at least ``own_hit_db_threshold`` dB above the median onset level.
    """
    if peak_db.size == 0:
        return np.zeros(0, dtype=bool)
    return peak_db >= float(np.median(peak_db)) + own_hit_db_threshold
