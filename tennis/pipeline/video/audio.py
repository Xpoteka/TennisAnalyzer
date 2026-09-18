"""Video stage ``audio``: ball impacts in the soundtrack (see :mod:`tennis.util.audio`).

A racket hitting the ball, and the ball bouncing, are sharp broadband transients: the most
precise timing cue there is (a few milliseconds, against 33 ms between frames at 30 fps).
The shot stage combines these onsets with the ball track and the players' wrists; on its
own an onset may also be a ball from the next court or a dropped racket.

Writes ``onsets.parquet`` (times on the video PTS timeline) and ``envelope.npy``, the onset
strength at 200 per second, used to line up several videos of one session. A video without
sound gets empty outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from tennis.util.audio import OnsetParams, compute_envelope, pick_onsets, read_wav_mono
from tennis.util.io import atomic_path, write_parquet

if TYPE_CHECKING:
    from tennis.pipeline import VideoContext

OUTPUTS = ("onsets.parquet", "envelope.npy")
SCHEMA_VERSION = 1


def run(ctx: VideoContext) -> None:
    meta = ctx.metadata()
    cfg = ctx.config.audio
    params = OnsetParams(
        highpass_hz=cfg.highpass_hz,
        highpass_order=cfg.highpass_order,
        onset_k=cfg.onset_k,
        min_separation_s=cfg.min_separation_s,
        threshold_window_s=cfg.threshold_window_s,
        amplitude_window_s=cfg.amplitude_window_s,
        min_prominence_db=cfg.min_prominence_db,
    )
    wav = ctx.path("audio.wav")
    columns: dict[str, pa.Array] = {
        "t": pa.array([], pa.float64()),
        "strength": pa.array([], pa.float32()),
        "peak_db": pa.array([], pa.float32()),
        "prominence_db": pa.array([], pa.float32()),
    }
    envelope = np.zeros(0, np.float32)
    rate = 0.0
    if meta.get("has_audio") and wav.is_file():
        sr, samples = read_wav_mono(wav)
        env = compute_envelope(samples, sr, params.highpass_hz, params.highpass_order)
        ctx.progress(0.7)
        onsets = pick_onsets(env, params)
        start = float(meta.get("audio_start_s") or 0.0)
        columns = {
            "t": pa.array(onsets.time_s + start, pa.float64()),
            "strength": pa.array(onsets.strength, pa.float32()),
            "peak_db": pa.array(onsets.peak_db, pa.float32()),
            "prominence_db": pa.array(onsets.prominence_db, pa.float32()),
        }
        envelope = env.strength.astype(np.float32)
        rate = env.frame_rate
        ctx.log(
            "onsets",
            count=len(onsets),
            per_minute=round(len(onsets) / max(1e-9, samples.size / sr / 60), 1),
        )
    write_parquet(
        pa.table(columns),
        ctx.path("onsets.parquet"),
        stage=ctx.stage,
        config_hash=ctx.config.section_hash("audio"),
        schema_version=SCHEMA_VERSION,
        extra={"envelope_rate": str(rate)},
    )
    with atomic_path(ctx.path("envelope.npy")) as tmp, tmp.open("wb") as fh:
        np.save(fh, envelope)
