"""Video stage ``ingest``: read the video's metadata, frame timestamps and audio.

Time conventions used by every later stage:

* Frame times are **PTS seconds** from the container, never ``frame_index / fps``.
  ``frame_times.parquet`` lists the PTS of every frame; frame index ``i`` means row ``i``.
* ``video_start_s`` is the PTS of the first video frame.
* ``audio.wav`` sample 0 sits at PTS ``audio_start_s``, so a time ``t`` in the WAV maps to
  PTS ``t + audio_start_s``.

Audio is optional: without it the hit detection relies on the ball and the players only,
and several videos cannot be synced by sound.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from tennis import __version__
from tennis.errors import UserError
from tennis.util import video
from tennis.util.io import atomic_path, write_json, write_parquet

if TYPE_CHECKING:
    from tennis.pipeline import VideoContext

SCHEMA_VERSION = 2
AUDIO_SAMPLE_RATE = 48_000
# audio.wav is written too when the video has sound; it is optional, so not listed.
OUTPUTS = ("metadata.json", "frame_times.parquet")
FRAME_TIMES_SCHEMA_VERSION = 1

# Footage outside this range is still processed; timing-dependent results (ball speed, hit
# times) just get less precise, so ingest warns.
FPS_SUPPORTED_MIN = 25.0
FPS_SUPPORTED_MAX = 240.0


def frame_rate_warnings(fps: float | None) -> list[str]:
    """Human-readable warnings for a frame rate outside the supported range."""
    if fps is None:
        return ["frame rate unknown; timing relies on frame timestamps only"]
    interval_ms = 1000.0 / fps
    if fps < FPS_SUPPORTED_MIN:
        return [
            f"low frame rate {fps:.2f} fps: frames are {interval_ms:.0f} ms apart, so ball "
            "speeds and hit times are less precise (50 fps or more is best)"
        ]
    if fps > FPS_SUPPORTED_MAX:
        return [
            f"high frame rate {fps:.2f} fps (supported up to {FPS_SUPPORTED_MAX:.0f}): "
            "analysis will be slow"
        ]
    return []


def build_metadata(probe: video.ProbeResult, intervals: video.FrameIntervals) -> dict[str, Any]:
    v = probe.video
    a = probe.audio
    assert v is not None
    created = probe.creation_time
    creation_source = "container"
    if created is None:
        created = video.file_creation_time(probe.path)
        creation_source = "filesystem"
    duration = probe.duration or v.duration or (a.duration if a else None)
    is_vfr = video.is_variable_frame_rate(v, intervals)
    warnings = frame_rate_warnings(v.fps_avg or v.fps_nominal)
    if a is None:
        warnings.append(
            "no audio track: hits are found from the ball and the players only, and this "
            "video cannot be synced to others by sound"
        )
    return {
        "warnings": warnings,
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": __version__,
        "source_path": str(probe.path),
        "source_size_bytes": probe.size_bytes,
        "container": probe.format_name,
        "duration_s": duration,
        "creation_time": created.isoformat(),
        "creation_time_source": creation_source,
        "video": {
            "stream_index": v.index,
            "codec": v.codec,
            "pix_fmt": v.pix_fmt,
            "width": v.width,
            "height": v.height,
            "rotation_deg": v.rotation,
            "fps_nominal": v.fps_nominal,
            "fps_avg": v.fps_avg,
            "is_vfr": is_vfr,
            "frame_interval_ms": {
                "frames": intervals.frames,
                "median": round(intervals.median_ms, 4),
                "p01": round(intervals.p01_ms, 4),
                "p99": round(intervals.p99_ms, 4),
            },
            "nb_frames": v.nb_frames,
            "start_s": v.start_time,
        },
        "audio": (
            {
                "stream_index": a.index,
                "source_codec": a.codec,
                "source_sample_rate": a.sample_rate,
                "source_channels": a.channels,
                "start_s": a.start_time,
                "wav_sample_rate": AUDIO_SAMPLE_RATE,
                "wav_channels": 1,
                "wav_sample_format": "s16",
            }
            if a is not None
            else None
        ),
        "has_audio": a is not None,
        "fps": v.fps_avg or v.fps_nominal,
        "is_vfr": is_vfr,
        "codec": v.codec,
        "width": v.width,
        "height": v.height,
        "video_start_s": v.start_time,
        "audio_start_s": a.start_time if a is not None else None,
    }


def run(ctx: VideoContext) -> None:
    src = ctx.source
    probe = video.probe(src)
    if probe.video is None:
        raise UserError(f"{src.name} has no video stream")
    audio = probe.audio
    if audio is not None and (audio.sample_rate <= 0 or audio.channels <= 0):
        audio = None

    pts = video.read_frame_pts(src, probe.video.index)
    if pts.size == 0:
        raise UserError(f"{src.name}: video stream {probe.video.index} has no frames")
    intervals = video.frame_intervals(pts)
    meta = build_metadata(dataclasses.replace(probe, audio=audio), intervals)
    ctx.log(
        "probed",
        duration_s=meta["duration_s"],
        resolution=f"{probe.video.width}x{probe.video.height}",
        fps=round(meta["fps"], 3) if meta["fps"] else None,
        vfr=meta["is_vfr"],
        codec=probe.video.codec,
    )
    for warning in meta["warnings"]:
        ctx.log(warning, level=logging.WARNING)
    ctx.progress(0.3)

    write_parquet(
        pa.table(
            {
                "frame_idx": pa.array(np.arange(pts.size, dtype=np.int64)),
                "pts": pa.array(pts),
            }
        ),
        ctx.path("frame_times.parquet"),
        stage=ctx.stage,
        config_hash="",
        schema_version=FRAME_TIMES_SCHEMA_VERSION,
    )
    ctx.log("frame times read", frames=int(pts.size))

    wav = ctx.path("audio.wav")
    if audio is not None:
        with atomic_path(wav) as tmp:
            video.extract_audio(src, tmp, audio.index, AUDIO_SAMPLE_RATE)
        ctx.log("audio extracted", path=wav.name, bytes=wav.stat().st_size)
    else:
        wav.unlink(missing_ok=True)
    ctx.progress(0.9)

    # metadata.json is written last so that it is never present without the other outputs.
    write_json(ctx.path("metadata.json"), meta)
