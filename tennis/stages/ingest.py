"""Stage 1: read video metadata and extract the audio track (spec section 6.1).

Time conventions written here and used by later stages:

* ``video_start_s`` is the PTS (seconds) of the first video frame. Frame times elsewhere in
  the pipeline are PTS values in the same timeline.
* ``audio.wav`` sample 0 sits at PTS ``audio_start_s``, so a time ``t`` in the WAV maps to
  PTS ``t + audio_start_s``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tennis import __version__
from tennis.errors import UserError
from tennis.util import video
from tennis.util.io import atomic_path, write_json

if TYPE_CHECKING:
    from tennis.stages import StageContext

SCHEMA_VERSION = 1
AUDIO_SAMPLE_RATE = 48_000
OUTPUTS = ("metadata.json", "audio.wav")


def build_metadata(
    probe: video.ProbeResult, intervals: video.FrameIntervals, config_hash: str
) -> dict[str, Any]:
    v = probe.video
    a = probe.audio
    assert v is not None and a is not None
    created = probe.creation_time
    creation_source = "container"
    if created is None:
        created = video.file_creation_time(probe.path)
        creation_source = "filesystem"
    duration = probe.duration or v.duration or a.duration
    is_vfr = video.is_variable_frame_rate(v, intervals)
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": __version__,
        "config_hash": config_hash,
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
                "sampled_frames": intervals.sampled_frames,
                "median": round(intervals.median_ms, 4),
                "p01": round(intervals.p01_ms, 4),
                "p99": round(intervals.p99_ms, 4),
            },
            "nb_frames": v.nb_frames,
            "start_s": v.start_time,
        },
        "audio": {
            "stream_index": a.index,
            "source_codec": a.codec,
            "source_sample_rate": a.sample_rate,
            "source_channels": a.channels,
            "start_s": a.start_time,
            "wav_sample_rate": AUDIO_SAMPLE_RATE,
            "wav_channels": 1,
            "wav_sample_format": "s16",
        },
        # Flat aliases for the fields the spec names directly.
        "fps": v.fps_nominal,
        "is_vfr": is_vfr,
        "resolution": [v.width, v.height],
        "codec": v.codec,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "video_start_s": v.start_time,
        "audio_start_s": a.start_time,
    }


def run(ctx: StageContext) -> None:
    src = ctx.session.source_link.resolve()
    probe = video.probe(src)
    if probe.video is None:
        raise UserError(f"{src.name} has no video stream")
    if probe.audio is None:
        raise UserError(
            f"{src.name} has no audio track; contact detection needs the microphone audio "
            "embedded in the video file"
        )
    if probe.audio.sample_rate <= 0 or probe.audio.channels <= 0:
        raise UserError(f"{src.name}: audio stream {probe.audio.index} has no usable samples")

    intervals = video.sample_frame_intervals(src, probe.video.index)
    meta = build_metadata(probe, intervals, ctx.config_hash)
    ctx.log(
        "probed",
        duration_s=meta["duration_s"],
        resolution=f"{probe.video.width}x{probe.video.height}",
        fps=meta["fps"],
        vfr=meta["is_vfr"],
        codec=probe.video.codec,
    )

    wav = ctx.session.path("audio.wav")
    with atomic_path(wav) as tmp:
        video.extract_audio(src, tmp, probe.audio.index, AUDIO_SAMPLE_RATE)
    ctx.log("audio extracted", path=wav.name, bytes=wav.stat().st_size)

    # metadata.json is written last so that it is never newer-and-valid without audio.wav.
    write_json(ctx.session.path("metadata.json"), meta)
