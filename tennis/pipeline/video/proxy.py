"""Video stage ``proxy``: a copy of the video that every browser can play.

Camera files are often HEVC, 10-bit or 4K, which Chrome and Firefox cannot play. The proxy is
H.264 8-bit with AAC audio, at most ``proxy.height`` pixels high, with the index at the front
so playback starts before the whole file has arrived. A source that browsers can already play
is linked instead of copied.

The browser's time 0 is the earliest stream start of the source. ``proxy.json`` records that
PTS as ``start_pts``, so a video time ``t`` plays at ``t - start_pts`` in the proxy.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

from tennis.util.io import atomic_path, write_json
from tennis.util.video import ProbeError, require_tool

if TYPE_CHECKING:
    from tennis.pipeline import VideoContext

OUTPUTS = ("proxy.mp4", "proxy.json")
PLAYABLE_CODECS = {"h264"}
PLAYABLE_PIX_FMTS = {"yuv420p", "yuvj420p"}
PLAYABLE_CONTAINERS = {".mp4", ".m4v", ".mov"}


def can_play_directly(meta: dict[str, object], suffix: str, max_height: int) -> bool:
    video = meta["video"]
    assert isinstance(video, dict)
    height = int(video["height"] or 0)
    return (
        video["codec"] in PLAYABLE_CODECS
        and video["pix_fmt"] in PLAYABLE_PIX_FMTS
        and suffix.lower() in PLAYABLE_CONTAINERS
        and height <= max(max_height, 1080)
        and int(video.get("rotation_deg") or 0) == 0
    )


def encoder_args(encoder: str, crf: int, height: int) -> list[str]:
    if encoder == "auto":
        encoder = "h264_videotoolbox" if sys.platform == "darwin" else "libx264"
    if encoder == "h264_videotoolbox":
        # Hardware encoder: no CRF; a bitrate that looks like CRF ~26 at this height.
        bitrate = max(1, round(3.5 * (height / 720) ** 2 * (26 / crf)))
        return ["-c:v", "h264_videotoolbox", "-b:v", f"{bitrate}M", "-allow_sw", "1"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]


def run(ctx: VideoContext) -> None:
    meta = ctx.metadata()
    cfg = ctx.config.proxy
    starts = [meta.get("video_start_s"), meta.get("audio_start_s")]
    start_pts = min((float(s) for s in starts if s is not None), default=0.0)
    out = ctx.path("proxy.mp4")

    if can_play_directly(meta, ctx.source.suffix, cfg.height):
        out.unlink(missing_ok=True)
        os.symlink(ctx.source, out)
        write_json(ctx.path("proxy.json"), {"start_pts": start_pts, "linked": True})
        ctx.log("source is browser-playable; linked")
        return

    duration = float(meta.get("duration_s") or 0.0)
    height = min(cfg.height, int(meta["video"]["height"] or cfg.height))
    height -= height % 2
    with atomic_path(out, keep_mtime_if_identical=False) as tmp:
        cmd = [
            require_tool("ffmpeg"), "-nostdin", "-hide_banner", "-v", "error", "-y",
            "-i", str(ctx.source),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", f"scale=-2:{height},format=yuv420p",
            *encoder_args(cfg.encoder, cfg.crf, height),
            "-c:a", "aac", "-b:a", "128k", "-ac", "2",
            "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats",
            str(tmp),
        ]  # fmt: skip
        _run_with_progress(cmd, duration, ctx)
    write_json(ctx.path("proxy.json"), {"start_pts": start_pts, "linked": False})
    ctx.log("proxy written", bytes=out.stat().st_size, height=height)


def _run_with_progress(cmd: list[str], duration: float, ctx: VideoContext) -> None:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and value.isdigit() and duration > 0:
            ctx.progress(int(value) / 1e6 / duration)
    _, err = proc.communicate()
    if proc.returncode != 0:
        raise ProbeError(f"ffmpeg could not write the proxy: {err.strip()[-400:]}")
