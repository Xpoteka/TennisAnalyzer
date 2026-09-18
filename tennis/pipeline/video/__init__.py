"""Per-video stages, in the order they run."""

from __future__ import annotations

from tennis.pipeline import VideoStage
from tennis.pipeline.video import ingest, proxy

VIDEO_STAGES: tuple[VideoStage, ...] = (
    VideoStage(
        "ingest", ingest.run, ingest.OUTPUTS, weight=0.3,
        title="Reading the video",
    ),
    VideoStage(
        "proxy", proxy.run, proxy.OUTPUTS, config_keys=("proxy",), weight=2.0,
        title="Making a browser-playable copy",
    ),
)  # fmt: skip
