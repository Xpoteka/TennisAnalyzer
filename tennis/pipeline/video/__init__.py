"""Per-video stages, in the order they run."""

from __future__ import annotations

from tennis.pipeline import VideoStage
from tennis.pipeline.video import audio, ball, court, ingest, people, proxy

VIDEO_STAGES: tuple[VideoStage, ...] = (
    VideoStage(
        "ingest", ingest.run, ingest.OUTPUTS, weight=0.3,
        title="Reading the video",
    ),
    VideoStage(
        "audio", audio.run, audio.OUTPUTS, config_keys=("audio",), weight=0.2,
        title="Listening for ball impacts",
    ),
    VideoStage(
        "proxy", proxy.run, proxy.OUTPUTS, config_keys=("proxy",), weight=2.0,
        title="Making a browser-playable copy",
    ),
    VideoStage(
        "court", court.run, court.OUTPUTS, config_keys=("court",), weight=0.5,
        title="Finding the court",
    ),
    VideoStage(
        "people", people.run, people.OUTPUTS, config_keys=("pose", "court"), weight=10.0,
        title="Tracking the players",
    ),
    VideoStage(
        "ball", ball.run, ball.OUTPUTS, config_keys=("ball",), weight=4.0,
        title="Tracking the ball",
    ),
)  # fmt: skip
