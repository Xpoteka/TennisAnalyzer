"""Several videos of one session: players across cameras, and shots seen twice."""

from __future__ import annotations

import numpy as np

from tennis.pipeline.session.identities import TrackInfo, two_players
from tennis.pipeline.session.shots import merge_views, resegment

LOOK = np.array([0.5, 0.5, 0.0, 0.0])


def track(
    video: int, tid: int, start: float, end: float, y: float, look: np.ndarray = LOOK
) -> TrackInfo:
    times = np.round(np.arange(start, end, 0.1), 1)
    xy = np.column_stack([np.zeros(len(times)), np.full(len(times), y)])
    return TrackInfo(video, tid, start, end, len(times), look, 5, times, xy)


def test_players_are_linked_across_cameras() -> None:
    # Camera 1 sees near (y=-12) and far (y=+12) players; camera 2 sees the same two, with
    # its own track ids and a different impression of the clothes.
    other_look = np.array([0.0, 0.0, 0.5, 0.5])
    tracks = [
        track(1, 1, 0, 60, -12), track(1, 2, 0, 60, 12),
        track(2, 7, 10, 60, -12, other_look), track(2, 8, 10, 60, 12, other_look),
    ]  # fmt: skip
    players = two_players(tracks)
    groups = [{(t.video, t.id) for t in g.tracks} for g in players]
    assert {(1, 1), (2, 7)} in groups
    assert {(1, 2), (2, 8)} in groups


def shot(
    video: int, t: float, label: str, stroke: str, speed: float | None = None
) -> dict[str, object]:
    r: dict[str, object] = {
        "video": video, "t": t, "label": label, "stroke": stroke,
        "metrics": {"knee_bend_deg": 140.0}, "sources": ["audio"],
        "quality": {"hit_score": 0.8}, "rally_key": (video, 0),
    }  # fmt: skip
    if speed is not None:
        r["speed_kmh"] = speed
    return r


def test_one_shot_seen_twice_becomes_one() -> None:
    shots = [
        shot(1, 10.00, "A", "serve"),
        shot(2, 10.05, "A", "serve", speed=150.0),
        shot(1, 11.30, "B", "forehand"),
        shot(2, 11.32, "B", "unknown"),
        shot(1, 20.00, "A", "serve"),
    ]
    merged = merge_views(shots)  # type: ignore[arg-type]
    assert len(merged) == 3
    assert merged[0]["speed_kmh"] == 150.0
    assert merged[1]["stroke"] == "forehand"
    resegment(merged)
    assert [r["rally_key"] for r in merged] == [(0, 0), (0, 0), (0, 1)]
