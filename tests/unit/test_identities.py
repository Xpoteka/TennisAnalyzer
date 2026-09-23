"""Telling the players apart from the tracks of one video."""

from __future__ import annotations

import numpy as np

from tennis.pipeline.session.identities import (
    TrackInfo,
    assign_by_side,
    concurrent_players,
    find_players,
    two_players,
)

DARK = np.array([0.1, 0.0, 0.8, 0.1])
LIGHT = np.array([0.1, 0.0, 0.1, 0.8])


def track(
    tid: int, start: float, end: float, y: float, look: np.ndarray, x: float = 0.0, hits: int = 0
) -> TrackInfo:
    times = np.round(np.arange(start, end, 0.1), 1)
    xy = np.column_stack([np.full(len(times), x), np.full(len(times), y)])
    return TrackInfo(1, tid, start, end, len(times), look, hits, times, xy)


def test_a_track_replaced_by_the_next_is_not_a_third_person() -> None:
    # The far player's track breaks every 10 s; the near one's every 15 s.
    tracks = [track(k, 10.0 * k, 10.0 * (k + 1), 12.0, LIGHT) for k in range(12)]
    tracks += [track(100 + k, 15.0 * k, 15.0 * (k + 1), -12.0, DARK) for k in range(8)]
    assert concurrent_players(tracks) == 2


def test_doubles_are_four() -> None:
    tracks = [track(k, 0.0, 60.0, y, LIGHT, x=x) for k, (x, y) in enumerate([(-3, 12), (3, 12)])]
    tracks += [
        track(10 + k, 0.0, 60.0, y, DARK, x=x) for k, (x, y) in enumerate([(-3, -12), (3, -12)])
    ]
    assert concurrent_players(tracks) == 4


def test_singles_chain_through_broken_tracks_and_changeovers() -> None:
    # Near player A (dark) and far player B (light) for 60 s, then they change ends.
    tracks = [
        track(1, 0, 30, -12, DARK), track(2, 0, 30, 12, LIGHT),
        track(3, 30.5, 60, -12, DARK), track(4, 30.5, 60, 12, LIGHT),
        track(5, 70, 100, 12, DARK), track(6, 70, 100, -12, LIGHT),
    ]  # fmt: skip
    players = two_players(tracks)
    ids = sorted(sorted(t.id for t in g.tracks) for g in players)
    assert ids == [[1, 3, 5], [2, 4, 6]]


def test_bystanders_on_the_bench_are_not_players() -> None:
    tracks = [
        track(1, 0, 60, -12, DARK, hits=10),
        track(2, 0, 60, 12, LIGHT, hits=10),
        track(3, 0, 60, -6, np.array([0.5, 0.5, 0.0, 0.0]), x=-8.5, hits=3),  # on the bench
    ]
    players, concurrent = find_players(tracks)
    assert concurrent == 2
    assert len(players) == 2
    assert all(t.id != 3 for g in players for t in g.tracks)


def test_short_tracks_go_to_the_player_on_that_side() -> None:
    a = track(1, 0, 60, -12, DARK)
    b = track(2, 0, 60, 12, LIGHT)
    players = two_players([a, b])
    stray = track(3, 20, 21, 11, np.zeros(4))  # one second at the far end: player B
    assign_by_side(players, [stray])
    owner = next(g for g in players if any(t.id == 2 for t in g.tracks))
    assert any(t.id == 3 for t in owner.tracks)
