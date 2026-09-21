"""Joining tracks into players."""

from __future__ import annotations

import time

import numpy as np

from tennis.pipeline.session.identities import (
    MERGE_MAX_DISTANCE,
    Group,
    TrackInfo,
    cluster,
)
from tennis.util.appearance import distance


def _tracks(n: int, seed: int) -> list[TrackInfo]:
    """Two players in different colours, seen in turns of a few seconds, with noise."""
    rng = np.random.default_rng(seed)
    colours = rng.random((2, 12))
    out = []
    for k in range(n):
        start = (k // 2) * 5.0 + rng.uniform(0, 1)
        times = np.round(start + np.arange(0, rng.uniform(2, 4), 0.1), 1)
        look = np.clip(colours[k % 2] + rng.normal(0, 0.08, 12), 0, None)
        out.append(
            TrackInfo(
                video=1,
                id=k,
                start=float(times[0]),
                end=float(times[-1]),
                samples=len(times),
                appearance=look,
                times=times,
            )
        )
    return out


def _cluster_slowly(tracks: list[TrackInfo]) -> list[Group]:
    """The plain version of the same rule: every pair of groups, again after every join."""
    groups = [Group([t]) for t in tracks]
    while True:
        best: tuple[float, int, int] | None = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = distance(groups[i].appearance, groups[j].appearance)
                if d > MERGE_MAX_DISTANCE or (best is not None and d >= best[0]):
                    continue
                if groups[i].overlaps(groups[j]):
                    continue
                best = (d, i, j)
        if best is None:
            return groups
        _, i, j = best
        groups[i] = Group(groups[i].tracks + groups[j].tracks)
        del groups[j]


def test_cluster_joins_as_the_plain_rule_does() -> None:
    for seed in range(4):
        tracks = _tracks(30, seed)
        fast = [[t.id for t in g.tracks] for g in cluster(tracks)]
        slow = [[t.id for t in g.tracks] for g in _cluster_slowly(tracks)]
        assert fast == slow
        assert len(fast) < 30  # it did join some


def test_tracks_seen_together_stay_apart() -> None:
    groups = cluster(_tracks(40, 9))
    for g in groups:
        for a in g.tracks:
            for b in g.tracks:
                assert a is b or not Group([a]).overlaps(Group([b]))


def test_cluster_copes_with_a_long_match() -> None:
    tracks = _tracks(1200, 3)
    started = time.monotonic()
    groups = cluster(tracks)
    assert time.monotonic() - started < 60
    assert sum(len(g.tracks) for g in groups) == 1200
    assert cluster([]) == [] and len(cluster(tracks[:1])) == 1
