"""Tell the two players apart across a session, including after they change ends.

Each pose window has a near-court and a far-court player. Their clothing descriptors
(``tennis.util.appearance``) are matched to two identities, A and B, with the constraint
that the near and far player of a window are different people. A is the near player of the
first window that has one. Players only change ends between games, so the per-window
assignment is smoothed with a majority vote over neighbouring windows, and stretches
shorter than ``min_side_duration_s`` between two longer ones are treated as mix-ups.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tennis.util.appearance import distance

Descriptor = npt.NDArray[np.float32]
A, B = 0, 1
NAMES = ("A", "B")


@dataclass(frozen=True)
class WindowLooks:
    window_id: int
    start: float
    near: Descriptor | None
    far: Descriptor | None


@dataclass(frozen=True)
class Assignment:
    near_identity: dict[int, int]  # window_id -> A or B
    margin: dict[int, float]  # cost difference behind each window's choice (before smoothing)
    prototypes: tuple[Descriptor | None, Descriptor | None]
    resolved: bool  # False when only one player was ever seen


def summarize(descriptors: npt.NDArray[np.float32]) -> Descriptor | None:
    """Median descriptor of the frames where a player was seen (all-zero rows are skipped)."""
    seen = descriptors[descriptors.sum(axis=1) > 0]
    if seen.shape[0] == 0:
        return None
    return np.asarray(np.median(seen, axis=0), np.float32)


def _cost(desc: Descriptor | None, proto: Descriptor | None) -> float:
    if desc is None or proto is None:
        return 0.0
    return distance(desc, proto)


def _merge_short_runs(values: list[int], starts: list[float], min_duration: float) -> list[int]:
    """Flip interior runs that last less than ``min_duration`` (start to next run's start).

    The first and last run are kept: the session may start or end just after a changeover.
    """
    values = list(values)
    while True:
        runs: list[tuple[int, int]] = []  # [first, last] indices
        for k, v in enumerate(values):
            if runs and values[runs[-1][0]] == v:
                runs[-1] = (runs[-1][0], k)
            else:
                runs.append((k, k))
        if len(runs) <= 2:
            return values

        def duration(r: tuple[int, int]) -> float:
            return starts[r[1] + 1] - starts[r[0]]

        shortest = min(range(1, len(runs) - 1), key=lambda i: duration(runs[i]))
        if duration(runs[shortest]) >= min_duration:
            return values
        first, last = runs[shortest]
        for k in range(first, last + 1):
            values[k] = 1 - values[k]


def assign_identities(
    looks: Sequence[WindowLooks],
    iterations: int = 10,
    smooth: int = 2,
    min_side_duration_s: float = 0.0,
) -> Assignment:
    ordered = sorted(looks, key=lambda w: w.start)
    first_near = next((w.near for w in ordered if w.near is not None), None)
    first_far = next((w for w in ordered if w.far is not None), None)
    if first_near is None or first_far is None:
        return Assignment({w.window_id: A for w in ordered}, {w.window_id: 0.0 for w in ordered},
                          (first_near, None), resolved=False)  # fmt: skip
    # B starts from a far player that does not look like A (the first one might be A,
    # e.g. if A walked to the far side before the first shot).
    far_list = [w.far for w in ordered if w.far is not None]
    proto_a: Descriptor = first_near
    proto_b: Descriptor = max(far_list, key=lambda d: distance(d, proto_a))

    choice: dict[int, int] = {}
    margin: dict[int, float] = {}
    for _ in range(iterations):
        new: dict[int, int] = {}
        for w in ordered:
            keep = _cost(w.near, proto_a) + _cost(w.far, proto_b)
            swap = _cost(w.near, proto_b) + _cost(w.far, proto_a)
            new[w.window_id] = A if keep <= swap else B
            margin[w.window_id] = abs(keep - swap)
        members: dict[int, list[Descriptor]] = {A: [], B: []}
        for w in ordered:
            near_id = new[w.window_id]
            if w.near is not None:
                members[near_id].append(w.near)
            if w.far is not None:
                members[1 - near_id].append(w.far)
        if members[A]:
            proto_a = np.asarray(np.median(np.stack(members[A]), axis=0), np.float32)
        if members[B]:
            proto_b = np.asarray(np.median(np.stack(members[B]), axis=0), np.float32)
        if new == choice:
            break
        choice = new

    ids = [w.window_id for w in ordered]
    raw = np.array([choice[i] for i in ids])
    smoothed = raw.copy()
    for k in range(len(ids)):
        lo, hi = max(0, k - smooth), min(len(ids), k + smooth + 1)
        votes = raw[lo:hi]
        if (votes == raw[k]).sum() * 2 < votes.size:  # clearly outvoted by its neighbours
            smoothed[k] = 1 - raw[k]
    starts = [w.start for w in ordered]
    final = _merge_short_runs([int(v) for v in smoothed], starts, min_side_duration_s)
    return Assignment(
        near_identity=dict(zip(ids, final, strict=True)),
        margin=margin,
        prototypes=(proto_a, proto_b),
        resolved=True,
    )


def side_segments(
    assignment: Mapping[int, int], starts: Mapping[int, float], ends: Mapping[int, float], me: int
) -> list[dict[str, object]]:
    """Merge consecutive windows into (start, end, side of `me`) segments."""
    segments: list[dict[str, object]] = []
    for wid in sorted(assignment, key=lambda w: starts[w]):
        side = "near" if assignment[wid] == me else "far"
        if segments and segments[-1]["side"] == side:
            segments[-1]["end"] = round(ends[wid], 3)
            segments[-1]["windows"] = int(segments[-1]["windows"]) + 1  # type: ignore[call-overload]
        else:
            segments.append({"start": round(starts[wid], 3), "end": round(ends[wid], 3),
                             "side": side, "windows": 1})  # fmt: skip
    return segments
