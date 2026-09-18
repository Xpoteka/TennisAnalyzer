"""Was a session a match or training? Decided from how the rallies are structured.

In a match:

- almost every rally starts with a serve (an overhead first shot);
- the same player serves several points in a row (a game), then the other;
- there are pauses of 10 to 30 seconds between points;
- rallies are short (most are under six shots).

In training, balls are fed or hit back and forth without serves, drills restart after a
few seconds, and cooperative baseline rallies go on for a long time. Serve practice has
serves without the server alternating by games.

Each observation shifts a score; the score becomes a probability. Sessions with very few
rallies stay undecided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RallyInfo:
    start: float
    end: float
    shots: int
    first_is_serve: bool
    server: int | None  # player index, when the first shot is a serve


@dataclass
class KindResult:
    kind: str  # match | training | unknown
    p_match: float
    evidence: dict[str, float]


MIN_RALLIES = 6


def serve_runs(servers: list[int]) -> list[int]:
    """Lengths of runs of consecutive rallies served by the same player."""
    runs: list[int] = []
    previous: int | None = None
    for s in servers:
        if runs and s == previous:
            runs[-1] += 1
        else:
            runs.append(1)
        previous = s
    return runs


MIN_SERVES = 6


def classify_session(rallies: list[RallyInfo]) -> KindResult:
    """Balls hit back between points form short rallies without a serve in a match too, so
    the decision rests on the rallies that start with a serve: who serves them in what
    order, and how far apart they are."""
    if len(rallies) < MIN_RALLIES:
        return KindResult("unknown", 0.5, {"rallies": float(len(rallies))})
    served = [r for r in rallies if r.first_is_serve]
    serve_share = len(served) / len(rallies)
    serve_gaps = np.diff([r.start for r in served]) if len(served) >= 2 else np.zeros(0)
    serve_gap = float(np.median(serve_gaps)) if len(serve_gaps) else 0.0
    long_share = float(np.mean([r.shots >= 12 for r in rallies]))
    servers = [r.server for r in served if r.server is not None]
    runs = serve_runs(servers)
    # A game is four or more points served by the same player, then the other one serves.
    game_like = float(np.mean([3 <= n <= 16 for n in runs])) if len(runs) >= 2 else 0.0

    z = -1.0
    if len(served) >= MIN_SERVES:
        z += 3.0 * (game_like - 0.3)
        z += 1.5 if 12.0 <= serve_gap <= 90.0 else -1.0
        z += 2.0 * (min(serve_share, 0.6) - 0.2)
    z -= 3.0 * max(0.0, long_share - 0.25)
    p = 1 / (1 + math.exp(-z))
    kind = "match" if p >= 0.6 else "training" if p <= 0.4 else "unknown"
    return KindResult(
        kind,
        round(p, 3),
        {
            "rallies": float(len(rallies)),
            "serves": float(len(served)),
            "serve_share": round(serve_share, 3),
            "serve_gap_s": round(serve_gap, 1),
            "long_rally_share": round(long_share, 3),
            "game_like_serve_runs": round(game_like, 3),
        },
    )
