"""Keep score from a list of points whose winners are only probably known.

What is observed per point, all of it noisy: who served, from which end of the court and
which service box (from the serve at the start of the rally), and a probability that the
server won the point (from how the rally ended: out, into the net, or not returned).

The rules of tennis fix much of the noise, so the whole match is decoded at once. The
hidden state is the score together with who serves and who stands at which end; every
observation is checked against what the rules require from that state:

1. **Games.** First to four points with a two-point lead. A player serves a whole game;
   games alternate the server. Serves come from the deuce court when the points of the
   game add up to an even number, from the ad court when odd.
2. **Tiebreaks.** At six games all: first to seven points with a two-point lead. The serve
   changes after the first point and then every two points; ends change every six points.
3. **Sets.** First to six games with a two-game lead, or seven after a tiebreak. Players
   change ends after the first game of a set and every two games from then on, so also
   when a set ends on an odd number of games.
4. **Match.** Best of three sets.

A detected point may not have been one (a fault counted as a point, a ball hit back
between points), and a point may have been missed; both are allowed at a cost. The most
likely path through the rules is found with a beam search over the points (Viterbi).

The result is a list of games (server, winner, point winners), the sets, and for every
detected point its winner (or None when it was not a point) and the score before it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache

POINT_NAMES = ("0", "15", "30", "40")

# Observation noise: how often a serve is credited to the wrong player, seen from the
# wrong end, or from the wrong service box (near the centre mark, or a second serve).
E_SERVER = 0.1
E_END = 0.05
E_BOX = 0.25
P_SPURIOUS = 0.1  # a detected point that was none (a fault, a ball hit back)
P_MISSED = 0.05  # a point the detector did not see, before any detected point
BEAM = 1500  # hypotheses kept per point

# State: sets (s0, s1), games in the set (g0, g1), points in the game (p0, p1),
# who serves the first point of the game, and player 0's end (-1 near, +1 far).
State = tuple[int, int, int, int, int, int, int, int]


@dataclass
class Point:
    server: int  # 0 or 1 (player index)
    p_server_wins: float  # 0..1
    rally: int | None = None  # the rally index this point came from
    end: int = 0  # the server's end of the court: -1 near, +1 far, 0 unknown
    box: int = 0  # the service box: +1 deuce, -1 ad, 0 unknown


@dataclass
class Game:
    server: int
    winner: int | None  # None: unfinished
    points: list[int]  # winner of each point (player index)
    tiebreak: bool = False


@dataclass
class Score:
    games: list[Game] = field(default_factory=list)
    sets: list[tuple[int, int]] = field(default_factory=list)  # finished sets
    current: tuple[int, int] = (0, 0)  # games in the set being played
    point_winners: list[int | None] = field(default_factory=list)  # per input point
    before: list[str] = field(default_factory=list)  # score text before each input point

    def text(self) -> str:
        parts = [f"{a}-{b}" for a, b in self.sets]
        if self.current != (0, 0) or not parts:
            parts.append(f"{self.current[0]}-{self.current[1]}")
        return " ".join(parts)


def _log(p: float) -> float:
    return math.log(min(max(p, 1e-6), 1 - 1e-6))


LOG_HALF = _log(0.5)
LOG_SPURIOUS, LOG_REAL = _log(P_SPURIOUS), _log(1 - P_SPURIOUS)
LOG_MISSED, LOG_NOT_MISSED = _log(P_MISSED), _log(1 - P_MISSED)
LOG_SERVER = (_log(E_SERVER), _log(1 - E_SERVER))
LOG_END = (_log(E_END), _log(1 - E_END))
LOG_BOX = (_log(E_BOX), _log(1 - E_BOX))


# --- The rules ---------------------------------------------------------------------------


def _tiebreak(st: State) -> bool:
    return st[2] == 6 and st[3] == 6


def _done(st: State) -> bool:
    return st[0] == 2 or st[1] == 2


def _server(st: State) -> int:
    """Who serves the next point."""
    first = st[6]
    if _tiebreak(st):
        k = st[4] + st[5]
        return first if k == 0 or ((k + 1) // 2) % 2 == 0 else 1 - first
    return first


@cache
def _advance(st: State, winner: int) -> tuple[State, int | None, tuple[int, int] | None]:
    """The state after ``winner`` takes a point: (state, game winner, finished set)."""
    s0, s1, g0, g1, p0, p1, first, end0 = st
    p = [p0, p1]
    p[winner] += 1
    g = [g0, g1]
    s = [s0, s1]
    if g0 == 6 and g1 == 6:
        if max(p) >= 7 and abs(p[0] - p[1]) >= 2:
            g[winner] += 1
            s[winner] += 1
            # Whoever served first in the tiebreak receives first in the next set, and
            # thirteen games is odd: the players change ends.
            return (s[0], s[1], 0, 0, 0, 0, 1 - first, -end0), winner, (g[0], g[1])
        if min(p) >= 12:  # a long tiebreak: keep the score small, the rhythm intact
            p = [p[0] - 6, p[1] - 6]
        if (p[0] + p[1]) % 6 == 0:
            end0 = -end0
        return (s0, s1, g0, g1, p[0], p[1], first, end0), None, None
    if max(p) >= 4 and abs(p[0] - p[1]) >= 2:
        g[winner] += 1
        if (g[0] + g[1]) % 2 == 1:
            end0 = -end0
        finished = None
        if max(g) >= 6 and abs(g[0] - g[1]) >= 2:
            s[winner] += 1
            finished = (g[0], g[1])
            g = [0, 0]
        return (s[0], s[1], g[0], g[1], 0, 0, 1 - first, end0), winner, finished
    while min(p) >= 4:  # past deuce the score collapses: 4-4 is deuce again
        p = [p[0] - 1, p[1] - 1]
    return (s0, s1, g0, g1, p[0], p[1], first, end0), None, None


# --- The observations --------------------------------------------------------------------


def _cues(st: State, obs: Point) -> float:
    """Log-likelihood of the serve cues of ``obs`` (server, end, box) from ``st``."""
    srv = _server(st)
    ll = LOG_SERVER[obs.server == srv]
    if obs.end:
        ll += LOG_END[obs.end == (st[7] if srv == 0 else -st[7])]
    if obs.box:
        ll += LOG_BOX[(obs.box > 0) == ((st[4] + st[5]) % 2 == 0)]
    return ll


def _real(st: State, winner: int, obs: Point) -> float:
    """Log-likelihood of ``obs`` being a real point from ``st`` won by ``winner``."""
    p = obs.p_server_wins if winner == _server(st) else 1 - obs.p_server_wins
    return _log(p) + _cues(st, obs)


def _spurious(st: State, obs: Point) -> float:
    """Log-likelihood of ``obs`` when it was not a point.

    Most such detections are a fault's second serve, or a ball served again after a let:
    the same player from the same end and box, so the serve cues fit the current state as
    well as a real point's would, and only the winner is a coin toss.
    """
    return LOG_HALF + _cues(st, obs)


# --- The decoder -------------------------------------------------------------------------

# For each state reached after a point: (where it came from, a missed point's winner
# before this point or None, this point's winner or None when it was spurious).
_Back = dict[State, tuple[State, int | None, int | None]]


def keep_score(points: list[Point]) -> Score:
    """Decode the whole match: games, tiebreaks and sets."""
    starts: list[State] = [(0, 0, 0, 0, 0, 0, srv, end) for srv in (0, 1) for end in (-1, 1)]
    states: dict[State, float] = dict.fromkeys(starts, 0.0)
    history: list[_Back] = []
    for obs in points:
        # A point the detector may have missed just before this one.
        pre: dict[State, tuple[float, State, int | None]] = {}
        for st, ll in states.items():
            pre[st] = (ll + LOG_NOT_MISSED, st, None)
            if _done(st):
                continue
            for w in (0, 1):
                st2 = _advance(st, w)[0]
                cand = ll + LOG_MISSED + LOG_HALF
                if st2 not in pre or cand > pre[st2][0]:
                    pre[st2] = (cand, st, w)
        nxt: dict[State, tuple[float, State, int | None, int | None]] = {}
        for st, (ll, origin, hidden) in pre.items():
            cand = ll + LOG_SPURIOUS + _spurious(st, obs)
            if st not in nxt or cand > nxt[st][0]:
                nxt[st] = (cand, origin, hidden, None)
            if _done(st):
                continue
            for w in (0, 1):
                st2 = _advance(st, w)[0]
                cand = ll + LOG_REAL + _real(st, w, obs)
                if st2 not in nxt or cand > nxt[st2][0]:
                    nxt[st2] = (cand, origin, hidden, w)
        if len(nxt) > BEAM:
            nxt = dict(sorted(nxt.items(), key=lambda kv: -kv[1][0])[:BEAM])
        history.append({st: (v[1], v[2], v[3]) for st, v in nxt.items()})
        states = {st: v[0] for st, v in nxt.items()}
    if not points:
        return Score()
    best = max(states, key=lambda k: states[k])
    actions: list[tuple[int | None, int | None]] = []
    for back in reversed(history):
        best, hidden, won = back[best]
        actions.append((hidden, won))
    actions.reverse()
    return _replay(best, actions)


def _text(sets: list[tuple[int, int]], st: State) -> str:
    first = st[6]
    p = (st[4], st[5])
    parts = [f"{a}-{b}" for a, b in sets]
    parts.append(f"{st[2]}-{st[3]}, {point_score_text(p[first], p[1 - first], _tiebreak(st))}")
    return " ".join(parts)


def _replay(start: State, actions: list[tuple[int | None, int | None]]) -> Score:
    """Walk the decoded path again to build the games, the sets and the running score."""
    score = Score()
    st = start
    game: list[int] = []
    game_first = st[6]
    game_tiebreak = False

    def take(w: int) -> None:
        nonlocal st, game, game_first, game_tiebreak
        if not game:
            game_first, game_tiebreak = st[6], _tiebreak(st)
        game.append(w)
        st, game_winner, finished = _advance(st, w)
        if game_winner is not None:
            score.games.append(Game(game_first, game_winner, game, tiebreak=game_tiebreak))
            game = []
        if finished is not None:
            score.sets.append(finished)

    for hidden, w in actions:
        if hidden is not None:
            take(hidden)
        score.before.append(_text(score.sets, st))
        score.point_winners.append(w)
        if w is not None:
            take(w)
    if game:
        score.games.append(Game(game_first, None, game, tiebreak=game_tiebreak))
    score.current = (st[2], st[3])
    return score


def point_score_text(a: int, b: int, tiebreak: bool = False) -> str:
    """The server-first point score of a game in progress: 15-30, deuce, advantage."""
    if tiebreak:
        return f"{a}-{b}"
    if a >= 3 and b >= 3:
        if a == b:
            return "deuce"
        return "advantage server" if a > b else "advantage receiver"
    return f"{POINT_NAMES[min(a, 3)]}-{POINT_NAMES[min(b, 3)]}"
