"""Keep score from a list of points whose winners are only probably known.

What is observed per point: who served (from the serve at the start of the rally), and a
probability that the server won it (from how the rally ended: out, into the net, or not
returned). Both are noisy.

Tennis structure fixes much of the noise:

1. **Games.** A player serves a whole game, so runs of points with the same server are
   games. Within a game, the winners must form a valid game: first to four points with a
   two-point lead. A dynamic program picks the most likely valid sequence of winners.
   A game that never finished (the video stopped) is allowed only at the end.
2. **Tiebreaks.** At six games all, serve changes after the first point and then every two
   points; the first to seven with a two-point lead wins.
3. **Sets.** The first to six games with a two-game lead, or seven after a tiebreak.

The result is a list of games (server, winner, point scores) and the running score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

POINT_NAMES = ("0", "15", "30", "40")


@dataclass
class Point:
    server: int  # 0 or 1 (player index)
    p_server_wins: float  # 0..1
    rally: int | None = None  # the rally index this point came from


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

    def text(self) -> str:
        parts = [f"{a}-{b}" for a, b in self.sets]
        if self.current != (0, 0) or not parts:
            parts.append(f"{self.current[0]}-{self.current[1]}")
        return " ".join(parts)


def _log(p: float) -> float:
    return math.log(min(max(p, 1e-6), 1 - 1e-6))


def _game_over(a: int, b: int, target: int) -> bool:
    return max(a, b) >= target and abs(a - b) >= 2


def best_game(p_server: list[float], *, must_finish: bool, target: int = 4) -> list[int]:
    """Most likely winners (0 = server, 1 = receiver) for one game's points.

    With ``must_finish``, the sequence must end exactly when the game is won. ``target`` is 4
    for a game and 7 for a tiebreak (where "server" means the player serving first).
    """
    # State: (server points, receiver points) -> (log-likelihood, path).
    states: dict[tuple[int, int], tuple[float, list[int]]] = {(0, 0): (0.0, [])}
    for p in p_server:
        nxt: dict[tuple[int, int], tuple[float, list[int]]] = {}
        for (a, b), (ll, path) in states.items():
            if _game_over(a, b, target):
                continue  # the game ended before this point: not allowed
            for who, lp in ((0, _log(p)), (1, _log(1 - p))):
                key = (a + 1, b) if who == 0 else (a, b + 1)
                # Scores past deuce collapse: 4-4 is deuce again, like 3-3.
                while min(key) >= target:
                    key = (key[0] - 1, key[1] - 1)
                cand = (ll + lp, [*path, who])
                if key not in nxt or cand[0] > nxt[key][0]:
                    nxt[key] = cand
        states = nxt
        if not states:
            break
    finished = {k: v for k, v in states.items() if _game_over(*k, target)}
    pool = finished if (must_finish and finished) else states
    if not pool:
        return [0 if p >= 0.5 else 1 for p in p_server]
    return max(pool.values(), key=lambda v: v[0])[1]


def split_games(points: list[Point]) -> list[list[int]]:
    """Group point indices into games by runs of the same server.

    A lone point by the other player inside a long run is more likely a missed or wrong
    serve detection than a one-point game, so it is folded into the run around it.
    """
    if not points:
        return []
    servers = [p.server for p in points]
    for i in range(1, len(servers) - 1):
        if servers[i - 1] == servers[i + 1] != servers[i]:
            servers[i] = servers[i - 1]
    runs: list[list[int]] = [[0]]
    for i in range(1, len(servers)):
        if servers[i] == servers[i - 1]:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def keep_score(points: list[Point]) -> Score:
    """Decode the whole match: games, tiebreaks and sets."""
    score = Score(point_winners=[None] * len(points))
    runs = split_games(points)
    games_a = games_b = 0
    k = 0
    while k < len(runs):
        run = runs[k]
        last_run = k == len(runs) - 1
        server = points[run[0]].server
        if games_a == 6 and games_b == 6:
            # Tiebreak: server changes after one point, then every two, so the runs are
            # short and alternating. Take points until someone has seven by two.
            idx = [i for r in runs[k:] for i in r]
            first = points[idx[0]].server
            probs = [
                points[i].p_server_wins
                if points[i].server == first
                else 1 - points[i].p_server_wins
                for i in idx
            ]
            winners = best_game(probs, must_finish=False, target=7)
            a = b = 0
            used = 0
            for w in winners:
                used += 1
                a, b = (a + 1, b) if w == 0 else (a, b + 1)
                if _game_over(a, b, 7):
                    break
            taken = idx[:used]
            game_winners = [first if w == 0 else 1 - first for w in winners[:used]]
            done = _game_over(a, b, 7)
            winner = (first if a > b else 1 - first) if done else None
            score.games.append(Game(first, winner, game_winners, tiebreak=True))
            for i, w in zip(taken, game_winners, strict=True):
                score.point_winners[i] = w
            if done:
                games_a, games_b = (7, 6) if winner == 0 else (6, 7)
                score.sets.append((games_a, games_b))
                games_a = games_b = 0
            # Skip the runs the tiebreak consumed.
            consumed = set(taken)
            while k < len(runs) and all(i in consumed for i in runs[k]):
                k += 1
            if k < len(runs) and any(i in consumed for i in runs[k]):
                runs[k] = [i for i in runs[k] if i not in consumed]
            continue
        probs = [points[i].p_server_wins for i in run]
        winners = best_game(probs, must_finish=not last_run)
        a = sum(1 for w in winners if w == 0)
        b = len(winners) - a
        finished = _game_over(a, b, 4) or (not last_run)
        game_winner = (server if winners[-1] == 0 else 1 - server) if finished else None
        point_winners = [server if w == 0 else 1 - server for w in winners]
        score.games.append(Game(server, game_winner, point_winners))
        for i, w in zip(run, point_winners, strict=True):
            score.point_winners[i] = w
        if game_winner is not None:
            if game_winner == 0:
                games_a += 1
            else:
                games_b += 1
            if (max(games_a, games_b) >= 6 and abs(games_a - games_b) >= 2) or max(
                games_a, games_b
            ) == 7:
                score.sets.append((games_a, games_b))
                games_a = games_b = 0
        k += 1
    score.current = (games_a, games_b)
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
