"""Score keeping from noisy point winners."""

from __future__ import annotations

import numpy as np

from tennis.analysis.scoring import Point, best_game, keep_score, point_score_text, split_games


def test_best_game_prefers_a_valid_game() -> None:
    # 4 points, the server seems to win 3 of them and the receiver 1: a server game needs
    # 4 points won, so one of the uncertain points must go to the server... unless the
    # game is unfinished.
    probs = [0.9, 0.9, 0.45, 0.9]
    assert best_game(probs, must_finish=True) == [0, 0, 0, 0]
    assert best_game(probs, must_finish=False) == [0, 0, 1, 0]


def test_deuce_game() -> None:
    probs = [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.9]  # 40-40, advantage, game
    assert best_game(probs, must_finish=True) == [0, 1, 0, 1, 0, 1, 0, 0]


def test_split_games_by_server_runs() -> None:
    servers = [0] * 4 + [1] * 6 + [0] * 5
    pts = [Point(s, 0.6) for s in servers]
    assert [len(g) for g in split_games(pts)] == [4, 6, 5]
    # One wrongly detected server inside a run is folded in.
    servers[6] = 0
    pts = [Point(s, 0.6) for s in servers]
    assert [len(g) for g in split_games(pts)] == [4, 6, 5]


def simulate_match(rng: np.random.Generator, noise: float) -> tuple[list[Point], list[int]]:
    """Points of a best-of-three match with true winners, and noisy observations."""
    points: list[Point] = []
    truth: list[int] = []
    server = 0
    sets = [0, 0]
    while max(sets) < 2:
        games = [0, 0]
        while not ((max(games) >= 6 and abs(games[0] - games[1]) >= 2) or max(games) == 7):
            if games == [6, 6]:
                pts = [0, 0]
                first = server
                k = 0
                while not (max(pts) >= 7 and abs(pts[0] - pts[1]) >= 2):
                    srv = first if (k + 1) // 2 % 2 == 0 else 1 - first
                    win = srv if rng.random() < 0.6 else 1 - srv
                    pts[win] += 1
                    p = 0.5 + (0.5 - noise) * (1 if win == srv else -1)
                    points.append(Point(srv, p))
                    truth.append(win)
                    k += 1
                games[0 if pts[0] > pts[1] else 1] += 1
                server = 1 - first
                break
            pts = [0, 0]
            while not (max(pts) >= 4 and abs(pts[0] - pts[1]) >= 2):
                win = server if rng.random() < 0.62 else 1 - server
                pts[win] += 1
                # The observation points the right way, with some noise.
                p = 0.5 + (0.5 - noise) * (1 if win == server else -1)
                p = float(np.clip(p + rng.normal(0, 0.15), 0.02, 0.98))
                points.append(Point(server, p))
                truth.append(win)
            games[0 if pts[0] > pts[1] else 1] += 1
            server = 1 - server
        sets[0 if games[0] > games[1] else 1] += 1
    return points, truth


def test_decodes_a_simulated_match() -> None:
    rng = np.random.default_rng(3)
    points, truth = simulate_match(rng, noise=0.2)
    score = keep_score(points)
    found = np.array(score.point_winners)
    assert (found == np.array(truth)).mean() > 0.85
    assert len(score.sets) >= 2


def test_point_score_text() -> None:
    assert point_score_text(2, 1) == "30-15"
    assert point_score_text(3, 3) == "deuce"
    assert point_score_text(4, 3) == "advantage server"
    assert point_score_text(5, 3, tiebreak=True) == "5-3"
