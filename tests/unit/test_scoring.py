"""Score keeping from noisy point winners."""

from __future__ import annotations

import numpy as np

from tennis.analysis.scoring import Point, keep_score, point_score_text


class Match:
    """A best-of-three match played by the rules, observed with noise."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        noise: float = 0.2,
        wrong_server: float = 0.0,
        wrong_end: float = 0.0,
        wrong_box: float = 0.0,
        spurious: float = 0.0,
        missed: float = 0.0,
    ) -> None:
        self.rng = rng
        self.noise = noise
        self.wrong_server, self.wrong_end, self.wrong_box = wrong_server, wrong_end, wrong_box
        self.spurious, self.missed = spurious, missed
        self.points: list[Point] = []
        self.truth: list[int | None] = []  # per observation; None: not a point
        self.sets: list[tuple[int, int]] = []
        self.games: list[tuple[int, int, int]] = []  # server, winner, points
        self.end0 = -1
        self.server = 0
        self.play()

    def observe(self, srv: int, win: int, k: int) -> None:
        rng = self.rng
        p = 0.5 + (0.5 - self.noise) * (1 if win == srv else -1)
        p = float(np.clip(p + rng.normal(0, 0.15), 0.02, 0.98))
        obs_srv = srv if rng.random() >= self.wrong_server else 1 - srv
        end = self.end0 if srv == 0 else -self.end0
        if rng.random() < self.wrong_end:
            end = -end
        box = 1 if k % 2 == 0 else -1
        if rng.random() < self.wrong_box:
            box = -box
        if rng.random() < self.spurious:  # a fault counted as a point: same serve again
            self.points.append(Point(obs_srv, float(rng.uniform(0.2, 0.8)), end=end, box=box))
            self.truth.append(None)
        if rng.random() < self.missed:
            return
        self.points.append(Point(obs_srv, p, end=end, box=box))
        self.truth.append(win)

    def play(self) -> None:
        rng = self.rng
        won = [0, 0]
        while max(won) < 2:
            games = [0, 0]
            while True:
                if games == [6, 6]:
                    pts = [0, 0]
                    first = self.server
                    k = 0
                    while not (max(pts) >= 7 and abs(pts[0] - pts[1]) >= 2):
                        srv = first if k == 0 or (k + 1) // 2 % 2 == 0 else 1 - first
                        win = srv if rng.random() < 0.6 else 1 - srv
                        self.observe(srv, win, k)
                        pts[win] += 1
                        k += 1
                        if k % 6 == 0:
                            self.end0 = -self.end0
                    games[0 if pts[0] > pts[1] else 1] += 1
                    self.games.append((first, 0 if pts[0] > pts[1] else 1, k))
                    self.server = 1 - first
                    self.end0 = -self.end0  # thirteen games
                    break
                pts = [0, 0]
                k = 0
                while not (max(pts) >= 4 and abs(pts[0] - pts[1]) >= 2):
                    win = self.server if rng.random() < 0.62 else 1 - self.server
                    self.observe(self.server, win, k)
                    pts[win] += 1
                    k += 1
                games[0 if pts[0] > pts[1] else 1] += 1
                self.games.append((self.server, 0 if pts[0] > pts[1] else 1, k))
                self.server = 1 - self.server
                if (games[0] + games[1]) % 2 == 1:
                    self.end0 = -self.end0
                if max(games) >= 6 and abs(games[0] - games[1]) >= 2:
                    break
            self.sets.append((games[0], games[1]))
            won[0 if games[0] > games[1] else 1] += 1


def accuracy(found: list[int | None], truth: list[int | None]) -> float:
    real = [(f, t) for f, t in zip(found, truth, strict=True) if t is not None]
    return sum(f == t for f, t in real) / len(real)


def test_decodes_a_clean_match() -> None:
    m = Match(np.random.default_rng(3))
    score = keep_score(m.points)
    assert score.sets == m.sets
    assert accuracy(score.point_winners, m.truth) > 0.9
    assert len(score.before) == len(m.points)
    assert score.before[0] == "0-0, 0-0"


def test_decodes_a_noisy_match() -> None:
    """Spurious points and wrong serve cues are absorbed by the rules. A missed point can
    still cost a game now and then: nothing in the rhythm of a match says who won a game,
    only how many were played."""
    for seed in (1, 2, 3):
        m = Match(
            np.random.default_rng(seed),
            noise=0.25,
            wrong_server=0.08,
            wrong_end=0.03,
            wrong_box=0.2,
            spurious=0.15,
            missed=0.05,
        )
        score = keep_score(m.points)
        assert len(score.games) == len(m.games), seed
        right = sum(f.winner == t[1] for f, t in zip(score.games, m.games, strict=True))
        assert right >= len(m.games) - 1, seed
        assert accuracy(score.point_winners, m.truth) > 0.85, seed
    # Faults counted as points do not change the number of games. The last game of a
    # video may end unfinished: nothing after it says how it went.
    m = Match(np.random.default_rng(4), noise=0.25, wrong_server=0.1, wrong_box=0.25, spurious=0.2)
    score = keep_score(m.points)
    assert len(score.games) == len(m.games)
    finished = [(f, t) for f, t in zip(score.games, m.games, strict=True) if f.winner is not None]
    assert sum(f.winner == t[1] for f, t in finished) >= len(finished) - 2


def test_tiebreak_found_from_the_rhythm_of_serves_and_ends() -> None:
    """A set at 6-5: the last game's points look like a coin toss, but what follows has
    the serve changing every two points and the ends every six. That is a tiebreak, so
    the doubtful game must have gone to the player who made it 6-6."""
    rng = np.random.default_rng(0)
    m = Match(rng)
    points: list[Point] = []
    truth: list[int] = []
    end0, server = -1, 0
    games = [0, 0]
    for g in range(12):  # alternate games to 6-6, four straight points each
        win = g % 2
        for k in range(4):
            p = 0.5 if g == 11 else (0.85 if win == server else 0.15)
            points.append(
                Point(server, p, end=end0 if server == 0 else -end0, box=1 if k % 2 == 0 else -1)
            )
            truth.append(win)
        games[win] += 1
        server = 1 - server
        if (games[0] + games[1]) % 2 == 1:
            end0 = -end0
    first, k, pts = server, 0, [0, 0]
    while not (max(pts) >= 7 and abs(pts[0] - pts[1]) >= 2):
        srv = first if k == 0 or (k + 1) // 2 % 2 == 0 else 1 - first
        win = 1 if k % 3 else 0  # 7-3 to player 1
        points.append(
            Point(
                srv,
                0.85 if win == srv else 0.15,
                end=end0 if srv == 0 else -end0,
                box=1 if k % 2 == 0 else -1,
            )
        )
        truth.append(win)
        pts[win] += 1
        k += 1
        if k % 6 == 0:
            end0 = -end0
    del m, rng
    score = keep_score(points)
    assert score.sets == [(6, 7)]
    assert score.games[-1].tiebreak
    assert score.point_winners == truth


def test_empty_and_text() -> None:
    assert keep_score([]).text() == "0-0"
    score = keep_score([Point(0, 0.9, end=-1, box=1), Point(0, 0.9, end=-1, box=-1)])
    assert score.before == ["0-0, 0-0", "0-0, 15-0"]
    assert score.point_winners == [0, 0]


def test_point_score_text() -> None:
    assert point_score_text(2, 1) == "30-15"
    assert point_score_text(3, 3) == "deuce"
    assert point_score_text(4, 3) == "advantage server"
    assert point_score_text(5, 3, tiebreak=True) == "5-3"
