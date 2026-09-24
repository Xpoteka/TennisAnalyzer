"""Compare the analysis with hand-made labels (``tennis eval-hits``).

Label files are CSV with a time column ``t`` (seconds or ``m:ss``) and optional columns.
Coarse labels, such as Wingfield's whole-second shot log, give each hit a window rather
than an instant: a label ``t`` with ``--offset 1 --resolution 1`` means the hit is somewhere
in ``[t + 1, t + 2]`` seconds of video time.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def parse_time(value: str) -> float:
    parts = value.strip().split(":")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def read_labels(path: Path) -> list[dict[str, str]]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return list(csv.DictReader(lines))


@dataclass
class Match:
    label_index: int
    detection_index: int


def match_windows(starts: np.ndarray, ends: np.ndarray, detections: np.ndarray) -> list[Match]:
    """Pair each label window with at most one detection inside it (greedy, in time order)."""
    order = np.argsort(detections)
    used = np.zeros(len(detections), bool)
    matches = []
    for i in np.argsort(starts):
        for j in order:
            if used[j]:
                continue
            if starts[i] <= detections[j] <= ends[i]:
                used[j] = True
                matches.append(Match(int(i), int(j)))
                break
            if detections[j] > ends[i]:
                break
    return matches


@dataclass
class HitReport:
    labels: int
    detections: int  # inside the labelled span
    matched: int

    @property
    def recall(self) -> float:
        return self.matched / self.labels if self.labels else 0.0

    @property
    def precision(self) -> float:
        return self.matched / self.detections if self.detections else 0.0

    def text(self) -> str:
        return (
            f"labels {self.labels}, detections {self.detections}, matched {self.matched}: "
            f"recall {self.recall:.1%}, precision {self.precision:.1%}"
        )


def evaluate_hits(
    hit_times: np.ndarray,
    label_times: np.ndarray,
    *,
    offset: float,
    resolution: float,
    tolerance: float = 0.25,
) -> tuple[HitReport, list[Match]]:
    starts = label_times + offset - tolerance
    ends = label_times + offset + resolution + tolerance
    span = (hit_times >= starts.min() - 2) & (hit_times <= ends.max() + 2)
    dets = hit_times[span]
    matches = match_windows(starts, ends, dets)
    return HitReport(len(label_times), len(dets), len(matches)), matches


STROKE_GROUPS = {
    "serve": "serve",
    "forehand": "forehand",
    "backhand": "backhand",
    "volley_forehand": "volley",
    "volley_backhand": "volley",
    "volley": "volley",
    "overhead": "overhead",
    "unknown": "unknown",
}


@dataclass
class ShotReport:
    hits: HitReport
    player_accuracy: float | None
    player_mapping: dict[str, str]
    stroke_accuracy: float | None
    confusion: dict[str, dict[str, int]]

    def text(self) -> str:
        lines = ["hits: " + self.hits.text()]
        if self.player_accuracy is not None:
            mapping = ", ".join(f"{k}={v}" for k, v in sorted(self.player_mapping.items()))
            lines.append(f"hitter: {self.player_accuracy:.1%} right ({mapping})")
        if self.stroke_accuracy is not None:
            lines.append(f"stroke: {self.stroke_accuracy:.1%} right")
            kinds = sorted(
                {k for row in self.confusion.values() for k in row} | set(self.confusion)
            )
            lines.append("  label \\ found " + " ".join(f"{k[:8]:>9}" for k in kinds))
            for truth in sorted(self.confusion):
                row = self.confusion[truth]
                lines.append(
                    f"  {truth[:14]:<14} " + " ".join(f"{row.get(k, 0):>9}" for k in kinds)
                )
        return "\n".join(lines)


def evaluate_shots(
    shot_t: np.ndarray,
    shot_player: list[str | None],
    shot_stroke: list[str],
    labels: list[dict[str, str]],
    *,
    offset: float,
    resolution: float,
) -> ShotReport:
    """Labels need ``t``; ``player`` and ``stroke`` columns are compared when present."""
    label_t = np.array([parse_time(r["t"]) for r in labels])
    report, matches = evaluate_hits(shot_t, label_t, offset=offset, resolution=resolution)
    span = (shot_t >= label_t.min() + offset - 2.25) & (
        shot_t <= label_t.max() + offset + resolution + 2.25
    )
    idx = np.nonzero(span)[0]
    pairs = [(labels[m.label_index], int(idx[m.detection_index])) for m in matches]

    mapping: dict[str, str] = {}
    player_acc = None
    if pairs and "player" in labels[0]:
        votes: dict[str, dict[str, int]] = {}
        for lab, j in pairs:
            found = shot_player[j]
            if found is None:
                continue
            votes.setdefault(found, {}).setdefault(lab["player"], 0)
            votes[found][lab["player"]] += 1
        mapping = {found: max(v, key=lambda k: v[k]) for found, v in votes.items()}
        right = sum(1 for lab, j in pairs if mapping.get(shot_player[j] or "") == lab["player"])
        player_acc = right / len(pairs)

    stroke_acc = None
    confusion: dict[str, dict[str, int]] = {}
    if pairs and "stroke" in labels[0]:
        right = 0
        for lab, j in pairs:
            truth = STROKE_GROUPS.get(lab["stroke"], lab["stroke"])
            found = STROKE_GROUPS.get(shot_stroke[j], shot_stroke[j])
            confusion.setdefault(truth, {}).setdefault(found, 0)
            confusion[truth][found] += 1
            right += truth == found
        stroke_acc = right / len(pairs)
    return ShotReport(report, player_acc, mapping, stroke_acc, confusion)


def parse_score(text: str) -> tuple[int, int]:
    """Games won so far by each player from a score text like ``"7-6 4-3, 30-15"``."""
    games = text.split(",")[0].strip()
    a = b = 0
    for part in games.split():
        if "-" not in part:
            continue
        x, y = part.split("-", 1)
        try:
            a += int(x)
            b += int(y)
        except ValueError:
            continue
    return a, b


@dataclass
class GameReport:
    checkpoints: int
    exact: int  # checkpoints where both game counts were right
    error_games: float  # mean absolute error in games, summed over both players
    found_final: str
    truth_final: tuple[int, int]
    self_first: bool

    def text(self) -> str:
        return (
            f"games: {self.exact}/{self.checkpoints} checkpoints exact, "
            f"{self.error_games:.2f} games off on average; "
            f"found {self.found_final or '?'} (self {'first' if self.self_first else 'second'}), "
            f"scoreboard ends at self {self.truth_final[0]} - other {self.truth_final[1]}"
        )


def evaluate_games(
    points: list[tuple[float, str]],
    checkpoints: list[tuple[float, int, int]],
    *,
    final: str = "",
) -> GameReport:
    """``points``: (start time, score text before the point). ``checkpoints``: (t, self, other).

    At each checkpoint the score before the first point starting after ``t`` is read; the
    scoreboard was updated between points, so that point's score is what it should show.
    After the last point, the final score stands (a checkpoint in the break after a set
    may be minutes before the next point).
    """
    found: list[tuple[int, int] | None] = []
    for t, _, _ in checkpoints:
        nxt = next((p for p in points if p[0] >= t), None)
        if nxt is not None:
            found.append(parse_score(nxt[1]))
        else:
            found.append(parse_score(final) if final else None)
    best: tuple[float, int, bool] | None = None
    for self_first in (True, False):
        err = 0.0
        exact = 0
        for (_, a, b), f in zip(checkpoints, found, strict=True):
            if f is None:
                err += a + b
                continue
            fa, fb = f if self_first else (f[1], f[0])
            err += abs(fa - a) + abs(fb - b)
            exact += fa == a and fb == b
        if best is None or err < best[0]:
            best = (err, exact, self_first)
    assert best is not None
    truth = checkpoints[-1][1:] if checkpoints else (0, 0)
    return GameReport(
        checkpoints=len(checkpoints),
        exact=best[1],
        error_games=best[0] / max(1, len(checkpoints)),
        found_final=final,
        truth_final=(int(truth[0]), int(truth[1])),
        self_first=best[2],
    )
