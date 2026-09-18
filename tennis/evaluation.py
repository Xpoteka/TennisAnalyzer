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
