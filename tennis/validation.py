"""Helpers for validating pipeline output against hand-made labels (spec section 10.3)."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tennis.errors import UserError

TIME_COLUMNS = ("t", "time", "time_s", "t_video", "timestamp", "seconds")
_CLOCK_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d*)?)$")


def parse_time(value: str) -> float:
    """Seconds from ``12.5``, ``1:02.345`` (m:ss) or ``0:01:02.345`` (h:mm:ss)."""
    text = value.strip()
    try:
        seconds = float(text)
    except ValueError:
        match = _CLOCK_RE.match(text)
        if not match:
            raise ValueError(f"not a time: {value!r}") from None
        hours, minutes, secs = match.groups()
        seconds = int(hours or 0) * 3600 + int(minutes) * 60 + float(secs)
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError(f"not a valid time: {value!r}")
    return seconds


def read_time_labels(path: Path) -> npt.NDArray[np.float64]:
    """Read a CSV of event times (video time, seconds from the first frame), sorted.

    The time column is the first one named t, time, time_s, t_video, timestamp or seconds;
    without a recognised header the first column is used. Blank lines and lines starting
    with ``#`` are ignored.
    """
    if not path.is_file():
        raise UserError(f"labels file not found: {path}")
    with path.open(newline="") as fh:
        rows = [
            r for r in csv.reader(fh) if r and r[0].strip() and not r[0].lstrip().startswith("#")
        ]
    if not rows:
        raise UserError(f"{path}: no labels")

    column = 0
    header = [c.strip().lower() for c in rows[0]]
    named = [i for i, c in enumerate(header) if c in TIME_COLUMNS]
    if named:
        column, rows = named[0], rows[1:]
    else:
        try:
            parse_time(rows[0][0])
        except ValueError:
            rows = rows[1:]  # unrecognised header: use the first column

    times = []
    for row in rows:
        try:
            times.append(parse_time(row[column]))
        except (ValueError, IndexError) as exc:
            raise UserError(f"{path}: bad row {','.join(row)!r}: {exc}") from exc
    if not times:
        raise UserError(f"{path}: no labels")
    return np.sort(np.asarray(times, dtype=np.float64))


@dataclass(frozen=True)
class MatchResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    offsets_s: npt.NDArray[np.float64]  # predicted - truth, for matched pairs
    # (prediction, label) index pairs into the *sorted* inputs, in label order.
    pairs: tuple[tuple[int, int], ...] = ()

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match_events(
    predicted: npt.NDArray[np.float64],
    truth: npt.NDArray[np.float64],
    tolerance_s: float,
    *,
    resolution_s: float = 0.0,
    offset_s: float = 0.0,
) -> MatchResult:
    """One-to-one matching of predicted event times to labels.

    Label ``t`` accepts predictions in ``[t + offset_s - tolerance_s,
    t + offset_s + resolution_s + tolerance_s]``. Pairs closest to the centre of their
    window are matched first. ``offsets_s`` holds prediction minus window start, and
    ``pairs`` the matched (prediction, label) indices into the sorted inputs, so callers
    can compare something other than the times (see ``eval-classifier``).
    """
    pred = np.sort(np.asarray(predicted, dtype=np.float64))
    starts = np.sort(np.asarray(truth, dtype=np.float64)) + offset_s
    pairs: list[tuple[float, int, int]] = []
    for j, start in enumerate(starts):
        lo = int(np.searchsorted(pred, start - tolerance_s, side="left"))
        hi = int(np.searchsorted(pred, start + resolution_s + tolerance_s, side="right"))
        center = start + resolution_s / 2
        pairs.extend((abs(pred[i] - center), i, j) for i in range(lo, hi))
    pairs.sort()
    used_pred: set[int] = set()
    used_true: set[int] = set()
    offsets = []
    matched: list[tuple[int, int]] = []
    for _, i, j in pairs:
        if i in used_pred or j in used_true:
            continue
        used_pred.add(i)
        used_true.add(j)
        offsets.append(pred[i] - starts[j])
        matched.append((i, j))
    tp = len(offsets)
    return MatchResult(
        true_positives=tp,
        false_positives=int(pred.size) - tp,
        false_negatives=int(starts.size) - tp,
        offsets_s=np.asarray(offsets, dtype=np.float64),
        pairs=tuple(sorted(matched, key=lambda p: p[1])),
    )


STROKE_COLUMNS = ("stroke", "stroke_type", "shot", "type", "label")
PLAYER_COLUMNS = ("player", "who", "hitter")
SELF_VALUES = {"self", "me", "you", "own", "1", "true", "yes"}


def _rows(path: Path, what: str) -> list[list[str]]:
    if not path.is_file():
        raise UserError(f"{what} file not found: {path}")
    with path.open(newline="") as fh:
        rows = [
            r for r in csv.reader(fh) if r and r[0].strip() and not r[0].lstrip().startswith("#")
        ]
    if not rows:
        raise UserError(f"{path}: no rows")
    return rows


def read_stroke_labels(path: Path, player: str = "self") -> list[tuple[float, str]]:
    """Read ``(time, stroke type)`` pairs from a stroke-label CSV, sorted by time.

    Two shapes are accepted, and the header decides which:

    * the Wingfield export (``t,player,stroke,...``), of which only the rows for ``player``
      are kept. The default, ``self``, also accepts the spellings in ``SELF_VALUES``
      (``me``, ``you``, ``own``, ...); any other name is matched literally, so a partner's
      strokes can be scored by passing theirs;
    * a plain ``t,stroke`` file with no player column, where every row is yours.

    Stroke names are lower-cased and must be one of the spec's four types.
    """
    from tennis.stages.classify import STROKE_TYPES

    rows = _rows(path, "labels")
    header = [c.strip().lower() for c in rows[0]]
    time_col = next((i for i, c in enumerate(header) if c in TIME_COLUMNS), None)
    stroke_col = next((i for i, c in enumerate(header) if c in STROKE_COLUMNS), None)
    player_col = next((i for i, c in enumerate(header) if c in PLAYER_COLUMNS), None)
    if time_col is None or stroke_col is None:
        raise UserError(
            f"{path}: needs a time column ({'/'.join(TIME_COLUMNS[:3])}...) and a stroke "
            f"column ({'/'.join(STROKE_COLUMNS[:2])}...); found {', '.join(header)}"
        )
    known = set(STROKE_TYPES)
    # Asking for "self" - by any of its spellings - accepts every spelling of it, since a
    # file may say "me" where the caller said "self". Any other name matches literally.
    name = player.strip().lower()
    wanted = set(SELF_VALUES) if name in SELF_VALUES else {name}
    labels: list[tuple[float, str]] = []
    skipped = 0
    for row in rows[1:]:
        try:
            if player_col is not None and row[player_col].strip().lower() not in wanted:
                continue
            t = parse_time(row[time_col])
            stroke = row[stroke_col].strip().lower()
        except (ValueError, IndexError) as exc:
            raise UserError(f"{path}: bad row {','.join(row)!r}: {exc}") from exc
        if stroke not in known:
            skipped += 1
            continue
        labels.append((t, stroke))
    if not labels:
        raise UserError(
            f"{path}: no usable {player} labels"
            + (f" ({skipped} rows had an unknown stroke type)" if skipped else "")
        )
    return sorted(labels)


def read_word_labels(path: Path) -> list[tuple[float, str]]:
    """Read ``(time, label)`` pairs for spoken labels, sorted by time (``tennis eval-labels``)."""
    rows = _rows(path, "labels")
    header = [c.strip().lower() for c in rows[0]]
    time_col = next((i for i, c in enumerate(header) if c in TIME_COLUMNS), None)
    label_col = next((i for i, c in enumerate(header) if c in ("label", "word", "spoken")), None)
    if time_col is None or label_col is None:
        raise UserError(
            f"{path}: needs a time column and a 'label' column; found {', '.join(header)}"
        )
    out: list[tuple[float, str]] = []
    for row in rows[1:]:
        try:
            out.append((parse_time(row[time_col]), row[label_col].strip().lower()))
        except (ValueError, IndexError) as exc:
            raise UserError(f"{path}: bad row {','.join(row)!r}: {exc}") from exc
    if not out:
        raise UserError(f"{path}: no labels")
    return sorted(out)


def read_segments(path: Path) -> list[tuple[float, float]]:
    """Read ``start,end`` rows (video time) from a CSV. A header row is optional."""
    if not path.is_file():
        raise UserError(f"segments file not found: {path}")
    with path.open(newline="") as fh:
        rows = [
            r for r in csv.reader(fh) if r and r[0].strip() and not r[0].lstrip().startswith("#")
        ]
    segments = []
    for i, row in enumerate(rows):
        try:
            start, end = parse_time(row[0]), parse_time(row[1])
        except (ValueError, IndexError) as exc:
            if i == 0:
                continue  # header
            raise UserError(f"{path}: bad row {','.join(row)!r}: {exc}") from exc
        if end <= start:
            raise UserError(f"{path}: segment {row[0]}-{row[1]} ends before it starts")
        segments.append((start, end))
    if not segments:
        raise UserError(f"{path}: no segments")
    return segments
