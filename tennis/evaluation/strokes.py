"""Score stage 5's stroke types against labelled shots (``tennis eval-classifier``).

The labels have their own clock (Wingfield's shot log is whole seconds on a clock offset
from the video), so labels are matched to classified swings with the same machinery the
contact tools use: ``LabelSpec`` describes the clock, ``make_scorer`` restricts scoring to
the labelled segments, and ``match_events`` pairs them one to one. **Only matched pairs
count towards accuracy**: a label with no classified swing is a miss of the *detector*,
not of the classifier, and is reported separately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tennis.errors import UserError
from tennis.session import Session
from tennis.stages.classify import STROKE_TYPES
from tennis.stages.contacts import LabelScorer


@dataclass(frozen=True)
class ConfusionMatrix:
    """``counts[truth][predicted]`` over the stroke types that appear in either."""

    labels: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]

    @property
    def total(self) -> int:
        return sum(sum(row) for row in self.counts)

    @property
    def correct(self) -> int:
        return sum(self.counts[i][i] for i in range(len(self.labels)))

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def support(self, i: int) -> int:
        return sum(self.counts[i])

    def predicted(self, i: int) -> int:
        return sum(row[i] for row in self.counts)

    def precision(self, i: int) -> float:
        denom = self.predicted(i)
        return self.counts[i][i] / denom if denom else 0.0

    def recall(self, i: int) -> float:
        denom = self.support(i)
        return self.counts[i][i] / denom if denom else 0.0

    def f1(self, i: int) -> float:
        p, r = self.precision(i), self.recall(i)
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def macro_f1(self) -> float:
        present = [i for i in range(len(self.labels)) if self.support(i) or self.predicted(i)]
        return sum(self.f1(i) for i in present) / len(present) if present else 0.0


def confusion_matrix(pairs: Sequence[tuple[str, str]]) -> ConfusionMatrix:
    """``pairs`` are (truth, predicted) stroke names."""
    seen = {name for pair in pairs for name in pair}
    labels = tuple([s for s in STROKE_TYPES if s in seen] + sorted(seen - set(STROKE_TYPES)))
    index = {name: i for i, name in enumerate(labels)}
    counts = [[0] * len(labels) for _ in labels]
    for truth, predicted in pairs:
        counts[index[truth]][index[predicted]] += 1
    return ConfusionMatrix(labels, tuple(tuple(row) for row in counts))


@dataclass(frozen=True)
class ClassifierEval:
    session_id: str
    classifier_version: str
    handedness: str
    labels_in_range: int
    classified_in_range: int
    matched: int
    unmatched_labels: int
    unmatched_swings: int
    matrix: ConfusionMatrix
    two_handed: int
    accuracy_target: float = 0.90

    @property
    def meets_target(self) -> bool:
        return self.matrix.accuracy >= self.accuracy_target


def evaluate_classifier(
    session: Session,
    scorer: LabelScorer,
    labels: Sequence[tuple[float, str]],
) -> ClassifierEval:
    """``labels`` are (time on the labels' clock, stroke type), as ``read_stroke_labels`` reads."""
    path = session.path("strokes.parquet")
    if not path.exists():
        raise UserError(f"session '{session.id}' has no strokes.parquet; run 'tennis process'")
    table = pq.read_table(path)
    meta = {
        k.decode().removeprefix("tennis."): v.decode()
        for k, v in (table.schema.metadata or {}).items()
    }
    rows = [r for r in table.to_pylist() if r["stroke_type"] is not None]
    rows.sort(key=lambda r: float(r["t_contact"]))

    times = np.array([float(r["t_contact"]) for r in rows], np.float64)
    in_range = scorer.in_range(times) if times.size else np.zeros(0, bool)
    chosen = [r for r, keep in zip(rows, in_range, strict=True) if keep]
    predicted_times = np.array([float(r["t_contact"]) for r in chosen], np.float64)

    # ``truth_index`` says which labels the scorer kept; both it and ``truth_pts`` follow
    # the order of ``labels``, which ``read_stroke_labels`` sorts by time, and
    # ``match_events`` indexes the same sorted array. So pair j is label ``truth_index[j]``.
    truth_strokes = kept_strokes(labels, scorer)
    result = scorer.score(predicted_times)

    pairs: list[tuple[str, str]] = []
    for pred_i, truth_j in result.pairs:
        pairs.append((truth_strokes[truth_j], str(chosen[pred_i]["stroke_type"])))
    matrix = confusion_matrix(pairs)
    return ClassifierEval(
        session_id=session.id,
        classifier_version=meta.get("classifier_version", "unknown"),
        handedness=meta.get("handedness", "unknown"),
        labels_in_range=int(scorer.truth_pts.size),
        classified_in_range=len(chosen),
        matched=len(pairs),
        unmatched_labels=result.false_negatives,
        unmatched_swings=result.false_positives,
        matrix=matrix,
        two_handed=sum(1 for r in chosen if r["two_handed"]),
    )


def kept_strokes(labels: Sequence[tuple[float, str]], scorer: LabelScorer) -> list[str]:
    """Stroke names of the labels the scorer kept, in ``truth_pts`` order."""
    if len(scorer.truth_index) != scorer.truth_pts.size:
        raise UserError(
            "this scorer does not record which labels it kept; build it with make_scorer"
        )
    if max(scorer.truth_index, default=-1) >= len(labels):
        raise UserError("the scorer was built from a different label file")
    return [labels[i][1] for i in scorer.truth_index]


def format_classifier_eval(ev: ClassifierEval, markdown: bool = False) -> str:
    m = ev.matrix
    header = ["truth \\ predicted", *m.labels, "support", "recall"]
    rows = [
        [
            name,
            *(str(m.counts[i][j]) for j in range(len(m.labels))),
            str(m.support(i)),
            f"{m.recall(i):.3f}",
        ]
        for i, name in enumerate(m.labels)
    ]
    rows.append(
        [
            "precision",
            *(f"{m.precision(j):.3f}" for j in range(len(m.labels))),
            str(m.total),
            f"{m.accuracy:.3f}",
        ]
    )
    summary = [
        f"session {ev.session_id}: classifier {ev.classifier_version}, {ev.handedness}-handed",
        f"{ev.labels_in_range} labels and {ev.classified_in_range} classified swings in range; "
        f"{ev.matched} matched pairs scored "
        f"({ev.unmatched_labels} labels and {ev.unmatched_swings} swings unmatched)",
        f"accuracy {m.accuracy:.3f} ({m.correct}/{m.total}), macro F1 {m.macro_f1:.3f} "
        f"({'meets' if ev.meets_target else 'does not meet'} the "
        f"{ev.accuracy_target:.0%} target)",
        f"{ev.two_handed} of the classified swings were two-handed",
    ]
    if markdown:
        lines = [f"# Stroke classifier evaluation: {ev.session_id}", ""]
        lines += [f"- {line}" for line in summary] + [""]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines) + "\n"
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    lines = [*summary, ""]
    for n, row in enumerate([header, *rows]):
        cells = [c.ljust(w) if i == 0 else c.rjust(w)
                 for i, (c, w) in enumerate(zip(row, widths, strict=True))]  # fmt: skip
        lines.append("  ".join(cells))
        if n == 0 or n == len(rows) - 1:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def write_classifier_eval(ev: ClassifierEval, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_classifier_eval(ev, markdown=True))
