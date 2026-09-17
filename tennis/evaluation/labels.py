"""Score stage 7's voice labels against a hand-written truth file (``tennis eval-labels``).

The M8 criterion is "at least 80% of spoken label words matched correctly". A word counts
as matched when a stored label was produced for it at about the right time **and** carries
the right label. The truth file lists what was actually said and when::

    t,label
    3:12.4,good
    3:20.0,late

Two failure modes are reported apart, because they have different fixes: a word the
transcriber never produced (or that landed outside the tolerance) is a *miss*, and a word
that produced the wrong vocabulary entry is a *confusion*. Stored labels with no truth
word behind them are *spurious*.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tennis.errors import UserError
from tennis.evaluation.strokes import ConfusionMatrix, confusion_matrix
from tennis.session import Session
from tennis.stages.labels import read_labels

DEFAULT_TOLERANCE_S = 2.0


@dataclass(frozen=True)
class LabelEval:
    session_id: str
    spoken: int
    stored: int
    matched: int  # a stored label was found near the spoken word
    correct: int  # ... and it was the right label
    spurious: int  # stored labels with no spoken word near them
    matrix: ConfusionMatrix
    swings_labelled: int
    tolerance_s: float
    target: float = 0.80

    @property
    def match_rate(self) -> float:
        return self.correct / self.spoken if self.spoken else 0.0

    @property
    def missed(self) -> int:
        return self.spoken - self.matched

    @property
    def confused(self) -> int:
        return self.matched - self.correct

    @property
    def meets_target(self) -> bool:
        return self.match_rate >= self.target


def evaluate_labels(
    session: Session,
    spoken: Sequence[tuple[float, str]],
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    video_start_s: float = 0.0,
) -> LabelEval:
    """``spoken`` is (time the word was said in video time, label), as ``read_word_labels``
    reads it. Times are moved onto the PTS timeline with ``video_start_s``."""
    if not session.path("labels.parquet").exists():
        raise UserError(
            f"session '{session.id}' has no labels.parquet; run 'tennis process' with "
            "labels.enabled (it is skipped by --no-labels)"
        )
    if not spoken:
        raise UserError("no spoken labels to evaluate against")
    stored = sorted(read_labels(session), key=lambda r: float(r["t_word"]))
    voice = [r for r in stored if r["source"] == "voice"]
    times = np.array([float(r["t_word"]) for r in voice], dtype=np.float64)

    used: set[int] = set()
    pairs: list[tuple[str, str]] = []
    matched = 0
    for t, truth in sorted(spoken):
        t_pts = t + video_start_s
        candidates = [
            i for i in range(len(voice)) if i not in used and abs(times[i] - t_pts) <= tolerance_s
        ]
        if not candidates:
            continue
        # Prefer a candidate that agrees with the truth, so a nearby unrelated word does
        # not eat the match and turn a correct label into both a miss and a spurious one.
        best = min(
            candidates,
            key=lambda i: (str(voice[i]["label"]) != truth, abs(times[i] - t_pts)),
        )
        used.add(best)
        matched += 1
        pairs.append((truth, str(voice[best]["label"])))

    matrix = confusion_matrix(pairs)
    return LabelEval(
        session_id=session.id,
        spoken=len(spoken),
        stored=len(stored),
        matched=matched,
        correct=matrix.correct,
        spurious=len(voice) - len(used),
        matrix=matrix,
        swings_labelled=len({int(r["swing_id"]) for r in stored}),
        tolerance_s=tolerance_s,
    )


def format_label_eval(ev: LabelEval, markdown: bool = False) -> str:
    m = ev.matrix
    summary = [
        f"session {ev.session_id}: {ev.spoken} spoken label words, {ev.stored} stored labels "
        f"on {ev.swings_labelled} swing(s), tolerance {ev.tolerance_s:g}s",
        f"matched {ev.correct}/{ev.spoken} correctly ({ev.match_rate:.1%}; "
        f"{'meets' if ev.meets_target else 'does not meet'} the {ev.target:.0%} target)",
        f"{ev.missed} never transcribed or too far off, {ev.confused} heard as another "
        f"label, {ev.spurious} stored labels with nothing spoken near them",
    ]
    if not m.labels:
        return "\n".join(summary)
    header = ["spoken \\ stored", *m.labels, "spoken", "recall"]
    rows = [
        [
            name,
            *(str(m.counts[i][j]) for j in range(len(m.labels))),
            str(m.support(i)),
            f"{m.recall(i):.3f}",
        ]
        for i, name in enumerate(m.labels)
    ]
    if markdown:
        lines = [f"# Voice-label evaluation: {ev.session_id}", ""]
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
        if n == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def write_label_eval(ev: LabelEval, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_label_eval(ev, markdown=True))
