"""Score the stored own-hit flags against labels, and sweep the confirmation settings.

Unlike ``tune-contacts`` this re-runs nothing: it reads ``contacts.parquet``,
``swing_info.parquet`` and the wrist speeds in ``swings.parquet``, and re-applies only the
cheap wrist-speed confirmation rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tennis.errors import UserError
from tennis.session import Session
from tennis.stages.clean import Swing, confirm, confirmation_peak, racket_side
from tennis.stages.contacts import LabelScorer
from tennis.validation import MatchResult

DEFAULT_SPEEDS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
DEFAULT_WINDOWS = (0.1, 0.15, 0.2, 0.3)


@dataclass(frozen=True)
class EvalRow:
    name: str
    selected: int
    result: MatchResult
    min_speed: float | None = None
    window_s: float | None = None
    qc_only: bool = False
    wrist: str | None = None


@dataclass(frozen=True)
class ContactEval:
    session_id: str
    labels: int
    covered_s: float
    fixed: list[EvalRow]  # all onsets, first pass, stored confirmation
    sweep: list[EvalRow]  # confirmation settings, best first
    current: tuple[float, float, str]


def evaluate_contacts(
    session: Session,
    scorer: LabelScorer,
    current: tuple[float, float, str],
    handedness: str,
    speeds: Sequence[float] = DEFAULT_SPEEDS,
    windows: Sequence[float] = DEFAULT_WINDOWS,
    wrists: Sequence[str] = ("either", "racket"),
) -> ContactEval:
    """``current`` is (min_speed, window_s, wrist) of the config that produced the files."""
    for name in ("contacts.parquet", "swing_info.parquet", "swings.parquet"):
        if not session.path(name).exists():
            raise UserError(f"session '{session.id}' has no {name}; run 'tennis process' first")
    contacts = pq.read_table(session.path("contacts.parquet")).to_pydict()
    info = pq.read_table(session.path("swing_info.parquet")).to_pylist()
    t_by_id = dict(zip(contacts["contact_id"], contacts["t_audio"], strict=True))

    t_all = np.asarray(contacts["t_audio"], np.float64)
    first = np.asarray(contacts["is_self_audio"], bool)
    in_all = scorer.in_range(t_all)
    stored = np.array(
        [t_by_id[r["contact_id"]] for r in info if r["is_self_confirmed"]], np.float64
    )
    stored_qc = np.array(
        [t_by_id[r["contact_id"]] for r in info if r["is_self_confirmed"] and r["qc_pass"]],
        np.float64,
    )

    def scored(
        name: str,
        t: np.ndarray,
        min_speed: float | None = None,
        window_s: float | None = None,
        qc_only: bool = False,
        wrist: str | None = None,
    ) -> EvalRow:
        mask = scorer.in_range(t)
        return EvalRow(
            name, int(mask.sum()), scorer.score(t[mask]), min_speed, window_s, qc_only, wrist
        )

    loud = {int(c) for c, f in zip(contacts["contact_id"], first, strict=True) if f}
    both = np.array(
        [t_by_id[r["contact_id"]] for r in info
         if r["is_self_confirmed"] and r["contact_id"] in loud],
        np.float64,
    )  # fmt: skip
    fixed = [
        EvalRow("all onsets", int(in_all.sum()), scorer.score(t_all[in_all])),
        scored("first pass (audio level)", t_all[first]),
        scored("confirmed (stored)", stored),
        scored("confirmed + QC pass (stored)", stored_qc),
        scored("first pass + confirmed", both),
    ]

    frames = pq.read_table(
        session.path("swings.parquet"),
        columns=["swing_id", "t_rel", "l_wrist_speed", "r_wrist_speed"],
    )
    ids = frames.column("swing_id").to_numpy()
    t_rel = frames.column("t_rel").to_numpy()
    speed = {s: frames.column(f"{s}_wrist_speed").to_numpy().astype(np.float64) for s in "lr"}
    order = np.argsort(ids, kind="stable")
    bounds = np.flatnonzero(np.diff(ids[order])) + 1
    series = {int(ids[chunk[0]]): chunk for chunk in np.split(order, bounds) if chunk.size}
    side = racket_side(handedness)

    sweep: list[EvalRow] = []
    for wrist in wrists:
        for window in windows:
            peaks: dict[int, tuple[float, float]] = {}
            for sid, rows in series.items():
                signal = (
                    speed[side][rows]
                    if wrist == "racket"
                    else np.fmax(speed["l"][rows], speed["r"][rows])
                )
                peaks[sid] = confirmation_peak(t_rel[rows], signal, window)
            for min_speed in speeds:
                swings = []
                for r in info:
                    values = {k: _nan(v) for k, v in r.items()}
                    peak = peaks.get(r["swing_id"], (float("nan"), float("nan")))
                    values["wrist_peak_speed"], values["wrist_peak_offset_s"] = peak
                    swings.append(Swing(r["swing_id"], r["contact_id"], r["t_contact"],
                                        np.zeros(0, np.int64), 0, info=values))  # fmt: skip
                confirm(swings, min_speed)
                for qc_only in (False, True):
                    t = np.array(
                        [s.t_contact for s in swings
                         if s.info["is_self_confirmed"] and (s.info["qc_pass"] or not qc_only)],
                        np.float64,
                    )  # fmt: skip
                    label = f"confirmed{' + QC' if qc_only else ''}"
                    sweep.append(scored(label, t, min_speed, window, qc_only, wrist))
    sweep.sort(key=lambda r: (-r.result.f1, -min(r.result.precision, r.result.recall)))
    return ContactEval(
        session_id=session.id,
        labels=int(scorer.truth_pts.size),
        covered_s=sum(b - a for a, b in scorer.segments),
        fixed=fixed,
        sweep=sweep,
        current=current,
    )


def _nan(value: object) -> object:
    return float("nan") if value is None else value


def format_contact_eval(ev: ContactEval, top: int = 12, markdown: bool = False) -> str:
    headers = [
        "selection", "wrist", "min_speed", "window_s", "selected", "precision", "recall", "f1",
    ]  # fmt: skip

    def row(r: EvalRow) -> list[str]:
        return [
            r.name,
            r.wrist or "-",
            "-" if r.min_speed is None else f"{r.min_speed:g}",
            "-" if r.window_s is None else f"{r.window_s:g}",
            str(r.selected),
            f"{r.result.precision:.3f}",
            f"{r.result.recall:.3f}",
            f"{r.result.f1:.3f}",
        ]

    rows = [row(r) for r in ev.fixed] + [row(r) for r in ev.sweep[:top]]
    summary = (
        f"session {ev.session_id}: {ev.labels} labeled own hits, {ev.covered_s:.0f} s evaluated; "
        f"stored with min_speed={ev.current[0]:g}, window_s={ev.current[1]:g}, "
        f"wrist={ev.current[2]}"
    )
    if markdown:
        lines = [f"# Own-hit evaluation: {ev.session_id}", "", summary, ""]
        lines += ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines) + "\n"
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    lines = [summary, ""]
    for i, r in enumerate([headers, *rows]):
        cells = [c.ljust(w) if j < 2 else c.rjust(w)
                 for j, (c, w) in enumerate(zip(r, widths, strict=True))]  # fmt: skip
        lines.append("  ".join(cells))
        if i == len(ev.fixed):
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def write_contact_eval(ev: ContactEval, path: Path, top: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_contact_eval(ev, top=top, markdown=True))
