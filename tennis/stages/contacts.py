"""Stage 2: ball-contact detection from audio, plus the tuning tool (spec section 6.2).

Output ``contacts.parquet`` (schema_version 1), one row per detected onset:

==================  =======  ==========================================================
contact_id          int64    sequential, in time order
t_audio             float64  onset time on the video PTS timeline (WAV time + audio_start_s)
frame_idx           int64    frame whose PTS is closest to ``t_audio``
t_video             float64  PTS of that frame
onset_strength      float32  onset envelope value at the detection
threshold           float32  adaptive threshold at the detection
peak_db             float64  peak 5 ms RMS level (dBFS) of the high-passed audio, +-20 ms
prominence_db       float64  peak_db minus the local background level
is_self_audio       bool     first pass: loud relative to the session median
is_self_confirmed   bool     second pass (wrist speed); null until stage 4 exists
==================  =======  ==========================================================
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from tennis.config import AudioConfig
from tennis.errors import UserError
from tennis.session import Session
from tennis.util.audio import (
    OnsetParams,
    Onsets,
    classify_self,
    compute_envelope,
    detect_onsets,
    pick_onsets,
    read_wav_mono,
)
from tennis.util.io import read_json, write_parquet
from tennis.util.video import nearest_frames
from tennis.validation import MatchResult, match_events

if TYPE_CHECKING:
    from tennis.stages import StageContext

SCHEMA_VERSION = 1
INPUTS = ("audio.wav", "metadata.json", "frame_times.parquet")
OUTPUTS = ("contacts.parquet",)

DEFAULT_GRID_K = (3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
DEFAULT_GRID_CUTOFF = (400.0, 800.0, 1500.0, 3000.0)
DEFAULT_GRID_DB = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)
DEFAULT_TOLERANCE_S = 0.040


def onset_params(audio: AudioConfig) -> OnsetParams:
    return OnsetParams(
        highpass_hz=audio.highpass_hz,
        highpass_order=audio.highpass_order,
        onset_k=audio.onset_k,
        min_separation_s=audio.min_separation_s,
        threshold_window_s=audio.threshold_window_s,
        amplitude_window_s=audio.amplitude_window_s,
        min_prominence_db=audio.min_prominence_db,
    )


def read_frame_pts(session: Session) -> npt.NDArray[np.float64]:
    table = pq.read_table(session.path("frame_times.parquet"), columns=["pts"])
    return np.asarray(table.column("pts").to_numpy(), dtype=np.float64)


def build_table(
    onsets: Onsets,
    is_self: npt.NDArray[np.bool_],
    audio_start_s: float,
    frame_pts: npt.NDArray[np.float64],
) -> pa.Table:
    t_audio = onsets.time_s + audio_start_s
    frame_idx = nearest_frames(frame_pts, t_audio) if len(onsets) else np.zeros(0, dtype=np.int64)
    n = len(onsets)
    return pa.table(
        {
            "contact_id": pa.array(np.arange(n, dtype=np.int64)),
            "t_audio": pa.array(t_audio, type=pa.float64()),
            "frame_idx": pa.array(frame_idx, type=pa.int64()),
            "t_video": pa.array(frame_pts[frame_idx], type=pa.float64()),
            "onset_strength": pa.array(onsets.strength, type=pa.float32()),
            "threshold": pa.array(onsets.threshold, type=pa.float32()),
            "peak_db": pa.array(onsets.peak_db, type=pa.float64()),
            "prominence_db": pa.array(onsets.prominence_db, type=pa.float64()),
            "is_self_audio": pa.array(is_self, type=pa.bool_()),
            "is_self_confirmed": pa.nulls(n, type=pa.bool_()),
        }
    )


def run(ctx: StageContext) -> None:
    session = ctx.session
    audio_cfg = ctx.config.audio
    meta = read_json(session.path("metadata.json"))
    audio_start_s = float(meta.get("audio_start_s", 0.0))
    sr, samples = read_wav_mono(session.path("audio.wav"))
    frame_pts = read_frame_pts(session)

    onsets = detect_onsets(samples, sr, onset_params(audio_cfg))
    is_self = classify_self(onsets.peak_db, audio_cfg.own_hit_db_threshold)
    table = build_table(onsets, is_self, audio_start_s, frame_pts)
    write_parquet(
        table,
        session.path("contacts.parquet"),
        stage=ctx.stage,
        config_hash=ctx.config_hash,
        schema_version=SCHEMA_VERSION,
    )

    minutes = samples.size / sr / 60 if samples.size else 0.0
    ctx.log(
        "contacts detected",
        onsets=len(onsets),
        self_audio=int(is_self.sum()),
        per_minute=round(len(onsets) / minutes, 1) if minutes else 0.0,
        median_db=round(float(np.median(onsets.peak_db)), 1) if len(onsets) else None,
    )
    if len(onsets) == 0:
        ctx.log(
            "no onsets found; check the audio track and audio.* settings", level=logging.WARNING
        )


# --- tuning ------------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelSpec:
    """How to read label times.

    A label ``t`` (video time) says the event happened in
    ``[t + offset_s, t + offset_s + resolution_s]``; a detection matches if it falls in that
    window widened by ``tolerance_s`` on both sides. Hand labels use offset 0 and
    resolution 0. Coarse external data (e.g. whole-second shot logs) uses a resolution
    of 1 s and whatever offset its clock has relative to the video.
    """

    times_s: npt.NDArray[np.float64]
    tolerance_s: float = DEFAULT_TOLERANCE_S
    resolution_s: float = 0.0
    offset_s: float = 0.0


@dataclass(frozen=True)
class TuneRow:
    highpass_hz: float
    onset_k: float
    own_hit_db_threshold: float | None  # None: every onset counts (target "any")
    onsets_in_range: int
    selected_in_range: int
    result: MatchResult  # selected detections vs labels
    recall_any: float  # recall when every onset counts, i.e. the detector's ceiling

    @property
    def median_offset_ms(self) -> float | None:
        offsets = self.result.offsets_s
        return float(np.median(offsets) * 1000) if offsets.size else None

    @property
    def key(self) -> tuple[float, float, float | None]:
        return (self.highpass_hz, self.onset_k, self.own_hit_db_threshold)


@dataclass(frozen=True)
class TuneReport:
    session_id: str
    target: str
    labels: int
    segments: list[tuple[float, float]]
    spec: LabelSpec
    current: tuple[float, float, float | None]
    rows: list[TuneRow]

    def best(self) -> TuneRow:
        return self.rows[0]

    @property
    def covered_s(self) -> float:
        return sum(end - start for start, end in self.segments)


def _in_segments(
    times: npt.NDArray[np.float64], segments: Sequence[tuple[float, float]]
) -> npt.NDArray[np.bool_]:
    mask = np.zeros(times.shape, dtype=bool)
    for start, end in segments:
        mask |= (times >= start) & (times <= end)
    return mask


def tune_contacts(
    session: Session,
    audio_cfg: AudioConfig,
    labels: LabelSpec,
    *,
    target: str = "self",
    ks: Sequence[float] = DEFAULT_GRID_K,
    cutoffs: Sequence[float] = DEFAULT_GRID_CUTOFF,
    db_thresholds: Sequence[float] = DEFAULT_GRID_DB,
    segments: Sequence[tuple[float, float]] | None = None,
    margin_s: float = 1.0,
) -> TuneReport:
    """Precision/recall of contact detection over a parameter grid.

    ``target="self"`` scores the first-pass own-hit selection (``is_self_audio``);
    ``target="any"`` scores every onset, for labels that cover all hits by both players.

    Only detections and labels inside ``segments`` count, so partly labeled sessions
    work. Segments use the labels' clock (``labels.offset_s`` is applied to them too).
    Without segments, the range is the first/last label window +- ``margin_s``.
    Detection always runs on the whole session, so the median used by
    the dB threshold matches a real run.
    """
    if target not in ("self", "any"):
        raise UserError(f"unknown target {target!r}: use 'self' or 'any'")
    for name in ("audio.wav", "metadata.json"):
        if not session.path(name).exists():
            raise UserError(f"session '{session.id}' has no {name}; run 'tennis process' first")
    if labels.times_s.size == 0:
        raise UserError("no labels to tune against")
    meta = read_json(session.path("metadata.json"))
    video_start = float(meta.get("video_start_s", 0.0))
    audio_start = float(meta.get("audio_start_s", 0.0))

    windows = labels.times_s + labels.offset_s
    if segments is None:
        segments = [
            (
                max(0.0, float(windows[0]) - margin_s),
                float(windows[-1]) + labels.resolution_s + margin_s,
            )
        ]
    else:
        segments = [(a + labels.offset_s, b + labels.offset_s) for a, b in segments]
    segments = sorted((float(a), float(b)) for a, b in segments)
    if any(b <= a for a, b in segments):
        raise UserError("every evaluation segment must end after it starts")
    centers = windows + labels.resolution_s / 2
    truth = labels.times_s[_in_segments(centers, segments)] + video_start
    if truth.size == 0:
        raise UserError("no labels fall inside the evaluation segments")
    pts_segments = [(a + video_start, b + video_start) for a, b in segments]

    def match(pred: npt.NDArray[np.float64]) -> MatchResult:
        return match_events(
            pred,
            truth,
            labels.tolerance_s,
            resolution_s=labels.resolution_s,
            offset_s=labels.offset_s,
        )

    grid: Sequence[float | None] = list(db_thresholds) if target == "self" else [None]
    sr, samples = read_wav_mono(session.path("audio.wav"))
    base_params = onset_params(audio_cfg)
    rows: list[TuneRow] = []
    for cutoff in cutoffs:
        envelope = compute_envelope(samples, sr, cutoff, audio_cfg.highpass_order)
        for k in ks:
            onsets = pick_onsets(envelope, replace(base_params, onset_k=k))
            t = onsets.time_s + audio_start
            in_range = _in_segments(t, pts_segments)
            any_result = match(t[in_range])
            for db in grid:
                selected = in_range if db is None else in_range & classify_self(onsets.peak_db, db)
                rows.append(
                    TuneRow(
                        highpass_hz=cutoff,
                        onset_k=k,
                        own_hit_db_threshold=db,
                        onsets_in_range=int(in_range.sum()),
                        selected_in_range=int(selected.sum()),
                        result=any_result if db is None else match(t[selected]),
                        recall_any=any_result.recall,
                    )
                )
    rows.sort(
        key=lambda r: (
            -r.result.f1,
            -min(r.result.precision, r.result.recall),
            abs(r.onset_k - audio_cfg.onset_k),
            r.highpass_hz,
            r.own_hit_db_threshold or 0.0,
        )
    )
    current_db = audio_cfg.own_hit_db_threshold if target == "self" else None
    return TuneReport(
        session_id=session.id,
        target=target,
        labels=int(truth.size),
        segments=list(segments),
        spec=labels,
        current=(audio_cfg.highpass_hz, audio_cfg.onset_k, current_db),
        rows=rows,
    )


def _fmt_offset(value: float | None) -> str:
    return f"{value:+.1f}" if value is not None else "-"


def format_tune_report(report: TuneReport, top: int = 15, markdown: bool = False) -> str:
    """Render the tuning results as a plain-text or markdown table."""
    headers = [
        "cutoff_hz", "k", "db", "precision", "recall", "f1",
        "recall_any", "selected", "onsets", "offset_ms", "",
    ]  # fmt: skip
    current = next((r for r in report.rows if r.key == report.current), None)
    shown = report.rows[:top]
    if current is not None and current not in shown:
        shown = [*shown, current]
    table = []
    for r in shown:
        db = r.own_hit_db_threshold
        table.append(
            [
                f"{r.highpass_hz:g}",
                f"{r.onset_k:g}",
                "-" if db is None else f"{db:g}",
                f"{r.result.precision:.3f}",
                f"{r.result.recall:.3f}",
                f"{r.result.f1:.3f}",
                f"{r.recall_any:.3f}",
                str(r.selected_in_range),
                str(r.onsets_in_range),
                _fmt_offset(r.median_offset_ms),
                "current config" if r is current else "",
            ]
        )

    spec = report.spec
    best = report.best()
    ok = best.result.precision >= 0.9 and best.result.recall >= 0.9
    what = "own hits" if report.target == "self" else "hits (all players)"
    window = f"tolerance +-{spec.tolerance_s * 1000:.0f} ms"
    if spec.resolution_s or spec.offset_s:
        window = (
            f"label window [t{spec.offset_s:+g}, t{spec.offset_s + spec.resolution_s:+g}] s, "
            + window
        )
    setting = f"highpass_hz={best.highpass_hz:g} onset_k={best.onset_k:g}"
    if best.own_hit_db_threshold is not None:
        setting += f" own_hit_db_threshold={best.own_hit_db_threshold:g}"
    summary = [
        f"session {report.session_id}: {report.labels} labeled {what} in "
        f"{len(report.segments)} segment(s), {report.covered_s:.0f} s total; {window}",
        f"best: {setting} -> precision {best.result.precision:.3f}, recall "
        f"{best.result.recall:.3f} ({'meets' if ok else 'does not meet'} the 0.90/0.90 target)",
        "recall_any = recall if every onset counted (detector ceiling); "
        "offset_ms = median detected time minus label window start.",
    ]
    snippet = ["audio:", f"  highpass_hz: {best.highpass_hz:g}", f"  onset_k: {best.onset_k:g}"]
    if best.own_hit_db_threshold is not None:
        snippet.append(f"  own_hit_db_threshold: {best.own_hit_db_threshold:g}")

    if markdown:
        lines = [f"# Contact detection tuning: {report.session_id}", ""]
        lines += [f"- {s}" for s in summary]
        lines += ["", "| " + " | ".join(h or "note" for h in headers) + " |"]
        lines += ["|" + "---|" * len(headers)]
        lines += ["| " + " | ".join(row) + " |" for row in table]
        lines += ["", "Suggested config:", "", "```yaml", *snippet, "```", ""]
        return "\n".join(lines)

    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    lines = [*summary, ""]
    for row in [headers, *table]:
        lines.append("  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)).rstrip())
    lines += ["", "suggested config:", *snippet]
    return "\n".join(lines)


def write_tune_report(report: TuneReport, path: Path, top: int = 15) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_tune_report(report, top=top, markdown=True))
