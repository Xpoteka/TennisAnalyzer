"""HTML reports (spec section 6.9): the per-session report and the cross-session trends.

Both are one self-contained file. Plotly's script is inlined once and every figure is
rendered without its own copy, so the page opens offline with no network and no sibling
files. Clips are linked by relative path, so the report and its ``clips/`` folder travel
together.

Cross-session numbers come from DuckDB over ``data/sessions/*/metrics.parquet``, read with
``read_parquet(glob, filename = true)`` so each row knows which session it came from. The
per-session aggregates are read from the ``metrics_summary.parquet`` stage 6 already wrote,
rather than recomputed here.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from tennis.config import Config
from tennis.errors import UserError
from tennis.session import Session, list_sessions, sessions_root
from tennis.stages.clips import INDEX_NAME
from tennis.stages.labels import read_labels
from tennis.stages.metrics import REGISTRY
from tennis.util.io import read_json

GOOD_LABEL = "good"
TEMPLATES = Path(__file__).parent / "templates"


# --- cross-session data -----------------------------------------------------------------------


@dataclass(frozen=True)
class SessionMetrics:
    """One session's per-stroke-type means, as the trends and deltas need them."""

    session_id: str
    date: str
    means: dict[tuple[str, str], float]  # (stroke type, metric) -> mean
    stds: dict[tuple[str, str], float]
    counts: dict[str, int]  # stroke type -> swings


def session_date(session: Session) -> str:
    """The session's date: its id starts with one, otherwise the ingest creation time."""
    head = session.id[:10]
    try:
        dt.date.fromisoformat(head)
        return head
    except ValueError:
        pass
    path = session.path("metadata.json")
    if path.exists():
        created = str(read_json(path).get("creation_time") or "")
        if len(created) >= 10:
            return created[:10]
    return session.id


def read_summary(session: Session) -> SessionMetrics | None:
    path = session.path("metrics_summary.parquet")
    if not path.exists():
        return None
    rows = pq.read_table(path).to_pylist()
    means: dict[tuple[str, str], float] = {}
    stds: dict[tuple[str, str], float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        key = (str(row["stroke_type"]), str(row["metric"]))
        means[key] = float(row["mean"])
        stds[key] = float(row["std"]) if row["std"] is not None else float("nan")
        counts[key[0]] = max(counts.get(key[0], 0), int(row["count"]))
    return SessionMetrics(session.id, session_date(session), means, stds, counts)


def history(
    config: Config, before: str | None = None, since: str | None = None
) -> list[SessionMetrics]:
    """Every session with metrics, oldest first, optionally limited by date."""
    out = []
    for session in list_sessions(config.paths.data_root):
        summary = read_summary(session)
        if summary is None:
            continue
        if since and summary.date < since:
            continue
        if before and summary.session_id == before:
            continue
        out.append(summary)
    return sorted(out, key=lambda s: (s.date, s.session_id))


def rolling_baseline(
    past: Sequence[SessionMetrics], rolling_sessions: int
) -> dict[tuple[str, str], float]:
    """Mean of each (stroke type, metric) over the most recent ``rolling_sessions``."""
    recent = list(past)[-rolling_sessions:]
    values: dict[tuple[str, str], list[float]] = {}
    for summary in recent:
        for key, value in summary.means.items():
            if np.isfinite(value):
                values.setdefault(key, []).append(value)
    return {key: float(np.mean(v)) for key, v in values.items() if v}


def trend_table(config: Config, since: str | None = None) -> list[dict[str, Any]]:
    """Per-session mean and std of every metric, from DuckDB over every session's Parquet.

    Falls back to reading the files one by one when DuckDB is not available, so the report
    never fails over an optional dependency.
    """
    root = sessions_root(config.paths.data_root)
    pattern = str(root / "*" / "metrics.parquet")
    try:
        import duckdb
    except ImportError:  # pragma: no cover - duckdb is a declared dependency
        return _trend_table_without_duckdb(config, since)
    if not list(root.glob("*/metrics.parquet")):
        return []
    # Only finite values are aggregated. A metric that does not apply to a stroke type is
    # NaN on every row, and DuckDB's stddev_samp raises "out of range" on a group of NaNs
    # rather than returning one; avg would quietly return NaN. This also matches what
    # stage 6 put in metrics_summary.parquet, and what the fallback below computes.
    finite = 'CASE WHEN isfinite("{name}") THEN "{name}" END'
    metric_columns = ", ".join(
        f'avg({finite.format(name=name)}) AS "{name}_mean", '
        f'stddev_samp({finite.format(name=name)}) AS "{name}_std", '
        f'count({finite.format(name=name)}) AS "{name}_count"'
        for name in REGISTRY
    )
    query = f"""
        SELECT
            regexp_extract(filename, '([^/]+)/metrics\\.parquet$', 1) AS session_id,
            stroke_type,
            count(*) AS swings,
            {metric_columns}
        FROM read_parquet(?, filename = true)
        GROUP BY session_id, stroke_type
        ORDER BY session_id, stroke_type
    """
    with duckdb.connect() as connection:
        result = connection.execute(query, [pattern])
        columns = [d[0] for d in result.description]
        rows = [dict(zip(columns, r, strict=True)) for r in result.fetchall()]
    dates = {s.id: session_date(s) for s in list_sessions(config.paths.data_root)}
    for row in rows:
        row["date"] = dates.get(str(row["session_id"]), str(row["session_id"]))
    rows = [r for r in rows if not since or str(r["date"]) >= since]
    return sorted(rows, key=lambda r: (str(r["date"]), str(r["session_id"])))


def _trend_table_without_duckdb(config: Config, since: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in history(config, since=since):
        for stroke_type, count in summary.counts.items():
            row: dict[str, Any] = {
                "session_id": summary.session_id,
                "date": summary.date,
                "stroke_type": stroke_type,
                "swings": count,
            }
            for name in REGISTRY:
                row[f"{name}_mean"] = summary.means.get((stroke_type, name))
                row[f"{name}_std"] = summary.stds.get((stroke_type, name))
                row[f"{name}_count"] = count
            rows.append(row)
    return rows


# --- label analysis -----------------------------------------------------------------------------


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardized difference of two groups' means; NaN when either is too small."""
    x = np.array([v for v in a if np.isfinite(v)], dtype=np.float64)
    y = np.array([v for v in b if np.isfinite(v)], dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    pooled = np.sqrt(((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1))
                     / (x.size + y.size - 2))  # fmt: skip
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / pooled)


def label_analysis(
    rows: Sequence[dict[str, Any]], labels_by_swing: dict[int, list[str]]
) -> list[dict[str, Any]]:
    """Cohen's d between the ``good`` swings and each other label, per metric, by |d|."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for label in labels_by_swing.get(int(row["swing_id"]), ()):
            groups.setdefault(label, []).append(row)
    good = groups.get(GOOD_LABEL, [])
    if not good:
        return []
    out = []
    for label, subset in sorted(groups.items()):
        if label == GOOD_LABEL:
            continue
        for name in REGISTRY:
            d = cohens_d([float(r[name]) for r in good], [float(r[name]) for r in subset])
            if np.isfinite(d):
                out.append(
                    {
                        "label": label,
                        "metric": name,
                        "unit": REGISTRY[name].unit,
                        "d": d,
                        "n_good": len(good),
                        "n_label": len(subset),
                    }
                )
    return sorted(out, key=lambda r: -abs(float(r["d"])))


# --- figures --------------------------------------------------------------------------------------


def _plotly() -> Any:
    import plotly.graph_objects as go

    return go


def figure_html(fig: Any, div_id: str) -> str:
    """One figure as a div, without its own copy of plotly.js."""
    html_text: str = fig.to_html(
        full_html=False, include_plotlyjs=False, div_id=div_id, default_width="100%"
    )
    return html_text


def plotly_script() -> str:
    from plotly.offline import get_plotlyjs

    return f"<script>{get_plotlyjs()}</script>"


def distribution_figures(rows: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    """A histogram per metric, one trace per stroke type."""
    go = _plotly()
    figures = []
    types = sorted({str(r["stroke_type"]) for r in rows})
    for name, meta in REGISTRY.items():
        fig = go.Figure()
        any_data = False
        for stroke_type in types:
            values = [
                float(r[name])
                for r in rows
                if r["stroke_type"] == stroke_type and np.isfinite(float(r[name]))
            ]
            if not values:
                continue
            any_data = True
            fig.add_trace(go.Histogram(x=values, name=stroke_type, opacity=0.65, nbinsx=20))
        if not any_data:
            continue
        fig.update_layout(
            barmode="overlay",
            template="plotly_white",
            height=280,
            margin={"l": 50, "r": 20, "t": 40, "b": 40},
            title=f"{name} ({meta.unit})" if meta.unit else name,
            legend={"orientation": "h", "y": -0.2},
        )
        figures.append((name, figure_html(fig, f"dist-{name}")))
    return figures


def trend_figures(
    rows: Sequence[dict[str, Any]], metrics: Sequence[str], highlight: str | None = None
) -> list[tuple[str, str]]:
    """Session mean with a +- std band, one line per stroke type, per metric."""
    go = _plotly()
    figures = []
    for name in metrics:
        if name not in REGISTRY:
            continue
        fig = go.Figure()
        any_data = False
        for stroke_type in sorted({str(r["stroke_type"]) for r in rows}):
            series = [r for r in rows if r["stroke_type"] == stroke_type]
            x = [str(r["date"]) for r in series]
            mean = [_float(r.get(f"{name}_mean")) for r in series]
            std = [_float(r.get(f"{name}_std")) for r in series]
            if not any(np.isfinite(v) for v in mean):
                continue
            any_data = True
            upper = [m + (s if np.isfinite(s) else 0.0) for m, s in zip(mean, std, strict=True)]
            lower = [m - (s if np.isfinite(s) else 0.0) for m, s in zip(mean, std, strict=True)]
            fig.add_trace(go.Scatter(x=[*x, *reversed(x)], y=[*upper, *reversed(lower)],
                                     fill="toself", mode="lines", line={"width": 0},
                                     opacity=0.15, showlegend=False, hoverinfo="skip",
                                     name=f"{stroke_type} band"))  # fmt: skip
            fig.add_trace(go.Scatter(x=x, y=mean, mode="lines+markers", name=stroke_type))
        if not any_data:
            continue
        if highlight is not None:
            fig.add_vline(x=highlight, line={"color": "#d62728", "width": 1, "dash": "dot"})
        fig.update_layout(
            template="plotly_white",
            height=320,
            margin={"l": 50, "r": 20, "t": 40, "b": 40},
            title=f"{name} ({REGISTRY[name].unit})" if REGISTRY[name].unit else name,
            legend={"orientation": "h", "y": -0.25},
        )
        figures.append((name, figure_html(fig, f"trend-{name}")))
    return figures


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


# --- the per-session report --------------------------------------------------------------------


@dataclass
class MetricRow:
    metric: str
    unit: str
    stroke_type: str
    count: int
    mean: float
    std: float
    median: float
    p10: float
    p90: float
    delta: float = float("nan")
    direction: str = ""  # "good", "bad" or "" when nobody has said which way is better


@dataclass
class SessionReport:
    session_id: str
    date: str
    summary: dict[str, Any]
    metrics: list[MetricRow]
    trends: list[tuple[str, str]] = field(default_factory=list)
    distributions: list[tuple[str, str]] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)
    clips: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    baseline_sessions: list[str] = field(default_factory=list)


def build_session_report(session: Session, config: Config) -> SessionReport:
    for name in ("metrics.parquet", "metrics_summary.parquet", "swing_info.parquet"):
        if not session.path(name).exists():
            raise UserError(f"session '{session.id}' has no {name}; run 'tennis process' first")
    rows = pq.read_table(session.path("metrics.parquet")).to_pylist()
    summary_rows = pq.read_table(session.path("metrics_summary.parquet")).to_pylist()
    labels_by_swing: dict[int, list[str]] = {}
    for row in read_labels(session):
        labels_by_swing.setdefault(int(row["swing_id"]), []).append(str(row["label"]))

    past = history(config, before=session.id)
    baseline = rolling_baseline(past, config.report.rolling_sessions)
    directions = config.report.metric_direction

    metrics: list[MetricRow] = []
    for row in summary_rows:
        key = (str(row["stroke_type"]), str(row["metric"]))
        delta = float("nan")
        if key in baseline:
            delta = float(row["mean"]) - baseline[key]
        wanted = directions.get(key[1])
        direction = ""
        if wanted and np.isfinite(delta) and delta != 0:
            improving = (delta > 0) == (wanted == "up")
            direction = "good" if improving else "bad"
        metrics.append(
            MetricRow(
                metric=key[1],
                unit=str(row["unit"]),
                stroke_type=key[0],
                count=int(row["count"]),
                mean=float(row["mean"]),
                std=_float(row["std"]),
                median=float(row["median"]),
                p10=float(row["p10"]),
                p90=float(row["p90"]),
                delta=delta,
                direction=direction,
            )
        )

    trend_rows = trend_table(config)
    report = SessionReport(
        session_id=session.id,
        date=session_date(session),
        summary=_session_summary(session, rows, labels_by_swing),
        metrics=metrics,
        trends=trend_figures(
            trend_rows, config.report.trend_metrics, highlight=session_date(session)
        ),
        distributions=distribution_figures(rows),
        labels=label_analysis(rows, labels_by_swing),
        clips=_clip_entries(session),
        diagnostics=_diagnostics(session),
        baseline_sessions=[s.session_id for s in past[-config.report.rolling_sessions :]],
    )
    return report


def _session_summary(
    session: Session, rows: Sequence[dict[str, Any]], labels_by_swing: dict[int, list[str]]
) -> dict[str, Any]:
    meta = _json_or_empty(session, "metadata.json")
    info = pq.read_table(session.path("swing_info.parquet")).to_pylist()
    players = _json_or_empty(session, "players.json")
    contacts = 0
    if session.path("contacts.parquet").exists():
        contacts = pq.read_metadata(session.path("contacts.parquet")).num_rows
    passed = sum(1 for r in info if r["qc_pass"])
    confirmed = sum(1 for r in info if r.get("is_self_confirmed"))
    per_type: dict[str, int] = {}
    for row in rows:
        key = str(row["stroke_type"])
        per_type[key] = per_type.get(key, 0) + 1
    label_counts: dict[str, int] = {}
    for values in labels_by_swing.values():
        for value in values:
            label_counts[value] = label_counts.get(value, 0) + 1
    return {
        "duration_s": _float(meta.get("duration_s")),
        "fps": _float(meta.get("fps")),
        "resolution": meta.get("resolution"),
        "contacts": contacts,
        "own_swings": len(info),
        "confirmed": confirmed,
        "qc_pass": passed,
        "qc_pass_rate": passed / len(info) if info else float("nan"),
        "measured": len(rows),
        "per_stroke_type": per_type,
        "outliers": sum(1 for r in rows if r.get("is_outlier")),
        "labels": label_counts,
        "me": players.get("me"),
        "me_reason": players.get("me_reason"),
        "handedness": (players.get("handedness") or {}).get("hand"),
        "sides": players.get("sides") or [],
    }


def _clip_entries(session: Session) -> list[dict[str, Any]]:
    path = session.path(INDEX_NAME)
    if not path.exists():
        return []
    index = read_json(path)
    return [c for c in index.get("clips", []) if c.get("rendered")]


def _json_or_empty(session: Session, name: str) -> dict[str, Any]:
    path = session.path(name)
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _diagnostics(session: Session) -> dict[str, Any]:
    meta = _json_or_empty(session, "metadata.json")
    info = pq.read_table(session.path("swing_info.parquet")).to_pylist()
    out: dict[str, Any] = {
        "ingest_warnings": meta.get("warnings") or [],
        "swaps": sum(int(r.get("swap_count") or 0) for r in info),
        "track_resets": sum(int(r.get("track_reset_count") or 0) for r in info),
        "far_side_swings": sum(1 for r in info if (r.get("player_side") or "near") == "far"),
        "qc_reasons": {},
    }
    reasons: dict[str, int] = {}
    for row in info:
        reason = str(row.get("qc_reason") or "")
        if reason and not row.get("qc_pass"):
            reasons[reason] = reasons.get(reason, 0) + 1
    out["qc_reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:10])

    if session.path("contacts.parquet").exists():
        contacts = pq.read_table(session.path("contacts.parquet"), columns=["is_self_audio"])
        first_pass = contacts.column("is_self_audio").to_pylist()
        out["contacts"] = len(first_pass)
        out["first_pass_own_hits"] = sum(1 for v in first_pass if v)
    out["confirmed_own_hits"] = sum(1 for r in info if r.get("is_self_confirmed"))

    keypoints = session.path("keypoints.parquet")
    if keypoints.exists():
        # The report must not fail over a diagnostic. An older keypoints file may not have
        # the slot column, and a hand-built one may have neither.
        names = set(pq.read_schema(keypoints).names)
        if "detected" in names:
            wanted = ["detected"] + (["slot"] if "slot" in names else [])
            table = pq.read_table(keypoints, columns=wanted).to_pydict()
            slots = table.get("slot") or ["near"] * len(table["detected"])
            for slot in ("near", "far"):
                flags = [d for s, d in zip(slots, table["detected"], strict=True) if s == slot]
                if flags:
                    out[f"{slot}_detection_rate"] = sum(1 for f in flags if f) / len(flags)

    players_path = session.path("players.json")
    if players_path.exists():
        players = read_json(players_path)
        margins = [float(w.get("margin") or 0.0) for w in players.get("windows") or []]
        out["identity_resolved"] = players.get("identities_resolved")
        out["identity_margin_median"] = float(np.median(margins)) if margins else float("nan")
        out["identity_margin_min"] = float(np.min(margins)) if margins else float("nan")
        out["hits"] = players.get("hits")
    return out


# --- rendering ------------------------------------------------------------------------------------


def render(template_name: str, **context: Any) -> str:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(["html", "xml", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["number"] = _format_number
    env.filters["signed"] = _format_signed
    env.filters["duration"] = _format_duration
    text: str = env.get_template(template_name).render(
        plotly_script=plotly_script(),
        generated_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        **context,
    )
    return text


def _format_number(value: Any, digits: int = 2) -> str:
    number = _float(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "-"


def _format_signed(value: Any, digits: int = 2) -> str:
    number = _float(value)
    return f"{number:+.{digits}f}" if np.isfinite(number) else "-"


def _format_duration(value: Any) -> str:
    seconds = _float(value)
    if not np.isfinite(seconds):
        return "-"
    return f"{int(seconds // 60)} min {int(seconds % 60):02d} s"


def write_session_report(session: Session, config: Config, out: Path | None = None) -> Path:
    report = build_session_report(session, config)
    text = render("report.html.j2", report=report, metrics_meta=REGISTRY)
    out = out or session.path("report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def write_trends_report(config: Config, since: str | None = None, out: Path | None = None) -> Path:
    rows = trend_table(config, since=since)
    if not rows:
        raise UserError(
            f"no session under {sessions_root(config.paths.data_root)} has metrics.parquet; "
            "run 'tennis process' first"
        )
    sessions = sorted({(str(r["date"]), str(r["session_id"])) for r in rows})
    text = render(
        "trends.html.j2",
        rows=rows,
        sessions=sessions,
        since=since,
        figures=trend_figures(rows, list(REGISTRY)),
        metrics_meta=REGISTRY,
    )
    out = out or config.paths.data_root / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out
