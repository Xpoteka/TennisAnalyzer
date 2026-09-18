"""A player's profile: numbers across all their sessions, the trend, and insights."""

from __future__ import annotations

import warnings
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from tennis.analysis.technique import METRIC_LABELS, insights

STROKES = ("serve", "forehand", "backhand", "volley_forehand", "volley_backhand", "overhead")


def _round(v: float | None, n: int = 1) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), n)


def _stats(values: list[float]) -> dict[str, float | None]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {"avg": None, "max": None, "n": 0}
    return {"avg": _round(float(np.mean(vals))), "max": _round(float(np.max(vals))), "n": len(vals)}


def player_profile(
    shots: list[dict[str, Any]],
    metrics: dict[int, dict[str, float]],
    sessions: list[dict[str, Any]],
) -> dict[str, Any]:
    """``shots``: the player's shots (dicts with the Shot fields and ``session_id``);
    ``metrics``: technique values per shot id; ``sessions``: id, recorded_at, kind."""
    by_stroke: dict[str, dict[str, Any]] = {}
    for stroke in STROKES:
        mine = [s for s in shots if s["stroke"] == stroke]
        if not mine:
            continue
        speed = _stats([s["speed_kmh"] for s in mine if s["speed_kmh"] is not None])
        landed = [s["in_court"] for s in mine if s["in_court"] is not None]
        clear = [s["net_clearance_m"] for s in mine if s["net_clearance_m"] is not None]
        heights = [s["contact_height_m"] for s in mine if s["contact_height_m"] is not None]
        tech: dict[str, float | None] = {}
        for name in METRIC_LABELS:
            vals = [metrics[s["id"]][name] for s in mine if name in metrics.get(s["id"], {})]
            if len(vals) >= 5:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    tech[name] = _round(float(np.median(vals)), 3)
        by_stroke[stroke] = {
            "count": len(mine),
            "speed_avg": speed["avg"],
            "speed_max": speed["max"],
            "n_speed": speed["n"],
            "in_pct": _round(sum(landed) / len(landed), 3) if landed else None,
            "n_in": len(landed),
            "net_clearance_avg": _round(float(np.mean(clear)), 2) if clear else None,
            "n_clear": len(clear),
            "contact_height_m": _round(float(np.median(heights)), 2) if heights else None,
            "technique": tech,
        }

    all_metrics: dict[str, list[float]] = defaultdict(list)
    for s in shots:
        for name, value in metrics.get(s["id"], {}).items():
            all_metrics[name].append(value)

    # One point per session, oldest first, for the trend charts.
    trend = []
    for sess in sorted(sessions, key=lambda x: x["recorded_at"] or datetime.min):
        mine = [s for s in shots if s["session_id"] == sess["id"]]
        if not mine:
            continue

        def avg(stroke: str, key: str, rows: list[dict[str, Any]] = mine) -> float | None:
            vals = [r[key] for r in rows if r["stroke"] == stroke and r[key] is not None]
            return _round(float(np.mean(vals))) if len(vals) >= 3 else None

        landed = [s["in_court"] for s in mine if s["in_court"] is not None]
        knee = [
            metrics[s["id"]]["knee_bend_deg"]
            for s in mine
            if "knee_bend_deg" in metrics.get(s["id"], {})
        ]
        split = [
            metrics[s["id"]]["split_step"] for s in mine if "split_step" in metrics.get(s["id"], {})
        ]
        trend.append(
            {
                "session_id": sess["id"],
                "recorded_at": sess["recorded_at"],
                "kind": sess["kind"],
                "shots": len(mine),
                "forehand_kmh": avg("forehand", "speed_kmh"),
                "backhand_kmh": avg("backhand", "speed_kmh"),
                "serve_kmh": avg("serve", "speed_kmh"),
                "in_pct": _round(sum(landed) / len(landed), 3) if len(landed) >= 5 else None,
                "knee_bend_deg": _round(float(np.median(knee))) if len(knee) >= 5 else None,
                "split_step_pct": _round(float(np.mean(split)), 3) if len(split) >= 5 else None,
            }
        )

    return {
        "by_stroke": by_stroke,
        "technique": {
            name: {
                "label": label,
                "unit": unit,
                "median": _round(float(np.median(vals)), 3) if len(vals) >= 5 else None,
                "n": len(vals),
            }
            for name, (label, unit) in METRIC_LABELS.items()
            for vals in [all_metrics.get(name, [])]
        },
        "trend": trend,
        "insights": [{"text": i.text, "level": i.level} for i in insights(by_stroke, all_metrics)],
    }
