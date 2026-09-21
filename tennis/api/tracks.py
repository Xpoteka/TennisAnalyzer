"""What the review player draws over a video: ball, poses and feet, in slices of time.

Everything is in the video's own pixels and PTS seconds, as the stages wrote it. Pixels are
rounded to whole numbers and court positions to centimetres, which keeps a slice small.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from tennis.vision import court as court_mod

MAX_SLICE_S = 30.0
KP_CONF_MIN = 0.3  # a keypoint below this is sent as missing


@lru_cache(maxsize=8)
def _table(path: str, mtime_ns: int, columns: tuple[str, ...]) -> dict[str, Any]:
    """The columns of one parquet file, sorted by time. Cached until the file changes."""
    table = pq.read_table(path, columns=list(columns))
    t = table.column("t").to_numpy()
    order = np.argsort(t, kind="stable")
    out: dict[str, Any] = {"t": t[order]}
    for name in columns:
        if name == "t":
            continue
        col = table.column(name)
        if str(col.type).startswith("list"):
            # Keypoint lists all have one length, so they become one (rows, keypoints) array.
            flat = col.combine_chunks().flatten().to_numpy(zero_copy_only=False)
            width = len(flat) // max(len(t), 1)
            out[name] = flat.reshape(len(t), width)[order] if len(t) else flat.reshape(0, 0)
        else:
            out[name] = col.to_numpy()[order]
    return out


def _load(path: Path, columns: tuple[str, ...]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _table(str(path), path.stat().st_mtime_ns, columns)


def _span(t: Any, start: float, end: float) -> range:
    lo, hi = np.searchsorted(t, [start, end])
    return range(int(lo), int(hi))


def _ints(values: Any, conf: Any) -> list[int | None]:
    seen = (conf >= KP_CONF_MIN) & np.isfinite(values)
    whole = np.rint(np.where(seen, values, 0)).astype(int).tolist()
    return [v if ok else None for v, ok in zip(whole, seen.tolist(), strict=True)]


def tracks_slice(directory: Path, start: float, end: float) -> dict[str, Any]:
    """Ball, poses and feet between two PTS times (``end`` exclusive)."""
    end = min(end, start + MAX_SLICE_S)
    ball_rows: list[list[Any]] = []
    ball = _load(directory / "ball.parquet", ("t", "x", "y", "track"))
    if ball is not None:
        for i in _span(ball["t"], start, end):
            ball_rows.append(
                [round(float(ball["t"][i]), 3), round(float(ball["x"][i])),
                 round(float(ball["y"][i])), int(ball["track"][i])]
            )  # fmt: skip

    pose_rows: list[list[Any]] = []
    motion = _load(directory / "motion.parquet", ("t", "track", "kp_x", "kp_y", "kp_c"))
    if motion is not None:
        for i in _span(motion["t"], start, end):
            conf = motion["kp_c"][i]
            pose_rows.append(
                [round(float(motion["t"][i]), 3), int(motion["track"][i]),
                 _ints(motion["kp_x"][i], conf), _ints(motion["kp_y"][i], conf)]
            )  # fmt: skip

    feet_rows: list[list[Any]] = []
    people = _load(
        directory / "people.parquet", ("t", "track_id", "foot_px", "foot_py", "court_x", "court_y")
    )
    if people is not None:
        for i in _span(people["t"], start, end):
            px, py = float(people["foot_px"][i]), float(people["foot_py"][i])
            cx, cy = float(people["court_x"][i]), float(people["court_y"][i])
            if not (np.isfinite(px) and np.isfinite(py)):
                continue
            on_court = np.isfinite(cx) and np.isfinite(cy)
            feet_rows.append(
                [round(float(people["t"][i]), 3), int(people["track_id"][i]), round(px), round(py),
                 round(cx, 2) if on_court else None, round(cy, 2) if on_court else None]
            )  # fmt: skip
    return {"start": start, "end": end, "ball": ball_rows, "poses": pose_rows, "feet": feet_rows}


def court_outline(court: dict[str, Any] | None) -> list[list[list[int]]]:
    """Every painted line and the net's foot as pixel polylines; empty without a court."""
    if not court or not court.get("H"):
        return []
    cal = court_mod.Calibration.from_json(court)
    lines = [*court_mod.COURT_LINES.values(), (-court_mod.HALF_DW, 0.0, court_mod.HALF_DW, 0.0)]
    out = []
    for x1, y1, x2, y2 in lines:
        s = np.linspace(0, 1, 12)  # several points, so lens distortion bends the line
        px = cal.court_to_image(np.stack([x1 + (x2 - x1) * s, y1 + (y2 - y1) * s], axis=1))
        if np.all(np.isfinite(px)):
            out.append([[round(float(x)), round(float(y))] for x, y in px])
    return out


def court_points(court: dict[str, Any] | None, xy: list[tuple[float, float]]) -> list[Any]:
    """Ground points in court metres -> pixels (None where it cannot be projected)."""
    if not court or not court.get("H") or not xy:
        return [None] * len(xy)
    px = court_mod.Calibration.from_json(court).court_to_image(np.asarray(xy, np.float64))
    return [
        [round(float(x)), round(float(y))] if np.isfinite(x) and np.isfinite(y) else None
        for x, y in px
    ]
