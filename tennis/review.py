"""Pose review video: sampled windows with the skeleton drawn, for the M3 acceptance check.

The M3 criterion is reviewed by eye: the skeleton must follow the player through at least
95% of the frames of 20 sampled swings. This module renders those swings into one video
and writes a per-window summary (frames, detected ratio, track resets).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from tennis.config import Config
from tennis.errors import UserError
from tennis.pose_backends.base import KEYPOINT_NAMES
from tennis.session import Session
from tennis.stages.clean import resolved_handedness
from tennis.stages.pose import frame_size, read_frame_pts
from tennis.util.frames import FrameReader, Window
from tennis.util.io import read_json
from tennis.util.overlay import CONTACT, VideoWriter, draw_border, draw_label, draw_pose

CONTACT_HIGHLIGHT_S = 0.05


@dataclass(frozen=True)
class WindowSummary:
    window_id: int
    start: float
    end: float
    frames: int
    detected: int
    resets: int

    @property
    def detected_ratio(self) -> float:
        return self.detected / self.frames if self.frames else 0.0


def sample_windows(window_ids: list[int], count: int, seed: int) -> list[int]:
    unique = sorted(set(window_ids))
    if count >= len(unique):
        return unique
    rng = np.random.default_rng(seed)
    return sorted(int(w) for w in rng.choice(unique, size=count, replace=False))


def render_pose_preview(
    session: Session,
    config: Config,
    *,
    count: int = 20,
    seed: int = 0,
    speed: float = 1.0,
    out: Path | None = None,
) -> tuple[Path, list[WindowSummary]]:
    path = session.path("keypoints.parquet")
    if not path.exists():
        raise UserError(f"session '{session.id}' has no keypoints.parquet; run 'tennis process'")
    table = pq.read_table(path)
    if table.num_rows == 0:
        raise UserError(f"session '{session.id}' has no pose frames to review")
    data = table.to_pydict()
    slots = data.get("slot") or ["near"] * table.num_rows
    hand = resolved_handedness(session, config)
    meta = read_json(session.path("metadata.json"))
    width, height = frame_size(meta)
    frame_pts = read_frame_pts(session.path("frame_times.parquet"))
    contacts = pq.read_table(session.path("contacts.parquet"), columns=["t_audio"])
    contact_times = np.sort(np.asarray(contacts.column("t_audio").to_numpy(), np.float64))

    chosen = sample_windows(data["window_id"], count, seed)
    rows_by_frame: dict[tuple[int, str], int] = {
        (int(f), slot): i for i, (f, slot) in enumerate(zip(data["frame_idx"], slots, strict=True))
    }
    fps = float(meta.get("fps") or 30.0) * speed
    out = out or session.dir / "debug" / "pose_preview.mp4"
    reader = FrameReader(
        session.source_link.resolve(),
        frame_pts,
        width,
        height,
        video_start_s=float(meta.get("video_start_s", 0.0)),
        hwaccel=config.pose.hwaccel,
    )
    kp_cols = [[f"{n}_{c}" for c in ("x", "y", "conf")] for n in KEYPOINT_NAMES]

    summaries: list[WindowSummary] = []
    with VideoWriter(out, width, height, fps) as writer:
        for n, wid in enumerate(chosen, start=1):
            idx = [i for i, w in enumerate(data["window_id"]) if w == wid and slots[i] == "near"]
            t = [data["t_video"][i] for i in idx]
            window = Window(wid, min(t), max(t))
            detected = sum(bool(data["detected"][i]) for i in idx)
            resets = sum(bool(data["track_reset"][i]) for i in idx)
            summary = WindowSummary(wid, window.start, window.end, len(idx), detected, resets)
            summaries.append(summary)
            for frame in reader.read([window]):
                image = frame.image.copy()
                for slot in ("far", "near"):
                    other = rows_by_frame.get((frame.index, slot))
                    if other is None or not data["detected"][other]:
                        continue
                    kp = np.array(
                        [[data[c][other] for c in cols] for cols in kp_cols], dtype=np.float32
                    )
                    bbox = tuple(
                        data[c][other] for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
                    )
                    draw_pose(image, kp, bbox, hand, config.pose.kp_conf_min)
                row = rows_by_frame.get((frame.index, "near"))
                found = row is not None and bool(data["detected"][row])
                near = np.searchsorted(contact_times, frame.pts)
                is_contact = any(
                    0 <= j < contact_times.size
                    and abs(contact_times[j] - frame.pts) <= CONTACT_HIGHLIGHT_S
                    for j in (near - 1, near)
                )
                if is_contact:
                    draw_border(image, CONTACT, max(4, height // 120))
                lines = [
                    f"swing {n}/{len(chosen)}  window {wid}  t={frame.pts:.2f}s  "
                    f"frame {frame.index}",
                    f"tracked {summary.detected}/{summary.frames} "
                    f"({summary.detected_ratio:.0%})  resets {summary.resets}",
                ]
                if row is not None and bool(data["track_reset"][row]):
                    lines.append("TRACK RESET")
                if not found:
                    lines.append("NO PLAYER")
                draw_label(image, lines, (80, 80, 255) if not found else (255, 255, 255))
                writer.write(image)

    with out.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window_id", "start_s", "end_s", "frames", "detected", "resets"])
        for s in summaries:
            w.writerow([s.window_id, f"{s.start:.3f}", f"{s.end:.3f}", s.frames, s.detected,
                        s.resets])  # fmt: skip
    return out, summaries


def render_swing_plots(
    session: Session,
    config: Config,
    *,
    count: int = 6,
    seed: int = 0,
    confirmed_only: bool = True,
    out: Path | None = None,
) -> Path:
    """Raw vs cleaned racket-wrist trajectories and wrist speed for sampled swings (HTML)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from tennis.stages.clean import racket_side

    for name in ("swings.parquet", "swing_info.parquet", "keypoints.parquet"):
        if not session.path(name).exists():
            raise UserError(f"session '{session.id}' has no {name}; run 'tennis process'")
    info = pq.read_table(session.path("swing_info.parquet")).to_pylist()
    pool = [r for r in info if r["qc_pass"] and (r["is_self_confirmed"] or not confirmed_only)]
    if not pool:
        raise UserError(
            "no swings to plot (none pass QC" + (" and are confirmed)" if confirmed_only else ")")
        )
    chosen = [pool[i] for i in sample_windows(list(range(len(pool))), count, seed)]

    side = racket_side(resolved_handedness(session, config))
    wrist = f"{side}_wrist"
    swings = pq.read_table(session.path("swings.parquet")).to_pydict()
    raw_table = pq.read_table(session.path("keypoints.parquet"))
    raw = raw_table.select(
        [c for c in ("frame_idx", "slot", f"{wrist}_x", f"{wrist}_y", f"{wrist}_conf")
         if c in raw_table.column_names]
    ).to_pydict()  # fmt: skip
    raw_slots = raw.get("slot") or ["near"] * len(raw["frame_idx"])
    raw_by_frame = {(f, sl): i for i, (f, sl) in enumerate(zip(raw["frame_idx"], raw_slots,
                                                                strict=True))}  # fmt: skip

    fig = make_subplots(
        rows=len(chosen), cols=2, shared_xaxes=False, vertical_spacing=0.04,
        subplot_titles=[
            title
            for r in chosen
            for title in (
                f"swing {r['swing_id']} @ {r['t_contact']:.1f}s: {wrist} (px)",
                f"wrist speed, peak {r['wrist_peak_speed']:.1f} torso/s",
            )
        ],
    )  # fmt: skip
    for n, r in enumerate(chosen, start=1):
        idx = [i for i, s in enumerate(swings["swing_id"]) if s == r["swing_id"]]
        t = [swings["t_rel"][i] for i in idx]
        for axis, color in (("x", "#1f77b4"), ("y", "#2ca02c")):
            raw_vals = []
            for i in idx:
                j = raw_by_frame.get((swings["frame_idx"][i], swings["player_side"][i]))
                ok = j is not None and (raw[f"{wrist}_conf"][j] or 0) >= config.pose.kp_conf_min
                raw_vals.append(raw[f"{wrist}_{axis}"][j] if ok else None)
            fig.add_trace(go.Scatter(x=t, y=raw_vals, mode="markers",
                                     marker={"size": 4, "color": color, "opacity": 0.5},
                                     name=f"raw {axis}", legendgroup=f"raw{axis}",
                                     showlegend=n == 1), row=n, col=1)  # fmt: skip
            fig.add_trace(go.Scatter(x=t, y=[swings[f"{wrist}_p{axis}"][i] for i in idx],
                                     mode="lines", line={"color": color},
                                     name=f"cleaned {axis}", legendgroup=f"clean{axis}",
                                     showlegend=n == 1), row=n, col=1)  # fmt: skip
        for s, label, color in (("l", "left wrist", "#ff7f0e"), ("r", "right wrist", "#9467bd")):
            fig.add_trace(go.Scatter(x=t, y=[swings[f"{s}_wrist_speed"][i] for i in idx],
                                     mode="lines", line={"color": color},
                                     name=label, legendgroup=label,
                                     showlegend=n == 1), row=n, col=2)  # fmt: skip
        for col in (1, 2):
            fig.add_vline(x=0, line={"color": "red", "width": 1}, row=n, col=col)
        fig.update_yaxes(autorange="reversed", row=n, col=1)  # image y points down
    fig.update_layout(height=260 * len(chosen), width=1100, template="plotly_white",
                      title=f"Swing trajectories: {session.id} (red line = audio contact; "
                            "x axis = seconds from contact)")  # fmt: skip
    out = out or session.dir / "debug" / "swing_trajectories.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    return out


def render_players(
    session: Session, config: Config, *, per_player: int = 6, out: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """A thumbnail sheet of both identities (from ``players.json``) for checking who is who."""
    import cv2

    from tennis.util.identity import NAMES

    path = session.path("players.json")
    if not path.exists():
        raise UserError(f"session '{session.id}' has no players.json; run 'tennis process'")
    players = read_json(path)
    windows = players.get("windows") or []
    if not windows:
        raise UserError("no pose windows to show")
    table = pq.read_table(
        session.path("keypoints.parquet"),
        columns=["frame_idx", "t_video", "window_id", "slot", "detected",
                 "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"],
    ).to_pydict()  # fmt: skip
    meta = read_json(session.path("metadata.json"))
    width, height = frame_size(meta)
    frame_pts = read_frame_pts(session.path("frame_times.parquet"))
    reader = FrameReader(session.source_link.resolve(), frame_pts, width, height,
                         video_start_s=float(meta.get("video_start_s", 0.0)),
                         hwaccel=config.pose.hwaccel)  # fmt: skip

    by_window: dict[tuple[int, str], list[int]] = {}
    for i, (wid, slot, det) in enumerate(
        zip(table["window_id"], table["slot"], table["detected"], strict=True)
    ):
        if det:
            by_window.setdefault((wid, slot), []).append(i)

    tile_h, tile_w = 220, 110
    rows_img = []
    for name in NAMES:
        # Windows where this identity was seen, spread over the session, both sides.
        seen = []
        for w in windows:
            slot = "near" if w["near"] == name else "far"
            rows = by_window.get((w["window_id"], slot))
            if rows:
                seen.append((w, slot, rows[len(rows) // 2]))
        picks = [seen[int(k)] for k in np.linspace(0, len(seen) - 1, min(per_player, len(seen)))]
        tiles = []
        for _w, slot, row in picks:
            t = table["t_video"][row]
            frame = next(iter(reader.read([Window(0, t, t)])), None)
            tile = np.zeros((tile_h, tile_w, 3), np.uint8)
            if frame is not None:
                x1, y1, x2, y2 = (int(table[c][row]) for c in ("bbox_x1", "bbox_y1",
                                                              "bbox_x2", "bbox_y2"))  # fmt: skip
                crop = frame.image[max(0, y1) : y2, max(0, x1) : x2]
                if crop.size:
                    scale = min(tile_w / crop.shape[1], (tile_h - 30) / crop.shape[0])
                    small = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)),
                                              max(1, int(crop.shape[0] * scale))))  # fmt: skip
                    tile[30 : 30 + small.shape[0], : small.shape[1]] = small
            text = f"{name} {slot} {int(t // 60)}:{int(t % 60):02d}"
            cv2.putText(tile, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255),
                        1, cv2.LINE_AA)  # fmt: skip
            tiles.append(tile)
        while len(tiles) < per_player:
            tiles.append(np.zeros((tile_h, tile_w, 3), np.uint8))
        label = np.zeros((tile_h, 90, 3), np.uint8)
        you = " (you)" if players.get("me") == name else ""
        cv2.putText(label, f"{name}{you}", (5, tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 200, 255) if you else (255, 255, 255), 1, cv2.LINE_AA)  # fmt: skip
        rows_img.append(np.hstack([label, *tiles]))
    out = out or session.dir / "debug" / "players.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows_img))
    return out, players
