"""Pose review video: sampled windows with the skeleton drawn, for the M3 acceptance check.

The M3 criterion is reviewed by eye: the skeleton must follow the player through at least
95% of the frames of 20 sampled swings. This module renders those swings into one video
and writes a per-window summary (frames, detected ratio, track resets).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tennis.config import Config
from tennis.errors import UserError
from tennis.pose_backends.base import KEYPOINT_NAMES
from tennis.session import Session
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
    meta = read_json(session.path("metadata.json"))
    width, height = frame_size(meta)
    frame_pts = read_frame_pts(session.path("frame_times.parquet"))
    contacts = pq.read_table(session.path("contacts.parquet"), columns=["t_audio"])
    contact_times = np.sort(np.asarray(contacts.column("t_audio").to_numpy(), np.float64))

    chosen = sample_windows(data["window_id"], count, seed)
    rows_by_frame = {int(f): i for i, f in enumerate(data["frame_idx"])}
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
            idx = [i for i, w in enumerate(data["window_id"]) if w == wid]
            t = [data["t_video"][i] for i in idx]
            window = Window(wid, min(t), max(t))
            detected = sum(bool(data["detected"][i]) for i in idx)
            resets = sum(bool(data["track_reset"][i]) for i in idx)
            summary = WindowSummary(wid, window.start, window.end, len(idx), detected, resets)
            summaries.append(summary)
            for frame in reader.read([window]):
                image = frame.image.copy()
                row = rows_by_frame.get(frame.index)
                found = row is not None and bool(data["detected"][row])
                if row is not None and found:
                    kp = np.array(
                        [[data[c][row] for c in cols] for cols in kp_cols], dtype=np.float32
                    )
                    bbox = tuple(data[c][row] for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
                    draw_pose(image, kp, bbox, config.player.handedness, config.pose.kp_conf_min)
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
