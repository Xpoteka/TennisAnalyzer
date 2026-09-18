"""Review videos: the analysis drawn over the footage, to check it by eye.

``tennis review <video-id>`` writes ``videos/<id>/debug/review.mp4`` with:

- the court lines found by the ``court`` stage (red), and the net (yellow);
- every tracked person, with their skeleton and track number;
- the ball, with a trail of the last half second (a colour per tracklet);
- a white flash in the corner on every audio onset.
"""

from __future__ import annotations

import colorsys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

from tennis.pipeline import video_dir
from tennis.pipeline.video.court import draw as draw_court
from tennis.util.frames import FrameReader, Window
from tennis.util.io import read_json
from tennis.util.overlay import VideoWriter, draw_label, draw_pose
from tennis.vision.court import Calibration


def _color(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def render(data_root: Path, video_id: int, source: Path, start: float, duration: float) -> Path:
    d = video_dir(data_root, video_id)
    meta = read_json(d / "metadata.json")
    pts = pq.read_table(d / "frame_times.parquet", columns=["pts"]).column("pts").to_numpy()
    w, h = int(meta["width"]), int(meta["height"])
    if int(meta["video"].get("rotation_deg") or 0) % 180 == 90:
        w, h = h, w
    court = read_json(d / "court.json") if (d / "court.json").is_file() else {}
    cal = Calibration.from_json(court) if court.get("H") else None

    people: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if (d / "people.parquet").is_file():
        rows = pq.read_table(d / "people.parquet").to_pylist()
        for r in rows:
            people[r["frame_idx"]].append(r)
    sampled = sorted(people)
    ball: dict[int, tuple[float, float, int]] = {}
    if (d / "ball.parquet").is_file():
        for r in pq.read_table(d / "ball.parquet").to_pylist():
            ball[r["frame_idx"]] = (r["x"], r["y"], r["track"])
    onsets = (
        np.asarray(pq.read_table(d / "onsets.parquet").column("t").to_numpy())
        if (d / "onsets.parquet").is_file()
        else np.zeros(0)
    )

    t0 = float(pts[0]) + start
    window = Window(0, t0, t0 + duration)
    reader = FrameReader(source, pts, w, h, float(meta.get("video_start_s") or 0.0))
    out = d / "debug" / "review.mp4"
    fps = float(meta.get("fps") or 30.0)
    trail = max(3, int(fps * 0.5))
    with VideoWriter(out, w, h, fps, out_height=min(h, 1080)) as writer:
        for frame in reader.read([window]):
            img = frame.image.copy()
            if cal is not None:
                img = draw_court(img, cal)
            k = bisect_right(sampled, frame.index) - 1
            for p in people.get(sampled[k], []) if k >= 0 else []:
                kp = np.stack([p["kp_x"], p["kp_y"], p["kp_c"]], axis=1).astype(np.float32)
                draw_pose(img, kp, (p["x1"], p["y1"], p["x2"], p["y2"]), "right", 0.3)
                cv2.putText(
                    img, f"#{p['track_id']}", (int(p["x1"]), int(p["y1"]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
                )  # fmt: skip
            for f in range(frame.index - trail, frame.index + 1):
                if f in ball:
                    x, y, tr = ball[f]
                    r = 6 if f == frame.index else 3
                    cv2.circle(img, (int(x), int(y)), r, _color(tr), -1 if f == frame.index else 1)
            if onsets.size and np.min(np.abs(onsets - frame.pts)) < 0.05:
                cv2.rectangle(img, (w - 60, 10), (w - 10, 60), (255, 255, 255), -1)
            draw_label(img, [f"{frame.pts - pts[0]:7.2f} s  frame {frame.index}"])
            writer.write(img)
    return out
