"""Video stage ``ball``: the ball's image position in every frame it can be found in.

Every frame is decoded (at most 1280 pixels wide), candidates come from three-frame motion
differencing and are linked into trajectories (see :mod:`tennis.vision.ball`). People found
by the ``people`` stage mark where arms and rackets move, so those blobs count less.

Writes ``ball.parquet``: ``frame_idx``, ``t``, ``x``, ``y`` (source pixels), ``track`` (the
tracklet id) and ``strength``.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tennis.util.io import write_parquet
from tennis.vision.ball import DetectorParams, Linker, candidates, choose_ball, prepare

if TYPE_CHECKING:
    from tennis.pipeline import VideoContext

OUTPUTS = ("ball.parquet",)
SCHEMA_VERSION = 1
WORK_WIDTH = 1280


def person_boxes(
    ctx: VideoContext, scale: float
) -> tuple[list[int], list[list[tuple[float, ...]]]]:
    """Sampled frame indices and the (scaled) boxes of the people in each."""
    t = pq.read_table(ctx.path("people.parquet"), columns=["frame_idx", "x1", "y1", "x2", "y2"])
    by_frame: dict[int, list[tuple[float, ...]]] = defaultdict(list)
    cols = [t.column(c).to_numpy() for c in ("frame_idx", "x1", "y1", "x2", "y2")]
    for f, x1, y1, x2, y2 in zip(*cols, strict=True):
        by_frame[int(f)].append((x1 * scale, y1 * scale, x2 * scale, y2 * scale))
    frames = sorted(by_frame)
    return frames, [by_frame[f] for f in frames]


def run(ctx: VideoContext) -> None:
    cfg = ctx.config.ball
    reader = ctx.frame_reader(out_width=WORK_WIDTH)
    scale = reader.scale
    area = reader.width * reader.height
    params = DetectorParams(min_area=cfg.min_area_px, max_area=cfg.max_area_frac * area)
    sampled, boxes = person_boxes(ctx, scale)
    linker = Linker(cfg.max_gap_frames)
    pts = ctx.frame_pts()
    total = len(pts)

    window: list[tuple[int, np.ndarray, np.ndarray]] = []  # (frame, gray, hsv), last three
    for n, frame in enumerate(reader.read(ctx.whole_video()), start=1):
        gray, hsv = prepare(frame.image)
        window.append((frame.index, gray, hsv))
        if len(window) > 3:
            window.pop(0)
        if len(window) == 3:
            (_, g0, _), (f1, g1, h1), (_, g2, _) = window
            k = bisect_right(sampled, f1) - 1
            people = boxes[k] if k >= 0 else []
            linker.update(f1, candidates(g0, g1, g2, h1, people, params))  # type: ignore[arg-type]
        if n % 200 == 0:
            ctx.progress(n / total, f"{n} of {total} frames")

    tracklets = linker.close()
    ball = choose_ball(tracklets)
    frames = sorted(ball)
    table = pa.table(
        {
            "frame_idx": pa.array(frames, pa.int64()),
            "t": pa.array([float(pts[f]) for f in frames], pa.float64()),
            "x": pa.array([ball[f][0] / scale for f in frames], pa.float32()),
            "y": pa.array([ball[f][1] / scale for f in frames], pa.float32()),
            "track": pa.array([ball[f][2] for f in frames], pa.int32()),
            "strength": pa.array([ball[f][3] for f in frames], pa.float32()),
        }
    )
    write_parquet(
        table,
        ctx.path("ball.parquet"),
        stage=ctx.stage,
        config_hash=ctx.config.section_hash("ball"),
        schema_version=SCHEMA_VERSION,
    )
    ctx.log(
        "ball found",
        frames=len(frames),
        share=round(len(frames) / max(1, total), 3),
        tracklets=len(tracklets),
    )
