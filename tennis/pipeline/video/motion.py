"""Video stage ``motion``: every tracked player's pose in every frame.

The ``people`` stage finds and follows players ten times a second. Here, every frame is
decoded, each player on court is cut out around where the tracker places them (with room
for the racket), enlarged, and run through the pose model. That gives the swing in full:
the wrist's speed peaks at the exact moment of a hit, which tells hits from bounces, and
technique needs every frame of the stroke.

Only tracks lasting a second or more count, at most ``MAX_PLAYERS`` per frame (the largest).

Writes ``motion.parquet``: ``frame_idx``, ``t``, ``track``, keypoints (``kp_x``, ``kp_y``,
``kp_c``, source pixels) and ``height`` (the player's box height, pixels).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tennis.pose_backends import PersonPose, create_backend
from tennis.util.io import write_parquet

if TYPE_CHECKING:
    from tennis.pipeline import VideoContext

OUTPUTS = ("motion.parquet",)
SCHEMA_VERSION = 1
CROP_HEIGHT = 288
CROP_IMGSZ = 320
MIN_TRACK_S = 1.0
MAX_GAP_S = 0.25  # a player's box is only placed within this of a tracker sample
MAX_PLAYERS = 4


class TrackBoxes:
    def __init__(self, t: np.ndarray, boxes: np.ndarray) -> None:
        self.t = t
        self.boxes = boxes

    def at(self, t: float) -> np.ndarray | None:
        i = int(np.searchsorted(self.t, t))
        near = min((abs(self.t[j] - t) for j in (i - 1, i) if 0 <= j < len(self.t)), default=np.inf)
        if near > MAX_GAP_S:
            return None
        return np.array([np.interp(t, self.t, self.boxes[:, k]) for k in range(4)])


def load_tracks(ctx: VideoContext) -> dict[int, TrackBoxes]:
    d = pq.read_table(
        ctx.path("people.parquet"), columns=["t", "track_id", "x1", "y1", "x2", "y2"]
    ).to_pydict()
    ids = np.asarray(d["track_id"])
    times = np.asarray(d["t"], np.float64)
    boxes = np.stack([np.asarray(d[k], np.float64) for k in ("x1", "y1", "x2", "y2")], axis=1)
    out = {}
    for tid in np.unique(ids):
        m = ids == tid
        if times[m].max() - times[m].min() < MIN_TRACK_S:
            continue
        order = np.argsort(times[m])
        out[int(tid)] = TrackBoxes(times[m][order], boxes[m][order])
    return out


def crop_for(box: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    """Around the player, with room for an outstretched racket on either side and above."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_top, pad_bottom = max(bw * 0.8, bh * 0.45), bh * 0.35, bh * 0.08
    return (
        int(max(0, x1 - pad_x)),
        int(max(0, y1 - pad_top)),
        int(min(w, x2 + pad_x)),
        int(min(h, y2 + pad_bottom)),
    )


def best_person(people: list[PersonPose], expected: np.ndarray) -> PersonPose | None:
    """The detection overlapping the expected box most (at least a little)."""
    best, best_iou = None, 0.15
    ea = (expected[2] - expected[0]) * (expected[3] - expected[1])
    for p in people:
        a = p.bbox
        ix = max(0.0, min(a[2], expected[2]) - max(a[0], expected[0]))
        iy = max(0.0, min(a[3], expected[3]) - max(a[1], expected[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + ea - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = p, iou
    return best


def run(ctx: VideoContext) -> None:
    cfg = ctx.config.pose
    tracks = load_tracks(ctx)
    rows: dict[str, list[Any]] = {
        k: [] for k in ("frame_idx", "t", "track", "kp_x", "kp_y", "kp_c", "height")
    }
    if tracks:
        backend, device = create_backend(
            cfg.model_copy(update={"imgsz": CROP_IMGSZ}), ctx.data_root
        )
        reader = ctx.frame_reader()
        total = len(ctx.frame_pts())
        pending: list[tuple[int, float, int, tuple[int, int, int, int], float, np.ndarray]] = []
        images: list[np.ndarray] = []

        def flush() -> None:
            found = backend.infer(images) if images else []
            for (f, t, k, c, z, expected), people in zip(pending, found, strict=True):
                mapped = [_uncrop(p, c, z) for p in people]
                person = best_person(mapped, expected)
                if person is None:
                    continue
                kp = person.keypoints
                rows["frame_idx"].append(f)
                rows["t"].append(t)
                rows["track"].append(k)
                rows["kp_x"].append(kp[:, 0].tolist())
                rows["kp_y"].append(kp[:, 1].tolist())
                rows["kp_c"].append(kp[:, 2].tolist())
                rows["height"].append(float(expected[3] - expected[1]))
            pending.clear()
            images.clear()

        for n, frame in enumerate(reader.read(ctx.whole_video()), start=1):
            active = []
            for k, tb in tracks.items():
                box = tb.at(frame.pts)
                if box is not None:
                    active.append((float((box[2] - box[0]) * (box[3] - box[1])), k, box))
            for _, k, box in sorted(active, key=lambda a: -a[0])[:MAX_PLAYERS]:
                c = crop_for(box, reader.width, reader.height)
                patch = frame.image[c[1] : c[3], c[0] : c[2]]
                if patch.shape[0] < 8 or patch.shape[1] < 8:
                    continue
                z = CROP_HEIGHT / patch.shape[0]
                resized = cv2.resize(
                    patch,
                    None,
                    fx=z,
                    fy=z,
                    interpolation=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR,
                )
                images.append(np.asarray(resized, np.uint8))
                pending.append((frame.index, frame.pts, k, c, z, box))
            if len(images) >= cfg.batch_size * 2:
                flush()
            if n % 300 == 0:
                ctx.progress(n / total, f"{n} of {total} frames")
        flush()
        ctx.log("motion", rows=len(rows["t"]), tracks=len(tracks), device=device)
    f32 = pa.float32()
    table = pa.table(
        {
            "frame_idx": pa.array(rows["frame_idx"], pa.int64()),
            "t": pa.array(rows["t"], pa.float64()),
            "track": pa.array(rows["track"], pa.int32()),
            **{k: pa.array(rows[k], pa.list_(f32)) for k in ("kp_x", "kp_y", "kp_c")},
            "height": pa.array(rows["height"], f32),
        }
    )
    write_parquet(
        table,
        ctx.path("motion.parquet"),
        stage=ctx.stage,
        config_hash=ctx.config.section_hash("pose"),
        schema_version=SCHEMA_VERSION,
    )


def _uncrop(p: PersonPose, c: tuple[int, int, int, int], z: float) -> PersonPose:
    kp = p.keypoints.copy()
    kp[:, 0] = kp[:, 0] / z + c[0]
    kp[:, 1] = kp[:, 1] / z + c[1]
    x1, y1, x2, y2 = p.bbox
    return PersonPose(
        bbox=(x1 / z + c[0], y1 / z + c[1], x2 / z + c[0], y2 / z + c[1]),
        bbox_conf=p.bbox_conf,
        keypoints=kp,
    )
