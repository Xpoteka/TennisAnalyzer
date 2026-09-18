"""Video stage ``people``: every person on court, with pose, through the whole video.

Frames are analysed at ``pose.sample_fps`` (about 10 per second). That is enough to follow
where players are and how they move; the precise moment of each hit is analysed again at
the full frame rate later, around the hits only.

For each person found: the box, 17 keypoints, a clothing-colour descriptor, the point
between the feet, and that point on the court in metres when the court is known. People
off court (spectators, players on the next court, a coach by the fence) are dropped when
the court is known; otherwise the largest people in the frame are kept.

Writes ``people.parquet``, one row per person per analysed frame, with a ``track_id``
from :mod:`tennis.vision.tracking`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from tennis.pose_backends import (
    KEYPOINT_NAMES,
    PersonPose,
    PoseBackend,
    create_backend,
    resolve_model_path,
)
from tennis.util.appearance import descriptor, torso_box
from tennis.util.frames import Frame
from tennis.util.io import read_json, write_parquet
from tennis.vision import court as court_mod
from tennis.vision.detect import PERSON, Box, Detector
from tennis.vision.tracking import Detection, Tracker

if TYPE_CHECKING:
    from tennis.pipeline import VideoContext

OUTPUTS = ("people.parquet",)
SCHEMA_VERSION = 1
KP = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
# How far outside the lines someone still counts as on court (metres): the run-off area.
RUNOFF_SIDE_M = 5.0
RUNOFF_BACK_M = 8.0
MAX_PEOPLE = 6
MIN_HEIGHT_FRAC = 0.03  # without a court: ignore people smaller than this share of the frame
FAR_IMGSZ = 1280
FAR_MIN_CONF = 0.25
TIGHT_HEIGHT = 256  # far players are enlarged to about this height for their pose
TIGHT_IMGSZ = 320


def on_court(xy: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    return np.asarray(
        (np.abs(xy[:, 0]) <= court_mod.HALF_DW + RUNOFF_SIDE_M)
        & (np.abs(xy[:, 1]) <= court_mod.HALF_L + RUNOFF_BACK_M)
    )


def foot_point(p: PersonPose, kp_conf_min: float) -> tuple[float, float]:
    """Between the ankles when both are seen, else the bottom centre of the box."""
    la, ra = p.keypoints[KP["l_ankle"]], p.keypoints[KP["r_ankle"]]
    if la[2] >= kp_conf_min and ra[2] >= kp_conf_min:
        return float((la[0] + ra[0]) / 2), float(max(la[1], ra[1]))
    x1, _, x2, y2 = p.bbox
    return float((x1 + x2) / 2), float(y2)


def select_people(
    people: list[PersonPose],
    cal: court_mod.Calibration | None,
    frame_h: int,
    kp_conf_min: float,
) -> list[tuple[PersonPose, tuple[float, float], tuple[float, float] | None]]:
    """The people to keep in one frame, with their foot point in pixels and on court."""
    if not people:
        return []
    feet = np.array([foot_point(p, kp_conf_min) for p in people])
    if cal is not None:
        xy = cal.image_to_court(feet)
        keep = on_court(xy) & np.isfinite(xy).all(axis=1)
        chosen = [i for i in np.argsort([-p.area for p in people]) if keep[i]][:MAX_PEOPLE]
        return [
            (people[i], (feet[i, 0], feet[i, 1]), (float(xy[i, 0]), float(xy[i, 1])))
            for i in chosen
        ]
    tall = [i for i, p in enumerate(people) if p.bbox[3] - p.bbox[1] >= MIN_HEIGHT_FRAC * frame_h]
    chosen = sorted(tall, key=lambda i: -people[i].area)[:4]
    return [(people[i], (feet[i, 0], feet[i, 1]), None) for i in chosen]


def far_crop(cal: court_mod.Calibration, w: int, h: int) -> tuple[int, int, int, int] | None:
    """The image box around the far half of the court, with room for standing players.

    Far players are often only a few dozen pixels tall, too small for the pose model at full
    frame. Running it again on this crop, enlarged, finds them. None when the far half is
    big enough in the frame anyway.
    """
    if not cal.has_camera:
        return None
    xs = np.array([-1, 1]) * (court_mod.HALF_DW + 2.5)
    ys = np.array([court_mod.HALF_L * 0.35, court_mod.HALF_L + 5.0])
    corners = np.array([[x, y, z] for x in xs for y in ys for z in (0.0, 2.3)])
    p = cal.world_to_image(corners)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) == 0:
        return None
    x1, y1 = np.clip(p.min(axis=0), 0, [w, h])
    x2, y2 = np.clip(p.max(axis=0), 0, [w, h])
    if (x2 - x1) < 32 or (y2 - y1) < 32 or (x2 - x1) > 0.7 * w or (y2 - y1) > 0.5 * h:
        return None
    return int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))


def merge_people(
    full: list[PersonPose], crop: list[PersonPose], iou_max: float = 0.4
) -> list[PersonPose]:
    """Full-frame people plus crop people that are not already among them."""
    out = list(full)
    for c in crop:
        cb = np.array(c.bbox)
        if all(_iou(cb, np.array(f.bbox)) < iou_max for f in full):
            out.append(c)
    return out


def _iou(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _from_crop(p: PersonPose, dx: int, dy: int, zoom: float) -> PersonPose:
    """A person found in an enlarged crop, in full-frame pixels."""
    kp = p.keypoints.copy()
    kp[:, 0] = kp[:, 0] / zoom + dx
    kp[:, 1] = kp[:, 1] / zoom + dy
    x1, y1, x2, y2 = (v / zoom for v in p.bbox)
    return PersonPose(
        bbox=(x1 + dx, y1 + dy, x2 + dx, y2 + dy), bbox_conf=p.bbox_conf, keypoints=kp
    )


def crop_zoom(box: tuple[int, int, int, int], imgsz: int) -> float:
    """How much to enlarge the far crop: to fill the model input, at most 3 times."""
    return float(min(3.0, max(1.0, imgsz / max(box[2] - box[0], box[3] - box[1]))))


def far_people(
    frames: list[Frame],
    crop: tuple[int, int, int, int],
    detector: Detector,
    backend: PoseBackend,
) -> list[list[PersonPose]]:
    """People in the far crop: boxes from the detector, keypoints from tight pose crops."""
    x1, y1, x2, y2 = crop
    zoom = crop_zoom(crop, FAR_IMGSZ)
    zoomed = [
        np.asarray(
            cv2.resize(
                f.image[y1:y2, x1:x2], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC
            ),
            np.uint8,
        )
        for f in frames
    ]
    found = detector.infer(zoomed, imgsz=FAR_IMGSZ, conf=FAR_MIN_CONF, classes=[PERSON])
    tight: list[npt.NDArray[np.uint8]] = []
    where: list[tuple[int, Box, tuple[float, float, float]]] = []  # frame, box, (ox, oy, zoom)
    for i, (frame, boxes) in enumerate(zip(frames, found, strict=True)):
        h, w = frame.image.shape[:2]
        for b in boxes:
            box = Box(
                b.x1 / zoom + x1,
                b.y1 / zoom + y1,
                b.x2 / zoom + x1,
                b.y2 / zoom + y1,
                b.conf,
                b.cls,
            )
            bw, bh = box.x2 - box.x1, box.y2 - box.y1
            cx1, cy1 = int(max(0, box.x1 - 0.5 * bw)), int(max(0, box.y1 - 0.2 * bh))
            cx2, cy2 = int(min(w, box.x2 + 0.5 * bw)), int(min(h, box.y2 + 0.2 * bh))
            if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                continue
            z = TIGHT_HEIGHT / (cy2 - cy1)
            patch = cv2.resize(
                frame.image[cy1:cy2, cx1:cx2], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC
            )
            tight.append(np.asarray(patch, np.uint8))
            where.append((i, box, (cx1, cy1, z)))
    out: list[list[PersonPose]] = [[] for _ in frames]
    poses = backend.infer(tight) if tight else []
    for (i, box, (ox, oy, z)), people in zip(where, poses, strict=True):
        bb = np.array([box.x1, box.y1, box.x2, box.y2])
        best = max(
            (_from_crop(p, int(ox), int(oy), z) for p in people),
            key=lambda q: _iou(np.array(q.bbox), bb),
            default=None,
        )
        if best is not None and _iou(np.array(best.bbox), bb) > 0.3:
            kp = best.keypoints
        else:
            kp = np.zeros((len(KEYPOINT_NAMES), 3), np.float32)
        out[i].append(
            PersonPose(bbox=(box.x1, box.y1, box.x2, box.y2), bbox_conf=box.conf, keypoints=kp)
        )
    return out


def load_calibration(ctx: VideoContext) -> court_mod.Calibration | None:
    data = read_json(ctx.path("court.json"))
    return court_mod.Calibration.from_json(data) if data.get("found") else None


def run(ctx: VideoContext) -> None:
    cfg = ctx.config.pose
    cal = load_calibration(ctx)
    meta = ctx.metadata()
    fps = float(meta.get("fps") or 30.0)
    every = max(1, round(fps / cfg.sample_fps))
    reader = ctx.frame_reader(every=every)
    total = max(1, len(ctx.frame_pts()) // every)
    backend, device = create_backend(cfg, ctx.data_root)
    crop = far_crop(cal, reader.width, reader.height) if cal is not None else None
    detector = (
        Detector(resolve_model_path(cfg.detector_model, ctx.data_root), device)
        if crop is not None
        else None
    )
    # Tight crops of far players are small: a small model input is enough and much faster.
    tight_backend = (
        create_backend(cfg.model_copy(update={"imgsz": TIGHT_IMGSZ}), ctx.data_root)[0]
        if crop is not None
        else None
    )
    ctx.log("pose model loaded", model=cfg.model, device=device, every=every, far_crop=crop)
    tracker = Tracker(in_metres=cal is not None)

    rows: dict[str, list[Any]] = {
        k: []
        for k in (
            "frame_idx", "t", "track_id", "x1", "y1", "x2", "y2", "conf",
            "foot_px", "foot_py", "court_x", "court_y", "kp_x", "kp_y", "kp_c", "appearance",
        )
    }  # fmt: skip
    batch: list[Any] = []
    done = 0

    def flush() -> None:
        nonlocal done
        if not batch:
            return
        results = backend.infer([f.image for f in batch])
        if crop is not None and detector is not None and tight_backend is not None:
            far = far_people(batch, crop, detector, tight_backend)
            results = [merge_people(full, extra) for full, extra in zip(results, far, strict=True)]
        for frame, people in zip(batch, results, strict=True):
            kept = select_people(people, cal, reader.height, cfg.kp_conf_min)
            dets = []
            for p, foot_px, foot_court in kept:
                look = descriptor(frame.image, torso_box(p.bbox, p.keypoints, cfg.kp_conf_min))
                if foot_court is not None:
                    foot = np.array(foot_court)
                else:
                    foot = np.array(foot_px) / reader.height
                d = Detection(
                    t=frame.pts,
                    foot=foot,
                    bbox=np.array(p.bbox),
                    appearance=look.astype(np.float64),
                )
                dets.append((d, p, foot_px, foot_court))
            tracker.update([d for d, *_ in dets])
            for d, p, foot_px, foot_court in dets:
                rows["frame_idx"].append(frame.index)
                rows["t"].append(frame.pts)
                rows["track_id"].append(d.track_id)
                for k, v in zip(("x1", "y1", "x2", "y2"), p.bbox, strict=True):
                    rows[k].append(v)
                rows["conf"].append(p.bbox_conf)
                rows["foot_px"].append(foot_px[0])
                rows["foot_py"].append(foot_px[1])
                rows["court_x"].append(foot_court[0] if foot_court else None)
                rows["court_y"].append(foot_court[1] if foot_court else None)
                rows["kp_x"].append(p.keypoints[:, 0].tolist())
                rows["kp_y"].append(p.keypoints[:, 1].tolist())
                rows["kp_c"].append(p.keypoints[:, 2].tolist())
                rows["appearance"].append(d.appearance.tolist())
        done += len(batch)
        batch.clear()
        ctx.progress(done / total, f"{done} of {total} frames")

    for frame in reader.read(ctx.whole_video()):
        batch.append(frame)
        if len(batch) >= cfg.batch_size:
            flush()
    flush()

    f32 = pa.float32()
    table = pa.table(
        {
            "frame_idx": pa.array(rows["frame_idx"], pa.int64()),
            "t": pa.array(rows["t"], pa.float64()),
            "track_id": pa.array(rows["track_id"], pa.int32()),
            **{k: pa.array(rows[k], f32) for k in ("x1", "y1", "x2", "y2", "conf")},
            **{k: pa.array(rows[k], f32) for k in ("foot_px", "foot_py", "court_x", "court_y")},
            **{k: pa.array(rows[k], pa.list_(f32)) for k in ("kp_x", "kp_y", "kp_c", "appearance")},
        }
    )
    write_parquet(
        table,
        ctx.path("people.parquet"),
        stage=ctx.stage,
        config_hash=ctx.config.section_hash("pose"),
        schema_version=SCHEMA_VERSION,
        extra={"every": str(every), "device": device, "court": str(cal is not None)},
    )
    ctx.log("people found", rows=table.num_rows, tracks=tracker.next_id)
