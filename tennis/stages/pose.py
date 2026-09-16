"""Stage 3: pose extraction around contacts (spec section 6.3).

Output ``keypoints.parquet`` (schema_version 1), one row per decoded frame inside a window:

====================  =======  ======================================================
frame_idx             int64    row index into frame_times.parquet
t_video               float64  PTS of the frame
window_id             int32    merged analysis window
detected              bool     a player was selected in this frame
track_reset           bool     the selection fell back to the first-frame rule
n_persons             int16    people the backend found in the frame
bbox_x1..bbox_y2      float32  selected player's box, pixels (NaN if not detected)
bbox_conf             float32
<kp>_x, <kp>_y        float32  keypoint position, pixels (COCO-17, e.g. r_wrist_x)
<kp>_conf             float32  keypoint confidence (0 if not detected)
====================  =======  ======================================================

Parquet metadata also records the frame size, backend, model, device and whether the
crop refinement pass ran.
"""

from __future__ import annotations

import logging
import queue
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from tennis.config import Config
from tennis.pose_backends import (
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    PersonPose,
    PoseBackend,
    check_backend,
    create_backend,
    resolve_model_path,
)
from tennis.session import SOURCE_INPUT
from tennis.util.frames import (
    PTS_TOLERANCE_S,
    Frame,
    FrameReader,
    Window,
    group_windows,
    merge_windows,
)
from tennis.util.io import read_json, write_parquet
from tennis.util.tracking import PlayerTracker, crop_box, iou
from tennis.util.video import ProbeError

if TYPE_CHECKING:
    from tennis.stages import StageContext

SCHEMA_VERSION = 1
INPUTS = ("contacts.parquet", "frame_times.parquet", "metadata.json", SOURCE_INPUT)
OUTPUTS = ("keypoints.parquet",)
CROP_REFINE_FACTOR = 2.5  # auto-refine when the frame's long side exceeds imgsz by this
CROP_MATCH_IOU = 0.3
PROGRESS_EVERY_S = 30.0

BBOX_COLUMNS = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
KEYPOINT_COLUMNS = tuple(f"{n}_{c}" for n in KEYPOINT_NAMES for c in ("x", "y", "conf"))


def frame_size(meta: dict[str, object]) -> tuple[int, int]:
    """Decoded (upright) frame size: ffmpeg applies the container rotation."""
    video = meta["video"]
    assert isinstance(video, dict)
    width, height = int(video["width"]), int(video["height"])
    if int(video.get("rotation_deg", 0)) % 180 == 90:
        width, height = height, width
    return width, height


def read_frame_pts(path: object) -> npt.NDArray[np.float64]:
    return np.asarray(pq.read_table(path, columns=["pts"]).column("pts").to_numpy(), np.float64)


def select_windows(
    contacts: pa.Table, config: Config, frame_pts: npt.NDArray[np.float64]
) -> list[Window]:
    times = np.asarray(contacts.column("t_audio").to_numpy(), dtype=np.float64)
    if config.pose.contacts == "self_audio":
        mask = np.asarray(contacts.column("is_self_audio").to_numpy(zero_copy_only=False), bool)
        times = times[mask]
    if frame_pts.size == 0:
        return []
    return merge_windows(
        times.tolist(),
        config.windows.pre_s,
        config.windows.post_s,
        float(frame_pts[0]),
        float(frame_pts[-1]),
    )


def should_refine(mode: str, width: int, height: int, imgsz: int) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return max(width, height) > CROP_REFINE_FACTOR * imgsz


@dataclass
class _Rows:
    frame_idx: list[int] = field(default_factory=list)
    t_video: list[float] = field(default_factory=list)
    window_id: list[int] = field(default_factory=list)
    detected: list[bool] = field(default_factory=list)
    track_reset: list[bool] = field(default_factory=list)
    n_persons: list[int] = field(default_factory=list)
    boxes: list[npt.NDArray[np.float32]] = field(default_factory=list)  # (5,) incl. conf
    keypoints: list[npt.NDArray[np.float32]] = field(default_factory=list)  # (17, 3)

    def add(self, frame: Frame, n: int, person: PersonPose | None, reset: bool,
            keypoints: npt.NDArray[np.float32] | None) -> None:  # fmt: skip
        self.frame_idx.append(frame.index)
        self.t_video.append(frame.pts)
        self.window_id.append(frame.window_id)
        self.n_persons.append(n)
        self.track_reset.append(reset)
        self.detected.append(person is not None)
        if person is None or keypoints is None:
            self.boxes.append(np.array([np.nan] * 4 + [0.0], np.float32))
            kp = np.full((NUM_KEYPOINTS, 3), np.nan, np.float32)
            kp[:, 2] = 0.0
            self.keypoints.append(kp)
        else:
            self.boxes.append(np.array([*person.bbox, person.bbox_conf], np.float32))
            self.keypoints.append(keypoints.astype(np.float32))

    def __len__(self) -> int:
        return len(self.frame_idx)

    def table(self) -> pa.Table:
        n = len(self)
        boxes = np.stack(self.boxes) if n else np.zeros((0, 5), np.float32)
        kps = np.stack(self.keypoints) if n else np.zeros((0, NUM_KEYPOINTS, 3), np.float32)
        order = np.argsort(np.asarray(self.frame_idx, np.int64), kind="stable")
        columns: dict[str, pa.Array] = {
            "frame_idx": pa.array(np.asarray(self.frame_idx, np.int64)[order]),
            "t_video": pa.array(np.asarray(self.t_video, np.float64)[order]),
            "window_id": pa.array(np.asarray(self.window_id, np.int32)[order]),
            "detected": pa.array(np.asarray(self.detected, bool)[order]),
            "track_reset": pa.array(np.asarray(self.track_reset, bool)[order]),
            "n_persons": pa.array(np.asarray(self.n_persons, np.int16)[order]),
        }
        for i, name in enumerate(BBOX_COLUMNS):
            columns[name] = pa.array(boxes[order, i])
        columns["bbox_conf"] = pa.array(boxes[order, 4])
        for k, kp_name in enumerate(KEYPOINT_NAMES):
            for c, suffix in enumerate(("x", "y", "conf")):
                columns[f"{kp_name}_{suffix}"] = pa.array(kps[order, k, c])
        return pa.table(columns)


def _refine_with_crops(
    backend: PoseBackend,
    frames: Sequence[Frame],
    selected: Sequence[PersonPose | None],
    pad: float,
) -> tuple[list[npt.NDArray[np.float32] | None], int]:
    """Keypoints from a second pass on full-resolution crops around each selected player."""
    out: list[npt.NDArray[np.float32] | None] = [
        None if p is None else p.keypoints for p in selected
    ]
    jobs = []
    for i, (frame, person) in enumerate(zip(frames, selected, strict=True)):
        if person is None:
            continue
        h, w = frame.image.shape[:2]
        x1, y1, x2, y2 = crop_box(person.bbox, pad, w, h)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        jobs.append((i, x1, y1, frame.image[y1:y2, x1:x2]))
    if not jobs:
        return out, 0
    results = backend.infer([crop for *_, crop in jobs])
    refined = 0
    for (i, x1, y1, _), people in zip(jobs, results, strict=True):
        target = selected[i]
        assert target is not None
        best, best_iou = None, CROP_MATCH_IOU
        for p in people:
            mapped = (p.bbox[0] + x1, p.bbox[1] + y1, p.bbox[2] + x1, p.bbox[3] + y1)
            score = iou(mapped, target.bbox)
            if score >= best_iou:
                best, best_iou = p, score
        if best is not None:
            kp = best.keypoints.copy()
            kp[:, 0] += x1
            kp[:, 1] += y1
            out[i] = kp
            refined += 1
    return out, refined


def extract(
    ctx: StageContext,
    backend: PoseBackend,
    reader: FrameReader,
    windows: Sequence[Window],
    refine: bool,
) -> tuple[_Rows, dict[str, int]]:
    cfg = ctx.config.pose
    tracker = PlayerTracker(reader.height, cfg.near_court_min_y, cfg.track_iou_min)
    rows = _Rows()
    stats = {"frames": 0, "detected": 0, "resets": 0, "refined": 0, "failed_windows": 0}
    current_window = -1
    started = last_report = time.perf_counter()

    def flush(batch: list[Frame]) -> None:
        nonlocal current_window
        people_per_frame = backend.infer([f.image for f in batch])
        selected: list[PersonPose | None] = []
        resets: list[bool] = []
        for frame, people in zip(batch, people_per_frame, strict=True):
            if frame.window_id != current_window:
                tracker.reset()
                current_window = frame.window_id
            sel = tracker.update(people)
            selected.append(None if sel.index is None else people[sel.index])
            resets.append(sel.track_reset)
        keypoints: list[npt.NDArray[np.float32] | None]
        if refine:
            keypoints, n_refined = _refine_with_crops(backend, batch, selected, cfg.crop_pad)
            stats["refined"] += n_refined
        else:
            keypoints = [None if p is None else p.keypoints for p in selected]
        for frame, people, person, reset, kp in zip(
            batch, people_per_frame, selected, resets, keypoints, strict=True
        ):
            rows.add(frame, len(people), person, reset, kp)
            stats["frames"] += 1
            stats["detected"] += person is not None
            stats["resets"] += reset
        report_progress()

    def report_progress() -> None:
        nonlocal last_report
        now = time.perf_counter()
        if now - last_report < PROGRESS_EVERY_S:
            return
        last_report = now
        ctx.log(
            "progress",
            window=current_window,
            windows=len(windows),
            frames=stats["frames"],
            fps=round(stats["frames"] / (now - started), 1),
        )

    for group in group_windows(windows, cfg.seek_gap_s):
        batch: list[Frame] = []
        try:
            for frame in reader.read(group):
                batch.append(frame)
                if len(batch) >= cfg.batch_size:
                    flush(batch)
                    batch = []
            if batch:
                flush(batch)
        except (ProbeError, queue.Empty) as exc:
            if batch:
                flush(batch)
            stats["failed_windows"] += len(group)
            ctx.log(
                "decoding failed; skipping windows",
                level=logging.WARNING,
                windows=[w.id for w in group],
                error=str(exc),
            )
    return rows, stats


def run(ctx: StageContext) -> None:
    session, config = ctx.session, ctx.config
    check_backend(config.pose.backend)
    meta = read_json(session.path("metadata.json"))
    width, height = frame_size(meta)
    frame_pts = read_frame_pts(session.path("frame_times.parquet"))
    contacts = pq.read_table(session.path("contacts.parquet"))
    windows = select_windows(contacts, config, frame_pts)

    starts = np.searchsorted(frame_pts, [w.start - PTS_TOLERANCE_S for w in windows], "left")
    ends = np.searchsorted(frame_pts, [w.end + PTS_TOLERANCE_S for w in windows], "right")
    planned = int(np.sum(ends - starts)) if windows else 0
    refine = should_refine(config.pose.crop_refine, width, height, config.pose.imgsz)
    if windows:
        # Loading a model is slow (and may download it), so only do it when needed.
        backend, device = create_backend(config.pose, config.paths.data_root)
        backend_name = backend.name
    else:
        backend, device, backend_name = None, "none", config.pose.backend
    ctx.log(
        "starting",
        backend=backend_name,
        model=config.pose.model,
        device=device,
        contacts=config.pose.contacts,
        windows=len(windows),
        frames=planned,
        frame_size=f"{width}x{height}",
        crop_refine=refine,
    )
    if not windows:
        ctx.log("no contacts selected; writing an empty keypoints table", level=logging.WARNING)

    reader = FrameReader(
        session.source_link.resolve(),
        frame_pts,
        width,
        height,
        video_start_s=float(meta.get("video_start_s", 0.0)),
        hwaccel=config.pose.hwaccel,
    )
    started = time.perf_counter()
    if backend is None:
        rows, stats = _Rows(), {"frames": 0, "detected": 0, "resets": 0, "refined": 0,
                                "failed_windows": 0}  # fmt: skip
    else:
        rows, stats = extract(ctx, backend, reader, windows, refine)
    elapsed = time.perf_counter() - started

    write_parquet(
        rows.table(),
        session.path("keypoints.parquet"),
        stage=ctx.stage,
        config_hash=ctx.config_hash,
        schema_version=SCHEMA_VERSION,
        extra={
            "frame_width": str(width),
            "frame_height": str(height),
            "backend": backend_name,
            "model": str(resolve_model_path(config.pose.model, config.paths.data_root).name),
            "device": device,
            "crop_refine": str(refine).lower(),
        },
    )
    frames = stats["frames"]
    ctx.log(
        "pose extracted",
        frames=frames,
        planned=planned,
        detected_ratio=round(stats["detected"] / frames, 3) if frames else None,
        track_resets=stats["resets"],
        refined=stats["refined"],
        failed_windows=stats["failed_windows"],
        fps=round(frames / elapsed, 1) if elapsed > 0 else None,
    )
