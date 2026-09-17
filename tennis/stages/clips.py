"""Stage 8: clips of the swings worth looking at (spec section 6.8).

Rendering a clip means decoding the source video, so only the selected swings are decoded,
in one pass over the file in time order.

Which swings
------------
* every outlier (``metrics.is_outlier``);
* the swing closest to the median of each stroke type, as something to compare against;
* every labelled swing, at most ``clips.max_per_label`` per label (voice labels arrive
  with stage 7; without them this rule selects nothing).

Output
------
``clips/<swing_id>.mp4``, plus ``clips/index.json`` listing every clip and why it was
chosen. **``clips/index.json`` is the stage's declared output, not the directory**: a
directory's modification time changes whenever any file inside it does, which the
fingerprint cache would read as "changed" forever.

Each frame carries the skeleton (racket side coloured), the stroke type and a few metrics,
and the contact frame gets a red border. ``clips.slow_motion`` writes the same frames at
``fps * clips.slow_motion_rate`` instead of real time.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq

from tennis.config import Config
from tennis.pose_backends.base import KEYPOINT_NAMES
from tennis.session import SOURCE_INPUT, Session
from tennis.stages.clean import resolved_handedness
from tennis.stages.labels import read_labels
from tennis.stages.metrics import REGISTRY
from tennis.stages.pose import frame_size, read_frame_pts
from tennis.util.frames import FrameReader, Window
from tennis.util.io import read_json, write_json
from tennis.util.overlay import CONTACT, VideoWriter, draw_border, draw_label, draw_pose

if TYPE_CHECKING:
    from tennis.stages import StageContext

SCHEMA_VERSION = 1
INPUTS = (
    SOURCE_INPUT,
    "metrics.parquet",
    "swings.parquet",
    "swing_info.parquet",
    "strokes.parquet",
    "frame_times.parquet",
    "metadata.json",
)
OPTIONAL_INPUTS = ("labels.parquet",)
OUTPUTS = ("clips/index.json",)
CONFIG_KEYS = ("clips", "player", "pose.hwaccel", "pose.kp_conf_min", "labels.enabled")

INDEX_NAME = "clips/index.json"
CLIPS_DIR = "clips"
# Metrics written on the clip, when the stroke type has them.
SHOWN_METRICS = ("peak_wrist_speed", "contact_height", "elbow_angle_contact", "knee_flex_min")


@dataclass
class Selection:
    swing_id: int
    t_contact: float
    stroke_type: str
    reasons: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def select_swings(
    rows: Sequence[dict[str, Any]],
    labels_by_swing: dict[int, list[str]],
    max_per_label: int,
) -> list[Selection]:
    """The swings to render, in time order, each with the reasons it was picked.

    A swing picked by more than one rule is rendered once and lists every reason.
    """
    chosen: dict[int, Selection] = {}

    def pick(row: dict[str, Any], reason: str) -> Selection:
        swing_id = int(row["swing_id"])
        entry = chosen.get(swing_id)
        if entry is None:
            entry = Selection(
                swing_id=swing_id,
                t_contact=float(row["t_contact"]),
                stroke_type=str(row["stroke_type"]),
                labels=list(labels_by_swing.get(swing_id, ())),
                metrics={
                    name: float(row[name])
                    for name in SHOWN_METRICS
                    if name in row and np.isfinite(float(row[name]))
                },
            )
            chosen[swing_id] = entry
        if reason not in entry.reasons:
            entry.reasons.append(reason)
        return entry

    for row in rows:
        if row.get("is_outlier"):
            pick(row, "outlier")

    for stroke_type in sorted({str(r["stroke_type"]) for r in rows}):
        subset = [r for r in rows if r["stroke_type"] == stroke_type]
        typical = median_swing(subset)
        if typical is not None:
            pick(typical, f"median {stroke_type}")

    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for label in labels_by_swing.get(int(row["swing_id"]), ()):
            by_label.setdefault(label, []).append(row)
    for label, subset in sorted(by_label.items()):
        for row in sorted(subset, key=lambda r: float(r["t_contact"]))[:max_per_label]:
            pick(row, f"label {label}")

    return sorted(chosen.values(), key=lambda s: s.t_contact)


def median_swing(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The swing closest to the median of its stroke type, over the metrics it has.

    Each metric is standardized first, so a metric measured in degrees does not outweigh
    one measured in torso lengths. Ties break on the lower swing id, so this is stable.
    """
    names = [
        name for name in REGISTRY if all(name in r and np.isfinite(float(r[name])) for r in rows)
    ]
    if not rows or not names:
        return rows[0] if rows else None
    x = np.array([[float(r[name]) for name in names] for r in rows], dtype=np.float64)
    scale = x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    distance = np.abs((x - np.median(x, axis=0)) / scale).sum(axis=1)
    order = sorted(range(len(rows)), key=lambda i: (distance[i], int(rows[i]["swing_id"])))
    return rows[order[0]]


def clip_label_lines(selection: Selection) -> list[str]:
    stroke = selection.stroke_type
    if selection.labels:
        stroke += "  [" + ", ".join(selection.labels) + "]"
    lines = [f"swing {selection.swing_id}  {stroke}", ", ".join(selection.reasons)]
    shown = []
    for name in SHOWN_METRICS:
        if name in selection.metrics:
            unit = REGISTRY[name].unit
            shown.append(f"{name} {selection.metrics[name]:.2f}{unit and ' ' + unit}")
    if shown:
        lines.append("  ".join(shown))
    return lines


def _pixel_frames(session: Session, swing_ids: set[int]) -> dict[int, dict[int, np.ndarray]]:
    """Cleaned pixel keypoints of the selected swings: swing id -> frame index -> (17, 3)."""
    columns = ["swing_id", "frame_idx"]
    for name in KEYPOINT_NAMES:
        columns += [f"{name}_px", f"{name}_py", f"{name}_conf"]
    table = pq.read_table(session.path("swings.parquet"), columns=columns).to_pydict()
    out: dict[int, dict[int, np.ndarray]] = {}
    arrays = {c: np.asarray(table[c]) for c in columns}
    for i, swing_id in enumerate(arrays["swing_id"]):
        sid = int(swing_id)
        if sid not in swing_ids:
            continue
        kp = np.empty((len(KEYPOINT_NAMES), 3), np.float32)
        for k, name in enumerate(KEYPOINT_NAMES):
            kp[k] = (
                arrays[f"{name}_px"][i],
                arrays[f"{name}_py"][i],
                arrays[f"{name}_conf"][i],
            )
        out.setdefault(sid, {})[int(arrays["frame_idx"][i])] = kp
    return out


def clip_path(session: Session, swing_id: int) -> Path:
    return session.dir / CLIPS_DIR / f"{swing_id}.mp4"


def swing_labels(session: Session, config: Config) -> dict[int, list[str]]:
    """Voice labels per swing, or nothing when ``labels.enabled`` is off.

    A ``labels.parquet`` left behind by an earlier run is ignored when the labels stage is
    turned off in the config, so clips and the report do not quietly keep using labels that
    the config says not to produce. ``--no-labels`` is deliberately not the same thing: it
    skips the slow transcription for one run without changing what the report says.
    """
    if not config.labels.enabled:
        return {}
    out: dict[int, list[str]] = {}
    for row in read_labels(session):
        out.setdefault(int(row["swing_id"]), []).append(str(row["label"]))
    return out


def run(ctx: StageContext) -> None:
    session, config = ctx.session, ctx.config
    rows = pq.read_table(session.path("metrics.parquet")).to_pylist()
    labels_by_swing = swing_labels(session, config)
    selections = select_swings(rows, labels_by_swing, config.clips.max_per_label)

    directory = session.dir / CLIPS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    keep = {f"{s.swing_id}.mp4" for s in selections}
    for stale in directory.glob("*.mp4"):
        if stale.name not in keep:
            stale.unlink()

    index = _render(ctx, session, config, selections)
    write_json(session.path(INDEX_NAME), index)
    ctx.log(
        "clips rendered",
        selected=len(selections),
        rendered=sum(1 for c in index["clips"] if c["rendered"]),
        outliers=sum(1 for s in selections if "outlier" in s.reasons),
        labelled=sum(1 for s in selections if s.labels),
    )


def _render(
    ctx: StageContext, session: Session, config: Config, selections: Sequence[Selection]
) -> dict[str, Any]:
    meta = read_json(session.path("metadata.json"))
    width, height = frame_size(meta)
    fps = float(meta.get("fps") or 30.0)
    if config.clips.slow_motion:
        fps *= config.clips.slow_motion_rate
    frame_pts = read_frame_pts(session.path("frame_times.parquet"))
    hand = resolved_handedness(session, config)
    poses = _pixel_frames(session, {s.swing_id for s in selections})
    reader = FrameReader(
        session.source_link.resolve(),
        frame_pts,
        width,
        height,
        video_start_s=float(meta.get("video_start_s", 0.0)),
        hwaccel=config.pose.hwaccel,
    )
    pre, post = config.clips.pre_s, config.clips.post_s
    last_pts = float(frame_pts[-1]) if frame_pts.size else 0.0

    clips: list[dict[str, Any]] = []
    for selection in selections:
        start = max(0.0, selection.t_contact - pre)
        end = min(last_pts, selection.t_contact + post)
        entry: dict[str, Any] = {
            "swing_id": selection.swing_id,
            "t_contact": round(selection.t_contact, 3),
            "stroke_type": selection.stroke_type,
            "reasons": selection.reasons,
            "labels": selection.labels,
            "metrics": {k: round(v, 4) for k, v in selection.metrics.items()},
            "file": f"{selection.swing_id}.mp4",
            "start_s": round(start, 3),
            "end_s": round(end, 3),
            "frames": 0,
            "rendered": False,
        }
        clips.append(entry)
        path = clip_path(session, selection.swing_id)
        contact_frame = -1
        if frame_pts.size:
            contact_frame = int(np.argmin(np.abs(frame_pts - selection.t_contact)))
        try:
            frames = _write_clip(
                reader,
                path,
                selection,
                poses.get(selection.swing_id, {}),
                Window(selection.swing_id, start, end),
                Output(width, height, fps, config.clips.height),
                contact_frame,
                hand,
                config.pose.kp_conf_min,
            )
        except Exception as exc:  # one unreadable clip must not stop the stage
            ctx.log(
                "clip failed",
                level=logging.WARNING,
                swing_id=selection.swing_id,
                error=repr(exc),
            )
            path.unlink(missing_ok=True)
            entry["error"] = f"{type(exc).__name__}: {exc}"
            continue
        entry["frames"] = frames
        entry["rendered"] = frames > 0
        if not frames:
            path.unlink(missing_ok=True)

    return {
        "schema_version": SCHEMA_VERSION,
        "session": session.id,
        "fps": round(fps, 4),
        "slow_motion": config.clips.slow_motion,
        "height": config.clips.height,
        "clips": clips,
    }


@dataclass(frozen=True)
class Output:
    """Source frame size, playback rate and the height to scale the file down to."""

    width: int
    height: int
    fps: float
    out_height: int


def _write_clip(
    reader: FrameReader,
    path: Path,
    selection: Selection,
    poses: dict[int, np.ndarray],
    window: Window,
    output: Output,
    contact_frame: int,
    hand: str,
    kp_conf_min: float,
) -> int:
    lines = clip_label_lines(selection)
    written = 0
    with VideoWriter(
        path, output.width, output.height, output.fps, out_height=output.out_height
    ) as writer:
        for frame in reader.read([window]):
            image = frame.image.copy()
            kp = poses.get(frame.index)
            if kp is not None:
                draw_pose(image, kp, None, hand, kp_conf_min)
            if frame.index == contact_frame:
                draw_border(image, CONTACT, max(4, output.height // 120))
            draw_label(image, [*lines, f"t{frame.pts - selection.t_contact:+.2f}s"])
            writer.write(image)
            written += 1
    return written
