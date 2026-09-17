"""Stage 4: cleaning, normalization, QC and own-hit confirmation (spec section 6.4).

Cleaning runs once per pose window (a continuous run of frames), in the spec's order:
confidence mask -> left/right swap fix -> gap filling -> smoothing. Each candidate contact
then becomes a swing: the cleaned frames within ``[t - pre_s, t + post_s]``, normalized
to the hip midpoint at contact and the median torso length.

Outputs
-------
``swings.parquet`` (schema_version 1), long format, one row per swing per frame:

* ``swing_id``, ``contact_id``, ``frame_idx``, ``t_video``, ``t_rel`` (``t_video - t_contact``,
  where ``t_contact`` is the contact's audio onset on the PTS timeline)
* ``detected``, ``track_reset``, ``valid`` (racket-side shoulder, elbow and wrist and both
  hips present after cleaning)
* ``<kp>_x``, ``<kp>_y``: normalized (origin hip midpoint at contact, unit median torso
  length, y up); ``<kp>_px``, ``<kp>_py``: cleaned pixel positions; ``<kp>_conf``
* ``l_wrist_vx``, ``l_wrist_vy``, ``l_wrist_speed`` (and ``r_``): normalized units per second
* per-swing QC repeated on each row: ``valid_frame_ratio``, ``swap_count``,
  ``track_reset_count``, ``qc_pass``

``swing_info.parquet`` (schema_version 1), one row per swing: identifiers, contact time and
frame, normalization (``origin_px_x``, ``origin_px_y``, ``scale_px``), QC fields and reason,
and the confirmation result (``wrist_peak_speed``, ``wrist_peak_offset_s``,
``is_self_confirmed``).

``players.json``: which tracked player is you, how that was decided, which side you were on
over time, and your racket hand.

Who is who
----------
Stage 3 tracks a near-court and a far-court player. Their clothing colors are matched to
two identities (``tennis.util.identity``), so the pipeline follows you when you change
ends. Each contact is attributed to the player whose wrist speed peaks highest around it.
You are the identity whose attributed hits are clearly louder on the clip-on mic
(``player.identity_loudness_db``); otherwise the near player at the start
(``player.identity`` overrides both). Swings use your keypoints wherever you are, but
far-side swings are not measured (they fail QC with "far side"). With
``player.handedness: auto`` the racket hand is the wrist that peaks faster at your own
near-side hits.

The spec writes ``is_self_confirmed`` back into ``contacts.parquet``. That file is an input
of stages 3 and 4, so rewriting it would make them stale forever; the flag lives in
``swing_info.parquet`` instead (contacts that are not swings count as not confirmed).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute
import pyarrow.parquet as pq

from tennis.config import CleaningConfig, Config
from tennis.pose_backends.base import KEYPOINT_NAMES, NUM_KEYPOINTS
from tennis.stages.pose import read_frame_pts
from tennis.util import appearance, identity
from tennis.util.cleaning import KP, apply_swaps, fix_swaps, mask_low_confidence
from tennis.util.filters import fill_gaps, one_euro, savgol
from tennis.util.geometry import distance, midpoint
from tennis.util.io import read_json, write_json, write_parquet
from tennis.util.tracking import iou

if TYPE_CHECKING:
    from tennis.session import Session
    from tennis.stages import StageContext

SCHEMA_VERSION = 1
INPUTS = ("keypoints.parquet", "contacts.parquet", "frame_times.parquet")
OUTPUTS = ("swings.parquet", "swing_info.parquet", "players.json")
CONFIG_KEYS = (
    "player",
    "cleaning",
    "windows",
    "pose.kp_conf_min",
    "pose.contacts",
    "audio.wrist_confirm_window_s",
    "audio.wrist_confirm_min_speed",
    "audio.wrist_confirm_wrist",
)

FloatArray = npt.NDArray[np.float64]


def racket_side(handedness: str) -> str:
    """Keypoint prefix of the racket hand: "r" or "l"."""
    if handedness not in ("right", "left"):
        raise ValueError(f"handedness must be resolved to right or left, got {handedness!r}")
    return "r" if handedness == "right" else "l"


def resolved_handedness(session: Session, config: Config) -> str:
    """The configured hand, or the one stage 4 inferred (right if it is not known yet)."""
    if config.player.handedness != "auto":
        return config.player.handedness
    path = session.path("players.json")
    if path.exists():
        value = read_json(path).get("handedness", {}).get("hand")
        if value in ("right", "left"):
            return str(value)
    return "right"


@dataclass
class Poses:
    """Keypoints of all decoded frames, in frame order, after cleaning."""

    frame_idx: npt.NDArray[np.int64]
    t: FloatArray
    window_id: npt.NDArray[np.int32]
    detected: npt.NDArray[np.bool_]
    track_reset: npt.NDArray[np.bool_]
    xy: FloatArray  # (N, 17, 2) cleaned pixel positions
    conf: FloatArray  # (N, 17)
    swaps: npt.NDArray[np.int64]  # (N,) swaps made in each frame
    looks: npt.NDArray[np.float32]  # (N, appearance.SIZE); zeros when unknown
    boxes: FloatArray  # (N, 4) player box, NaN when not detected


def load_keypoints(table: pa.Table, slot: str = "near") -> Poses:
    """The rows of one tracked slot (files without a slot column only have the near one)."""
    if "slot" in table.column_names:
        table = table.filter(pa.compute.equal(table["slot"], slot))
    elif slot != "near":
        table = table.slice(0, 0)
    cols = {name: table.column(name).to_numpy(zero_copy_only=False)
            for name in table.column_names if name != "appearance"}  # fmt: skip
    n = table.num_rows
    if "appearance" in table.column_names and n:
        looks = np.stack(table.column("appearance").to_numpy(zero_copy_only=False)).astype(
            np.float32
        )
    else:
        looks = np.zeros((n, appearance.SIZE), np.float32)
    xy = np.empty((n, NUM_KEYPOINTS, 2))
    conf = np.empty((n, NUM_KEYPOINTS))
    for k, name in enumerate(KEYPOINT_NAMES):
        xy[:, k, 0] = cols[f"{name}_x"]
        xy[:, k, 1] = cols[f"{name}_y"]
        conf[:, k] = cols[f"{name}_conf"]
    boxes = np.full((n, 4), np.nan)
    for i, c in enumerate(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
        if c in cols:
            boxes[:, i] = cols[c]
    order = np.argsort(cols["frame_idx"], kind="stable")
    return Poses(
        frame_idx=np.asarray(cols["frame_idx"], np.int64)[order],
        t=np.asarray(cols["t_video"], np.float64)[order],
        window_id=np.asarray(cols["window_id"], np.int32)[order],
        detected=np.asarray(cols["detected"], bool)[order],
        track_reset=np.asarray(cols["track_reset"], bool)[order],
        xy=xy[order],
        conf=np.nan_to_num(conf[order], nan=0.0),
        swaps=np.zeros(n, np.int64),
        looks=looks[order],
        boxes=boxes[order],
    )


def torso_lengths(xy: FloatArray) -> FloatArray:
    shoulders = midpoint(xy[:, KP["l_shoulder"]], xy[:, KP["r_shoulder"]])
    hips = midpoint(xy[:, KP["l_hip"]], xy[:, KP["r_hip"]])
    return distance(shoulders, hips)


def _nanmedian(values: FloatArray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def clean_run(
    xy: FloatArray, conf: FloatArray, t: FloatArray, cfg: CleaningConfig, kp_conf_min: float
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int64]]:
    """Clean one continuous run of frames. Returns positions, confidences, swaps per frame."""
    masked = mask_low_confidence(xy, conf, kp_conf_min)
    fixed, swap_mask = fix_swaps(masked, cfg.swap_improvement_ratio)
    conf = apply_swaps(conf, swap_mask)
    n = xy.shape[0]
    flat = fill_gaps(fixed.reshape(n, -1), t, cfg.max_gap_frames)
    if cfg.smoother == "one_euro":
        # beta multiplies speed; the spec's value is meant for torso-normalized units.
        scale = _nanmedian(torso_lengths(flat.reshape(n, NUM_KEYPOINTS, 2)))
        beta = cfg.one_euro.beta / scale if np.isfinite(scale) and scale > 0 else 0.0
        e = cfg.one_euro
        flat = one_euro(flat, t, e.min_cutoff, beta, e.d_cutoff, zero_phase=e.zero_phase)
    else:
        flat = savgol(flat, cfg.savgol.window, cfg.savgol.order)
    return flat.reshape(n, NUM_KEYPOINTS, 2), conf, swap_mask.sum(axis=1)


def clean_poses(poses: Poses, cfg: CleaningConfig, kp_conf_min: float) -> Poses:
    xy = np.full_like(poses.xy, np.nan)
    conf = poses.conf.copy()
    swaps = np.zeros(poses.frame_idx.size, np.int64)
    for wid in np.unique(poses.window_id):
        rows = np.flatnonzero(poses.window_id == wid)
        xy[rows], conf[rows], swaps[rows] = clean_run(
            poses.xy[rows], poses.conf[rows], poses.t[rows], cfg, kp_conf_min
        )
    return Poses(poses.frame_idx, poses.t, poses.window_id, poses.detected,
                 poses.track_reset, xy, conf, swaps, poses.looks, poses.boxes)  # fmt: skip


def drop_duplicate_far(near: Poses, far: Poses, max_iou: float = 0.3) -> int:
    """Forget far-slot detections that are really the near player (overlapping boxes).

    Modifies ``far`` in place (before cleaning) and returns how many frames were dropped.
    """
    if not far.frame_idx.size or not near.frame_idx.size:
        return 0
    row_of = {int(f): i for i, f in enumerate(near.frame_idx)}
    dropped = 0
    for j, f in enumerate(far.frame_idx):
        i = row_of.get(int(f))
        if i is None or not (near.detected[i] and far.detected[j]):
            continue
        a, b = near.boxes[i], far.boxes[j]
        if iou((a[0], a[1], a[2], a[3]), (b[0], b[1], b[2], b[3])) > max_iou:
            far.detected[j] = False
            far.xy[j] = np.nan
            far.conf[j] = 0.0
            far.looks[j] = 0.0
            dropped += 1
    return dropped


@dataclass
class Swing:
    swing_id: int
    contact_id: int
    t_contact: float
    rows: npt.NDArray[np.int64]  # into Poses
    expected_frames: int
    info: dict[str, Any] = field(default_factory=dict)
    columns: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    poses: Poses | None = None  # the slot the rows index into
    slot: str = "near"


def build_swing(
    swing: Swing,
    poses: Poses,
    config: Config,
    frame_interval_s: float,
    side: str,
    wrist_mode: str | None = None,
) -> None:
    """Normalize one swing, compute wrist velocities, QC and the confirmation signal.

    ``side`` is the racket-hand prefix ("r" or "l"); ``wrist_mode`` overrides
    ``audio.wrist_confirm_wrist`` for the confirmation signal.
    """
    swing.poses = poses
    qc = config.cleaning.qc
    rows = swing.rows
    info: dict[str, Any] = {
        "swing_id": swing.swing_id,
        "contact_id": swing.contact_id,
        "t_contact": swing.t_contact,
        "n_frames": int(rows.size),
        "expected_frames": swing.expected_frames,
        "contact_frame_idx": None,
        "origin_px_x": np.nan,
        "origin_px_y": np.nan,
        "scale_px": np.nan,
        "valid_frame_ratio": 0.0,
        "swap_count": int(poses.swaps[rows].sum()),
        "track_reset_count": int(poses.track_reset[rows].sum()),
        "wrist_peak_speed": np.nan,
        "wrist_peak_offset_s": np.nan,
    }
    swing.info = info
    if rows.size == 0:
        info.update(qc_pass=False, qc_reason="no pose frames")
        return

    t = poses.t[rows]
    xy = poses.xy[rows]
    t_rel = t - swing.t_contact
    c = int(np.argmin(np.abs(t_rel)))
    contact_ok_time = abs(t_rel[c]) <= max(frame_interval_s, 1e-3)
    info["contact_frame_idx"] = int(poses.frame_idx[rows[c]])

    hips = midpoint(xy[:, KP["l_hip"]], xy[:, KP["r_hip"]])
    hip_ok: npt.NDArray[np.bool_] = np.asarray(np.isfinite(hips).all(axis=1))
    origin_note = ""
    if hip_ok[c]:
        origin = hips[c]
    elif hip_ok.any():
        nearest = np.flatnonzero(hip_ok)[np.argmin(np.abs(t_rel[hip_ok]))]
        origin = hips[nearest]
        origin_note = "origin from nearest frame with hips"
    else:
        origin = np.array([np.nan, np.nan])
    scale = _nanmedian(torso_lengths(xy))
    info.update(origin_px_x=float(origin[0]), origin_px_y=float(origin[1]), scale_px=scale)

    norm = np.empty_like(xy)
    norm[..., 0] = (xy[..., 0] - origin[0]) / scale
    norm[..., 1] = -(xy[..., 1] - origin[1]) / scale

    speeds: dict[str, FloatArray] = {}
    columns: dict[str, npt.NDArray[Any]] = {}
    for s in ("l", "r"):
        w = norm[:, KP[f"{s}_wrist"]]
        if rows.size >= 2:
            vx = np.gradient(w[:, 0], t)
            vy = np.gradient(w[:, 1], t)
        else:
            vx = vy = np.full(rows.size, np.nan)
        speeds[s] = np.hypot(vx, vy)
        columns[f"{s}_wrist_vx"] = vx
        columns[f"{s}_wrist_vy"] = vy
        columns[f"{s}_wrist_speed"] = speeds[s]

    racket = [KP[f"{side}_shoulder"], KP[f"{side}_elbow"], KP[f"{side}_wrist"]]
    needed = [*racket, KP["l_hip"], KP["r_hip"]]
    valid = poses.detected[rows] & np.isfinite(xy[:, needed]).all(axis=(1, 2))
    ratio = float(valid.sum() / swing.expected_frames) if swing.expected_frames else 0.0
    contact_ok = contact_ok_time and bool(np.isfinite(xy[c, racket]).all())

    reasons = []
    if not np.isfinite(scale) or scale <= 0:
        reasons.append("no torso length")
    if not np.isfinite(origin).all():
        reasons.append("no hips")
    if ratio < qc.min_valid_frame_ratio:
        reasons.append(f"valid frames {ratio:.0%}")
    if not contact_ok:
        reasons.append("racket arm missing at contact")
    if info["track_reset_count"] > qc.max_track_resets:
        reasons.append(f"{info['track_reset_count']} track resets")
    info.update(
        valid_frame_ratio=min(1.0, ratio),
        qc_pass=not reasons,
        qc_reason="; ".join(reasons) or origin_note,
    )

    audio = config.audio
    mode = wrist_mode or audio.wrist_confirm_wrist
    signal = speeds[side] if mode == "racket" else np.fmax(speeds["l"], speeds["r"])
    peak_speed, peak_offset = confirmation_peak(t_rel, signal, audio.wrist_confirm_window_s)
    info.update(wrist_peak_speed=peak_speed, wrist_peak_offset_s=peak_offset)

    columns.update(
        t_rel=t_rel,
        valid=valid,
    )
    for k, name in enumerate(KEYPOINT_NAMES):
        columns[f"{name}_x"] = norm[:, k, 0]
        columns[f"{name}_y"] = norm[:, k, 1]
        columns[f"{name}_px"] = xy[:, k, 0]
        columns[f"{name}_py"] = xy[:, k, 1]
        columns[f"{name}_conf"] = poses.conf[rows, k]
    swing.columns = columns


def confirmation_peak(t_rel: FloatArray, speed: FloatArray, window_s: float) -> tuple[float, float]:
    """The highest local maximum of ``speed`` within ``|t_rel| <= window_s``.

    Returns (speed, t_rel) of that peak; (0.0, NaN) if the wrist was tracked in the window
    but its speed has no peak there (e.g. still rising); (NaN, NaN) if it was not tracked.
    """
    near = np.flatnonzero(np.abs(t_rel) <= window_s + 1e-9)
    values = np.where(np.isfinite(speed), speed, -np.inf)
    if near.size == 0 or not np.isfinite(speed[near]).any():
        return float("nan"), float("nan")
    best_speed, best_offset = 0.0, float("nan")
    for i in near:
        v = values[i]
        left = values[i - 1] if i > 0 else -np.inf
        right = values[i + 1] if i + 1 < values.size else -np.inf
        if np.isfinite(v) and v >= left and v >= right and v > best_speed:
            best_speed, best_offset = float(v), float(t_rel[i])
    return best_speed, best_offset


def confirm(swings: Sequence[Swing], min_speed: float) -> None:
    """Second pass: the wrist speed must peak near the onset (see ``confirmation_peak``)
    and reach ``min_speed``.

    A contact attributed to the other player (``hitter == "other"``) is never confirmed.
    Several nearby contacts can share one wrist peak (a hit and the bounce before it);
    only the contact closest to that peak is confirmed.
    """
    for s in swings:
        speed = s.info["wrist_peak_speed"]
        if not np.isfinite(speed):
            s.info["is_self_confirmed"] = None
            continue
        s.info["is_self_confirmed"] = bool(
            np.isfinite(s.info["wrist_peak_offset_s"])
            and speed >= min_speed
            and s.info.get("hitter", "self") != "other"
        )
    confirmed = sorted(
        (s for s in swings if s.info["is_self_confirmed"]),
        key=lambda s: s.t_contact + s.info["wrist_peak_offset_s"],
    )
    groups: list[list[Swing]] = []
    for s in confirmed:
        peak = s.t_contact + s.info["wrist_peak_offset_s"]
        if groups:
            last = groups[-1][0]
            if abs(peak - (last.t_contact + last.info["wrist_peak_offset_s"])) < 1e-6:
                groups[-1].append(s)
                continue
        groups.append([s])
    for group in groups:
        best = min(group, key=lambda s: abs(s.info["wrist_peak_offset_s"]))
        for s in group:
            if s is not best:
                s.info["is_self_confirmed"] = False


INFO_SCHEMA = pa.schema(
    [
        ("swing_id", pa.int64()),
        ("contact_id", pa.int64()),
        ("t_contact", pa.float64()),
        ("contact_frame_idx", pa.int64()),
        ("n_frames", pa.int32()),
        ("expected_frames", pa.int32()),
        ("origin_px_x", pa.float64()),
        ("origin_px_y", pa.float64()),
        ("scale_px", pa.float64()),
        ("valid_frame_ratio", pa.float64()),
        ("swap_count", pa.int32()),
        ("track_reset_count", pa.int32()),
        ("qc_pass", pa.bool_()),
        ("qc_reason", pa.string()),
        ("wrist_peak_speed", pa.float64()),
        ("wrist_peak_offset_s", pa.float64()),
        ("is_self_confirmed", pa.bool_()),
        ("player_side", pa.string()),
        ("hitter", pa.string()),
        ("near_peak_speed", pa.float64()),
        ("far_peak_speed", pa.float64()),
    ]
)


def info_table(swings: Sequence[Swing]) -> pa.Table:
    return pa.Table.from_pylist([{k: s.info.get(k) for k in INFO_SCHEMA.names} for s in swings],
                                schema=INFO_SCHEMA)  # fmt: skip


def swings_table(swings: Sequence[Swing]) -> pa.Table:
    parts: dict[str, list[npt.NDArray[Any]]] = {}
    for s in swings:
        if not s.columns or s.poses is None:
            continue
        poses = s.poses
        n = s.rows.size
        base: dict[str, npt.NDArray[Any]] = {
            "swing_id": np.full(n, s.swing_id, np.int64),
            "contact_id": np.full(n, s.contact_id, np.int64),
            "frame_idx": poses.frame_idx[s.rows],
            "t_video": poses.t[s.rows],
            "detected": poses.detected[s.rows],
            "track_reset": poses.track_reset[s.rows],
            "player_side": np.full(n, s.slot, object),
            **s.columns,
            "valid_frame_ratio": np.full(n, s.info["valid_frame_ratio"]),
            "swap_count": np.full(n, s.info["swap_count"], np.int32),
            "track_reset_count": np.full(n, s.info["track_reset_count"], np.int32),
            "qc_pass": np.full(n, s.info["qc_pass"], bool),
        }
        for name, values in base.items():
            parts.setdefault(name, []).append(values)
    if not parts:
        return _empty_swings_table()
    columns = {}
    for name, chunks in parts.items():
        values = np.concatenate(chunks)
        if values.dtype == np.float64 and name not in ("t_video", "t_rel"):
            values = values.astype(np.float32)
        columns[name] = pa.array(values, pa.string() if name == "player_side" else None)
    return pa.table(columns)


def _empty_swings_table() -> pa.Table:
    fields = [
        ("swing_id", pa.int64()), ("contact_id", pa.int64()), ("frame_idx", pa.int64()),
        ("t_video", pa.float64()), ("detected", pa.bool_()), ("track_reset", pa.bool_()),
        ("player_side", pa.string()),
        ("l_wrist_vx", pa.float32()), ("l_wrist_vy", pa.float32()),
        ("l_wrist_speed", pa.float32()), ("r_wrist_vx", pa.float32()),
        ("r_wrist_vy", pa.float32()), ("r_wrist_speed", pa.float32()),
        ("t_rel", pa.float64()), ("valid", pa.bool_()),
    ]  # fmt: skip
    for name in KEYPOINT_NAMES:
        fields += [(f"{name}_{c}", pa.float32()) for c in ("x", "y", "px", "py", "conf")]
    fields += [
        ("valid_frame_ratio", pa.float32()), ("swap_count", pa.int32()),
        ("track_reset_count", pa.int32()), ("qc_pass", pa.bool_()),
    ]  # fmt: skip
    return pa.schema(fields).empty_table()


def candidate_contacts(contacts: pa.Table, config: Config) -> pa.Table:
    """The contacts pose was extracted for (see ``pose.contacts``)."""
    if config.pose.contacts == "self_audio":
        mask = contacts.column("is_self_audio").to_numpy(zero_copy_only=False).astype(bool)
        return contacts.filter(pa.array(mask))
    return contacts


def _window_bounds(poses: Poses) -> dict[int, tuple[float, float]]:
    bounds: dict[int, tuple[float, float]] = {}
    for wid in np.unique(poses.window_id):
        t = poses.t[poses.window_id == wid]
        bounds[int(wid)] = (float(t[0]), float(t[-1]))
    return bounds


def _window_looks(near: Poses, far: Poses) -> list[identity.WindowLooks]:
    looks = []
    for wid, (start, _) in _window_bounds(near).items():
        n_rows = (near.window_id == wid) & near.detected
        f_rows = (far.window_id == wid) & far.detected
        looks.append(identity.WindowLooks(
            wid, start,
            identity.summarize(near.looks[n_rows]),
            identity.summarize(far.looks[f_rows]) if far.frame_idx.size else None,
        ))  # fmt: skip
    return looks


def _rows_between(poses: Poses, lo: float, hi: float) -> npt.NDArray[np.int64]:
    a = int(np.searchsorted(poses.t, lo - 1e-6, side="left"))
    b = int(np.searchsorted(poses.t, hi + 1e-6, side="right"))
    return np.arange(a, b, dtype=np.int64)


def _decide_me(
    config: Config, hitter_ids: Sequence[int | None], peak_db: Sequence[float]
) -> tuple[int, str, dict[str, Any]]:
    """Which identity is you, and why."""
    choice = config.player.identity
    if choice in ("A", "B"):
        return identity.NAMES.index(choice), "config", {}
    stats: dict[str, Any] = {}
    if choice == "auto":
        levels = {i: [db for h, db in zip(hitter_ids, peak_db, strict=True) if h == i]
                  for i in (identity.A, identity.B)}  # fmt: skip
        stats = {identity.NAMES[i]: {"hits": len(v),
                                     "median_db": round(float(np.median(v)), 1) if v else None}
                 for i, v in levels.items()}  # fmt: skip
        if all(len(v) >= 5 for v in levels.values()):
            diff = float(np.median(levels[identity.A]) - np.median(levels[identity.B]))
            stats["difference_db"] = round(diff, 1)
            if abs(diff) >= config.player.identity_loudness_db:
                return (identity.A if diff > 0 else identity.B), "louder hits", stats
    return identity.A, "near player at the start", stats


def _decide_hand(config: Config, votes: dict[str, int]) -> tuple[str, str]:
    if config.player.handedness != "auto":
        return config.player.handedness, "config"
    if votes["left"] + votes["right"] < 5 or votes["left"] == votes["right"]:
        return "right", "not enough own near-side hits; assumed right"
    return ("left" if votes["left"] > votes["right"] else "right"), "faster wrist at own hits"


def run(ctx: StageContext) -> None:
    session, config = ctx.session, ctx.config
    table = pq.read_table(session.path("keypoints.parquet"))
    kp_min = config.pose.kp_conf_min
    raw_near, raw_far = load_keypoints(table, "near"), load_keypoints(table, "far")
    duplicates = drop_duplicate_far(raw_near, raw_far)
    near = clean_poses(raw_near, config.cleaning, kp_min)
    far = clean_poses(raw_far, config.cleaning, kp_min)
    all_contacts = pq.read_table(session.path("contacts.parquet"))
    contacts = candidate_contacts(all_contacts, config)
    frame_pts = read_frame_pts(session.path("frame_times.parquet"))
    intervals = np.diff(frame_pts)
    frame_interval = float(np.median(intervals)) if intervals.size else 0.0
    pre, post = config.windows.pre_s, config.windows.post_s
    min_speed = config.audio.wrist_confirm_min_speed

    # 1. Who is who, window by window.
    assignment = identity.assign_identities(
        _window_looks(near, far), min_side_duration_s=config.player.min_side_duration_s
    )
    bounds = _window_bounds(near)

    # 2. Who hit each contact: the slot whose wrist speed peaks highest around it.
    times = contacts.column("t_audio").to_numpy()
    ids = contacts.column("contact_id").to_numpy()
    db_by_id = dict(zip(all_contacts.column("contact_id").to_pylist(),
                        all_contacts.column("peak_db").to_pylist(), strict=True))  # fmt: skip
    probes: list[dict[str, Any]] = []
    for contact_id, t_contact in zip(ids, times, strict=True):
        near_rows = _rows_between(near, t_contact - pre, t_contact + post)
        wid = int(near.window_id[near_rows[0]]) if near_rows.size else None
        near_id = assignment.near_identity.get(wid, identity.A) if wid is not None else identity.A
        peaks: dict[str, tuple[float, float, Swing]] = {}
        for slot, poses in (("near", near), ("far", far)):
            rows = _rows_between(poses, t_contact - pre, t_contact + post)
            probe = Swing(0, int(contact_id), float(t_contact), rows, rows.size, slot=slot)
            try:
                build_swing(probe, poses, config, frame_interval, "r", wrist_mode="either")
            except Exception:
                continue
            peaks[slot] = (probe.info["wrist_peak_speed"], probe.info["wrist_peak_offset_s"], probe)
        scored = {k: v[0] for k, v in peaks.items() if np.isfinite(v[0]) and np.isfinite(v[1])}
        hitter_slot = max(scored, key=scored.__getitem__) if scored else None
        hitter_id = None
        if hitter_slot is not None and scored[hitter_slot] >= min_speed:
            hitter_id = near_id if hitter_slot == "near" else 1 - near_id
        probes.append({"contact_id": int(contact_id), "t": float(t_contact), "near_id": near_id,
                       "peaks": peaks, "hitter_id": hitter_id})  # fmt: skip

    # 3. Which identity is you, and your racket hand.
    me, me_reason, loudness = _decide_me(
        config,
        [p["hitter_id"] for p in probes],
        [float(db_by_id.get(p["contact_id"], np.nan)) for p in probes],
    )
    votes = {"left": 0, "right": 0}
    for p in probes:
        if p["hitter_id"] == me and p["near_id"] == me and "near" in p["peaks"]:
            probe = p["peaks"]["near"][2]
            t_rel = probe.columns.get("t_rel")
            if t_rel is None:
                continue
            window = config.audio.wrist_confirm_window_s
            left = confirmation_peak(t_rel, probe.columns["l_wrist_speed"], window)[0]
            right = confirmation_peak(t_rel, probe.columns["r_wrist_speed"], window)[0]
            if np.isfinite(left) and np.isfinite(right) and left != right:
                votes["left" if left > right else "right"] += 1
    hand, hand_reason = _decide_hand(config, votes)
    side = racket_side(hand)

    # 4. Your swings, from wherever you are.
    swings: list[Swing] = []
    for swing_id, p in enumerate(probes):
        slot = "near" if p["near_id"] == me else "far"
        poses = near if slot == "near" else far
        t_contact = p["t"]
        rows = _rows_between(poses, t_contact - pre, t_contact + post)
        expected = int(
            np.searchsorted(frame_pts, t_contact + post + 1e-6, side="right")
            - np.searchsorted(frame_pts, t_contact - pre - 1e-6, side="left")
        )
        swing = Swing(swing_id, p["contact_id"], t_contact, rows, expected, slot=slot)
        try:
            build_swing(swing, poses, config, frame_interval, side)
        except Exception as exc:  # one bad swing must not stop the stage
            ctx.log("swing failed", level=logging.WARNING, swing_id=swing_id, error=repr(exc))
            swing.columns = {}
            swing.info.update(qc_pass=False, qc_reason=f"error: {exc}")
        if slot == "far":
            reasons = ["far side (not measured)", swing.info.get("qc_reason") or ""]
            swing.info.update(qc_pass=False, qc_reason="; ".join(r for r in reasons if r))
        hitter = p["hitter_id"]
        swing.info.update(
            player_side=slot,
            hitter=None if hitter is None else ("self" if hitter == me else "other"),
            near_peak_speed=p["peaks"].get("near", (np.nan,))[0],
            far_peak_speed=p["peaks"].get("far", (np.nan,))[0],
        )
        swings.append(swing)

    confirm(swings, min_speed)

    sides = identity.side_segments(
        assignment.near_identity, {w: b[0] for w, b in bounds.items()},
        {w: b[1] for w, b in bounds.items()}, me,
    )  # fmt: skip
    players = {
        "schema_version": 1,
        "me": identity.NAMES[me],
        "me_reason": me_reason,
        "identities_resolved": assignment.resolved,
        "loudness": loudness,
        "handedness": {"hand": hand, "reason": hand_reason, "votes": votes},
        "sides": sides,
        "windows": [
            {
                "window_id": wid,
                "start": round(b[0], 3),
                "end": round(b[1], 3),
                "near": identity.NAMES[assignment.near_identity.get(wid, identity.A)],
                "margin": round(assignment.margin.get(wid, 0.0), 3),
            }
            for wid, b in sorted(bounds.items(), key=lambda kv: kv[1][0])
        ],
        "hits": {
            "self": sum(p["hitter_id"] == me for p in probes),
            "other": sum(p["hitter_id"] is not None and p["hitter_id"] != me for p in probes),
            "unknown": sum(p["hitter_id"] is None for p in probes),
        },
        "prototypes": {
            name: None if proto is None else [round(float(v), 5) for v in proto]
            for name, proto in zip(identity.NAMES, assignment.prototypes, strict=True)
        },
    }

    write_parquet(swings_table(swings), session.path("swings.parquet"),
                  stage=ctx.stage, config_hash=ctx.config_hash,
                  schema_version=SCHEMA_VERSION, extra={"racket_side": side})  # fmt: skip
    write_parquet(info_table(swings), session.path("swing_info.parquet"),
                  stage=ctx.stage, config_hash=ctx.config_hash,
                  schema_version=SCHEMA_VERSION, extra={"racket_side": side})  # fmt: skip
    write_json(session.path("players.json"), players)

    passed = sum(bool(s.info["qc_pass"]) for s in swings)
    confirmed = sum(bool(s.info.get("is_self_confirmed")) for s in swings)
    far_side = sum(s.slot == "far" for s in swings)
    ctx.log(
        "players identified",
        me=identity.NAMES[me],
        reason=me_reason,
        hand=hand,
        hand_votes=f"{votes['left']}L/{votes['right']}R",
        end_changes=max(0, len(sides) - 1),
        far_duplicates_dropped=duplicates,
    )
    ctx.log(
        "swings cleaned",
        swings=len(swings),
        far_side=far_side,
        qc_pass=passed,
        qc_pass_rate=round(passed / len(swings), 3) if swings else None,
        confirmed=confirmed,
        confirmed_and_qc=sum(
            bool(s.info.get("is_self_confirmed")) and bool(s.info["qc_pass"]) for s in swings
        ),
        swaps=int(near.swaps.sum()),
    )
