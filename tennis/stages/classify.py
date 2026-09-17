"""Stage 5: stroke classification (spec section 6.5).

Each swing is labelled ``serve``, ``forehand``, ``backhand`` or ``volley``, plus a
``two_handed`` flag. Only swings that are yours (``is_self_confirmed``), measurable
(``qc_pass``) and on the near side are classified; the rest keep a null ``stroke_type``
so that the row still exists and downstream joins stay total.

Output
------
``strokes.parquet`` (schema_version 1), one row per swing:

* ``swing_id``, ``contact_id``, ``t_contact``
* ``stroke_type`` (null when not classified), ``two_handed`` (null likewise)
* ``classifier_version``, ``rule`` (which rule fired, or why the swing was skipped)
* the features the rules used: ``racket_wrist_x``, ``racket_wrist_y``,
  ``other_wrist_x``, ``other_wrist_y``, ``nose_y``, ``wrist_travel``, ``wrist_gap``,
  ``bbox_bottom_y``, ``player_side``, ``handedness``

The spec has stage 5 rewrite ``swings.parquet`` in place. That file is an input of this
stage, so rewriting it would make the stage permanently stale under the fingerprint cache
(see ARCHITECTURE.md). A separate file joined on ``swing_id`` avoids that.

Rules
-----
Evaluated in order, in normalized units (origin at the hip midpoint at contact, one unit =
the median torso length, y up, x in image direction):

1. **serve**: the racket wrist is more than ``serve_wrist_above_nose`` above the nose.
2. **volley**: the racket wrist travelled less than ``volley_travel_max`` over the
   0.5 s before contact *and* the player's box bottom is above
   ``volley_bbox_bottom_max_y`` of the image height (they are close to the net).
3. **forehand**: the racket wrist is on the racket-hand side of the hip midpoint.
   Normalized x is image x, so from behind the baseline a right-handed player's forehand
   has ``racket_wrist_x > 0`` and a left-handed player's has ``racket_wrist_x < 0``.
4. **backhand**: anything else.

``two_handed`` is true when the two wrists are within ``two_handed_max_dist`` of each
other at contact. ``player.camera_side: side_on`` is rejected: the sign rule in 3 assumes
the camera is behind the baseline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tennis.config import ClassifyConfig
from tennis.errors import UserError
from tennis.session import Session
from tennis.util.io import read_json, write_parquet

if TYPE_CHECKING:
    from tennis.stages import StageContext

SCHEMA_VERSION = 1
INPUTS = ("swings.parquet", "swing_info.parquet", "keypoints.parquet", "metadata.json")
OUTPUTS = ("strokes.parquet",)
CONFIG_KEYS = ("player", "classify")

STROKE_TYPES: tuple[str, ...] = ("serve", "forehand", "backhand", "volley")
SwingInfo = dict[str, Any]  # one row of swing_info.parquet
VOLLEY_TRAVEL_WINDOW_S = 0.5


@dataclass(frozen=True)
class SwingFeatures:
    """Per-swing values the rules need, computed once from the swing's frames."""

    swing_id: int
    contact_id: int
    t_contact: float
    handedness: str  # "right" or "left"
    player_side: str  # "near" or "far"
    racket_wrist_x: float
    racket_wrist_y: float
    other_wrist_x: float
    other_wrist_y: float
    nose_y: float
    wrist_travel: float  # racket-wrist path length over [-0.5, 0] s, torso lengths
    wrist_gap: float  # distance between the wrists at contact
    bbox_bottom_y: float  # player's box bottom as a fraction of the image height

    @property
    def racket_sign(self) -> float:
        """+1 when the racket hand is on the image-right side of the body, else -1."""
        return 1.0 if self.handedness == "right" else -1.0


@dataclass(frozen=True)
class StrokeResult:
    stroke_type: str | None
    two_handed: bool | None
    rule: str


class Classifier(Protocol):
    version: str

    def classify(self, swing: SwingFeatures) -> StrokeResult: ...


class RuleClassifier:
    """The spec's rule cascade (see the module docstring). v1 of the classifier."""

    version = "rule-1"

    def __init__(self, config: ClassifyConfig) -> None:
        self.config = config

    def classify(self, swing: SwingFeatures) -> StrokeResult:
        cfg = self.config
        wrist_x, wrist_y = swing.racket_wrist_x, swing.racket_wrist_y
        if not (np.isfinite(wrist_x) and np.isfinite(wrist_y)):
            return StrokeResult(None, None, "racket wrist missing at contact")
        two_handed: bool | None = (
            bool(swing.wrist_gap <= cfg.two_handed_max_dist)
            if np.isfinite(swing.wrist_gap)
            else None
        )
        if np.isfinite(swing.nose_y) and wrist_y > swing.nose_y + cfg.serve_wrist_above_nose:
            return StrokeResult("serve", two_handed, "wrist above nose")
        at_net = np.isfinite(swing.bbox_bottom_y) and (
            swing.bbox_bottom_y <= cfg.volley_bbox_bottom_max_y
        )
        short = np.isfinite(swing.wrist_travel) and swing.wrist_travel < cfg.volley_travel_max
        if short and at_net:
            return StrokeResult("volley", two_handed, "short wrist travel near the net")
        if wrist_x * swing.racket_sign > 0:
            return StrokeResult("forehand", two_handed, "wrist on the racket-hand side")
        return StrokeResult("backhand", two_handed, "wrist on the other side")


ClassifierFactory = Callable[[ClassifyConfig], Classifier]
_REGISTRY: dict[str, ClassifierFactory] = {}


def register_classifier(name: str, factory: ClassifierFactory) -> ClassifierFactory:
    """Register a classifier factory under ``name`` (same pattern as the pose backends)."""
    previous = _REGISTRY.get(name)
    _REGISTRY[name] = factory
    return previous if previous is not None else factory


def available_classifiers() -> list[str]:
    return sorted(_REGISTRY)


def get_classifier(config: ClassifyConfig) -> Classifier:
    factory = _REGISTRY.get(config.classifier)
    if factory is None:
        raise UserError(
            f"unknown classify.classifier '{config.classifier}'; "
            f"available: {', '.join(available_classifiers())}"
        )
    return factory(config)


register_classifier("rule", RuleClassifier)


# --- features -----------------------------------------------------------------------------


def _box_bottoms(session: Session, frame_height: float) -> dict[tuple[int, str], float]:
    """Box bottom of each tracked player per frame, as a fraction of the image height."""
    path = session.path("keypoints.parquet")
    if not path.exists() or frame_height <= 0:
        return {}
    columns = ["frame_idx", "bbox_y2"]
    names = pq.read_schema(path).names
    if not all(c in names for c in columns):
        return {}
    if "slot" in names:
        columns.append("slot")
    table = pq.read_table(path, columns=columns).to_pydict()
    slots = table.get("slot") or ["near"] * len(table["frame_idx"])
    out: dict[tuple[int, str], float] = {}
    for frame, slot, y2 in zip(table["frame_idx"], slots, table["bbox_y2"], strict=True):
        if y2 is not None and np.isfinite(y2):
            out[(int(frame), str(slot))] = float(y2) / frame_height
    return out


def _path_length(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    return float(np.hypot(np.diff(x[ok]), np.diff(y[ok])).sum())


def swing_features(
    rows: dict[str, np.ndarray],
    info: SwingInfo,
    handedness: str,
    bbox_bottoms: dict[tuple[int, str], float],
    frame_height: float,
) -> SwingFeatures:
    """Features of one swing. ``rows`` holds its frames, sorted by ``t_rel``."""
    side = "r" if handedness == "right" else "l"
    other = "l" if side == "r" else "r"
    t_rel = rows["t_rel"]
    contact = int(np.argmin(np.abs(t_rel))) if t_rel.size else -1
    player_side = str(info.get("player_side") or "near")

    def at_contact(column: str) -> float:
        if contact < 0 or column not in rows:
            return float("nan")
        return float(rows[column][contact])

    travel = float("nan")
    if t_rel.size:
        window = (t_rel >= -VOLLEY_TRAVEL_WINDOW_S - 1e-9) & (t_rel <= 1e-9)
        travel = _path_length(rows[f"{side}_wrist_x"][window], rows[f"{side}_wrist_y"][window])

    racket = np.array([at_contact(f"{side}_wrist_x"), at_contact(f"{side}_wrist_y")])
    opposite = np.array([at_contact(f"{other}_wrist_x"), at_contact(f"{other}_wrist_y")])
    gap = float(np.hypot(*(racket - opposite))) if np.isfinite(opposite).all() else float("nan")

    bottom = float("nan")
    frame = info.get("contact_frame_idx")
    if frame is not None:
        bottom = bbox_bottoms.get((int(frame), player_side), float("nan"))
    if not np.isfinite(bottom) and contact >= 0 and frame_height > 0:
        # No stored box: use the lowest keypoint the cleaner kept at the contact frame.
        pixels = np.array([rows[c][contact] for c in rows if c.endswith("_py")], dtype=np.float64)
        finite = pixels[np.isfinite(pixels)]
        if finite.size:
            bottom = float(finite.max()) / frame_height

    return SwingFeatures(
        swing_id=int(info["swing_id"]),
        contact_id=int(info["contact_id"]),
        t_contact=float(info["t_contact"]),
        handedness=handedness,
        player_side=player_side,
        racket_wrist_x=float(racket[0]),
        racket_wrist_y=float(racket[1]),
        other_wrist_x=float(opposite[0]),
        other_wrist_y=float(opposite[1]),
        nose_y=at_contact("nose_y"),
        wrist_travel=travel,
        wrist_gap=gap,
        bbox_bottom_y=bottom,
    )


def is_classifiable(info: SwingInfo) -> tuple[bool, str]:
    """Whether a swing gets a stroke type, and why not when it doesn't."""
    if not info.get("is_self_confirmed"):
        return False, "not a confirmed own hit"
    if not info.get("qc_pass"):
        return False, "failed QC"
    if (info.get("player_side") or "near") != "near":
        return False, "far side (not measured)"
    return True, ""


# --- output -------------------------------------------------------------------------------

SCHEMA = pa.schema(
    [
        ("swing_id", pa.int64()),
        ("contact_id", pa.int64()),
        ("t_contact", pa.float64()),
        ("stroke_type", pa.string()),
        ("two_handed", pa.bool_()),
        ("classifier_version", pa.string()),
        ("rule", pa.string()),
        ("player_side", pa.string()),
        ("handedness", pa.string()),
        ("racket_wrist_x", pa.float64()),
        ("racket_wrist_y", pa.float64()),
        ("other_wrist_x", pa.float64()),
        ("other_wrist_y", pa.float64()),
        ("nose_y", pa.float64()),
        ("wrist_travel", pa.float64()),
        ("wrist_gap", pa.float64()),
        ("bbox_bottom_y", pa.float64()),
    ]
)


def strokes_table(rows: Sequence[tuple[SwingFeatures, StrokeResult]], version: str) -> pa.Table:
    records = []
    for features, result in rows:
        record = asdict(features)
        record.update(
            stroke_type=result.stroke_type,
            two_handed=result.two_handed,
            classifier_version=version,
            rule=result.rule,
        )
        records.append({name: record.get(name) for name in SCHEMA.names})
    return pa.Table.from_pylist(records, schema=SCHEMA)


def group_rows(table: pa.Table) -> dict[int, dict[str, np.ndarray]]:
    """The frames of each swing, keyed by ``swing_id`` and sorted by ``t_rel``."""
    data = {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}
    ids = data["swing_id"]
    order = np.argsort(ids, kind="stable")
    bounds = np.flatnonzero(np.diff(ids[order])) + 1
    out: dict[int, dict[str, np.ndarray]] = {}
    for chunk in np.split(order, bounds):
        if not chunk.size:
            continue
        chunk = chunk[np.argsort(data["t_rel"][chunk], kind="stable")]
        out[int(ids[chunk[0]])] = {name: values[chunk] for name, values in data.items()}
    return out


def run(ctx: StageContext) -> None:
    from tennis.stages.clean import resolved_handedness

    session, config = ctx.session, ctx.config
    if config.player.camera_side != "behind_baseline":
        raise UserError(
            f"stroke classification needs player.camera_side: behind_baseline, got "
            f"'{config.player.camera_side}'; the forehand/backhand rule is not specified "
            "for a side-on camera"
        )
    handedness = resolved_handedness(session, config)
    classifier = get_classifier(config.classify)
    meta = read_json(session.path("metadata.json"))
    frame_height = float((meta.get("video") or {}).get("height") or 0.0)
    bbox_bottoms = _box_bottoms(session, frame_height)

    info_rows = pq.read_table(session.path("swing_info.parquet")).to_pylist()
    columns = _needed_columns(session, handedness)
    frames = group_rows(pq.read_table(session.path("swings.parquet"), columns=columns))

    rows: list[tuple[SwingFeatures, StrokeResult]] = []
    counts: dict[str, int] = {}
    for info in info_rows:
        swing_id = int(info["swing_id"])
        ok, why = is_classifiable(info)
        empty = {name: np.zeros(0) for name in columns}
        try:
            features = swing_features(
                frames.get(swing_id, empty), info, handedness, bbox_bottoms, frame_height
            )
        except Exception as exc:  # one bad swing must not stop the stage
            ctx.log("swing failed", level=logging.WARNING, swing_id=swing_id, error=repr(exc))
            continue
        result = classifier.classify(features) if ok else StrokeResult(None, None, why)
        counts[result.stroke_type or "unclassified"] = (
            counts.get(result.stroke_type or "unclassified", 0) + 1
        )
        rows.append((features, result))

    write_parquet(
        strokes_table(rows, classifier.version),
        session.path("strokes.parquet"),
        stage=ctx.stage,
        config_hash=ctx.config_hash,
        schema_version=SCHEMA_VERSION,
        extra={"classifier_version": classifier.version, "handedness": handedness},
    )
    ctx.log(
        "strokes classified",
        classifier=classifier.version,
        handedness=handedness,
        two_handed=sum(1 for _, r in rows if r.two_handed),
        **{k: counts.get(k, 0) for k in (*STROKE_TYPES, "unclassified")},
    )


def _needed_columns(session: Session, handedness: str) -> list[str]:
    names = set(pq.read_schema(session.path("swings.parquet")).names)
    wanted = ["swing_id", "t_rel", "nose_y"]
    for side in ("l", "r"):
        wanted += [f"{side}_wrist_x", f"{side}_wrist_y"]
    wanted += [c for c in names if c.endswith("_py")]
    missing = [c for c in wanted if c not in names]
    if missing:
        raise UserError(f"swings.parquet is missing columns: {', '.join(sorted(missing))}")
    return wanted


def read_strokes(session: Session) -> dict[int, dict[str, Any]]:
    """``strokes.parquet`` keyed by ``swing_id`` (used by later stages and the eval tool)."""
    path = session.path("strokes.parquet")
    if not path.exists():
        raise UserError(f"session '{session.id}' has no strokes.parquet; run 'tennis process'")
    return {int(r["swing_id"]): r for r in pq.read_table(path).to_pylist()}


def stroke_counts(table: pa.Table) -> dict[str, int]:
    """How many swings got each stroke type (nulls counted as ``unclassified``)."""
    counts = {name: 0 for name in (*STROKE_TYPES, "unclassified")}
    values = table.column("stroke_type").to_pylist() if table.num_rows else []
    for value in values:
        key = value if value in counts else "unclassified"
        counts[key] += 1
    return counts
