"""Stage 6: technique metrics, aggregation and outliers (spec section 6.6).

Every metric is a registered function of one swing, so adding one is a single edit here::

    @metric("elbow_angle_contact", applies_to={"serve", "forehand", "backhand", "volley"},
            unit="deg")
    def elbow_angle_contact(s: SwingFrame, racket: Side) -> float: ...

``SwingFrame`` gives time-indexed access to the swing's normalized keypoints: ``at(t_rel)``
for the nearest frame to a time, ``between(a, b)`` for a span, and ``contact`` for the
frame nearest ``t_rel = 0``. ``racket`` is ``"l"`` or ``"r"``.

Outputs
-------
``metrics.parquet`` (schema_version 1), one row per swing that is yours, passes QC and is
on the near side: ``swing_id``, ``contact_id``, ``t_contact``, ``stroke_type``,
``two_handed``, one float64 column per metric (NaN where the metric does not apply to that
stroke type or could not be computed), and ``is_outlier`` / ``outlier_score``.

``metrics_summary.parquet`` (schema_version 1), one row per stroke type per metric:
``count``, ``mean``, ``std`` (the spec's "consistency"), ``median``, ``p10``, ``p90``.
Keeping the aggregates in their own file means the report stage and ``tennis trends`` can
read them straight from disk instead of recomputing them per session.

The 2-D convention
------------------
Coordinates are the normalized ones from stage 4: the origin is the hip midpoint at
contact, one unit is the swing's median torso length, ``y`` points up and ``x`` is image x
(so with the camera behind the baseline, the player's right is ``+x``).

``contact_forward`` carries a caveat that is worth repeating wherever it is read: with the
camera behind the baseline, moving **up the court** and moving **up** both show up as a
smaller image y, so a single camera cannot separate depth from height. The two metrics
therefore use different references - ``contact_height`` is measured from the ground
(the ankle midpoint) and ``contact_forward`` from the body centre (the hip midpoint) - and
``player.forward_sign`` flips the latter for a mirrored setup. See ARCHITECTURE.md.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from tennis.config import MetricsConfig
from tennis.errors import UserError
from tennis.pose_backends.base import KEYPOINT_NAMES
from tennis.session import Session
from tennis.stages.classify import STROKE_TYPES
from tennis.util.cleaning import KP
from tennis.util.geometry import angle_deg, distance, midpoint
from tennis.util.io import write_parquet

if TYPE_CHECKING:
    from tennis.stages import StageContext

SCHEMA_VERSION = 1
INPUTS = ("swings.parquet", "swing_info.parquet", "strokes.parquet")
OUTPUTS = ("metrics.parquet", "metrics_summary.parquet")
CONFIG_KEYS = ("player", "metrics")

FloatArray = npt.NDArray[np.float64]
Side = Literal["l", "r"]
ALL_STROKES = frozenset(STROKE_TYPES)
GROUNDSTROKES = frozenset({"forehand", "backhand"})
SWINGING = frozenset({"serve", "forehand", "backhand"})
NAN = float("nan")


# --- time-indexed access to one swing ------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    """One frame of a swing: normalized keypoints, plus the time it was taken."""

    t_rel: float
    xy: FloatArray  # (17, 2)

    def __getitem__(self, name: str) -> FloatArray:
        return np.asarray(self.xy[KP[name]], dtype=np.float64)

    def point(self, side: Side, part: str) -> FloatArray:
        return np.asarray(self.xy[KP[f"{side}_{part}"]], dtype=np.float64)

    def hips(self) -> FloatArray:
        return midpoint(self["l_hip"], self["r_hip"])

    def shoulders(self) -> FloatArray:
        return midpoint(self["l_shoulder"], self["r_shoulder"])

    def ankles(self) -> FloatArray:
        return midpoint(self["l_ankle"], self["r_ankle"])


@dataclass(frozen=True)
class SwingFrame:
    """The frames of one swing, sorted by ``t_rel``, with the stored wrist speeds."""

    swing_id: int
    t_rel: FloatArray
    xy: FloatArray  # (T, 17, 2) normalized
    speeds: dict[str, FloatArray]  # "l" / "r" wrist speed, torso lengths per second
    # Per-run context a metric may need. Keeping it on the swing lets every metric stay a
    # plain function of (swing, racket side) instead of reaching for the config.
    forward_sign: float = 1.0
    unit_turn_ratio: float = 0.85

    def __len__(self) -> int:
        return int(self.t_rel.size)

    def at(self, t_rel: float, tolerance_s: float = 0.2) -> Pose | None:
        """The frame nearest ``t_rel``, or None when nothing is that close."""
        if not len(self):
            return None
        i = int(np.argmin(np.abs(self.t_rel - t_rel)))
        if abs(self.t_rel[i] - t_rel) > tolerance_s:
            return None
        return Pose(float(self.t_rel[i]), self.xy[i])

    def between(self, a: float, b: float) -> SwingFrame:
        """The frames with ``a <= t_rel <= b`` (endpoints included, within rounding)."""
        mask = (self.t_rel >= a - 1e-9) & (self.t_rel <= b + 1e-9)
        return SwingFrame(
            self.swing_id,
            self.t_rel[mask],
            self.xy[mask],
            {side: values[mask] for side, values in self.speeds.items()},
            self.forward_sign,
            self.unit_turn_ratio,
        )

    @property
    def contact(self) -> Pose | None:
        return self.at(0.0, tolerance_s=float("inf"))

    @property
    def start(self) -> float:
        return float(self.t_rel[0]) if len(self) else NAN

    @property
    def end(self) -> float:
        return float(self.t_rel[-1]) if len(self) else NAN

    def series(self, name: str, axis: int) -> FloatArray:
        return np.asarray(self.xy[:, KP[name], axis], dtype=np.float64)

    def track(self, side: Side, part: str) -> FloatArray:
        """(T, 2) positions of one keypoint over the span."""
        return np.asarray(self.xy[:, KP[f"{side}_{part}"]], dtype=np.float64)

    def speed(self, side: Side) -> FloatArray:
        return self.speeds[side]


def other_side(side: Side) -> Side:
    return "l" if side == "r" else "r"


# --- the registry --------------------------------------------------------------------------

MetricFn = Callable[[SwingFrame, Side], float]


@dataclass(frozen=True)
class Metric:
    name: str
    applies_to: frozenset[str]
    unit: str
    description: str
    fn: MetricFn

    def compute(self, swing: SwingFrame, racket: Side, stroke_type: str) -> float:
        if stroke_type not in self.applies_to:
            return NAN
        value = self.fn(swing, racket)
        return float(value) if np.isfinite(value) else NAN


REGISTRY: dict[str, Metric] = {}


def metric(
    name: str, *, applies_to: Iterable[str], unit: str = "", description: str = ""
) -> Callable[[MetricFn], MetricFn]:
    """Register a metric. ``applies_to`` names the stroke types it is defined for."""

    def register(fn: MetricFn) -> MetricFn:
        unknown = set(applies_to) - ALL_STROKES
        if unknown:
            raise ValueError(f"metric {name}: unknown stroke type(s) {sorted(unknown)}")
        if name in REGISTRY:
            raise ValueError(f"metric {name} is already registered")
        REGISTRY[name] = Metric(
            name=name,
            applies_to=frozenset(applies_to),
            unit=unit,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
            fn=fn,
        )
        return fn

    return register


def metric_names() -> list[str]:
    """Registered metrics, in registration order (the column order of the output)."""
    return list(REGISTRY)


# --- the metrics ---------------------------------------------------------------------------


def _finite(values: FloatArray) -> FloatArray:
    return values[np.isfinite(values)]


def _knee_angles(pose: Pose) -> FloatArray:
    return np.asarray(
        [
            float(angle_deg(pose[f"{s}_hip"], pose[f"{s}_knee"], pose[f"{s}_ankle"]))
            for s in ("l", "r")
        ],
        dtype=np.float64,
    )


def _mean_knee_angle(pose: Pose) -> float:
    """Mean of the two knee angles; NaN when neither knee is tracked."""
    finite = _finite(_knee_angles(pose))
    return float(finite.mean()) if finite.size else NAN


def _shoulder_width(pose: Pose) -> float:
    return float(distance(pose["l_shoulder"], pose["r_shoulder"]))


def _shoulder_widths(swing: SwingFrame) -> FloatArray:
    left = swing.xy[:, KP["l_shoulder"]]
    right = swing.xy[:, KP["r_shoulder"]]
    return np.asarray(np.hypot(left[:, 0] - right[:, 0], left[:, 1] - right[:, 1]))


@metric(
    "contact_height",
    applies_to=ALL_STROKES,
    unit="torso",
    description="Height of the racket wrist above the ground (ankle midpoint) at contact.",
)
def contact_height(swing: SwingFrame, racket: Side) -> float:
    pose = swing.contact
    if pose is None:
        return NAN
    return float(pose.point(racket, "wrist")[1] - pose.ankles()[1])


@metric(
    "contact_forward",
    applies_to=ALL_STROKES,
    unit="torso",
    description="Racket wrist up-court of the body centre at contact (see the 2-D caveat).",
)
def contact_forward(swing: SwingFrame, racket: Side) -> float:
    pose = swing.contact
    if pose is None:
        return NAN
    return float(swing.forward_sign * (pose.point(racket, "wrist")[1] - pose.hips()[1]))


@metric(
    "contact_lateral",
    applies_to=ALL_STROKES,
    unit="torso",
    description="Racket wrist to the racket-hand side of the body centre at contact.",
)
def contact_lateral(swing: SwingFrame, racket: Side) -> float:
    pose = swing.contact
    if pose is None:
        return NAN
    sign = 1.0 if racket == "r" else -1.0
    return float(sign * (pose.point(racket, "wrist")[0] - pose.hips()[0]))


@metric(
    "contact_reach",
    applies_to=ALL_STROKES,
    unit="torso",
    description="Shoulder-to-wrist distance at contact: how far the arm is extended.",
)
def contact_reach(swing: SwingFrame, racket: Side) -> float:
    pose = swing.contact
    if pose is None:
        return NAN
    return float(distance(pose.point(racket, "shoulder"), pose.point(racket, "wrist")))


@metric(
    "elbow_angle_contact",
    applies_to=ALL_STROKES,
    unit="deg",
    description="Racket-arm elbow angle (shoulder-elbow-wrist) at contact; 180 is straight.",
)
def elbow_angle_contact(swing: SwingFrame, racket: Side) -> float:
    pose = swing.contact
    if pose is None:
        return NAN
    return float(
        angle_deg(
            pose.point(racket, "shoulder"), pose.point(racket, "elbow"), pose.point(racket, "wrist")
        )
    )


@metric(
    "elbow_angle_min",
    applies_to=ALL_STROKES,
    unit="deg",
    description="Smallest racket-arm elbow angle in the 0.5 s before contact.",
)
def elbow_angle_min(swing: SwingFrame, racket: Side) -> float:
    span = swing.between(-0.5, 0.0)
    if not len(span):
        return NAN
    angles = angle_deg(
        span.track(racket, "shoulder"), span.track(racket, "elbow"), span.track(racket, "wrist")
    )
    finite = _finite(np.asarray(angles, dtype=np.float64))
    return float(finite.min()) if finite.size else NAN


@metric(
    "knee_flex_min",
    applies_to=ALL_STROKES,
    unit="deg",
    description="Deepest knee bend in the swing: the smallest mean knee angle of any frame.",
)
def knee_flex_min(swing: SwingFrame, racket: Side) -> float:
    values = [_mean_knee_angle(Pose(float(t), xy))
              for t, xy in zip(swing.t_rel, swing.xy, strict=True)]  # fmt: skip
    finite = _finite(np.asarray(values, dtype=np.float64))
    return float(finite.min()) if finite.size else NAN


@metric(
    "knee_flex_contact",
    applies_to=ALL_STROKES,
    unit="deg",
    description="Mean knee angle at contact; smaller means more bend left in the legs.",
)
def knee_flex_contact(swing: SwingFrame, racket: Side) -> float:
    pose = swing.contact
    return _mean_knee_angle(pose) if pose is not None else NAN


@metric(
    "peak_wrist_speed",
    applies_to=ALL_STROKES,
    unit="torso/s",
    description="Highest racket-wrist speed in the swing window.",
)
def peak_wrist_speed(swing: SwingFrame, racket: Side) -> float:
    finite = _finite(swing.speed(racket))
    return float(finite.max()) if finite.size else NAN


@metric(
    "peak_speed_offset",
    applies_to=ALL_STROKES,
    unit="s",
    description="When the racket wrist was fastest, relative to contact (negative = before).",
)
def peak_speed_offset(swing: SwingFrame, racket: Side) -> float:
    speed = swing.speed(racket)
    if not np.isfinite(speed).any():
        return NAN
    return float(swing.t_rel[int(np.nanargmax(np.where(np.isfinite(speed), speed, -np.inf)))])


@metric(
    "shoulder_turn_proxy_min",
    applies_to=SWINGING,
    unit="ratio",
    description="Narrowest apparent shoulder width, as a fraction of its width 1 s before.",
)
def shoulder_turn_proxy_min(swing: SwingFrame, racket: Side) -> float:
    ratios = shoulder_turn_ratios(swing)
    finite = _finite(ratios)
    return float(finite.min()) if finite.size else NAN


def shoulder_turn_ratios(swing: SwingFrame, reference_t: float = -1.0) -> FloatArray:
    """Apparent shoulder width over the swing, divided by its width at ``reference_t``.

    NaN throughout when the window starts later than ``reference_t``: without that frame
    there is no baseline to compare against, and the spec's proxy is meaningless.
    """
    if not len(swing) or swing.start > reference_t + 1e-9:
        return np.full(len(swing), NAN)
    base = swing.at(reference_t, tolerance_s=0.1)
    if base is None:
        return np.full(len(swing), NAN)
    width = _shoulder_width(base)
    if not np.isfinite(width) or width <= 0:
        return np.full(len(swing), NAN)
    return _shoulder_widths(swing) / width


@metric(
    "unit_turn_lead_time",
    applies_to=GROUNDSTROKES,
    unit="s",
    description="How long before contact the shoulders first turned away from the camera.",
)
def unit_turn_lead_time(swing: SwingFrame, racket: Side) -> float:
    ratios = shoulder_turn_ratios(swing)
    if not np.isfinite(ratios).any():
        return NAN
    threshold = swing.unit_turn_ratio
    before = swing.t_rel <= 1e-9
    turned = np.flatnonzero(before & np.isfinite(ratios) & (ratios <= threshold))
    if not turned.size:
        return NAN
    return float(-swing.t_rel[turned[0]])


@metric(
    "follow_through_height",
    applies_to=SWINGING,
    unit="torso",
    description="Racket wrist height above the hips at the end of the swing window.",
)
def follow_through_height(swing: SwingFrame, racket: Side) -> float:
    if not len(swing):
        return NAN
    pose = swing.at(swing.end, tolerance_s=float("inf"))
    if pose is None:
        return NAN
    return float(pose.point(racket, "wrist")[1] - pose.hips()[1])


@metric(
    "torso_lean_contact",
    applies_to=ALL_STROKES,
    unit="deg",
    description="Tilt of the hip-to-shoulder line from vertical at contact, towards the "
    "racket side.",
)
def torso_lean_contact(swing: SwingFrame, racket: Side) -> float:
    pose = swing.contact
    if pose is None:
        return NAN
    hips, shoulders = pose.hips(), pose.shoulders()
    dx, dy = shoulders[0] - hips[0], shoulders[1] - hips[1]
    if not (np.isfinite(dx) and np.isfinite(dy)) or (dx == 0 and dy == 0):
        return NAN
    sign = 1.0 if racket == "r" else -1.0
    return float(sign * np.degrees(np.arctan2(dx, dy)))


# --- building the table ---------------------------------------------------------------------


def load_swings(
    session: Session, forward_sign: float = 1.0, unit_turn_ratio: float = 0.85
) -> dict[int, SwingFrame]:
    """Every swing in ``swings.parquet`` as a ``SwingFrame``, keyed by ``swing_id``."""
    columns = ["swing_id", "t_rel", "l_wrist_speed", "r_wrist_speed"]
    columns += [f"{name}_{axis}" for name in KEYPOINT_NAMES for axis in ("x", "y")]
    path = session.path("swings.parquet")
    missing = [c for c in columns if c not in set(pq.read_schema(path).names)]
    if missing:
        raise UserError(
            f"{path.name} is missing columns: {', '.join(missing)}; it was written by an "
            "older version of stage 4, so rerun with --from-stage 4"
        )
    table = pq.read_table(path, columns=columns)
    data = {name: table.column(name).to_numpy(zero_copy_only=False) for name in columns}
    ids = np.asarray(data["swing_id"], np.int64)
    order = np.argsort(ids, kind="stable")
    bounds = np.flatnonzero(np.diff(ids[order])) + 1
    out: dict[int, SwingFrame] = {}
    for chunk in np.split(order, bounds):
        if not chunk.size:
            continue
        chunk = chunk[np.argsort(np.asarray(data["t_rel"])[chunk], kind="stable")]
        xy = np.empty((chunk.size, len(KEYPOINT_NAMES), 2), np.float64)
        for k, name in enumerate(KEYPOINT_NAMES):
            xy[:, k, 0] = data[f"{name}_x"][chunk]
            xy[:, k, 1] = data[f"{name}_y"][chunk]
        out[int(ids[chunk[0]])] = SwingFrame(
            swing_id=int(ids[chunk[0]]),
            t_rel=np.asarray(data["t_rel"], np.float64)[chunk],
            xy=xy,
            speeds={s: np.asarray(data[f"{s}_wrist_speed"], np.float64)[chunk] for s in ("l", "r")},
            forward_sign=forward_sign,
            unit_turn_ratio=unit_turn_ratio,
        )
    return out


def compute_metrics(swing: SwingFrame, racket: Side, stroke_type: str) -> dict[str, float]:
    return {name: m.compute(swing, racket, stroke_type) for name, m in REGISTRY.items()}


def metrics_schema() -> pa.Schema:
    fields = [
        ("swing_id", pa.int64()),
        ("contact_id", pa.int64()),
        ("t_contact", pa.float64()),
        ("stroke_type", pa.string()),
        ("two_handed", pa.bool_()),
    ]
    fields += [(name, pa.float64()) for name in metric_names()]
    fields += [("outlier_score", pa.float64()), ("is_outlier", pa.bool_())]
    return pa.schema(fields)


SUMMARY_SCHEMA = pa.schema(
    [
        ("stroke_type", pa.string()),
        ("metric", pa.string()),
        ("unit", pa.string()),
        ("count", pa.int64()),
        ("mean", pa.float64()),
        ("std", pa.float64()),
        ("median", pa.float64()),
        ("p10", pa.float64()),
        ("p90", pa.float64()),
    ]
)


def summarize(rows: Sequence[dict[str, Any]]) -> pa.Table:
    """Count, mean, std, median, p10 and p90 per stroke type and metric.

    ``std`` is the sample standard deviation (the spec's "consistency"); it is NaN for a
    single swing. Stroke types are listed in the spec's order, then any others, so the
    output is the same for the same input.
    """
    seen = [r["stroke_type"] for r in rows]
    types = [s for s in STROKE_TYPES if s in seen] + sorted(set(seen) - ALL_STROKES)
    records = []
    for stroke_type in types:
        subset = [r for r in rows if r["stroke_type"] == stroke_type]
        for name in metric_names():
            values = _finite(np.array([r[name] for r in subset], dtype=np.float64))
            if not values.size:
                continue
            records.append(
                {
                    "stroke_type": stroke_type,
                    "metric": name,
                    "unit": REGISTRY[name].unit,
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else NAN,
                    "median": float(np.median(values)),
                    "p10": float(np.percentile(values, 10)),
                    "p90": float(np.percentile(values, 90)),
                }
            )
    return pa.Table.from_pylist(records, schema=SUMMARY_SCHEMA)


# --- outliers ------------------------------------------------------------------------------


@dataclass
class OutlierReport:
    scores: dict[int, float] = field(default_factory=dict)
    flagged: set[int] = field(default_factory=set)
    methods: dict[str, str] = field(default_factory=dict)  # stroke type -> method used


def _distances(x: FloatArray, ridge: float, min_for_covariance: int) -> tuple[FloatArray, str]:
    """Mahalanobis distance of each row from the mean, with a deterministic fallback.

    With fewer than ``min_for_covariance`` swings a covariance matrix is not worth
    estimating, so each column is standardized and the plain Euclidean norm is used.
    Otherwise the covariance gets a small ridge (scaled by its trace) before inversion, so
    a collinear or degenerate set of metrics still produces finite distances.
    """
    centered = x - x.mean(axis=0)
    if x.shape[0] < min_for_covariance:
        scale = x.std(axis=0, ddof=1) if x.shape[0] > 1 else np.zeros(x.shape[1])
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        return np.asarray(np.linalg.norm(centered / scale, axis=1)), "standardized euclidean"
    cov = np.cov(x, rowvar=False)
    cov = np.atleast_2d(cov)
    trace = float(np.trace(cov))
    cov = cov + np.eye(cov.shape[0]) * ridge * max(trace / cov.shape[0], 1.0)
    inverse = np.linalg.pinv(cov)
    quadratic = np.einsum("ij,jk,ik->i", centered, inverse, centered)
    return np.sqrt(np.maximum(quadratic, 0.0)), "mahalanobis"


def find_outliers(rows: Sequence[dict[str, Any]], config: MetricsConfig) -> OutlierReport:
    """Score swings per stroke type on ``metrics.outlier_metrics`` and flag the extremes.

    Deterministic: no sampling, no random state. A swing whose outlier metrics are not all
    finite gets no score and is never flagged.
    """
    report = OutlierReport()
    unknown = [name for name in config.outlier_metrics if name not in REGISTRY]
    if unknown:
        raise UserError(
            f"metrics.outlier_metrics names unknown metric(s) {', '.join(unknown)}; "
            f"available: {', '.join(metric_names())}"
        )
    for stroke_type in sorted({r["stroke_type"] for r in rows}):
        subset = [r for r in rows if r["stroke_type"] == stroke_type]
        usable = [r for r in subset if all(np.isfinite(r[name]) for name in config.outlier_metrics)]
        if len(usable) < 2 or not config.outlier_metrics:
            report.methods[stroke_type] = "not enough swings"
            continue
        x = np.array(
            [[float(r[name]) for name in config.outlier_metrics] for r in usable],
            dtype=np.float64,
        )
        scores, method = _distances(x, config.outlier_ridge, config.min_swings_for_covariance)
        report.methods[stroke_type] = method
        # Strictly above the cutoff: with a flat score distribution (every swing the same
        # distance from the mean) the percentile equals every score, and ">=" would flag
        # the whole stroke type.
        cutoff = float(np.percentile(scores, config.outlier_percentile))
        for r, score in zip(usable, scores, strict=True):
            report.scores[int(r["swing_id"])] = float(score)
            if score > cutoff:
                report.flagged.add(int(r["swing_id"]))
    return report


# --- the stage -------------------------------------------------------------------------------


def measurable(info: dict[str, Any]) -> bool:
    """A swing is measured when it is a confirmed own hit that passed QC on the near side."""
    return bool(
        info.get("is_self_confirmed")
        and info.get("qc_pass")
        and (info.get("player_side") or "near") == "near"
    )


def run(ctx: StageContext) -> None:
    from tennis.stages.clean import racket_side, resolved_handedness

    session, config = ctx.session, ctx.config
    racket: Side = "r" if racket_side(resolved_handedness(session, config)) == "r" else "l"
    forward_sign = float(config.player.forward_sign)

    info_by_id = {
        int(r["swing_id"]): r for r in pq.read_table(session.path("swing_info.parquet")).to_pylist()
    }
    strokes = {
        int(r["swing_id"]): r for r in pq.read_table(session.path("strokes.parquet")).to_pylist()
    }
    swings = load_swings(session, forward_sign, config.metrics.unit_turn_ratio)

    rows: list[dict[str, Any]] = []
    skipped = 0
    for swing_id, info in sorted(info_by_id.items()):
        stroke = strokes.get(swing_id, {})
        stroke_type = stroke.get("stroke_type")
        if not measurable(info) or stroke_type is None:
            skipped += 1
            continue
        frame = swings.get(swing_id)
        if frame is None or not len(frame):
            ctx.log("swing has no frames", level=logging.WARNING, swing_id=swing_id)
            skipped += 1
            continue
        try:
            values = compute_metrics(frame, racket, str(stroke_type))
        except Exception as exc:  # one bad swing must not stop the stage
            ctx.log("swing failed", level=logging.WARNING, swing_id=swing_id, error=repr(exc))
            skipped += 1
            continue
        rows.append(
            {
                "swing_id": swing_id,
                "contact_id": int(info["contact_id"]),
                "t_contact": float(info["t_contact"]),
                "stroke_type": str(stroke_type),
                "two_handed": stroke.get("two_handed"),
                **values,
                "outlier_score": NAN,
                "is_outlier": False,
            }
        )

    outliers = find_outliers(rows, config.metrics)
    for row in rows:
        score = outliers.scores.get(int(row["swing_id"]), NAN)
        row["outlier_score"] = score
        row["is_outlier"] = int(row["swing_id"]) in outliers.flagged

    table = pa.Table.from_pylist(
        [{name: row.get(name) for name in metrics_schema().names} for row in rows],
        schema=metrics_schema(),
    )
    extra = {"racket_side": racket, "forward_sign": str(int(forward_sign))}
    for name, output in (
        ("metrics.parquet", table),
        ("metrics_summary.parquet", summarize(rows)),
    ):
        write_parquet(
            output,
            session.path(name),
            stage=ctx.stage,
            config_hash=ctx.config_hash,
            schema_version=SCHEMA_VERSION,
            extra=extra,
        )

    ctx.log(
        "metrics computed",
        swings=len(rows),
        skipped=skipped,
        metrics=len(REGISTRY),
        outliers=len(outliers.flagged),
        methods=outliers.methods,
        racket_side=racket,
    )


def read_metrics(session: Session) -> list[dict[str, Any]]:
    path = session.path("metrics.parquet")
    if not path.exists():
        raise UserError(f"session '{session.id}' has no metrics.parquet; run 'tennis process'")
    return [dict(r) for r in pq.read_table(path).to_pylist()]
