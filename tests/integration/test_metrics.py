"""Stage 6: one test per metric on hand-built geometry, plus outliers and determinism."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tennis.config import Config, MetricsConfig, PathsConfig, PlayerConfig
from tennis.errors import UserError
from tennis.pose_backends.base import KEYPOINT_NAMES
from tennis.session import Session, create_or_reuse_session
from tennis.stages import StageContext, metrics
from tennis.stages.metrics import REGISTRY, Side, SwingFrame
from tennis.util.cleaning import KP
from tennis.util.log import get_logger

FPS = 30.0
PRE, POST = 1.0, 0.5

# A plain standing pose in normalized units: hips at the origin, one torso length tall.
STANDING: dict[str, tuple[float, float]] = {
    "nose": (0.0, 1.5),
    "l_eye": (-0.05, 1.55), "r_eye": (0.05, 1.55),
    "l_ear": (-0.1, 1.5), "r_ear": (0.1, 1.5),
    "l_shoulder": (-0.3, 1.0), "r_shoulder": (0.3, 1.0),
    "l_elbow": (-0.4, 0.5), "r_elbow": (0.4, 0.5),
    "l_wrist": (-0.5, 0.0), "r_wrist": (0.5, 0.0),
    "l_hip": (-0.2, 0.0), "r_hip": (0.2, 0.0),
    "l_knee": (-0.2, -0.8), "r_knee": (0.2, -0.8),
    "l_ankle": (-0.2, -1.6), "r_ankle": (0.2, -1.6),
}  # fmt: skip


def pose_array(overrides: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
    xy = np.zeros((len(KEYPOINT_NAMES), 2))
    for name, point in {**STANDING, **(overrides or {})}.items():
        xy[KP[name]] = point
    return xy


def make_swing(
    frames: list[tuple[float, dict[str, tuple[float, float]]]],
    *,
    speeds: dict[str, list[float]] | None = None,
    forward_sign: float = 1.0,
    unit_turn_ratio: float = 0.85,
) -> SwingFrame:
    """A swing from explicit (t_rel, keypoint overrides) frames."""
    t_rel = np.array([t for t, _ in frames], dtype=np.float64)
    xy = (
        np.stack([pose_array(over) for _, over in frames])
        if frames
        else np.zeros((0, len(KEYPOINT_NAMES), 2))
    )
    default = list(np.zeros(len(frames)))
    return SwingFrame(
        swing_id=0,
        t_rel=t_rel,
        xy=xy,
        speeds={
            side: np.array((speeds or {}).get(side, default), dtype=np.float64)
            for side in ("l", "r")
        },
        forward_sign=forward_sign,
        unit_turn_ratio=unit_turn_ratio,
    )


def still(times: list[float], **overrides: tuple[float, float]) -> SwingFrame:
    return make_swing([(t, dict(overrides)) for t in times])


def value(name: str, swing: SwingFrame, racket: Side = "r") -> float:
    return REGISTRY[name].fn(swing, racket)


# --- one test per metric --------------------------------------------------------------------


def test_contact_height_is_measured_from_the_ground() -> None:
    # The racket wrist sits 1.0 above the hips; the ankles are 1.6 below them.
    swing = still([-0.1, 0.0, 0.1], r_wrist=(0.6, 1.0))
    assert value("contact_height", swing) == pytest.approx(2.6)


def test_contact_forward_is_measured_from_the_hips_and_follows_forward_sign() -> None:
    frames = [(t, {"r_wrist": (0.6, 0.8)}) for t in (-0.1, 0.0, 0.1)]
    assert value("contact_forward", make_swing(frames)) == pytest.approx(0.8)
    flipped = make_swing(frames, forward_sign=-1.0)
    assert value("contact_forward", flipped) == pytest.approx(-0.8)


def test_contact_lateral_is_positive_on_the_racket_side_for_both_hands() -> None:
    right = still([0.0], r_wrist=(0.9, 0.2))
    assert value("contact_lateral", right, "r") == pytest.approx(0.9)
    left = still([0.0], l_wrist=(-0.9, 0.2))
    assert value("contact_lateral", left, "l") == pytest.approx(0.9)


def test_contact_reach_is_the_shoulder_to_wrist_distance() -> None:
    swing = still([0.0], r_shoulder=(0.3, 1.0), r_wrist=(0.3, 0.2))
    assert value("contact_reach", swing) == pytest.approx(0.8)


def test_elbow_angle_contact_on_a_right_angle_and_a_straight_arm() -> None:
    bent = still([0.0], r_shoulder=(0.0, 1.0), r_elbow=(0.0, 0.0), r_wrist=(1.0, 0.0))
    assert value("elbow_angle_contact", bent) == pytest.approx(90.0)
    straight = still([0.0], r_shoulder=(0.0, 1.0), r_elbow=(0.0, 0.5), r_wrist=(0.0, 0.0))
    assert value("elbow_angle_contact", straight) == pytest.approx(180.0)


def test_elbow_angle_min_looks_only_at_the_half_second_before_contact() -> None:
    swing = make_swing(
        [
            (-0.8, {"r_shoulder": (0.0, 1.0), "r_elbow": (0.0, 0.0), "r_wrist": (0.0, 0.1)}),
            (-0.2, {"r_shoulder": (0.0, 1.0), "r_elbow": (0.0, 0.0), "r_wrist": (1.0, 0.0)}),
            (0.0, {"r_shoulder": (0.0, 1.0), "r_elbow": (0.0, 0.5), "r_wrist": (0.0, 0.0)}),
        ]
    )
    # The near-zero angle at -0.8 s is outside the window; the 90 degrees at -0.2 s wins.
    assert value("elbow_angle_min", swing) == pytest.approx(90.0)


def test_knee_flex_min_takes_the_deepest_frame_and_averages_both_legs() -> None:
    deep = {
        "l_hip": (-0.2, 0.0), "l_knee": (-0.2, -0.8), "l_ankle": (-1.0, -0.8),
        "r_hip": (0.2, 0.0), "r_knee": (0.2, -0.8), "r_ankle": (1.0, -0.8),
    }  # fmt: skip
    swing = make_swing([(-0.5, {}), (-0.2, deep), (0.0, {})])
    assert value("knee_flex_min", swing) == pytest.approx(90.0)
    assert value("knee_flex_contact", swing) == pytest.approx(180.0)


def test_peak_wrist_speed_and_its_offset_come_from_the_stored_speeds() -> None:
    swing = make_swing(
        [(t, {}) for t in (-0.2, -0.1, 0.0, 0.1)],
        speeds={"r": [1.0, 9.0, 4.0, 2.0], "l": [0.0, 0.0, 0.0, 0.0]},
    )
    assert value("peak_wrist_speed", swing) == pytest.approx(9.0)
    assert value("peak_speed_offset", swing) == pytest.approx(-0.1)


def test_shoulder_turn_proxy_min_is_relative_to_one_second_before_contact() -> None:
    narrow = {"l_shoulder": (-0.15, 1.0), "r_shoulder": (0.15, 1.0)}  # half the width
    swing = make_swing([(-1.0, {}), (-0.5, narrow), (0.0, {})])
    assert value("shoulder_turn_proxy_min", swing) == pytest.approx(0.5)


def test_shoulder_turn_proxy_is_nan_when_the_window_starts_too_late() -> None:
    swing = make_swing([(-0.6, {}), (0.0, {})])
    assert math.isnan(value("shoulder_turn_proxy_min", swing))
    assert math.isnan(value("unit_turn_lead_time", swing))


def test_unit_turn_lead_time_is_the_first_frame_below_the_ratio() -> None:
    narrow = {"l_shoulder": (-0.2, 1.0), "r_shoulder": (0.2, 1.0)}  # 2/3 of the width
    swing = make_swing(
        [(-1.0, {}), (-0.8, {}), (-0.6, narrow), (-0.4, narrow), (0.0, narrow)],
        unit_turn_ratio=0.85,
    )
    assert value("unit_turn_lead_time", swing) == pytest.approx(0.6)


def test_follow_through_height_uses_the_last_frame() -> None:
    swing = make_swing([(-0.5, {}), (0.0, {}), (0.5, {"r_wrist": (-0.2, 1.8)})])
    assert value("follow_through_height", swing) == pytest.approx(1.8)


def test_torso_lean_contact_is_zero_upright_and_signed_towards_the_racket_side() -> None:
    upright = still([0.0])
    assert value("torso_lean_contact", upright) == pytest.approx(0.0)
    leaning = {"l_shoulder": (0.7, 1.0), "r_shoulder": (1.3, 1.0)}  # shoulders 1.0 to the right
    swing = still([0.0], **leaning)  # type: ignore[arg-type]
    assert value("torso_lean_contact", swing, "r") == pytest.approx(45.0)
    assert value("torso_lean_contact", swing, "l") == pytest.approx(-45.0)


def test_every_registered_metric_is_covered_by_a_test() -> None:
    source = Path(__file__).read_text()
    missing = [name for name in REGISTRY if f'"{name}"' not in source]
    assert not missing, f"metrics without a test: {missing}"


def test_metrics_that_do_not_apply_to_a_stroke_type_are_nan() -> None:
    narrow = {"l_shoulder": (-0.2, 1.0), "r_shoulder": (0.2, 1.0)}
    swing = make_swing([(-1.0, {}), (-0.4, narrow), (0.0, narrow), (0.5, narrow)])
    assert math.isnan(REGISTRY["unit_turn_lead_time"].compute(swing, "r", "serve"))
    assert not math.isnan(REGISTRY["unit_turn_lead_time"].compute(swing, "r", "forehand"))
    assert math.isnan(REGISTRY["follow_through_height"].compute(swing, "r", "volley"))


def test_metrics_on_an_empty_swing_are_all_nan() -> None:
    swing = make_swing([])
    for name in REGISTRY:
        assert math.isnan(REGISTRY[name].compute(swing, "r", "forehand")), name


def test_registering_a_duplicate_metric_fails() -> None:
    with pytest.raises(ValueError, match="already registered"):
        metrics.metric("contact_height", applies_to={"serve"})(lambda s, r: 0.0)
    with pytest.raises(ValueError, match="unknown stroke type"):
        metrics.metric("new_one", applies_to={"smash"})(lambda s, r: 0.0)


# --- SwingFrame ------------------------------------------------------------------------------


def test_swing_frame_at_between_and_contact() -> None:
    swing = make_swing([(t, {}) for t in (-1.0, -0.5, 0.0, 0.25, 0.5)])
    assert swing.at(-0.5) is not None and swing.at(-0.5).t_rel == pytest.approx(-0.5)  # type: ignore[union-attr]
    assert swing.at(-0.4, tolerance_s=0.05) is None  # nothing that close
    assert swing.contact is not None and swing.contact.t_rel == 0.0
    assert len(swing.between(-0.5, 0.25)) == 3
    assert swing.start == -1.0 and swing.end == 0.5


# --- outliers --------------------------------------------------------------------------------


def _rows(values: list[float], name: str = "contact_height") -> list[dict[str, object]]:
    return [
        {"swing_id": i, "stroke_type": "forehand", name: v, "contact_forward": 0.0}
        for i, v in enumerate(values)
    ]


def test_outliers_flag_the_extreme_swings_deterministically() -> None:
    config = MetricsConfig(outlier_metrics=["contact_height"], outlier_percentile=90.0)
    rows = _rows([1.0] * 19 + [9.0])
    first = metrics.find_outliers(rows, config)
    assert first.flagged == {19}
    assert metrics.find_outliers(rows, config).flagged == first.flagged
    assert first.methods["forehand"] == "mahalanobis"


def test_few_swings_fall_back_to_standardized_euclidean() -> None:
    config = MetricsConfig(outlier_metrics=["contact_height"], outlier_percentile=90.0)
    report = metrics.find_outliers(_rows([1.0, 1.0, 1.0, 5.0]), config)
    assert report.methods["forehand"] == "standardized euclidean"
    assert report.flagged == {3}


def test_swings_with_a_missing_outlier_metric_are_never_flagged() -> None:
    config = MetricsConfig(outlier_metrics=["contact_height"], outlier_percentile=90.0)
    rows = _rows([1.0] * 19 + [float("nan")])
    report = metrics.find_outliers(rows, config)
    assert 19 not in report.flagged and 19 not in report.scores


def test_collinear_metrics_still_give_finite_scores() -> None:
    config = MetricsConfig(outlier_metrics=["contact_height", "contact_forward"])
    rows = [
        {"swing_id": i, "stroke_type": "forehand", "contact_height": float(i),
         "contact_forward": float(i)}  # perfectly correlated
        for i in range(12)
    ]  # fmt: skip
    report = metrics.find_outliers(rows, config)
    assert all(np.isfinite(v) for v in report.scores.values())


def test_unknown_outlier_metric_is_a_user_error() -> None:
    config = MetricsConfig(outlier_metrics=["nope"])
    with pytest.raises(UserError, match="unknown metric"):
        metrics.find_outliers(_rows([1.0, 2.0]), config)


# --- the stage --------------------------------------------------------------------------------


def _write_session(
    tmp_path: Path, data_root: Path, strokes: list[str | None], n_each: int = 1
) -> Session:
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"x")
    session = create_or_reuse_session(data_root, video, datetime(2026, 9, 20, tzinfo=UTC), "m")
    (session.dir / "metadata.json").write_text('{"video_start_s": 0.0, "fps": 30.0}')
    (session.dir / "players.json").write_text('{"handedness": {"hand": "right"}}')

    swing_rows: dict[str, list[object]] = {}
    info_rows: list[dict[str, object]] = []
    stroke_rows: list[dict[str, object]] = []
    swing_id = 0
    for stroke in strokes:
        for n in range(n_each):
            t_rel = np.arange(-PRE, POST + 1e-9, 1 / FPS)
            for i, t in enumerate(t_rel):
                # The racket wrist sweeps across the body and rises a little with n, so
                # the swings of one stroke type differ and one of them stands out.
                wrist = (0.9 * float(t) + 0.3, 0.2 + 0.1 * n * (n == n_each - 1))
                xy = pose_array({"r_wrist": wrist})
                swing_rows.setdefault("swing_id", []).append(swing_id)
                swing_rows.setdefault("t_rel", []).append(float(t))
                swing_rows.setdefault("l_wrist_speed", []).append(0.5)
                swing_rows.setdefault("r_wrist_speed", []).append(float(i))
                for k, name in enumerate(KEYPOINT_NAMES):
                    swing_rows.setdefault(f"{name}_x", []).append(float(xy[k, 0]))
                    swing_rows.setdefault(f"{name}_y", []).append(float(xy[k, 1]))
            info_rows.append(
                {
                    "swing_id": swing_id,
                    "contact_id": swing_id,
                    "t_contact": 5.0 + swing_id,
                    "qc_pass": True,
                    "is_self_confirmed": True,
                    "player_side": "near",
                }
            )
            stroke_rows.append({"swing_id": swing_id, "stroke_type": stroke, "two_handed": False})
            swing_id += 1
    pq.write_table(pa.table(swing_rows), session.path("swings.parquet"))
    pq.write_table(pa.Table.from_pylist(info_rows), session.path("swing_info.parquet"))
    pq.write_table(pa.Table.from_pylist(stroke_rows), session.path("strokes.parquet"))
    return session


def _run(session: Session, data_root: Path, **overrides: object) -> None:
    config = Config(paths=PathsConfig(data_root=data_root), **overrides)  # type: ignore[arg-type]
    metrics.run(StageContext(session, config, "h", get_logger(), "metrics"))


def test_stage_writes_a_row_per_measured_swing(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root, ["forehand", "backhand", None], n_each=2)
    _run(session, data_root)
    rows = pq.read_table(session.path("metrics.parquet")).to_pylist()
    assert len(rows) == 4  # the two unclassified swings are left out
    assert {r["stroke_type"] for r in rows} == {"forehand", "backhand"}
    for name in metrics.metric_names():
        assert name in rows[0]
    # unit_turn_lead_time applies to groundstrokes only and the window starts at -1.0 s.
    assert all(np.isfinite(r["contact_height"]) for r in rows)


def test_stage_skips_swings_that_are_not_measurable(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root, ["forehand"] * 3)
    info = pq.read_table(session.path("swing_info.parquet")).to_pylist()
    info[0]["qc_pass"] = False
    info[1]["player_side"] = "far"
    pq.write_table(pa.Table.from_pylist(info), session.path("swing_info.parquet"))
    _run(session, data_root)
    rows = pq.read_table(session.path("metrics.parquet")).to_pylist()
    assert [r["swing_id"] for r in rows] == [2]


def test_summary_has_a_row_per_stroke_type_and_metric(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root, ["forehand"], n_each=3)
    _run(session, data_root)
    summary = pq.read_table(session.path("metrics_summary.parquet")).to_pylist()
    by_metric = {r["metric"]: r for r in summary}
    assert by_metric["contact_height"]["count"] == 3
    assert by_metric["contact_height"]["std"] >= 0
    assert {r["stroke_type"] for r in summary} == {"forehand"}
    # The synthetic player never turns their shoulders, so that metric is NaN throughout
    # and gets no summary row at all: the summary only lists metrics with real values.
    assert "unit_turn_lead_time" not in by_metric
    assert set(by_metric) < set(metrics.metric_names())


def test_rerunning_produces_identical_files(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root, ["forehand", "backhand"], n_each=3)
    _run(session, data_root)
    before = {
        name: session.path(name).read_bytes()
        for name in ("metrics.parquet", "metrics_summary.parquet")
    }
    mtimes = {name: session.path(name).stat().st_mtime_ns for name in before}
    _run(session, data_root)
    for name, blob in before.items():
        # Byte-identical, metadata included: the config hash and pipeline version do not
        # change between the two runs either. (Arrow's Table.equals says NaN != NaN, so it
        # cannot be used here.)
        assert session.path(name).read_bytes() == blob
        # ... and an identical rewrite keeps its mtime, so stage 8 and 9 do not rerun.
        assert session.path(name).stat().st_mtime_ns == mtimes[name]


def test_forward_sign_flips_contact_forward(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root, ["forehand"])
    _run(session, data_root)
    plus = pq.read_table(session.path("metrics.parquet")).to_pylist()[0]["contact_forward"]
    _run(session, data_root, player=PlayerConfig(forward_sign=-1))
    minus = pq.read_table(session.path("metrics.parquet")).to_pylist()[0]["contact_forward"]
    assert minus == pytest.approx(-plus)


def test_inspect_prints_metrics_and_reports_a_missing_clip(tmp_path: Path, data_root: Path) -> None:
    from tennis.review import format_swing_inspection

    session = _write_session(tmp_path, data_root, ["forehand"], n_each=3)
    _run(session, data_root)
    config = Config(paths=PathsConfig(data_root=data_root))
    text, clip = format_swing_inspection(session, config, 1)
    assert "swing 1" in text and "forehand" in text and "contact_height" in text
    assert clip is None
    with pytest.raises(UserError, match="no measured swing"):
        format_swing_inspection(session, config, 99)
