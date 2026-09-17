"""Stage 5 on hand-built poses: the rule cascade, both racket hands, and eval-classifier."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tennis.config import ClassifyConfig, Config, PathsConfig, PlayerConfig
from tennis.errors import UserError
from tennis.pose_backends.base import KEYPOINT_NAMES
from tennis.session import Session, create_or_reuse_session
from tennis.stages import StageContext, classify
from tennis.stages.classify import RuleClassifier, StrokeResult, SwingFeatures
from tennis.util.cleaning import KP
from tennis.util.log import get_logger

FPS = 30.0
FRAME_HEIGHT = 720
PRE, POST = 1.0, 0.5


def features(**overrides: object) -> SwingFeatures:
    """A plain right-handed baseline forehand; override one field per rule under test."""
    base: dict[str, object] = {
        "swing_id": 0,
        "contact_id": 0,
        "t_contact": 1.0,
        "handedness": "right",
        "player_side": "near",
        "racket_wrist_x": 0.9,
        "racket_wrist_y": 0.2,
        "other_wrist_x": -0.5,
        "other_wrist_y": 0.1,
        "nose_y": 1.5,
        "wrist_travel": 3.0,
        "wrist_gap": 1.4,
        "bbox_bottom_y": 0.9,
    }
    base.update(overrides)
    return SwingFeatures(**base)  # type: ignore[arg-type]


def classify_one(**overrides: object) -> StrokeResult:
    return RuleClassifier(ClassifyConfig()).classify(features(**overrides))


# --- the rule cascade ---------------------------------------------------------------------


def test_serve_beats_every_other_rule() -> None:
    # Above the nose, short travel, at the net and on the backhand side: still a serve.
    result = classify_one(
        racket_wrist_y=2.0, nose_y=1.5, wrist_travel=0.1, bbox_bottom_y=0.3, racket_wrist_x=-1.0
    )
    assert result.stroke_type == "serve"


def test_wrist_just_below_the_serve_threshold_is_not_a_serve() -> None:
    assert classify_one(racket_wrist_y=1.79, nose_y=1.5).stroke_type != "serve"
    assert classify_one(racket_wrist_y=1.81, nose_y=1.5).stroke_type == "serve"


def test_volley_needs_both_short_travel_and_the_net() -> None:
    assert classify_one(wrist_travel=0.3, bbox_bottom_y=0.4).stroke_type == "volley"
    assert classify_one(wrist_travel=0.3, bbox_bottom_y=0.9).stroke_type == "forehand"
    assert classify_one(wrist_travel=3.0, bbox_bottom_y=0.4).stroke_type == "forehand"


@pytest.mark.parametrize(
    ("handedness", "wrist_x", "expected"),
    [
        ("right", 0.9, "forehand"),
        ("right", -0.9, "backhand"),
        ("left", -0.9, "forehand"),
        ("left", 0.9, "backhand"),
    ],
)
def test_forehand_sign_follows_the_racket_hand(
    handedness: str, wrist_x: float, expected: str
) -> None:
    assert classify_one(handedness=handedness, racket_wrist_x=wrist_x).stroke_type == expected


def test_two_handed_uses_the_gap_between_the_wrists() -> None:
    assert classify_one(wrist_gap=0.2).two_handed is True
    assert classify_one(wrist_gap=1.4).two_handed is False
    assert classify_one(wrist_gap=float("nan")).two_handed is None


def test_missing_racket_wrist_is_not_classified() -> None:
    result = classify_one(racket_wrist_x=float("nan"))
    assert result.stroke_type is None and "missing" in result.rule


def test_unknown_classifier_name() -> None:
    with pytest.raises(UserError, match=r"unknown classify\.classifier"):
        classify.get_classifier(ClassifyConfig(classifier="gbm"))


def test_registry_accepts_a_new_classifier() -> None:
    class Always:
        version = "always-1"

        def classify(self, swing: SwingFeatures) -> StrokeResult:
            return StrokeResult("volley", False, "test")

    previous = classify.register_classifier("test-only", lambda cfg: Always())
    try:
        assert "test-only" in classify.available_classifiers()
        got = classify.get_classifier(ClassifyConfig(classifier="test-only"))
        assert got.classify(features()).stroke_type == "volley"
    finally:
        classify._REGISTRY.pop("test-only", None)
        assert previous is not None


# --- the stage ----------------------------------------------------------------------------


def _pose(t_rel: float, kind: str, handedness: str) -> np.ndarray:
    """(17, 2) normalized keypoints of one frame of a swing of ``kind``."""
    side = "r" if handedness == "right" else "l"
    other = "l" if side == "r" else "r"
    sign = 1.0 if side == "r" else -1.0
    kp = np.zeros((17, 2))
    for name, (x, y) in {
        "nose": (0.0, 1.5),
        "l_shoulder": (-0.3, 1.0), "r_shoulder": (0.3, 1.0),
        "l_elbow": (-0.4, 0.4), "r_elbow": (0.4, 0.4),
        "l_hip": (-0.2, 0.0), "r_hip": (0.2, 0.0),
        "l_knee": (-0.22, -0.8), "r_knee": (0.22, -0.8),
        "l_ankle": (-0.24, -1.6), "r_ankle": (0.24, -1.6),
        "l_eye": (-0.05, 1.55), "r_eye": (0.05, 1.55),
        "l_ear": (-0.1, 1.5), "r_ear": (0.1, 1.5),
    }.items():  # fmt: skip
        kp[KP[name]] = (x, y)
    kp[KP[f"{other}_wrist"]] = (-0.5 * sign, 0.1)
    if kind == "serve":
        # The racket arm swings up and is well above the nose at contact.
        kp[KP[f"{side}_wrist"]] = (0.2 * sign, 2.0 + 2.0 * t_rel)
    elif kind == "volley":
        # A short punch: 0.2 units of travel over the whole second before contact.
        kp[KP[f"{side}_wrist"]] = ((0.8 + 0.2 * t_rel) * sign, 0.5)
    elif kind == "forehand":
        kp[KP[f"{side}_wrist"]] = ((0.9 + 2.0 * t_rel) * sign, 0.3)
    else:  # backhand: the racket wrist crosses to the other side of the body
        kp[KP[f"{side}_wrist"]] = ((-0.9 + 2.0 * t_rel) * sign, 0.3)
    return kp


def _write_session(tmp_path: Path, data_root: Path, kinds: list[str], handedness: str) -> Session:
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"x")
    session = create_or_reuse_session(data_root, video, datetime(2026, 9, 20, tzinfo=UTC), "c")
    (session.dir / "metadata.json").write_text(
        '{"video_start_s": 0.0, "audio_start_s": 0.0, "fps": 30.0, '
        f'"video": {{"width": 1280, "height": {FRAME_HEIGHT}}}}}'
    )
    (session.dir / "players.json").write_text(
        f'{{"me": "A", "handedness": {{"hand": "{handedness}"}}}}'
    )

    swing_rows: dict[str, list[float]] = {}
    info_rows = []
    kp_rows: dict[str, list[float]] = {}
    frame = 0
    for swing_id, kind in enumerate(kinds):
        t_contact = 5.0 + 10.0 * swing_id
        offsets = np.arange(-PRE, POST + 1e-9, 1 / FPS)
        for t_rel in offsets:
            kp = _pose(float(t_rel), kind, handedness)
            swing_rows.setdefault("swing_id", []).append(swing_id)
            swing_rows.setdefault("contact_id", []).append(swing_id)
            swing_rows.setdefault("frame_idx", []).append(frame)
            swing_rows.setdefault("t_rel", []).append(float(t_rel))
            for k, name in enumerate(KEYPOINT_NAMES):
                swing_rows.setdefault(f"{name}_x", []).append(kp[k, 0])
                swing_rows.setdefault(f"{name}_y", []).append(kp[k, 1])
                # Pixels: y down, hips at 400 px, one torso length = 100 px.
                swing_rows.setdefault(f"{name}_px", []).append(640 + 100 * kp[k, 0])
                swing_rows.setdefault(f"{name}_py", []).append(400 - 100 * kp[k, 1])
            if abs(t_rel) < 1e-9:
                kp_rows.setdefault("frame_idx", []).append(frame)
                kp_rows.setdefault("slot", []).append("near")  # type: ignore[arg-type]
                # A volley is played at the net, so the player's feet are high in the frame.
                kp_rows.setdefault("bbox_y2", []).append(
                    0.4 * FRAME_HEIGHT if kind == "volley" else 0.9 * FRAME_HEIGHT
                )
            frame += 1
        info_rows.append(
            {
                "swing_id": swing_id,
                "contact_id": swing_id,
                "t_contact": t_contact,
                "contact_frame_idx": frame - len(offsets) + int(np.argmin(np.abs(offsets))),
                "qc_pass": True,
                "is_self_confirmed": True,
                "player_side": "near",
            }
        )
    pq.write_table(pa.table(swing_rows), session.path("swings.parquet"))
    pq.write_table(pa.Table.from_pylist(info_rows), session.path("swing_info.parquet"))
    pq.write_table(pa.table(kp_rows), session.path("keypoints.parquet"))
    return session


KINDS = ["serve", "forehand", "backhand", "volley"]


@pytest.mark.parametrize("handedness", ["right", "left"])
def test_stage_classifies_every_stroke_for_both_hands(
    tmp_path: Path, data_root: Path, handedness: str
) -> None:
    session = _write_session(tmp_path, data_root, KINDS, handedness)
    config = Config(paths=PathsConfig(data_root=data_root))
    classify.run(StageContext(session, config, "h", get_logger(), "classify"))
    rows = pq.read_table(session.path("strokes.parquet")).to_pylist()
    assert [r["stroke_type"] for r in rows] == KINDS
    assert {r["handedness"] for r in rows} == {handedness}
    assert all(r["classifier_version"] == "rule-1" for r in rows)
    volley = rows[KINDS.index("volley")]
    # 0.2 units per second of travel, measured over the 0.5 s before contact.
    assert volley["wrist_travel"] == pytest.approx(0.1, abs=0.02)
    assert volley["bbox_bottom_y"] == pytest.approx(0.4)


def test_unclassifiable_swings_keep_their_row(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root, KINDS, "right")
    info = pq.read_table(session.path("swing_info.parquet")).to_pylist()
    info[0]["qc_pass"] = False
    info[1]["is_self_confirmed"] = False
    info[2]["player_side"] = "far"
    pq.write_table(pa.Table.from_pylist(info), session.path("swing_info.parquet"))
    config = Config(paths=PathsConfig(data_root=data_root))
    classify.run(StageContext(session, config, "h", get_logger(), "classify"))
    rows = pq.read_table(session.path("strokes.parquet")).to_pylist()
    assert len(rows) == 4
    assert [r["stroke_type"] for r in rows] == [None, None, None, "volley"]
    assert [r["rule"] for r in rows[:3]] == [
        "failed QC",
        "not a confirmed own hit",
        "far side (not measured)",
    ]


def test_side_on_camera_is_rejected(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root, KINDS, "right")
    config = Config(
        paths=PathsConfig(data_root=data_root), player=PlayerConfig(camera_side="side_on")
    )
    with pytest.raises(UserError, match="behind_baseline"):
        classify.run(StageContext(session, config, "h", get_logger(), "classify"))


def test_rerun_keeps_the_file_unchanged(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root, KINDS, "right")
    config = Config(paths=PathsConfig(data_root=data_root))
    ctx = StageContext(session, config, "h", get_logger(), "classify")
    classify.run(ctx)
    first = session.path("strokes.parquet").stat().st_mtime_ns
    classify.run(ctx)
    assert session.path("strokes.parquet").stat().st_mtime_ns == first


# --- eval-classifier ----------------------------------------------------------------------


def test_eval_classifier_confusion_matrix(tmp_path: Path, data_root: Path) -> None:
    from tennis.evaluation import evaluate_classifier, format_classifier_eval
    from tennis.stages.contacts import LabelSpec, make_scorer

    session = _write_session(tmp_path, data_root, KINDS, "right")
    config = Config(paths=PathsConfig(data_root=data_root))
    classify.run(StageContext(session, config, "h", get_logger(), "classify"))

    # Contacts are at 5, 15, 25 and 35 s. Call the volley a backhand to make one mistake,
    # and add a label at 45 s that no swing matches.
    labels = [(5.0, "serve"), (15.0, "forehand"), (25.0, "backhand"), (35.0, "backhand"),
              (45.0, "forehand")]  # fmt: skip
    spec = LabelSpec(np.array([t for t, _ in labels]), tolerance_s=0.5)
    ev = evaluate_classifier(session, make_scorer(session, spec), labels)
    assert ev.matched == 4
    assert ev.unmatched_labels == 1 and ev.unmatched_swings == 0
    assert ev.matrix.accuracy == pytest.approx(3 / 4)
    i = ev.matrix.labels.index("backhand")
    assert ev.matrix.recall(i) == pytest.approx(0.5)  # one of two backhand labels was right
    assert ev.matrix.precision(i) == pytest.approx(1.0)
    assert not ev.meets_target
    text = format_classifier_eval(ev)
    assert "accuracy 0.750" in text and "backhand" in text
    assert "| accuracy" in format_classifier_eval(ev, markdown=True).replace("- ", "| ")


def test_eval_classifier_handles_a_label_clock_offset(tmp_path: Path, data_root: Path) -> None:
    from tennis.evaluation import evaluate_classifier
    from tennis.stages.contacts import LabelSpec, make_scorer

    session = _write_session(tmp_path, data_root, KINDS, "right")
    config = Config(paths=PathsConfig(data_root=data_root))
    classify.run(StageContext(session, config, "h", get_logger(), "classify"))
    # Wingfield-style: whole seconds, one second early, so a label t means [t + 1, t + 2].
    labels = [(4.0, "serve"), (14.0, "forehand"), (24.0, "backhand"), (34.0, "volley")]
    spec = LabelSpec(
        np.array([t for t, _ in labels]), tolerance_s=0.04, resolution_s=1.0, offset_s=1.0
    )
    ev = evaluate_classifier(session, make_scorer(session, spec), labels)
    assert ev.matched == 4 and ev.matrix.accuracy == 1.0
    assert ev.meets_target


def test_read_stroke_labels(tmp_path: Path) -> None:
    from tennis.validation import read_stroke_labels

    path = tmp_path / "strokes.csv"
    path.write_text(
        "# from an export\n"
        "t,player,stroke,rally,shot\n"
        "20,self,forehand,1,2\n"
        "10,other,serve,1,1\n"
        "30,self,VOLLEY,1,3\n"
        "40,self,unknown,1,4\n"
    )
    assert read_stroke_labels(path) == [(20.0, "forehand"), (30.0, "volley")]

    plain = tmp_path / "plain.csv"
    plain.write_text("t,stroke\n1:05.5,serve\n2,backhand\n")
    assert read_stroke_labels(plain) == [(2.0, "backhand"), (65.5, "serve")]

    bad = tmp_path / "bad.csv"
    bad.write_text("t,note\n1,x\n")
    with pytest.raises(UserError, match="stroke column"):
        read_stroke_labels(bad)
