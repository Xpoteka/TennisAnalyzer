"""Stage 4 on hand-built keypoints: cleaning, normalization, QC and confirmation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from tennis.config import Config, PathsConfig
from tennis.pose_backends.base import KEYPOINT_NAMES
from tennis.session import Session, create_or_reuse_session
from tennis.stages import StageContext, clean
from tennis.util.cleaning import KP
from tennis.util.frames import merge_windows
from tennis.util.log import get_logger

FPS = 30.0
DURATION = 10.0
TORSO = 100.0  # pixels
HIP = (500.0, 400.0)
# t_contact -> scenario
FAST, BOUNCE, SLOW, MISSING = 2.0, 2.3, 5.0, 8.0


def _skeleton(t: float) -> np.ndarray:
    """(17, 3) pixels; a standing player, right-handed, facing away from the camera."""
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.9
    hx, hy = HIP
    offsets = {
        "nose": (0, -150), "l_eye": (-5, -155), "r_eye": (5, -155),
        "l_ear": (-10, -150), "r_ear": (10, -150),
        "l_shoulder": (-30, -TORSO), "r_shoulder": (30, -TORSO),
        "l_elbow": (-40, -60), "r_elbow": (40, -60),
        "l_wrist": (-45, -20), "r_wrist": (45, -20),
        "l_hip": (-20, 0), "r_hip": (20, 0),
        "l_knee": (-22, 80), "r_knee": (22, 80),
        "l_ankle": (-24, 160), "r_ankle": (24, 160),
    }  # fmt: skip
    for name, (dx, dy) in offsets.items():
        kp[KP[name], :2] = (hx + dx, hy + dy)
    wrist = KP["r_wrist"]
    # A fast forehand: 150 px sweep, fastest (30 torso lengths/s) exactly at FAST.
    kp[wrist, 0] += 150 * np.tanh((t - FAST) / 0.05)
    # A slow arm movement around SLOW: at most ~0.6 torso lengths/s.
    kp[wrist, 1] += 20 * np.sin(2 * np.pi * 0.5 * (t - SLOW)) if abs(t - SLOW) < 1.5 else 0
    if abs(t - MISSING) < 0.3:
        kp[wrist, 2] = 0.05  # the wrist is not visible around this contact
    return kp


def _write_session(tmp_path: Path, data_root: Path) -> Session:
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"x")
    session = create_or_reuse_session(data_root, video, datetime(2026, 9, 20, tzinfo=UTC), "c")
    pts = np.arange(int(DURATION * FPS)) / FPS
    pq.write_table(
        pa.table({"frame_idx": np.arange(pts.size), "pts": pts}),
        session.path("frame_times.parquet"),
    )
    times = [FAST, BOUNCE, SLOW, MISSING]
    pq.write_table(
        pa.table(
            {
                "contact_id": np.arange(4),
                "t_audio": np.array(times) + 0.004,  # onsets fall between frames
                "is_self_audio": [True] * 4,
                "peak_db": [-20.0, -30.0, -40.0, -25.0],
            }
        ),
        session.path("contacts.parquet"),
    )
    windows = merge_windows(times, 1.0, 0.5, 0.0, pts[-1])
    rows: dict[str, list[float]] = {c: [] for c in ("frame_idx", "t_video", "window_id",
                                                    "detected", "track_reset")}  # fmt: skip
    kps = []
    for w in windows:
        for i in np.flatnonzero((pts >= w.start - 1e-9) & (pts <= w.end + 1e-9)):
            kp = _skeleton(pts[i])
            detected = True
            if abs(pts[i] - (FAST - 0.5)) < 1e-6:  # one frame without a detection
                detected = False
                kp[:, :2] = np.nan
                kp[:, 2] = 0
            if abs(pts[i] - (FAST - 0.2)) < 1e-6:  # elbows swapped for one frame
                kp[[KP["l_elbow"], KP["r_elbow"]]] = kp[[KP["r_elbow"], KP["l_elbow"]]]
            rows["frame_idx"].append(int(i))
            rows["t_video"].append(float(pts[i]))
            rows["window_id"].append(w.id)
            rows["detected"].append(detected)
            rows["track_reset"].append(False)
            kps.append(kp)
    arr = np.array(kps)
    columns: dict[str, object] = {
        "frame_idx": pa.array(rows["frame_idx"], pa.int64()),
        "t_video": pa.array(rows["t_video"], pa.float64()),
        "window_id": pa.array(rows["window_id"], pa.int32()),
        "detected": pa.array(rows["detected"], pa.bool_()),
        "track_reset": pa.array(rows["track_reset"], pa.bool_()),
    }
    for k, name in enumerate(KEYPOINT_NAMES):
        for c, suffix in enumerate(("x", "y", "conf")):
            columns[f"{name}_{suffix}"] = pa.array(arr[:, k, c].astype(np.float32))
    pq.write_table(pa.table(columns), session.path("keypoints.parquet"))
    return session


@pytest.fixture
def cleaned(tmp_path: Path, data_root: Path) -> tuple[pa.Table, dict[int, dict[str, object]]]:
    session = _write_session(tmp_path, data_root)
    config = Config(paths=PathsConfig(data_root=data_root))
    clean.run(StageContext(session, config, "h", get_logger(), "clean"))
    info = pq.read_table(session.path("swing_info.parquet")).to_pylist()
    return pq.read_table(session.path("swings.parquet")), {r["contact_id"]: r for r in info}


def test_confirmation(cleaned: tuple[pa.Table, dict[int, dict[str, object]]]) -> None:
    _, info = cleaned
    fast, bounce, slow, missing = (info[i] for i in range(4))
    assert fast["is_self_confirmed"] is True
    assert fast["wrist_peak_offset_s"] == pytest.approx(0.0, abs=0.04)
    # Smoothing flattens this very fast (0.05 s) sweep somewhat; see the light-smoothing test.
    assert 8 < fast["wrist_peak_speed"] < 30
    # The bounce sees the same wrist peak, 0.3 s before it: too far, and not the closest.
    assert bounce["is_self_confirmed"] is False
    assert slow["is_self_confirmed"] is False
    assert slow["wrist_peak_speed"] < 3.0
    assert missing["is_self_confirmed"] in (False, None)


def test_light_smoothing_keeps_peak_speed(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root)
    config = Config.model_validate(
        {
            "paths": {"data_root": str(data_root)},
            "cleaning": {"one_euro": {"min_cutoff": 8.0, "beta": 1.0}},
        }
    )
    clean.run(StageContext(session, config, "h", get_logger(), "clean"))
    info = pq.read_table(session.path("swing_info.parquet")).to_pylist()
    assert info[0]["wrist_peak_speed"] == pytest.approx(30, rel=0.2)


def test_savgol_smoother(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root)
    config = Config.model_validate(
        {"paths": {"data_root": str(data_root)}, "cleaning": {"smoother": "savgol"}}
    )
    clean.run(StageContext(session, config, "h", get_logger(), "clean"))
    info = pq.read_table(session.path("swing_info.parquet")).to_pylist()
    assert info[0]["is_self_confirmed"] is True


def test_qc(cleaned: tuple[pa.Table, dict[int, dict[str, object]]]) -> None:
    _, info = cleaned
    fast, slow, missing = info[0], info[2], info[3]
    assert fast["qc_pass"] and slow["qc_pass"]
    assert fast["swap_count"] >= 1
    assert fast["expected_frames"] == 45 and fast["n_frames"] == 45
    assert fast["valid_frame_ratio"] == pytest.approx(44 / 45)
    assert fast["contact_frame_idx"] == 60
    assert not missing["qc_pass"]
    assert "racket arm missing at contact" in str(missing["qc_reason"])


def test_normalization(cleaned: tuple[pa.Table, dict[int, dict[str, object]]]) -> None:
    table, info = cleaned
    assert info[2]["scale_px"] == pytest.approx(TORSO, rel=1e-3)
    assert (info[2]["origin_px_x"], info[2]["origin_px_y"]) == pytest.approx(HIP, abs=0.5)
    rows = table.filter(pc.equal(table["contact_id"], 2)).to_pydict()
    c = int(np.argmin(np.abs(rows["t_rel"])))
    assert rows["t_rel"][c] == pytest.approx(-0.004, abs=1e-6)
    hip_x = (rows["l_hip_x"][c] + rows["r_hip_x"][c]) / 2
    hip_y = (rows["l_hip_y"][c] + rows["r_hip_y"][c]) / 2
    assert (hip_x, hip_y) == pytest.approx((0.0, 0.0), abs=0.01)
    assert rows["l_shoulder_y"][c] == pytest.approx(1.0, abs=0.01)  # y points up
    assert rows["l_ankle_y"][c] == pytest.approx(-1.6, abs=0.01)
    assert rows["r_shoulder_x"][c] == pytest.approx(0.3, abs=0.01)
    assert rows["r_shoulder_px"][c] == pytest.approx(HIP[0] + 30, abs=0.5)


def test_swings_table(cleaned: tuple[pa.Table, dict[int, dict[str, object]]]) -> None:
    table, info = cleaned
    assert table.num_rows == sum(int(r["n_frames"]) for r in info.values())
    for column in ("swing_id", "t_rel", "valid", "r_wrist_vx", "r_wrist_speed", "nose_conf",
                   "valid_frame_ratio", "swap_count", "track_reset_count", "qc_pass"):  # fmt: skip
        assert column in table.column_names
    fast = table.filter(pc.equal(table["contact_id"], 0)).to_pydict()
    # The swapped elbows were put back.
    frame = fast["frame_idx"].index(54)  # FAST - 0.2 s
    assert fast["r_elbow_px"][frame] > fast["l_elbow_px"][frame]
    # The undetected frame is interpolated (gap of 1) but not counted as valid.
    missing = fast["frame_idx"].index(45)
    assert not fast["valid"][missing]
    assert np.isfinite(fast["r_wrist_px"][missing])


def test_empty_keypoints(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root)
    table = pq.read_table(session.path("keypoints.parquet"))
    pq.write_table(table.slice(0, 0), session.path("keypoints.parquet"))
    config = Config(paths=PathsConfig(data_root=data_root))
    clean.run(StageContext(session, config, "h", get_logger(), "clean"))
    info = pq.read_table(session.path("swing_info.parquet")).to_pylist()
    assert len(info) == 4
    assert all(r["qc_reason"] == "no pose frames" for r in info)
    assert pq.read_table(session.path("swings.parquet")).num_rows == 0


def test_swing_plots_and_evaluation(tmp_path: Path, data_root: Path) -> None:
    from tennis.evaluation import evaluate_contacts, format_contact_eval
    from tennis.review import render_swing_plots
    from tennis.stages.contacts import LabelSpec, make_scorer

    session = _write_session(tmp_path, data_root)
    (session.dir / "metadata.json").write_text('{"video_start_s": 0.0, "audio_start_s": 0.0}')
    config = Config(paths=PathsConfig(data_root=data_root))
    clean.run(StageContext(session, config, "h", get_logger(), "clean"))

    out = render_swing_plots(session, config, count=3)
    html = out.read_text()
    assert out.name == "swing_trajectories.html" and "Swing trajectories" in html

    # The only real own hit is FAST; label it and score the stored flags.
    scorer = make_scorer(session, LabelSpec(np.array([FAST]), 0.04), [(0.0, 9.9)])
    ev = evaluate_contacts(session, scorer, (6.0, 0.2, "either"), "right",
                           speeds=(0.0, 6.0), windows=(0.2,))  # fmt: skip
    stored = {r.name: r.result for r in ev.fixed}
    assert stored["confirmed (stored)"].precision == 1.0
    assert stored["confirmed (stored)"].recall == 1.0
    assert stored["all onsets"].precision == pytest.approx(1 / 4)
    assert ev.sweep[0].result.f1 == 1.0
    assert "confirmed (stored)" in format_contact_eval(ev)
