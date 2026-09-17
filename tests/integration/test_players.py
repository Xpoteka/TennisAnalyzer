"""Stage 4 with two tracked players who change ends: identity, hit attribution, handedness."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tennis.config import Config
from tennis.pose_backends.base import KEYPOINT_NAMES
from tennis.session import Session, create_or_reuse_session
from tennis.stages import StageContext, clean
from tennis.util import appearance
from tennis.util.cleaning import KP
from tennis.util.frames import merge_windows
from tennis.util.io import read_json
from tennis.util.log import get_logger

FPS = 30.0
DURATION = 20.0
# (time, hitter, hitter's slot). A is left-handed and louder (the clip-on mic is on A).
HITS = [
    (2.0, "A", "near"), (3.0, "B", "far"), (4.0, "A", "near"), (5.0, "B", "far"),
    (6.0, "A", "near"), (7.0, "B", "far"), (8.0, "A", "near"), (9.0, "B", "far"),
    (10.0, "A", "near"),
    # They change ends here.
    (13.0, "B", "near"), (14.0, "A", "far"), (15.0, "B", "near"), (16.0, "A", "far"),
    (17.0, "B", "near"), (18.0, "A", "far"),
]  # fmt: skip
END_CHANGE = 11.5
LOOKS = {"A": (0, 140, 255), "B": (200, 60, 20)}  # orange, blue (BGR)
HAND = {"A": "l_wrist", "B": "r_wrist"}


def _look(color: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((20, 20, 3), np.uint8)
    img[:] = color
    return appearance.descriptor(img, (0, 0, 20, 20))


def _skeleton(t: float, player: str, slot: str) -> np.ndarray:
    scale, hip = (1.0, (500.0, 400.0)) if slot == "near" else (0.3, (500.0, 100.0))
    offsets = {
        "nose": (0, -150), "l_eye": (-5, -155), "r_eye": (5, -155), "l_ear": (-10, -150),
        "r_ear": (10, -150), "l_shoulder": (-30, -100), "r_shoulder": (30, -100),
        "l_elbow": (-40, -60), "r_elbow": (40, -60), "l_wrist": (-45, -20),
        "r_wrist": (45, -20), "l_hip": (-20, 0), "r_hip": (20, 0), "l_knee": (-22, 80),
        "r_knee": (22, 80), "l_ankle": (-24, 160), "r_ankle": (24, 160),
    }  # fmt: skip
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.9
    for name, (dx, dy) in offsets.items():
        kp[KP[name], :2] = (hip[0] + dx * scale, hip[1] + dy * scale)
    wrist = KP[HAND[player]]
    for n, (h, who, _) in enumerate(HITS):
        if who == player:  # a quick 150-torso-% sweep, alternating direction
            kp[wrist, 0] += (-1) ** n * 150 * scale * np.tanh((t - h) / 0.06)
    return kp


def _near_player(t: float) -> str:
    return "A" if t < END_CHANGE else "B"


def _write(tmp_path: Path, data_root: Path) -> Session:
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"x")
    session = create_or_reuse_session(data_root, video, datetime(2026, 9, 20, tzinfo=UTC), "p")
    pts = np.arange(int(DURATION * FPS)) / FPS
    pq.write_table(pa.table({"frame_idx": np.arange(pts.size), "pts": pts}),
                   session.path("frame_times.parquet"))  # fmt: skip
    times = [h for h, _, _ in HITS]
    pq.write_table(
        pa.table({
            "contact_id": np.arange(len(HITS)),
            "t_audio": np.array(times) + 0.004,
            "is_self_audio": [False] * len(HITS),
            "peak_db": [-15.0 if who == "A" else -35.0 for _, who, _ in HITS],
        }),
        session.path("contacts.parquet"),
    )  # fmt: skip
    cols: dict[str, list[object]] = {c: [] for c in (
        "frame_idx", "t_video", "window_id", "slot", "detected", "track_reset", "n_persons",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_conf",
    )}  # fmt: skip
    kps, looks = [], []
    for w in merge_windows(times, 1.0, 0.5, 0.0, float(pts[-1])):
        for i in np.flatnonzero((pts >= w.start - 1e-9) & (pts <= w.end + 1e-9)):
            t = float(pts[i])
            near = _near_player(t)
            for slot, player in (("near", near), ("far", "B" if near == "A" else "A")):
                kp = _skeleton(t, player, slot)
                for name, value in (("frame_idx", int(i)), ("t_video", t), ("window_id", w.id),
                                    ("slot", slot), ("detected", True), ("track_reset", False),
                                    ("n_persons", 2), ("bbox_conf", 0.9)):  # fmt: skip
                    cols[name].append(value)
                x1, y1 = kp[:, 0].min(), kp[:, 1].min()
                x2, y2 = kp[:, 0].max(), kp[:, 1].max()
                for name, value in zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
                                       (x1, y1, x2, y2), strict=True):  # fmt: skip
                    cols[name].append(float(value))
                kps.append(kp)
                looks.append(_look(LOOKS[player]))
    arr = np.array(kps, np.float32)
    table: dict[str, object] = {
        name: pa.array(values, pa.string() if name == "slot" else None)
        for name, values in cols.items()
    }
    for k, name in enumerate(KEYPOINT_NAMES):
        for c, suffix in enumerate(("x", "y", "conf")):
            table[f"{name}_{suffix}"] = pa.array(arr[:, k, c])
    flat = pa.array(np.array(looks, np.float32).ravel())
    table["appearance"] = pa.FixedSizeListArray.from_arrays(flat, appearance.SIZE)
    pq.write_table(pa.table(table), session.path("keypoints.parquet"))
    return session


def _run(session: Session, data_root: Path, **player: object) -> tuple[dict, dict]:  # type: ignore[type-arg]
    config = Config.model_validate(
        {
            "paths": {"data_root": str(data_root)},
            "pose": {"contacts": "all"},
            "player": {"min_side_duration_s": 0.0, **player},
        }
    )
    clean.run(StageContext(session, config, "h", get_logger(), "clean"))
    info = {
        r["contact_id"]: r for r in pq.read_table(session.path("swing_info.parquet")).to_pylist()
    }
    return read_json(session.path("players.json")), info


@pytest.fixture
def session(tmp_path: Path, data_root: Path) -> Session:
    return _write(tmp_path, data_root)


def test_identity_hand_and_sides(session: Session, data_root: Path) -> None:
    players, _ = _run(session, data_root)
    assert players["me"] == "A"
    assert players["me_reason"] == "louder hits"
    assert players["identities_resolved"] is True
    assert players["handedness"]["hand"] == "left"
    assert players["handedness"]["votes"]["left"] == 5
    assert [s["side"] for s in players["sides"]] == ["near", "far"]
    assert players["hits"] == {"self": 8, "other": 7, "unknown": 0}


def test_hits_are_attributed_and_far_side_not_measured(session: Session, data_root: Path) -> None:
    _, info = _run(session, data_root)
    for cid, (t, who, _slot) in enumerate(HITS):
        row = info[cid]
        my_side = "near" if _near_player(t) == "A" else "far"
        assert row["player_side"] == my_side, cid
        assert row["hitter"] == ("self" if who == "A" else "other"), cid
        assert row["is_self_confirmed"] is (who == "A"), cid
        if my_side == "far":
            assert not row["qc_pass"] and row["qc_reason"].startswith("far side"), cid
        elif who == "A":
            assert row["qc_pass"], (cid, row["qc_reason"])
    swings = pq.read_table(session.path("swings.parquet"))
    assert set(swings.column("player_side").to_pylist()) == {"near", "far"}
    assert pq.read_schema(session.path("swings.parquet")).metadata[b"tennis.racket_side"] == b"l"


def test_config_overrides(session: Session, data_root: Path) -> None:
    players, info = _run(session, data_root, identity="B", handedness="right")
    assert players["me"] == "B" and players["me_reason"] == "config"
    assert players["handedness"] == {"hand": "right", "reason": "config",
                                     "votes": players["handedness"]["votes"]}  # fmt: skip
    assert info[1]["hitter"] == "self" and info[1]["player_side"] == "far"
    assert info[0]["hitter"] == "other"

    players, _ = _run(session, data_root, identity="near_at_start")
    assert players["me"] == "A" and players["me_reason"] == "near player at the start"


def test_without_far_player_everyone_near_is_you(session: Session, data_root: Path) -> None:
    table = pq.read_table(session.path("keypoints.parquet"))
    near_only = table.filter(pa.compute.equal(table["slot"], "near"))
    pq.write_table(near_only, session.path("keypoints.parquet"))
    players, info = _run(session, data_root)
    assert players["identities_resolved"] is False
    assert players["me"] == "A"
    assert {r["player_side"] for r in info.values()} == {"near"}
