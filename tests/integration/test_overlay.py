"""What the review player asks for: a video's overlay and slices of its tracks."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from tennis.api.app import create_app
from tennis.db import session_scope
from tennis.db.models import Player, Session, SessionPlayer, Shot, Video
from tennis.pipeline import session_dir, video_dir
from tennis.util.io import write_json
from tennis.vision import court as court_mod

W, H = 1920, 1080
CORNERS = [(700, 300), (1220, 300), (1700, 1000), (220, 1000)]  # far-left, clockwise


@pytest.fixture
def client(data_root: Path) -> TestClient:
    return TestClient(create_app(data_root, run_worker=False))


def _session(data_root: Path, *, court: bool) -> tuple[int, int, int]:
    """A session with one video, one player on track 3, and a shot that bounced at the T."""
    cal = court_mod.from_corners(CORNERS, W, H)
    with session_scope(data_root) as db:
        session = Session()
        player = Player(name="Ann")
        db.add_all([session, player])
        db.flush()
        video = Video(
            session_id=session.id or 0,
            filename="a.mp4",
            path="a.mp4",
            width=W,
            height=H,
            court=cal.to_json() if court else None,
        )
        db.add(video)
        db.add(SessionPlayer(session_id=session.id or 0, player_id=player.id or 0, label="A"))
        db.flush()
        shot = Shot(
            session_id=session.id or 0,
            video_id=video.id or 0,
            player_id=player.id,
            t=5.0,
            bounce_x=0.0,
            bounce_y=court_mod.SERVICE_FROM_NET,
            in_court=False,
        )
        net = Shot(
            session_id=session.id or 0,
            video_id=video.id or 0,
            t=6.0,
            quality={"crosses_net": False},
        )
        db.add_all([shot, net])
        db.flush()
        ids = (session.id or 0, video.id or 0, shot.id or 0)
    write_json(
        session_dir(data_root, ids[0]) / "identities.json",
        {"tracks": {str(ids[1]): {"3": "A", "9": "Z"}}, "players": {}},
    )
    return ids


def _tracks(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    f32 = pa.float32()
    times = [0.0, 0.5, 9.99, 10.0, 12.0]
    pq.write_table(
        pa.table(
            {
                "t": pa.array(times, pa.float64()),
                "x": pa.array([100.4, 110.6, 120, 130, 140], f32),
                "y": pa.array([50, 51, 52, 53, 54], f32),
                "track": pa.array([1, 1, 1, 2, 2], pa.int32()),
            }
        ),
        directory / "ball.parquet",
    )
    nan = float("nan")
    pq.write_table(
        pa.table(
            {
                "t": pa.array([12.0, 1.0], pa.float64()),  # out of order on purpose
                "track": pa.array([3, 3], pa.int32()),
                "kp_x": pa.array([[5.0] * 17, [10.2] * 16 + [nan]], pa.list_(f32)),
                "kp_y": pa.array([[6.0] * 17, [20.7] * 17], pa.list_(f32)),
                "kp_c": pa.array([[0.9] * 17, [0.9, 0.1] + [0.9] * 15], pa.list_(f32)),
            }
        ),
        directory / "motion.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "t": pa.array([1.0, 2.0], pa.float64()),
                "track_id": pa.array([3, 3], pa.int32()),
                "foot_px": pa.array([900.0, 910.0], f32),
                "foot_py": pa.array([800.0, 805.0], f32),
                "court_x": pa.array([1.234, nan], f32),
                "court_y": pa.array([-11.0, nan], f32),
            }
        ),
        directory / "people.parquet",
    )


def test_tracks_come_in_slices(client: TestClient, data_root: Path) -> None:
    _, video_id, _ = _session(data_root, court=True)
    _tracks(video_dir(data_root, video_id))

    r = client.get(f"/api/videos/{video_id}/tracks", params={"start": 0, "end": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ball"] == [[0.0, 100, 50, 1], [0.5, 111, 51, 1], [9.99, 120, 52, 1]]
    [pose] = body["poses"]
    assert pose[:2] == [1.0, 3]
    assert pose[2] == [10, None] + [10] * 14 + [None]  # unsure or missing keypoints are left out
    assert pose[3][0] == 21
    assert body["feet"] == [[1.0, 3, 900, 800, 1.23, -11.0], [2.0, 3, 910, 805, None, None]]

    later = client.get(f"/api/videos/{video_id}/tracks", params={"start": 10, "end": 20}).json()
    assert [row[0] for row in later["ball"]] == [10.0, 12.0]
    assert [row[0] for row in later["poses"]] == [12.0]


def test_tracks_without_files_are_empty(client: TestClient, data_root: Path) -> None:
    _, video_id, _ = _session(data_root, court=False)
    video_dir(data_root, video_id).mkdir(parents=True)
    body = client.get(f"/api/videos/{video_id}/tracks").json()
    assert (body["ball"], body["poses"], body["feet"]) == ([], [], [])
    assert client.get("/api/videos/999/tracks").status_code == 404


def test_overlay(client: TestClient, data_root: Path) -> None:
    session_id, video_id, shot_id = _session(data_root, court=True)
    body = client.get(f"/api/videos/{video_id}/overlay").json()
    assert (body["width"], body["height"]) == (W, H)
    player_id = client.get(f"/api/sessions/{session_id}").json()["players"][0]["player_id"]
    assert body["players"] == {"3": player_id}  # track 9 belongs to nobody in this session
    # The far service line's middle lies between the far corners, below them in the image.
    x, y = body["bounces"][str(shot_id)]
    assert abs(x - W / 2) < 3 and 300 < y < 1000
    assert len(body["court"]) == len(court_mod.COURT_LINES) + 1
    far_left = min((p for line in body["court"] for p in line), key=lambda p: (p[1], p[0]))
    assert far_left == [700, 300]


def test_overlay_without_a_court(client: TestClient, data_root: Path) -> None:
    _, video_id, _ = _session(data_root, court=False)
    body = client.get(f"/api/videos/{video_id}/overlay").json()
    assert body["court"] == [] and body["bounces"] == {}


def test_shots_say_how_they_ended(client: TestClient, data_root: Path) -> None:
    session_id, _, _ = _session(data_root, court=True)
    shots = client.get(f"/api/sessions/{session_id}/shots").json()
    assert [s["outcome"] for s in shots] == ["out", "net"]
