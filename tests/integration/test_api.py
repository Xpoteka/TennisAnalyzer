"""The API end to end: upload a synthetic video, create a session, run its job."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tennis.api.app import create_app
from tennis.config import Config, PathsConfig
from tennis.util.log import get_logger
from tennis.worker import execute_job
from tests.conftest import MakeVideo, needs_ffmpeg

pytestmark = needs_ffmpeg


@pytest.fixture
def client(data_root: Path) -> TestClient:
    return TestClient(create_app(data_root, run_worker=False))


def _upload(client: TestClient, video: Path, piece: int = 50_000) -> str:
    data = video.read_bytes()
    size = len(data)
    status = client.get("/api/uploads", params={"name": video.name, "size": size}).json()
    assert status == {"done": False, "received": 0, "size": size}
    offset = 0
    result: dict[str, object] = {}
    while offset < size:
        chunk = data[offset : offset + piece]
        r = client.put(
            "/api/uploads",
            params={"name": video.name, "size": size, "offset": offset},
            content=chunk,
        )
        assert r.status_code == 200, r.text
        result = r.json()
        offset += len(chunk)
    assert result["done"] is True
    return str(result["path"])


def test_upload_analyze_and_read_back(
    client: TestClient, data_root: Path, make_video: MakeVideo
) -> None:
    video = make_video("session.mp4", seconds=2.0, fps=30)
    path = _upload(client, video)
    assert Path(path).read_bytes() == video.read_bytes()

    r = client.post("/api/sessions", json={"paths": [path]})
    assert r.status_code == 201, r.text
    session = r.json()
    assert session["status"] == "queued"
    assert session["job"]["status"] == "queued"
    assert [v["filename"] for v in session["videos"]] == ["session.mp4"]

    cfg = Config(paths=PathsConfig(data_root=data_root))
    execute_job(data_root, cfg, session["job"]["id"], get_logger())

    detail = client.get(f"/api/sessions/{session['id']}").json()
    assert detail["status"] == "ready", detail
    assert detail["job"]["status"] == "done"
    assert detail["job"]["progress"] == 1.0
    assert detail["recorded_at"].startswith("2026-09-20T18:30:00")
    assert detail["duration_s"] == pytest.approx(2.0, abs=0.1)
    (v,) = detail["videos"]
    assert v["status"] == "ready"
    assert v["fps"] == pytest.approx(30)
    assert v["sync_method"] == "single"
    # testsrc2 H.264 in MP4 plays in browsers: linked, not transcoded.
    proxy = client.get(v["proxy_url"], headers={"Range": "bytes=0-99"})
    assert proxy.status_code == 206
    assert len(proxy.content) == 100

    listing = client.get("/api/sessions").json()
    assert [s["id"] for s in listing] == [session["id"]]
    lib = client.get("/api/library").json()
    assert lib["videos"][0]["sessions"] == [session["id"]]

    assert client.delete(f"/api/sessions/{session['id']}").json() == {"deleted": session["id"]}
    assert client.get("/api/sessions").json() == []
    assert Path(path).is_file()  # the source video stays


def test_second_video_is_placed_by_creation_time(
    client: TestClient, data_root: Path, make_video: MakeVideo
) -> None:
    a = make_video("a.mp4", seconds=1.0, fps=30, creation_time="2026-09-20T18:30:00Z")
    b = make_video("b.mp4", seconds=1.0, fps=30, creation_time="2026-09-20T18:40:00Z")
    session = client.post("/api/sessions", json={"paths": [str(b), str(a)]}).json()
    cfg = Config(paths=PathsConfig(data_root=data_root))
    execute_job(data_root, cfg, session["job"]["id"], get_logger())
    detail = client.get(f"/api/sessions/{session['id']}").json()
    offsets = {v["filename"]: v["offset_s"] for v in detail["videos"]}
    assert offsets == {"a.mp4": 0.0, "b.mp4": 600.0}
    assert detail["duration_s"] == pytest.approx(601.0, abs=0.1)


def test_hevc_source_gets_a_transcoded_proxy(
    client: TestClient, data_root: Path, tmp_path: Path, make_video: MakeVideo
) -> None:
    import subprocess

    src = make_video("h.mp4", seconds=1.0, fps=30)
    mkv = tmp_path / "h.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c", "copy", str(mkv)], check=True
    )
    session = client.post("/api/sessions", json={"paths": [str(mkv)]}).json()
    cfg = Config.model_validate(
        {"paths": {"data_root": data_root}, "proxy": {"encoder": "libx264"}}
    )
    execute_job(data_root, cfg, session["job"]["id"], get_logger())
    (v,) = client.get(f"/api/sessions/{session['id']}").json()["videos"]
    proxy = data_root / "videos" / str(v["id"]) / "proxy.mp4"
    assert proxy.is_file() and not proxy.is_symlink()


def test_missing_and_non_video_paths_are_refused(client: TestClient, tmp_path: Path) -> None:
    assert (
        client.post("/api/sessions", json={"paths": [str(tmp_path / "x.mp4")]}).status_code == 404
    )
    (tmp_path / "notes.txt").write_text("x")
    r = client.post("/api/sessions", json={"paths": [str(tmp_path / "notes.txt")]})
    assert r.status_code == 415


def test_password_protects_the_api(data_root: Path) -> None:
    client = TestClient(create_app(data_root, run_worker=False, password="correct horse"))
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/me").json()["logged_in"] is False
    assert client.post("/api/login", json={"password": "correct horse"}).status_code == 200
    assert client.get("/api/sessions").status_code == 200


def test_cross_origin_writes_are_refused(client: TestClient) -> None:
    r = client.post("/api/sessions", json={"paths": ["/x"]}, headers={"Origin": "http://evil"})
    assert r.status_code == 403
