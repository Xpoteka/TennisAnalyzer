"""The web UI's HTTP surface: what it serves, what it refuses, and that it starts jobs."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tennis.config import load_config
from tennis.web.server import make_server


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[str, ThreadingHTTPServer]]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"paths:\n  data_root: {tmp_path / 'data'}\n  labels_dir: {tmp_path / 'labels'}\n"
    )
    (tmp_path / "data" / "sessions" / "2026-01-02_evening").mkdir(parents=True)
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "contacts_x.csv").write_text("1.0\n2.0\n")
    httpd, _ = make_server(load_config(config_path), config_path, tmp_path, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host}:{port}", httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


def get(base: str, path: str) -> Any:
    with urllib.request.urlopen(base + path) as response:
        return json.loads(response.read())


def post(base: str, path: str, data: Any, headers: dict[str, str] | None = None) -> Any:
    body = json.dumps(data).encode()
    request = urllib.request.Request(
        base + path, body, {"Content-Type": "application/json", **(headers or {})}
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def test_the_page_and_its_assets_are_served(server: tuple[str, ThreadingHTTPServer]) -> None:
    base, _ = server
    for path in ("/", "/static/app.js", "/static/style.css"):
        with urllib.request.urlopen(base + path) as response:
            assert response.status == 200
            assert response.read()


def test_meta_lists_the_commands_and_stages(server: tuple[str, ThreadingHTTPServer]) -> None:
    meta = get(server[0], "/api/meta")
    assert {c["name"] for c in meta["commands"]} >= {"process", "report", "trends"}
    assert meta["stages"][0]["name"] == "ingest"
    assert meta["config_path"].endswith("config.yaml")


def test_sessions_are_listed_with_a_status_per_stage(
    server: tuple[str, ThreadingHTTPServer],
) -> None:
    sessions = get(server[0], "/api/sessions")["sessions"]
    assert [s["id"] for s in sessions] == ["2026-01-02_evening"]
    assert sessions[0]["stages"]["ingest"] == "-"
    assert sessions[0]["n_swings"] == 0


def test_a_session_detail_carries_the_video_the_ui_needs_to_continue_it(
    server: tuple[str, ThreadingHTTPServer], tmp_path: Path
) -> None:
    """`tennis process` takes a video, not a session id, so the UI is told where the video is."""
    session = tmp_path / "data" / "sessions" / "2026-01-02_evening"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    (session / "source.mp4").symlink_to(video)

    detail = get(server[0], "/api/sessions/2026-01-02_evening")
    assert detail["source_path"] == str(video.resolve())
    assert detail["source_missing"] is False

    video.unlink()
    gone = get(server[0], "/api/sessions/2026-01-02_evening")
    assert gone["source_missing"] is True


def test_an_unprocessed_session_still_has_a_detail_page(
    server: tuple[str, ThreadingHTTPServer],
) -> None:
    detail = get(server[0], "/api/sessions/2026-01-02_evening")
    assert detail["counts"]["contacts"] == 0
    assert detail["summary"] == []
    assert get(server[0], "/api/sessions/2026-01-02_evening/swings")["swings"] == []


def test_an_unknown_session_is_a_404(server: tuple[str, ThreadingHTTPServer]) -> None:
    with pytest.raises(urllib.error.HTTPError) as info:
        get(server[0], "/api/sessions/nope")
    assert info.value.code == 404


def test_label_files_are_listed(server: tuple[str, ThreadingHTTPServer]) -> None:
    assert [f["name"] for f in get(server[0], "/api/labels")["files"]] == ["contacts_x.csv"]


def test_files_outside_the_served_roots_are_refused(
    server: tuple[str, ThreadingHTTPServer],
) -> None:
    for path in ("/files/data/%2e%2e/%2e%2e/etc/passwd", "/files/secrets/key"):
        with pytest.raises(urllib.error.HTTPError) as info:
            get(server[0], path)
        assert info.value.code in {403, 404}


def test_a_served_file_supports_range_requests(server: tuple[str, ThreadingHTTPServer]) -> None:
    base, _ = server
    request = urllib.request.Request(base + "/files/labels/contacts_x.csv")
    request.add_header("Range", "bytes=0-2")
    with urllib.request.urlopen(request) as response:
        assert response.status == 206
        assert response.read() == b"1.0"


def test_a_cross_origin_write_is_refused(server: tuple[str, ThreadingHTTPServer]) -> None:
    with pytest.raises(urllib.error.HTTPError) as info:
        post(server[0], "/api/jobs", {"command": "list"}, {"Origin": "http://evil.example"})
    assert info.value.code == 403


def test_an_unknown_command_is_refused(server: tuple[str, ThreadingHTTPServer]) -> None:
    with pytest.raises(urllib.error.HTTPError) as info:
        post(server[0], "/api/jobs", {"command": "rm -rf", "values": {}})
    assert info.value.code == 400


def test_a_job_runs_the_cli_and_keeps_its_output(server: tuple[str, ThreadingHTTPServer]) -> None:
    base, _ = server
    job = post(
        base, "/api/jobs", {"command": "players", "values": {"session_id": "2026-01-02_evening"}}
    )
    assert job["argv"][:2] == ["players", "2026-01-02_evening"]
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = get(base, f"/api/jobs/{job['id']}?since=0")
        if state["state"] in {"done", "failed", "cancelled"}:
            break
        time.sleep(0.2)
    # The session has no pose output, so the command fails - but it ran, and said why.
    assert state["state"] == "failed"
    assert state["returncode"] != 0
    assert any("2026-01-02_evening" in line for line in state["lines"])
    assert get(base, "/api/jobs")["jobs"][0]["id"] == job["id"]


def test_uploads_land_in_the_right_folder_and_check_the_suffix(
    server: tuple[str, ThreadingHTTPServer], tmp_path: Path
) -> None:
    base, _ = server
    request = urllib.request.Request(
        base + "/api/upload?kind=labels&name=said_x.csv",
        b"1.0,in\n",
        {"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read())
    assert Path(result["path"]) == (tmp_path / "labels" / "said_x.csv").resolve()
    assert Path(result["path"]).read_text() == "1.0,in\n"

    bad = urllib.request.Request(
        base + "/api/upload?kind=video&name=notes.txt",
        b"x",
        {"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(bad)
    assert info.value.code == 415


def test_an_invalid_config_is_not_saved(
    server: tuple[str, ThreadingHTTPServer], tmp_path: Path
) -> None:
    base, _ = server
    before = (tmp_path / "config.yaml").read_text()
    with pytest.raises(urllib.error.HTTPError) as info:
        post(base, "/api/config", {"text": "audio:\n  onset_k: not-a-number\n"})
    assert info.value.code == 400
    assert (tmp_path / "config.yaml").read_text() == before

    saved = post(base, "/api/config", {"text": before + "audio:\n  onset_k: 7\n"})
    assert saved["error"] is None
    assert "onset_k: 7" in (tmp_path / "config.yaml").read_text()


def test_the_config_view_can_create_a_missing_config(tmp_path: Path) -> None:
    """Started without a config file, the UI offers the example and saves a real one."""
    httpd, _ = make_server(load_config(None), None, tmp_path, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"
    try:
        before = get(base, "/api/config")
        assert before["exists"] is False
        assert before["path"] is None
        assert before["would_create"] == str(tmp_path / "config.yaml")
        assert "paths:" in before["text"]

        saved = post(base, "/api/config", {"text": "audio:\n  onset_k: 5\n"})
        assert saved["exists"] is True
        assert (tmp_path / "config.yaml").read_text() == "audio:\n  onset_k: 5\n"
        assert get(base, "/api/config")["path"] == str(tmp_path / "config.yaml")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_config_that_does_not_validate_is_not_created(tmp_path: Path) -> None:
    httpd, _ = make_server(load_config(None), None, tmp_path, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as info:
            post(base, "/api/config", {"text": "nonsense_section: 1\n"})
        assert info.value.code == 400
        assert not (tmp_path / "config.yaml").exists()
    finally:
        httpd.shutdown()
        httpd.server_close()


def _send(base: str, path: str, body: bytes, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(
        base + path, body, {"Content-Type": "application/octet-stream", **(headers or {})}
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def test_a_video_uploads_in_pieces_and_resumes_after_an_interruption(
    server: tuple[str, ThreadingHTTPServer], tmp_path: Path
) -> None:
    base, _ = server
    data = bytes(range(256)) * 40  # 10 KiB
    q = f"/api/upload?kind=video&name=my%20session.MP4&size={len(data)}"
    assert get(base, q) == {"done": False, "received": 0, "size": len(data)}

    first = _send(base, q + "&offset=0", data[:4000])
    assert first == {"done": False, "received": 4000, "size": len(data)}

    # A piece sent again after a dropped connection is refused, and says where to go on.
    with pytest.raises(urllib.error.HTTPError) as info:
        _send(base, q + "&offset=0", data[:4000])
    assert info.value.code == 409
    assert json.loads(info.value.read())["received"] == 4000

    # The page reloads and asks how far it got: the server kept the first piece.
    assert get(base, q)["received"] == 4000
    last = _send(base, q + "&offset=4000", data[4000:])
    assert last["done"] is True and last["reused"] is False
    final = tmp_path / "data" / "uploads" / "my_session.MP4"
    assert Path(last["path"]) == final.resolve()
    assert final.read_bytes() == data
    assert not list((tmp_path / "data" / "uploads" / ".partial").iterdir())

    # Dropping the same file again does not send it again.
    again = get(base, q)
    assert again["done"] is True and again["reused"] is True


def test_the_library_lists_videos_and_the_sessions_that_use_them(
    server: tuple[str, ThreadingHTTPServer], tmp_path: Path
) -> None:
    base, _ = server
    uploads = tmp_path / "data" / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "a.mp4").write_bytes(b"a" * 10)
    (uploads / "b.mov").write_bytes(b"b" * 20)
    (uploads / "notes.txt").write_text("not a video")
    session = tmp_path / "data" / "sessions" / "2026-01-02_evening"
    (session / "source.mp4").symlink_to("../../uploads/a.mp4")
    partial = uploads / ".partial"
    partial.mkdir()
    (partial / "c.mp4.1000.part").write_bytes(b"c" * 300)

    lib = get(base, "/api/videos")
    by_name = {v["name"]: v for v in lib["videos"]}
    assert set(by_name) == {"a.mp4", "b.mov"}
    assert by_name["a.mp4"]["sessions"] == ["2026-01-02_evening"]
    assert by_name["b.mov"]["sessions"] == []
    assert lib["partial"] == [
        {"name": "c.mp4", "size": 1000, "received": 300, "modified": lib["partial"][0]["modified"]}
    ]
    assert lib["disk"]["free"] > 0

    # The video can be downloaded, with ranges for seeking.
    with urllib.request.urlopen(base + by_name["b.mov"]["url"]) as response:
        assert response.read() == b"b" * 20

    deleted = post(base, "/api/videos/delete", {"name": "a.mp4"})
    assert deleted == {"deleted": "a.mp4", "sessions": ["2026-01-02_evening"]}
    assert not (uploads / "a.mp4").exists()
    with pytest.raises(urllib.error.HTTPError) as info:
        post(base, "/api/videos/delete", {"name": "../config.yaml"})
    assert info.value.code == 404
    assert (tmp_path / "config.yaml").exists()

    post(base, "/api/upload/discard?kind=video&name=c.mp4&size=1000", {})
    assert get(base, "/api/videos")["partial"] == []


def test_a_piece_larger_than_the_file_is_refused(server: tuple[str, ThreadingHTTPServer]) -> None:
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as info:
        _send(base, "/api/upload?kind=video&name=x.mp4&size=3&offset=0", b"abcd")
    assert info.value.code == 400


@pytest.fixture
def locked_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[str, ThreadingHTTPServer]]:
    from tennis.web import auth

    monkeypatch.setattr(auth, "FAILED_LOGIN_DELAY_S", 0.0)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"paths:\n  data_root: {tmp_path / 'data'}\n")
    httpd, _ = make_server(
        load_config(config_path), config_path, tmp_path, "127.0.0.1", 0, password="correct horse"
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def test_with_a_password_nothing_is_served_before_logging_in(
    locked_server: tuple[str, ThreadingHTTPServer], tmp_path: Path
) -> None:
    base, _ = locked_server
    for path in ("/api/sessions", "/api/videos", "/files/data/.ui_secret"):
        with pytest.raises(urllib.error.HTTPError) as info:
            urllib.request.urlopen(base + path)
        assert info.value.code == 401, path
    with pytest.raises(urllib.error.HTTPError) as info:
        post(base, "/api/jobs", {"command": "list", "values": {}})
    assert info.value.code == 401

    # The page itself sends the browser to the login form, which is served.
    opener = urllib.request.build_opener(_NoRedirect)
    with pytest.raises(urllib.error.HTTPError) as info:
        opener.open(base + "/")
    assert info.value.code == 303 and info.value.headers["Location"] == "/login"
    with urllib.request.urlopen(base + "/login") as response:
        assert b"Password" in response.read()

    with pytest.raises(urllib.error.HTTPError) as info:
        post(base, "/api/login", {"password": "wrong"})
    assert info.value.code == 401

    request = urllib.request.Request(
        base + "/api/login",
        json.dumps({"password": "correct horse"}).encode(),
        {"Content-Type": "application/json", "X-Forwarded-Proto": "https"},
    )
    with urllib.request.urlopen(request) as response:
        cookie = response.headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Secure" in cookie
    token = cookie.split(";")[0]

    request = urllib.request.Request(base + "/api/sessions", headers={"Cookie": token})
    with urllib.request.urlopen(request) as response:
        assert json.loads(response.read()) == {"sessions": []}
    assert get_with(base, "/api/meta", token)["auth"] is True

    # Logged in or not, the signing secret in the data folder is never served.
    with pytest.raises(urllib.error.HTTPError) as info:
        get_with(base, "/files/data/.ui_secret", token)
    assert info.value.code == 404

    # A tampered cookie is refused.
    with pytest.raises(urllib.error.HTTPError) as info:
        get_with(base, "/api/sessions", token[:-1] + ("0" if token[-1] != "0" else "1"))
    assert info.value.code == 401


def get_with(base: str, path: str, cookie: str) -> Any:
    request = urllib.request.Request(base + path, headers={"Cookie": cookie})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def test_a_write_through_a_reverse_proxy_is_same_origin(
    server: tuple[str, ThreadingHTTPServer],
) -> None:
    """The proxy talks to us as 127.0.0.1; the browser's host comes as X-Forwarded-Host."""
    base, _ = server
    result = post(
        base,
        "/api/config",
        {"text": "audio:\n  onset_k: 6\n"},
        {"Origin": "https://tennis.example.com", "X-Forwarded-Host": "tennis.example.com"},
    )
    assert result["ok"] is True


def test_serving_on_a_network_interface_needs_a_password(tmp_path: Path) -> None:
    from tennis.errors import UserError
    from tennis.web.server import serve

    with pytest.raises(UserError, match="without a password"):
        serve(load_config(None), None, tmp_path, host="0.0.0.0", port=0, open_browser=False)
    with pytest.raises(UserError, match="at least"):
        serve(
            load_config(None),
            None,
            tmp_path,
            host="0.0.0.0",
            port=0,
            open_browser=False,
            password="short",
        )


def test_a_refused_request_does_not_garble_the_next_one_on_the_connection(
    locked_server: tuple[str, ThreadingHTTPServer],
) -> None:
    """Browsers reuse connections: a body nobody read must not become the next request."""
    import http.client

    base, _ = locked_server
    conn = http.client.HTTPConnection(base.removeprefix("http://"))
    try:
        # Refused before its body is looked at: not logged in.
        conn.request("POST", "/api/logout", b"{}", {"Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.status == 401
        response.read()
        conn.request("POST", "/api/jobs", b'{"command": "list"}')
        response = conn.getresponse()
        assert response.status == 401
        response.read()
        conn.request("GET", "/login")
        response = conn.getresponse()
        assert response.status == 200
        response.read()
        # An upload piece refused unread closes the connection rather than leave it dirty.
        conn.request("POST", "/api/upload?kind=video&name=x.mp4&size=3&offset=0", b"abc")
        response = conn.getresponse()
        assert response.status == 401
        assert response.getheader("Connection") == "close"
    finally:
        conn.close()
