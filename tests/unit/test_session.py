from __future__ import annotations

import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tennis import __version__
from tennis.errors import UserError
from tennis.session import (
    Session,
    check_video_file,
    create_or_reuse_session,
    default_session_id,
    list_sessions,
    open_session,
    validate_session_id,
)

EVENING_UTC = datetime(2026, 9, 20, 18, 30, tzinfo=UTC)


def _video(tmp_path: Path, name: str = "a.mp4") -> Path:
    p = tmp_path / "raw" / name
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(b"not really a video")
    return p


@pytest.fixture
def toronto_tz(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TZ", "America/Toronto")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.parametrize(
    ("utc_hour", "expected"),
    [
        (8, "2026-09-20_night"),  # 04:00 local
        (9, "2026-09-20_morning"),  # 05:00
        (16, "2026-09-20_afternoon"),  # 12:00
        (21, "2026-09-20_evening"),  # 17:00
        (2, "2026-09-19_night"),  # 22:00 the day before
    ],
)
@pytest.mark.usefixtures("toronto_tz")
def test_default_session_id_uses_local_time(utc_hour: int, expected: str) -> None:
    assert default_session_id(datetime(2026, 9, 20, utc_hour, tzinfo=UTC)) == expected


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "-lead", "has space", "x" * 101])
def test_invalid_session_ids(bad: str) -> None:
    with pytest.raises(UserError):
        validate_session_id(bad)


def test_create_links_source_and_reuses(tmp_path: Path, data_root: Path) -> None:
    video = _video(tmp_path)
    s1 = create_or_reuse_session(data_root, video, EVENING_UTC)
    link = s1.dir / "source.mp4"
    assert link.is_symlink() and link.resolve() == video.resolve()
    assert video.read_bytes() == b"not really a video"
    s2 = create_or_reuse_session(data_root, video, EVENING_UTC)
    assert s2 == s1


def test_same_video_reused_even_with_other_creation_time(tmp_path: Path, data_root: Path) -> None:
    video = _video(tmp_path)
    s1 = create_or_reuse_session(data_root, video, EVENING_UTC)
    s2 = create_or_reuse_session(data_root, video, EVENING_UTC + timedelta(days=3))
    assert s1 == s2


def test_derived_id_collision_gets_suffix(tmp_path: Path, data_root: Path) -> None:
    a = create_or_reuse_session(data_root, _video(tmp_path, "a.mp4"), EVENING_UTC)
    b = create_or_reuse_session(data_root, _video(tmp_path, "b.MOV"), EVENING_UTC)
    assert b.id == f"{a.id}_2"
    assert (b.dir / "source.mov").is_symlink()


def test_explicit_id_collision_is_error(tmp_path: Path, data_root: Path) -> None:
    create_or_reuse_session(data_root, _video(tmp_path, "a.mp4"), EVENING_UTC, "s1")
    with pytest.raises(UserError, match="different video"):
        create_or_reuse_session(data_root, _video(tmp_path, "b.mp4"), EVENING_UTC, "s1")


def test_open_and_list(tmp_path: Path, data_root: Path) -> None:
    assert list_sessions(data_root) == []
    create_or_reuse_session(data_root, _video(tmp_path), EVENING_UTC, "s1")
    assert [s.id for s in list_sessions(data_root)] == ["s1"]
    assert open_session(data_root, "s1").id == "s1"
    with pytest.raises(UserError, match="unknown session"):
        open_session(data_root, "s2")


def test_check_video_file(tmp_path: Path) -> None:
    with pytest.raises(UserError, match="not found"):
        check_video_file(tmp_path / "missing.mp4")
    with pytest.raises(UserError, match="not a file"):
        check_video_file(tmp_path)
    txt = tmp_path / "notes.txt"
    txt.write_text("x")
    with pytest.raises(UserError, match="unsupported"):
        check_video_file(txt)
    assert check_video_file(_video(tmp_path)).is_absolute()


# --- caching ---------------------------------------------------------------------------


@pytest.fixture
def session(tmp_path: Path, data_root: Path) -> Session:
    return create_or_reuse_session(data_root, _video(tmp_path), EVENING_UTC, "cache")


def _touch(path: Path, mtime: float) -> None:
    if not path.exists():
        path.write_text("x")
    os.utime(path, (mtime, mtime))


def test_stale_reasons(session: Session) -> None:
    ins, outs = ("in.txt",), ("out.txt",)
    in_p, out_p = session.path("in.txt"), session.path("out.txt")
    _touch(in_p, 1_000)

    assert session.stale_reason("s", ins, outs, "h1") == "not run yet"

    _touch(out_p, 2_000)
    session.write_stamp("s", "h1")
    assert session.stale_reason("s", ins, outs, "h1") is None
    assert session.stale_reason("s", ins, outs, "h2") == "config changed"

    _touch(in_p, 3_000)
    assert "input newer" in (session.stale_reason("s", ins, outs, "h1") or "")

    _touch(out_p, 4_000)
    assert session.stale_reason("s", ins, outs, "h1") is None

    out_p.unlink()
    assert "output missing" in (session.stale_reason("s", ins, outs, "h1") or "")


def test_stale_on_version_change(session: Session) -> None:
    session.write_stamp("s", "h")
    stamp = session.stamp_path("s")
    stamp.write_text(stamp.read_text().replace(__version__, "0.0.0-old"))
    assert "pipeline version" in (session.stale_reason("s", (), (), "h") or "")


def test_source_input_follows_symlink(session: Session, tmp_path: Path) -> None:
    session.write_stamp("s", "h")
    out_p = session.path("out.txt")
    _touch(out_p, 2_000)
    raw = tmp_path / "raw" / "a.mp4"
    os.utime(raw, (1_000, 1_000))
    assert session.stale_reason("s", ("@source",), ("out.txt",), "h") is None
    os.utime(raw, (3_000, 3_000))
    assert session.stale_reason("s", ("@source",), ("out.txt",), "h") is not None
