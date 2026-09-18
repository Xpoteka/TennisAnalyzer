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
    relink_missing_sources,
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


def test_stale_reasons_with_fingerprints(session: Session) -> None:
    ins, outs = ("in.txt",), ("out.txt",)
    in_p, out_p = session.path("in.txt"), session.path("out.txt")
    _touch(in_p, 3_000)
    _touch(out_p, 1_000)  # older than the input: fine, the recorded fingerprints decide
    session.write_stamp("s", "h", session.fingerprints(ins), session.fingerprints(outs))
    assert session.stale_reason("s", ins, outs, "h") is None

    _touch(in_p, 3_001)
    assert session.stale_reason("s", ins, outs, "h") == "input changed: in.txt"
    _touch(in_p, 3_000)
    assert session.stale_reason("s", ins, outs, "h") is None

    out_p.write_text("edited by hand")
    _touch(out_p, 1_000)
    assert session.stale_reason("s", ins, outs, "h") == "output changed: out.txt"

    in_p.unlink()
    assert session.stale_reason("s", ins, outs, "h") == "input missing: in.txt"


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


def test_is_dataless_ignores_an_ordinary_file(tmp_path: Path) -> None:
    from tennis.util.io import is_dataless

    real = tmp_path / "clip.mp4"
    real.write_bytes(b"x" * 4096)
    assert is_dataless(real) is False
    assert is_dataless(tmp_path / "gone.mp4") is False

    empty = tmp_path / "empty.mp4"
    empty.touch()
    assert is_dataless(empty) is False


def test_a_cloud_placeholder_fails_fast_instead_of_stalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading an evicted iCloud file blocks for as long as the download takes, with no
    output at all, so it is rejected before ffmpeg ever sees the path."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x" * 4096)

    real_stat = Path.stat

    class _Dataless:
        def __init__(self, st: os.stat_result) -> None:
            self._st = st

        def __getattr__(self, name: str) -> object:
            return getattr(self._st, name)

        @property
        def st_flags(self) -> int:
            return 0x40000000  # SF_DATALESS

    def fake_stat(self: Path, *args: object, **kwargs: object) -> object:
        st = real_stat(self, *args, **kwargs)  # type: ignore[arg-type]
        return _Dataless(st) if self == video else st

    monkeypatch.setattr(Path, "stat", fake_stat)

    from tennis.util.io import is_dataless

    assert is_dataless(video) is True
    with pytest.raises(UserError, match="not on this disk"):
        check_video_file(video)


def test_an_uploaded_video_is_linked_relatively_so_the_data_folder_can_move(
    tmp_path: Path, data_root: Path
) -> None:
    video = data_root / "uploads" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not really a video")
    session = create_or_reuse_session(data_root, video, EVENING_UTC)
    link = session.dir / "source.mp4"
    assert os.readlink(link) == "../../uploads/a.mp4"

    moved = tmp_path / "elsewhere"
    data_root.rename(moved)
    assert (moved / "sessions" / session.id / "source.mp4").resolve() == (
        moved / "uploads" / "a.mp4"
    ).resolve()


def test_relink_points_sessions_at_a_moved_video(tmp_path: Path, data_root: Path) -> None:
    video = _video(tmp_path, "match.mp4")
    session = create_or_reuse_session(data_root, video, EVENING_UTC)
    st = video.stat()
    session.write_stamp("ingest", "h", inputs={"@source": [st.st_mtime_ns, st.st_size]})

    uploads = data_root / "uploads"
    uploads.mkdir()
    copy = uploads / "match.mp4"
    copy.write_bytes(video.read_bytes())
    os.utime(copy, ns=(st.st_atime_ns, st.st_mtime_ns))  # as rsync -t would
    video.unlink()

    dry = relink_missing_sources(data_root, [uploads], dry_run=True)
    assert [(r.session, r.found, r.same_file) for r in dry] == [(session.id, copy.resolve(), True)]
    assert not session.source_link.exists()

    relink_missing_sources(data_root, [uploads])
    assert session.source_target() == copy.resolve()
    assert os.readlink(session.source_link) == "../../uploads/match.mp4"
    assert relink_missing_sources(data_root, [uploads]) == []


def test_relink_skips_a_same_named_file_of_another_size(tmp_path: Path, data_root: Path) -> None:
    video = _video(tmp_path, "match.mp4")
    session = create_or_reuse_session(data_root, video, EVENING_UTC)
    st = video.stat()
    session.write_stamp("ingest", "h", inputs={"@source": [st.st_mtime_ns, st.st_size]})
    other = tmp_path / "other"
    other.mkdir()
    (other / "match.mp4").write_bytes(b"a different video altogether")
    video.unlink()

    [result] = relink_missing_sources(data_root, [other])
    assert result.found is None
    assert not session.source_link.exists()
