"""Ingest stage and pipeline runner on synthetic clips generated with ffmpeg."""

from __future__ import annotations

import json
import os
import time
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tennis.config import Config, PathsConfig
from tennis.errors import StageError
from tennis.session import Session, create_or_reuse_session
from tennis.stages import STAGES, run_pipeline, stage_status
from tennis.util.log import get_logger
from tests.conftest import MakeVideo, needs_ffmpeg

pytestmark = needs_ffmpeg

CREATED = datetime(2026, 9, 20, 18, 30, tzinfo=UTC)


def _config(data_root: Path) -> Config:
    return Config(paths=PathsConfig(data_root=data_root))


def _session(data_root: Path, video: Path, sid: str = "t") -> Session:
    return create_or_reuse_session(data_root, video, CREATED, sid)


def test_ingest_writes_metadata_and_audio(make_video: MakeVideo, data_root: Path) -> None:
    video = make_video(seconds=2.0, fps=120)
    session = _session(data_root, video)
    ran = run_pipeline(session, _config(data_root), get_logger())
    assert ran == ["ingest", "contacts", "pose"]
    assert (session.dir / "frame_times.parquet").exists()

    meta = json.loads((session.dir / "metadata.json").read_text())
    assert meta["fps"] == pytest.approx(120)
    assert meta["is_vfr"] is False
    assert meta["warnings"] == []
    assert meta["resolution"] == [320, 240]
    assert meta["codec"] == "h264"
    assert meta["audio_sample_rate"] == 48_000
    assert meta["duration_s"] == pytest.approx(2.0, abs=0.1)
    assert meta["creation_time"].startswith("2026-09-20T18:30:00")
    assert meta["creation_time_source"] == "container"
    assert meta["source_path"] == str(video.resolve())
    assert meta["config_hash"]
    assert meta["pipeline_version"]

    with wave.open(str(session.dir / "audio.wav")) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 48_000
        assert w.getsampwidth() == 2
        assert w.getnframes() / 48_000 == pytest.approx(2.0, abs=0.05)

    lines = (session.dir / "pipeline.log").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert any(e["event"] == "done" and e.get("stage") == "ingest" for e in events)
    assert not list(session.dir.glob(".*partial*"))


def test_ingest_is_cached_and_rerun_on_demand(make_video: MakeVideo, data_root: Path) -> None:
    video = make_video()
    session = _session(data_root, video)
    cfg = _config(data_root)
    assert run_pipeline(session, cfg, get_logger()) == ["ingest", "contacts", "pose"]
    assert run_pipeline(session, cfg, get_logger()) == []
    assert stage_status(session, STAGES[0], cfg) == "ok"

    both = ["ingest", "contacts", "pose"]
    assert run_pipeline(session, cfg, get_logger(), force=True) == both
    assert run_pipeline(session, cfg, get_logger(), from_stage=1) == both
    assert run_pipeline(session, cfg, get_logger(), from_stage=2) == ["contacts", "pose"]

    (session.dir / "audio.wav").unlink()
    assert stage_status(session, STAGES[0], cfg) == "stale"
    # ingest rewrites audio.wav, which makes contacts stale too.
    assert run_pipeline(session, cfg, get_logger()) == both

    # A newer source file (e.g. re-exported) invalidates everything downstream.
    now = time.time()
    for name in ("metadata.json", "audio.wav", "frame_times.parquet", "contacts.parquet"):
        os.utime(session.dir / name, (now - 100, now - 100))
    os.utime(video, (now - 50, now - 50))
    assert run_pipeline(session, cfg, get_logger()) == both

    # Changing an audio option reruns contacts and, since its output changed, pose.
    louder = cfg.model_copy(update={"audio": cfg.audio.model_copy(update={"onset_k": 9.0})})
    assert stage_status(session, STAGES[1], louder) == "stale"
    assert run_pipeline(session, louder, get_logger()) == ["contacts", "pose"]


def test_ingest_does_not_touch_source(make_video: MakeVideo, data_root: Path) -> None:
    video = make_video()
    before = (video.stat().st_mtime_ns, video.read_bytes())
    run_pipeline(_session(data_root, video), _config(data_root), get_logger())
    assert (video.stat().st_mtime_ns, video.read_bytes()) == before


def test_missing_audio_fails_with_clear_error(make_video: MakeVideo, data_root: Path) -> None:
    video = make_video("silent.mp4", audio=False)
    session = _session(data_root, video)
    with pytest.raises(StageError, match="no audio track") as info:
        run_pipeline(session, _config(data_root), get_logger())
    assert info.value.stage == "ingest"
    assert not (session.dir / "metadata.json").exists()
    assert session.read_stamp("ingest") is None


def test_vfr_is_detected(make_video: MakeVideo, data_root: Path) -> None:
    video = make_video("vfr.mp4", fps=60, vfr=True)
    session = _session(data_root, video)
    run_pipeline(session, _config(data_root), get_logger())
    meta = json.loads((session.dir / "metadata.json").read_text())
    assert meta["is_vfr"] is True


def test_missing_creation_time_falls_back_to_filesystem(
    make_video: MakeVideo, data_root: Path
) -> None:
    video = make_video("nodate.mp4", creation_time=None)
    session = _session(data_root, video)
    run_pipeline(session, _config(data_root), get_logger())
    meta = json.loads((session.dir / "metadata.json").read_text())
    assert meta["creation_time_source"] == "filesystem"


def test_corrupt_video_fails_ingest(tmp_path: Path, data_root: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 1024)
    session = _session(data_root, bad)
    with pytest.raises(StageError) as info:
        run_pipeline(session, _config(data_root), get_logger())
    assert info.value.stage == "ingest"


def test_low_frame_rate_is_processed_with_warning(make_video: MakeVideo, data_root: Path) -> None:
    video = make_video("24fps.mp4", fps=24)
    session = _session(data_root, video)
    assert run_pipeline(session, _config(data_root), get_logger()) == [
        "ingest",
        "contacts",
        "pose",
    ]
    meta = json.loads((session.dir / "metadata.json").read_text())
    assert meta["fps"] == pytest.approx(24)
    assert len(meta["warnings"]) == 1
    assert "low frame rate" in meta["warnings"][0]
    events = [json.loads(line) for line in (session.dir / "pipeline.log").read_text().splitlines()]
    assert any(e["level"] == "WARNING" and "low frame rate" in e["event"] for e in events)
