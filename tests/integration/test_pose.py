"""Frame decoding and the pose stage, with a model-free fake backend."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.io import wavfile

from tennis.config import Config, PathsConfig, PoseConfig
from tennis.pose_backends import Image, PersonPose, register_backend
from tennis.session import Session, create_or_reuse_session
from tennis.stages import run_pipeline
from tennis.stages.pose import KEYPOINT_COLUMNS, frame_size, should_refine
from tennis.util.frames import FrameReader, Window
from tennis.util.io import read_parquet_provenance
from tennis.util.log import get_logger
from tennis.util.video import frame_intervals, read_frame_pts
from tests.conftest import COUNTER_PERIOD, MakeVideo, frame_number, needs_ffmpeg
from tests.synth import render

pytestmark = needs_ffmpeg
FPS = 30


class FakeBackend:
    """A near-court player plus a far one. Keypoint x holds the decoded frame number."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def infer(self, frames: list[Image]) -> list[list[PersonPose]]:
        self.calls.append(len(frames))
        out = []
        for image in frames:
            h, w = image.shape[:2]
            n = frame_number(image)
            kp = np.zeros((17, 3), np.float32)
            kp[:, 0] = n
            kp[:, 1] = np.arange(17)
            kp[:, 2] = 0.9
            if min(h, w) < 40:  # a crop: return one person filling it
                out.append([PersonPose((0.0, 0.0, float(w), float(h)), 0.9, kp)])
                continue
            near = PersonPose((10.0 + n % 3, 20.0, 30.0, 46.0), 0.9, kp)
            far = PersonPose((40.0, 2.0, 44.0, 10.0), 0.8, np.zeros((17, 3), np.float32))
            out.append([far, near])
        return out


@pytest.fixture
def fake_backend() -> FakeBackend:
    backend = FakeBackend()
    register_backend("fake", lambda cfg, path, device: backend)
    return backend


def test_frame_reader_matches_frame_numbers(make_video: MakeVideo) -> None:
    video = make_video("counter.mp4", seconds=4.0, fps=FPS, counter=True, audio=False)
    pts = read_frame_pts(video, 0)
    reader = FrameReader(video, pts, 64, 48)
    windows = [Window(0, 0.5, 0.9), Window(1, 1.0, 1.2), Window(2, 3.0, 3.5)]
    frames = list(reader.read(windows))
    expected = [i for i, t in enumerate(pts) for w in windows if w.start <= t <= w.end]
    assert [f.index for f in frames] == expected
    for f in frames:
        assert frame_number(f.image) == f.index % COUNTER_PERIOD
        assert f.pts == pts[f.index]
        w = windows[f.window_id]
        assert w.start - 1e-3 <= f.pts <= w.end + 1e-3


def test_frame_reader_with_variable_frame_rate(make_video: MakeVideo) -> None:
    video = make_video("counter_vfr.mp4", seconds=4.0, fps=FPS, counter=True, vfr=True,
                       audio=False)  # fmt: skip
    pts = read_frame_pts(video, 0)
    assert frame_intervals(pts).spread > 0.5
    frames = list(FrameReader(video, pts, 64, 48).read([Window(0, 1.0, 2.0)]))
    assert len(frames) == np.sum((pts >= 1.0) & (pts <= 2.0))
    for f in frames:
        # Source frame n was shown at n / fps; dropped frames leave gaps in n.
        assert frame_number(f.image) == round(f.pts * FPS) % COUNTER_PERIOD


def _session_with_clicks(
    make_video: MakeVideo, tmp_path: Path, data_root: Path, times: list[float]
) -> Session:

    wav = tmp_path / "clicks.wav"
    wavfile.write(wav, 48_000, render(8.0, times, [-12.0] * len(times)))
    video = make_video(f"counter_clicks_{len(times)}.mp4", seconds=8.0, fps=FPS, counter=True,
                       audio_wav=wav)  # fmt: skip
    return create_or_reuse_session(data_root, video, datetime(2026, 9, 20, tzinfo=UTC), "p")


def test_pose_stage(
    make_video: MakeVideo, tmp_path: Path, data_root: Path, fake_backend: FakeBackend
) -> None:
    session = _session_with_clicks(make_video, tmp_path, data_root, [2.0, 2.4, 6.0])
    config = Config(
        paths=PathsConfig(data_root=data_root),
        pose=PoseConfig(backend="fake", contacts="all", batch_size=7, crop_refine="never"),
    )
    ran = run_pipeline(session, config, get_logger())
    assert ran == ["ingest", "contacts", "pose"]

    path = session.dir / "keypoints.parquet"
    table = pq.read_table(path)
    meta = read_parquet_provenance(path)
    assert meta["stage"] == "pose" and meta["backend"] == "fake"
    assert (meta["frame_width"], meta["frame_height"]) == ("64", "48")
    for column in ("frame_idx", "t_video", "window_id", "detected", "track_reset", "n_persons",
                   "bbox_x1", "bbox_conf", "r_wrist_x", "r_wrist_y", "r_wrist_conf"):  # fmt: skip
        assert column in table.column_names
    assert len(KEYPOINT_COLUMNS) == 51

    pts = pq.read_table(session.dir / "frame_times.parquet").column("pts").to_numpy()
    d = table.to_pydict()
    # Windows [1.0, 2.9] (two merged contacts) and [5.0, 6.5].
    expected = [i for i, t in enumerate(pts) if 1.0 - 0.01 <= t <= 2.91 or 4.99 <= t <= 6.51]
    assert abs(len(d["frame_idx"]) - len(expected)) <= 2  # click timing shifts window edges
    assert d["frame_idx"] == sorted(d["frame_idx"])
    assert set(d["window_id"]) == {0, 1}
    assert all(d["detected"]) and not any(d["track_reset"])
    assert set(d["n_persons"]) == {2}
    # The near player was chosen, and each row's keypoints came from its own frame.
    assert all(x2 == 30.0 for x2 in d["bbox_x2"])
    assert d["nose_x"] == [float(i % COUNTER_PERIOD) for i in d["frame_idx"]]
    assert d["r_ankle_y"][0] == 16.0
    assert max(fake_backend.calls) == 7

    # Cached on rerun; a pose option change reruns only pose.
    assert run_pipeline(session, config, get_logger()) == []
    changed = config.model_copy(
        update={"pose": config.pose.model_copy(update={"crop_refine": "always"})}
    )
    assert run_pipeline(session, changed, get_logger()) == ["pose"]
    refined = pq.read_table(path).to_pydict()
    assert read_parquet_provenance(path)["crop_refine"] == "true"
    # The crop pass saw the same frame; its keypoints are shifted by the crop's origin,
    # which is the box padded by 20% of its width (box x2 is always 30).
    for n, x1, nose in zip(refined["frame_idx"], refined["bbox_x1"], refined["nose_x"],
                           strict=True):  # fmt: skip
        assert nose == n % COUNTER_PERIOD + int(x1 - 0.2 * (30 - x1))


def test_pose_stage_self_contacts_only(
    make_video: MakeVideo, tmp_path: Path, data_root: Path, fake_backend: FakeBackend
) -> None:
    # All clicks are equally loud, so none is 6 dB above the median: no windows.
    session = _session_with_clicks(make_video, tmp_path, data_root, [2.0, 6.0])
    config = Config(paths=PathsConfig(data_root=data_root), pose=PoseConfig(backend="fake"))
    assert run_pipeline(session, config, get_logger()) == ["ingest", "contacts", "pose"]
    table = pq.read_table(session.dir / "keypoints.parquet")
    assert table.num_rows == 0
    assert table.schema.field("r_wrist_x").type == pa.float32()


def test_unknown_backend_fails_the_stage(make_video: MakeVideo, tmp_path: Path,
                                         data_root: Path) -> None:  # fmt: skip
    from tennis.errors import StageError

    session = _session_with_clicks(make_video, tmp_path, data_root, [2.0])
    config = Config(paths=PathsConfig(data_root=data_root), pose=PoseConfig(backend="nope"))
    with pytest.raises(StageError, match="unknown pose backend") as info:
        run_pipeline(session, config, get_logger())
    assert info.value.stage == "pose"


def test_frame_size_and_refine_rule() -> None:
    meta = {"video": {"width": 1920, "height": 1080, "rotation_deg": 90}}
    assert frame_size(meta) == (1080, 1920)
    assert frame_size({"video": {"width": 1920, "height": 1080}}) == (1920, 1080)
    assert should_refine("auto", 3840, 2160, 640)
    assert not should_refine("auto", 1280, 720, 640)
    assert should_refine("always", 10, 10, 640)
    assert not should_refine("never", 9999, 9999, 640)


def test_pose_preview(
    make_video: MakeVideo, tmp_path: Path, data_root: Path, fake_backend: FakeBackend
) -> None:
    from tennis.review import render_pose_preview
    from tennis.util.video import probe

    session = _session_with_clicks(make_video, tmp_path, data_root, [2.0, 6.0])
    config = Config(
        paths=PathsConfig(data_root=data_root),
        pose=PoseConfig(backend="fake", contacts="all", crop_refine="never"),
    )
    run_pipeline(session, config, get_logger())
    out, summaries = render_pose_preview(session, config, count=1, seed=3)
    assert out == session.dir / "debug" / "pose_preview.mp4"
    assert len(summaries) == 1 and summaries[0].detected_ratio == 1.0
    info = probe(out)
    assert info.video is not None and info.video.codec == "h264"
    assert info.video.nb_frames == summaries[0].frames
    assert out.with_suffix(".csv").read_text().startswith("window_id,start_s")
