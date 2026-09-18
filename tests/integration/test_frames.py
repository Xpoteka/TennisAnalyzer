"""FrameReader against a video whose brightness encodes the frame number."""

from __future__ import annotations

import numpy as np

from tennis.util.frames import FrameReader, Window
from tennis.util.video import read_frame_pts
from tests.conftest import MakeVideo, frame_number, needs_ffmpeg

pytestmark = needs_ffmpeg


def test_every_nth_frame_with_exact_indices(make_video: MakeVideo) -> None:
    video = make_video("counter.mp4", seconds=2.0, fps=30, counter=True)
    pts = read_frame_pts(video, 0)
    reader = FrameReader(video, pts, 64, 48, every=3)
    frames = list(reader.read([Window(0, float(pts[0]), float(pts[-1]))]))
    assert [f.index for f in frames] == list(range(0, len(pts), 3))
    for f in frames:
        assert frame_number(f.image) == f.index % 110
        assert f.pts == pts[f.index]


def test_scaled_frames(make_video: MakeVideo) -> None:
    video = make_video("big.mp4", seconds=0.5, fps=30)
    pts = read_frame_pts(video, 0)
    reader = FrameReader(video, pts, 320, 240, out_width=160)
    assert (reader.width, reader.height) == (160, 120)
    assert reader.scale == 0.5
    frames = list(reader.read([Window(0, float(pts[0]), float(pts[-1]))]))
    assert len(frames) == len(pts)
    assert frames[0].image.shape == (120, 160, 3)
    assert np.asarray(frames[0].image).mean() > 0
