"""Frame PTS parsed from ffmpeg's showinfo log lines."""

from __future__ import annotations

import pytest

from tennis.util.frames import ShowinfoParser

PREFIX = "[Parsed_showinfo_1 @ 0x7f8]"


def test_exact_pts_past_1000_s_from_old_ffmpeg() -> None:
    # FFmpeg < 7 prints pts_time with %.6g, so 1002.084417 s shows as 1002.08.
    parser = ShowinfoParser()
    assert parser.feed(f"{PREFIX} config in time_base: 1/24000, frame_rate: 24000/1001") is None
    line = f"{PREFIX} n:  12 pts:24050026 pts_time:1002.08  duration:1001 fmt:yuv420p"
    assert parser.feed(line) == pytest.approx(24050026 / 24000, abs=1e-9)


def test_pts_time_when_time_base_unknown() -> None:
    parser = ShowinfoParser()
    assert parser.feed(f"{PREFIX} n:   0 pts:  3003 pts_time:0.100100 duration:1001") == 0.1001


def test_other_lines_are_not_frames() -> None:
    parser = ShowinfoParser()
    assert parser.feed(f"{PREFIX} color_range:tv color_space:bt709") is None
    assert parser.feed(f"{PREFIX} config out time_base: 0/0, frame_rate: 0/0") is None
    assert parser.time_base is None
