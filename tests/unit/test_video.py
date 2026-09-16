from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tennis.stages.ingest import frame_rate_warnings
from tennis.util.video import (
    FrameIntervals,
    VideoStream,
    is_variable_frame_rate,
    parse_creation_time,
    parse_rate,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("120/1", 120.0), ("120000/1001", 120000 / 1001), ("0/0", None), ("", None), ("x", None)],
)
def test_parse_rate(raw: str, expected: float | None) -> None:
    assert parse_rate(raw) == expected


def test_parse_creation_time() -> None:
    assert parse_creation_time("2026-09-20T18:30:00.000000Z") == datetime(
        2026, 9, 20, 18, 30, tzinfo=UTC
    )
    assert parse_creation_time("2026-09-20 18:30:00") == datetime(2026, 9, 20, 18, 30, tzinfo=UTC)
    assert parse_creation_time("1970-01-01T00:00:00Z") is None
    assert parse_creation_time("garbage") is None
    assert parse_creation_time(None) is None


def _stream(nominal: float | None, avg: float | None) -> VideoStream:
    return VideoStream(0, "h264", 1, 1, 0, nominal, avg, 0.0, None, None, None)


def _intervals(median: float, p01: float, p99: float) -> FrameIntervals:
    return FrameIntervals(100, median, p01, p99)


def test_cfr_with_millisecond_rounding_is_not_vfr() -> None:
    assert not is_variable_frame_rate(_stream(120, 120), _intervals(8.0, 8.0, 9.0))


def test_uneven_intervals_are_vfr() -> None:
    assert is_variable_frame_rate(_stream(120, 120), _intervals(8.3, 8.3, 16.7))


def test_rate_mismatch_is_vfr() -> None:
    assert is_variable_frame_rate(_stream(120, 100), _intervals(8.3, 8.3, 8.3))


@pytest.mark.parametrize(
    ("fps", "fragment"),
    [
        (23.976, "low frame rate 23.98 fps"),
        (500.0, "high frame rate"),
        (None, "unknown"),
    ],
)
def test_frame_rate_warnings(fps: float | None, fragment: str) -> None:
    (warning,) = frame_rate_warnings(fps)
    assert fragment in warning


@pytest.mark.parametrize("fps", [30.0, 60.0, 119.88, 240.0])
def test_supported_frame_rates_do_not_warn(fps: float) -> None:
    assert frame_rate_warnings(fps) == []
