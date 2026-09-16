from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tennis.errors import UserError
from tennis.validation import match_events, parse_time, read_segments, read_time_labels


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("12.5", 12.5), ("1:02.345", 62.345), ("0:01:02.5", 62.5), ("1:00:00", 3600.0), (" 3 ", 3.0)],
)
def test_parse_time(text: str, seconds: float) -> None:
    assert parse_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "abc", "-1", "1:2:3:4", "nan"])
def test_parse_time_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        parse_time(text)


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "labels.csv"
    p.write_text(text)
    return p


def test_labels_with_named_column(tmp_path: Path) -> None:
    p = _write(tmp_path, "note,t\nfh,2.5\n# skipped\n\nbh,1:00.0\n")
    assert read_time_labels(p).tolist() == [2.5, 60.0]


def test_labels_without_header(tmp_path: Path) -> None:
    assert read_time_labels(_write(tmp_path, "3.0\n1.0\n")).tolist() == [1.0, 3.0]


def test_labels_with_unknown_header(tmp_path: Path) -> None:
    assert read_time_labels(_write(tmp_path, "when\n1.5\n")).tolist() == [1.5]


def test_labels_errors(tmp_path: Path) -> None:
    with pytest.raises(UserError, match="not found"):
        read_time_labels(tmp_path / "missing.csv")
    with pytest.raises(UserError, match="no labels"):
        read_time_labels(_write(tmp_path, "# nothing\n"))
    with pytest.raises(UserError, match="bad row"):
        read_time_labels(_write(tmp_path, "t\n1.0\noops\n"))


def test_match_events_one_to_one() -> None:
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.01, 1.02, 2.05, 3.0, 9.0])
    r = match_events(pred, truth, 0.04)
    assert (r.true_positives, r.false_positives, r.false_negatives) == (2, 3, 2)
    assert r.precision == pytest.approx(2 / 5)
    assert r.recall == pytest.approx(2 / 4)
    assert sorted(r.offsets_s.round(3).tolist()) == [0.0, 0.01]


def test_match_events_prefers_closest_pairs() -> None:
    # Matching in label order would pair 1.03 with 1.0; the closer label is 1.035.
    r = match_events(np.array([1.03]), np.array([1.0, 1.035]), 0.04)
    assert r.true_positives == 1
    assert r.offsets_s[0] == pytest.approx(-0.005)


def test_match_events_empty() -> None:
    r = match_events(np.array([]), np.array([1.0]), 0.04)
    assert (r.precision, r.recall, r.f1) == (0.0, 0.0, 0.0)


def test_match_events_with_label_window() -> None:
    # Whole-second labels whose clock is 1 s behind: t=5 means the event is in [6, 7].
    truth = np.array([5.0, 7.0])
    pred = np.array([6.4, 6.9, 8.95, 9.5])
    r = match_events(pred, truth, 0.04, resolution_s=1.0, offset_s=1.0)
    assert (r.true_positives, r.false_positives, r.false_negatives) == (2, 2, 0)
    assert sorted(r.offsets_s.round(2).tolist()) == [0.4, 0.95]


def test_read_segments(tmp_path: Path) -> None:
    p = tmp_path / "seg.csv"
    p.write_text("start,end\n0:21,0:28\n38,40.5\n")
    assert read_segments(p) == [(21.0, 28.0), (38.0, 40.5)]
    p.write_text("start,end\n5,4\n")
    with pytest.raises(UserError, match="ends before"):
        read_segments(p)
    p.write_text("start,end\n")
    with pytest.raises(UserError, match="no segments"):
        read_segments(p)
