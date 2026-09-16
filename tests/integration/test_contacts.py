"""Contact detection end to end: synthetic clicks muxed into a video, then the tuning CLI."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.io import wavfile

from tennis.cli import main
from tennis.config import Config, PathsConfig
from tennis.session import Session, create_or_reuse_session
from tennis.stages import run_pipeline
from tennis.util.io import read_parquet_provenance
from tennis.util.log import get_logger
from tests.conftest import MakeVideo, needs_ffmpeg
from tests.synth import render

pytestmark = needs_ffmpeg

SECONDS = 24.0
FPS = 30
# A rally: own hits (loud) alternate with the partner's (quiet), with bounces and footsteps
# in between. As in real sessions, own hits are a minority of all onsets, which the
# median-based first-pass rule relies on.
OWN = [1.50, 4.21, 7.03, 9.87, 12.62, 15.40, 18.19, 21.00]
OTHER = [2.85, 5.62, 8.44, 11.25, 14.01, 16.79, 19.60]
NOISE = [0.60, 2.10, 3.50, 4.90, 6.30, 7.70, 9.10, 10.50, 13.30, 17.50, 20.30, 22.40]
LEVELS = {
    **dict.fromkeys(OWN, -12.0),
    **dict.fromkeys(OTHER, -32.0),
    **dict.fromkeys(NOISE, -36.0),
}


@pytest.fixture(scope="module")
def click_video(make_video: MakeVideo, tmp_path_factory: pytest.TempPathFactory) -> Path:
    times = sorted(LEVELS)
    samples = render(SECONDS, times, [LEVELS[t] for t in times], noise_db=-58.0)
    wav = tmp_path_factory.mktemp("audio") / "clicks.wav"
    wavfile.write(wav, 48_000, samples)
    return make_video("clicks.mp4", seconds=SECONDS, fps=FPS, audio_wav=wav)


def _process(data_root: Path, video: Path) -> Session:
    session = create_or_reuse_session(data_root, video, datetime(2026, 9, 20, 18, tzinfo=UTC), "c")
    run_pipeline(session, Config(paths=PathsConfig(data_root=data_root)), get_logger())
    return session


def test_contacts_parquet(click_video: Path, data_root: Path) -> None:
    session = _process(data_root, click_video)
    path = session.dir / "contacts.parquet"
    table = pq.read_table(path)
    assert table.column_names == [
        "contact_id", "t_audio", "frame_idx", "t_video", "onset_strength", "threshold",
        "peak_db", "prominence_db", "is_self_audio", "is_self_confirmed",
    ]  # fmt: skip
    assert read_parquet_provenance(path)["stage"] == "contacts"
    df = table.to_pydict()

    t_audio = np.array(df["t_audio"])
    assert len(t_audio) == len(LEVELS)
    # AAC encoding adds a little smear; timing still well inside the +-40 ms target.
    assert np.abs(t_audio - np.array(sorted(LEVELS))).max() < 0.010

    is_self = np.array(df["is_self_audio"])
    assert t_audio[is_self] == pytest.approx(OWN, abs=0.010)
    assert all(v is None for v in df["is_self_confirmed"])

    frames = pq.read_table(session.dir / "frame_times.parquet").column("pts").to_numpy()
    t_video = np.array(df["t_video"])
    assert np.array_equal(t_video, frames[np.array(df["frame_idx"])])
    assert np.abs(t_video - t_audio).max() <= 0.5 / FPS + 1e-6
    assert df["contact_id"] == list(range(len(t_audio)))


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    with pytest.raises(SystemExit) as info:
        main(argv)
    out, err = capsys.readouterr()
    return int(info.value.code or 0), out, err


def test_tune_contacts_cli(
    click_video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("paths:\n  data_root: ./data\n")
    cfg = ["--config", str(config)]
    labels = tmp_path / "contacts_c.csv"
    labels.write_text("t,note\n" + "".join(f"0:{t:06.3f},own\n" for t in OWN))

    code, _, err = _run(["tune-contacts", "c", "--labels", str(labels), *cfg], capsys)
    assert code == 1 and "unknown session" in err

    code, _, err = _run(["process", str(click_video), "--session-id", "c", *cfg], capsys)
    assert code == 0, err

    report = tmp_path / "docs" / "M2.md"
    code, out, err = _run(
        [
            "tune-contacts", "c", "--labels", str(labels), *cfg,
            "--k", "4,6", "--cutoff", "800", "--db", "3,6,30", "--report", str(report),
        ],
        capsys,
    )  # fmt: skip
    assert code == 0, err
    assert "meets the 0.90/0.90 target" in out
    assert "current config" in out
    assert "own_hit_db_threshold: " in out
    rows = [line.split() for line in out.splitlines() if line.strip().startswith("800 ")]
    assert len(rows) == 6
    best = rows[0]
    assert (best[3], best[4]) == ("1.000", "1.000")
    too_strict = next(r for r in rows if r[2] == "30")
    assert too_strict[4] == "0.000"
    assert report.read_text().startswith("# Contact detection tuning: c")

    code, _, err = _run(["tune-contacts", "c", "--labels", str(labels), "--k", "x", *cfg], capsys)
    assert code == 1 and "--k" in err


def test_tune_all_hits_with_coarse_labels(
    click_video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Whole-second labels for both players, with a clock 1 s behind the video."""
    config = tmp_path / "config.yaml"
    config.write_text("paths:\n  data_root: ./data\n")
    cfg = ["--config", str(config)]
    code, _, err = _run(["process", str(click_video), "--session-id", "c", *cfg], capsys)
    assert code == 0, err

    hits = sorted(OWN + OTHER)
    labels = tmp_path / "all.csv"
    labels.write_text("t\n" + "".join(f"{int(t) - 1}\n" for t in hits))
    segments = tmp_path / "rallies.csv"
    # On the labels' clock, so 1-22 s of video: every hit is inside, and the noise events at
    # 0.6 s and 22.4 s are outside.
    segments.write_text("start,end\n0.0,21.0\n")

    code, out, err = _run(
        [
            "tune-contacts", "c", "--labels", str(labels), *cfg, "--target", "any",
            "--label-offset", "1", "--label-resolution", "1", "--segments", str(segments),
            "--k", "6", "--cutoff", "800",
        ],
        capsys,
    )  # fmt: skip
    assert code == 0, err
    assert "15 labeled hits (all players)" in out
    assert "label window [t+1, t+2] s" in out
    row = next(line.split() for line in out.splitlines() if line.strip().startswith("800 "))
    # All 15 hits found; the 10 noise clicks inside the rally are the false positives.
    assert row[2] == "-"
    assert row[4] == "1.000"
    assert row[3] == f"{15 / 25:.3f}"
    assert "own_hit_db_threshold" not in out

    code, _, err = _run(
        ["tune-contacts", "c", "--labels", str(labels), *cfg, "--segments", str(segments),
         "--start", "3"],
        capsys,
    )  # fmt: skip
    assert code == 1 and "not both" in err
