"""CLI behaviour and exit codes (spec section 7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tennis.cli import main
from tests.conftest import MakeVideo, implemented_stages, needs_ffmpeg


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    with pytest.raises(SystemExit) as info:
        main(argv)
    out, err = capsys.readouterr()
    code = info.value.code
    return (code if isinstance(code, int) else 1), out, err


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text("paths:\n  data_root: ./data\n")
    return p


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(["--version"], capsys)
    assert code == 0
    assert "tennis-analyzer" in out


def test_help_lists_all_commands(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(["--help"], capsys)
    assert code == 0
    for cmd in (
        "process",
        "report",
        "trends",
        "tune-contacts",
        "eval-classifier",
        "list",
        "inspect",
    ):
        assert cmd in out


@pytest.mark.parametrize(
    "argv",
    [
        ["process"],  # missing argument
        ["process", "x.mp4", "--bogus"],  # unknown option
        ["nope"],  # unknown command
        ["process", "/does/not/exist.mp4"],
    ],
)
def test_user_errors_exit_1(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    code, _, _ = _run(argv, capsys)
    assert code == 1


def test_bad_config_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("player:\n  handedness: both\n")
    code, _, err = _run(["list", "--config", str(cfg)], capsys)
    assert code == 1
    assert "player.handedness" in err


def test_unimplemented_command_exits_1(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = _run(["trends", "--config", str(config_file)], capsys)
    assert code == 1
    assert "M7" in err


def test_list_empty(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(["list", "--config", str(config_file)], capsys)
    assert code == 0
    assert "no sessions" in out


@needs_ffmpeg
def test_process_then_list(
    make_video: MakeVideo, config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video = make_video()
    code, _, err = _run(["process", str(video), "--config", str(config_file)], capsys)
    assert code == 0, err
    assert "session 2026-09-20_" in err
    assert f"done: {', '.join(implemented_stages())}" in err

    sessions = list((config_file.parent / "data" / "sessions").iterdir())
    assert len(sessions) == 1
    assert json.loads((sessions[0] / "metadata.json").read_text())["fps"] == pytest.approx(120)

    code, out, _ = _run(["list", "--config", str(config_file)], capsys)
    assert code == 0
    header, row = out.strip().splitlines()
    assert header.split()[:3] == ["session", "ingest", "contacts"]
    n = len(implemented_stages())
    assert row.split()[1 : n + 2] == ["ok"] * n + ["n/a"]

    code, _, err = _run(["process", str(video), "--config", str(config_file)], capsys)
    assert code == 0
    assert "everything up to date" in err

    code, _, err = _run(
        ["process", str(video), "--config", str(config_file), "--from-stage", str(n + 1)],
        capsys,
    )
    assert code == 1
    assert "not implemented yet" in err


@needs_ffmpeg
def test_stage_failure_exits_2(
    make_video: MakeVideo, config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video = make_video("silent.mp4", audio=False)
    code, _, err = _run(["process", str(video), "--config", str(config_file)], capsys)
    assert code == 2
    assert "stage 'ingest' failed" in err
