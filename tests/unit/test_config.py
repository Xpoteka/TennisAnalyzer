from __future__ import annotations

from pathlib import Path

import pytest

from tennis.config import Config, load_config
from tennis.errors import ConfigError

REPO = Path(__file__).resolve().parents[2]


def test_example_config_matches_defaults() -> None:
    cfg = load_config(REPO / "config.example.yaml")
    defaults = load_config(None, cwd=REPO / "tennis")  # no config.yaml there
    assert cfg.model_dump(exclude={"paths"}) == defaults.model_dump(exclude={"paths"})
    assert cfg.paths.data_root == REPO / "data"


def test_missing_file_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_default_file_in_cwd_is_used(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("player:\n  handedness: left\n")
    assert load_config(None, cwd=tmp_path).player.handedness == "left"


def test_empty_file_gives_defaults(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("")
    assert load_config(p).player.handedness == "right"


@pytest.mark.parametrize(
    ("yaml_text", "fragment"),
    [
        ("player:\n  handedness: both\n", "player.handedness"),
        ("audio:\n  highpass_hz: -1\n", "audio.highpass_hz"),
        ("audoi:\n  highpass_hz: 1\n", "audoi"),
        ("cleaning:\n  savgol: {window: 8}\n", "odd"),
        ("labels:\n  vocabulary: {good: [nice], bad: [nice]}\n", "mapped to both"),
        ("labels:\n  vocabulary: {good: [yes]}\n", "labels.vocabulary.good"),
        ("- a\n- b\n", "mapping"),
        ("player: [\n", "invalid YAML"),
    ],
)
def test_invalid_configs(tmp_path: Path, yaml_text: str, fragment: str) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ConfigError, match=fragment):
        load_config(p)


def test_relative_data_root_resolves_against_config_dir(tmp_path: Path) -> None:
    sub = tmp_path / "conf"
    sub.mkdir()
    (sub / "c.yaml").write_text("paths:\n  data_root: ../store\n")
    assert load_config(sub / "c.yaml").paths.data_root == tmp_path / "store"


def test_section_hash_only_depends_on_named_sections() -> None:
    base = Config()
    other = Config.model_validate({"pose": {"batch_size": 4}})
    assert base.section_hash("audio") == other.section_hash("audio")
    assert base.section_hash("pose") != other.section_hash("pose")
    assert base.section_hash() == other.section_hash()
    moved = Config.model_validate({"paths": {"data_root": "/elsewhere"}})
    assert base.section_hash("audio", "pose") == moved.section_hash("audio", "pose")
