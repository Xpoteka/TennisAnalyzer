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


def test_example_config_lists_every_option() -> None:
    """Spec section 12 asks for every option to be in the example file, commented."""
    import yaml

    raw = yaml.safe_load((REPO / "config.example.yaml").read_text())
    defaults = Config().model_dump(mode="json")

    def missing(actual: object, expected: object, prefix: str = "") -> list[str]:
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return []
        out = []
        for key, value in expected.items():
            if key not in actual:
                out.append(f"{prefix}{key}")
                continue
            out += missing(actual[key], value, f"{prefix}{key}.")
        return out

    assert missing(raw, defaults) == []


def test_missing_file_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_default_file_in_cwd_is_used(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("pose:\n  device: cpu\n")
    assert load_config(None, cwd=tmp_path).pose.device == "cpu"


def test_empty_file_gives_defaults(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("")
    assert load_config(p) == Config(paths=load_config(p).paths)


@pytest.mark.parametrize(
    ("yaml_text", "fragment"),
    [
        ("pose:\n  device: tpu\n", "pose.device"),
        ("audio:\n  highpass_hz: -1\n", "audio.highpass_hz"),
        ("audoi:\n  highpass_hz: 1\n", "audoi"),
        ("court:\n  min_quality: 2\n", "court.min_quality"),
        ("- a\n- b\n", "mapping"),
        ("pose: [\n", "invalid YAML"),
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
    # Field-level keys only depend on that field.
    k9 = Config.model_validate({"audio": {"onset_k": 9}})
    other_audio = Config.model_validate({"audio": {"min_prominence_db": 9}})
    assert base.section_hash("audio.onset_k") != k9.section_hash("audio.onset_k")
    assert base.section_hash("audio.onset_k") == other_audio.section_hash("audio.onset_k")
    assert base.section_hash("audio") != other_audio.section_hash("audio")
    # A whole section plus one of its fields hashes like the section alone.
    assert base.section_hash("audio", "audio.onset_k") == base.section_hash("audio")
    with pytest.raises(KeyError):
        base.section_hash("audio.nope")
    with pytest.raises(KeyError):
        base.section_hash("nope")
    moved = Config.model_validate({"paths": {"data_root": "/elsewhere"}})
    assert base.section_hash("audio", "pose") == moved.section_hash("audio", "pose")
