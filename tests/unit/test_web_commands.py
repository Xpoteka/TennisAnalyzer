"""The web UI's command catalogue: every command is real, and forms cannot inject arguments."""

from __future__ import annotations

from pathlib import Path

import pytest

from tennis.cli import app
from tennis.errors import UserError
from tennis.web.commands import COMMANDS, COMMANDS_BY_NAME, GROUPS, build_argv


def _cli_command_names() -> set[str]:
    return {info.name or info.callback.__name__ for info in app.registered_commands}


def test_every_offered_command_exists_in_the_cli() -> None:
    assert {c.name for c in COMMANDS} <= _cli_command_names()


def test_every_command_is_in_a_known_group() -> None:
    for command in COMMANDS:
        assert command.group in GROUPS, command.name


def test_field_flags_look_like_options() -> None:
    for command in COMMANDS:
        for field in command.fields:
            assert field.flag is None or field.flag.startswith("--"), (command.name, field.name)
            if field.kind == "bool":
                assert field.flag is not None, f"{command.name}.{field.name}"


def test_build_argv_puts_positionals_first_and_skips_empty_options() -> None:
    argv = build_argv(
        COMMANDS_BY_NAME["inspect"],
        {"session_id": "2026-01-01_evening", "swing_id": "42"},
        None,
    )
    assert argv == ["inspect", "2026-01-01_evening", "42", "--no-open"]


def test_build_argv_appends_the_servers_config() -> None:
    argv = build_argv(
        COMMANDS_BY_NAME["report"], {"session_id": "s", "no_clips": True}, Path("config.yaml")
    )
    assert argv == ["report", "s", "--no-clips", "--config", "config.yaml"]


def test_build_argv_ignores_fields_the_command_does_not_declare() -> None:
    argv = build_argv(COMMANDS_BY_NAME["players"], {"session_id": "s", "rm": "-rf"}, None)
    assert argv == ["players", "s"]


def test_build_argv_rejects_a_value_that_would_become_a_flag() -> None:
    with pytest.raises(UserError, match="must not start with"):
        build_argv(COMMANDS_BY_NAME["players"], {"session_id": "--force"}, None)


def test_build_argv_requires_the_required_fields() -> None:
    with pytest.raises(UserError, match="required"):
        build_argv(COMMANDS_BY_NAME["report"], {}, None)


def test_build_argv_checks_numbers_and_choices() -> None:
    with pytest.raises(UserError, match="must be a number"):
        build_argv(COMMANDS_BY_NAME["players"], {"session_id": "s", "count": "lots"}, None)
    with pytest.raises(UserError, match="must be one of"):
        build_argv(
            COMMANDS_BY_NAME["tune-contacts"],
            {"session_id": "s", "labels": "l.csv", "target": "everyone"},
            None,
        )


def test_a_false_bool_adds_nothing() -> None:
    assert build_argv(COMMANDS_BY_NAME["report"], {"session_id": "s", "no_clips": False}, None) == [
        "report",
        "s",
    ]
