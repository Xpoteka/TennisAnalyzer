"""Exception types that map to the CLI exit codes (spec section 7)."""

from __future__ import annotations


class TennisError(Exception):
    """Base class for all errors the CLI reports without a traceback."""

    exit_code = 1


class UserError(TennisError):
    """Bad arguments, missing files, or a request the pipeline cannot serve. Exit code 1."""

    exit_code = 1


class ConfigError(UserError):
    """The configuration file is missing, malformed, or fails validation. Exit code 1."""


class StageError(TennisError):
    """A pipeline stage failed. Exit code 2; the stage name is always reported."""

    exit_code = 2

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"stage '{stage}' failed: {message}")
        self.stage = stage
