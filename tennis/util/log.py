"""Structured logging: JSON lines to the session's pipeline.log, short lines to stderr."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER_NAME = "tennis"


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields: dict[str, Any] = getattr(record, "fields", {})
        stage = fields.get("stage")
        prefix = f"[{stage}] " if stage else ""
        extras = " ".join(f"{k}={v}" for k, v in fields.items() if k not in ("stage", "session"))
        level = "" if record.levelno == logging.INFO else f"{record.levelname}: "
        return f"{level}{prefix}{record.getMessage()}" + (f"  ({extras})" if extras else "")


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to the current ``sys.stderr`` at emit time, so it survives stream swaps."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self) -> Any:
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        pass


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not any(getattr(h, "_tennis_console", False) for h in logger.handlers):
        handler = _StderrHandler()
        handler.setFormatter(ConsoleFormatter())
        handler._tennis_console = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log(logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Log an event with structured fields."""
    logger.log(level, event, extra={"fields": fields})


@contextmanager
def session_log(logger: logging.Logger, path: Path) -> Iterator[None]:
    """Append JSON-lines records to ``path`` for the duration of the block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonLinesFormatter())
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
