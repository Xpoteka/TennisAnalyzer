"""The SQLite database at ``<data_root>/tennis.db``.

The web server and the analysis subprocess write to it at the same time, so it runs in WAL
mode with a busy timeout. Schema changes are additive: a new table is created and a new
nullable column is added on open, so an existing database keeps its data across updates.

JSON columns are not tracked for in-place changes: assign a new value (``row.summary =
{**row.summary, "x": 1}``) instead of mutating the old one.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event, inspect, text
from sqlalchemy.engine import Engine
from sqlmodel import Session as DbSession
from sqlmodel import SQLModel, create_engine

from tennis.db import models  # noqa: F401  (registers the tables)

DB_NAME = "tennis.db"

_engines: dict[Path, Engine] = {}
_lock = threading.Lock()


def db_path(data_root: Path) -> Path:
    return data_root / DB_NAME


def get_engine(data_root: Path) -> Engine:
    """One engine per database file, created (with its tables) on first use."""
    path = db_path(data_root).resolve()
    with _lock:
        engine = _engines.get(path)
        if engine is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(
                f"sqlite:///{path}",
                connect_args={"check_same_thread": False, "timeout": 30},
            )
            event.listen(engine, "connect", _on_connect)
            SQLModel.metadata.create_all(engine)
            _add_missing_columns(engine)
            _engines[path] = engine
        return engine


def dispose_engines() -> None:
    """Close every engine (tests use a fresh data root each)."""
    with _lock:
        for engine in _engines.values():
            engine.dispose()
        _engines.clear()


@contextmanager
def session_scope(data_root: Path) -> Iterator[DbSession]:
    """A database session that commits on success and rolls back on error."""
    with DbSession(get_engine(data_root), expire_on_commit=False) as db:
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise


def _on_connect(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _add_missing_columns(engine: Engine) -> None:
    """Add columns that exist in the models but not yet in the database file."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                ddl = column.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl}'))
