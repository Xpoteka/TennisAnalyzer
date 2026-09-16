"""Atomic file writes and Parquet provenance metadata.

Outputs are written to a temporary sibling and renamed into place, so a crashed stage never
leaves a half-written file that the cache would mistake for a finished one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tennis import __version__

META_PREFIX = "tennis."


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary path with the same suffix; move it to ``path`` on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.partial-{os.getpid()}{path.suffix}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, data: Any) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def provenance(stage: str, config_hash: str, schema_version: int) -> dict[str, str]:
    return {
        "pipeline_version": __version__,
        "config_hash": config_hash,
        "stage": stage,
        "schema_version": str(schema_version),
    }


def write_parquet(
    table: pa.Table, path: Path, *, stage: str, config_hash: str, schema_version: int
) -> None:
    """Write ``table`` with pipeline version, config hash and schema version in its metadata."""
    meta = dict(table.schema.metadata or {})
    for key, value in provenance(stage, config_hash, schema_version).items():
        meta[f"{META_PREFIX}{key}".encode()] = value.encode()
    table = table.replace_schema_metadata(meta)
    with atomic_path(path) as tmp:
        pq.write_table(table, tmp)


def read_parquet_provenance(path: Path) -> dict[str, str]:
    meta = pq.read_schema(path).metadata or {}
    return {
        k.decode()[len(META_PREFIX) :]: v.decode()
        for k, v in meta.items()
        if k.decode().startswith(META_PREFIX)
    }
