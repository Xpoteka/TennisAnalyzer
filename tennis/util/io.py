"""Atomic file writes and Parquet provenance metadata.

Outputs are written to a temporary sibling and renamed into place, so a crashed stage never
leaves a half-written file that the cache would mistake for a finished one.
"""

from __future__ import annotations

import filecmp
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tennis import __version__

META_PREFIX = "tennis."


@contextmanager
def atomic_path(path: Path, keep_mtime_if_identical: bool = True) -> Iterator[Path]:
    """Yield a temporary path with the same suffix; move it to ``path`` on success.

    If the new file is byte-identical to the old one, the old modification time is kept,
    so later stages see the file as unchanged.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.partial-{os.getpid()}{path.suffix}")
    try:
        yield tmp
        old_times = None
        if keep_mtime_if_identical and path.is_file() and filecmp.cmp(tmp, path, shallow=False):
            st = path.stat()
            old_times = (st.st_atime_ns, st.st_mtime_ns)
        os.replace(tmp, path)
        if old_times is not None:
            os.utime(path, ns=old_times)
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
    table: pa.Table,
    path: Path,
    *,
    stage: str,
    config_hash: str,
    schema_version: int,
    extra: Mapping[str, str] | None = None,
) -> None:
    """Write ``table`` with pipeline version, config hash and schema version in its metadata.

    ``extra`` entries are stored under the same ``tennis.`` prefix. If the file already
    holds exactly the same data, it is rewritten (fresh metadata) but keeps its old
    modification time, so a rerun that changes nothing does not invalidate later stages.
    """
    meta = dict(table.schema.metadata or {})
    entries = {**(extra or {}), **provenance(stage, config_hash, schema_version)}
    for key, value in entries.items():
        meta[f"{META_PREFIX}{key}".encode()] = value.encode()
    table = table.replace_schema_metadata(meta)
    unchanged_mtime = _mtime_if_same_data(path, table)
    with atomic_path(path, keep_mtime_if_identical=False) as tmp:
        pq.write_table(table, tmp)
    if unchanged_mtime is not None:
        os.utime(path, ns=unchanged_mtime)


def _mtime_if_same_data(path: Path, table: pa.Table) -> tuple[int, int] | None:
    if not path.exists():
        return None
    try:
        old = pq.read_table(path)
    except (OSError, ValueError):
        return None
    if not same_data(old, table):
        return None
    st = path.stat()
    return (st.st_atime_ns, st.st_mtime_ns)


def same_data(old: pa.Table, new: pa.Table) -> bool:
    """Whether two tables hold the same values, ignoring their metadata.

    This is ``Table.equals`` with one change: two NaNs in a float column count as equal.
    Arrow follows IEEE 754, where they do not, but a metric that could not be computed is
    the same result on the next run, and treating it as a change would make every stage
    that writes NaN (stage 4 and stage 6) invalidate its successors on every rerun.
    """
    if not old.schema.equals(new.schema, check_metadata=False):
        return False
    if old.num_rows != new.num_rows:
        return False
    for field in new.schema:
        a, b = old.column(field.name), new.column(field.name)
        if a.equals(b):
            continue
        if not pa.types.is_floating(field.type) or a.null_count != b.null_count:
            return False
        left = np.asarray(a.to_numpy(zero_copy_only=False), dtype=np.float64)
        right = np.asarray(b.to_numpy(zero_copy_only=False), dtype=np.float64)
        if not np.array_equal(left, right, equal_nan=True):
            return False
    return True


def read_parquet_provenance(path: Path) -> dict[str, str]:
    meta = pq.read_schema(path).metadata or {}
    return {
        k.decode()[len(META_PREFIX) :]: v.decode()
        for k, v in meta.items()
        if k.decode().startswith(META_PREFIX)
    }
