from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tennis import __version__
from tennis.util.io import atomic_path, read_parquet_provenance, write_parquet


def test_parquet_records_provenance(tmp_path: Path) -> None:
    path = tmp_path / "t.parquet"
    table = pa.table({"x": [1, 2]}).replace_schema_metadata({"keep": "me"})
    write_parquet(table, path, stage="contacts", config_hash="abc", schema_version=3)
    assert read_parquet_provenance(path) == {
        "pipeline_version": __version__,
        "config_hash": "abc",
        "stage": "contacts",
        "schema_version": "3",
    }
    assert pq.read_schema(path).metadata[b"keep"] == b"me"
    assert pq.read_table(path).column("x").to_pylist() == [1, 2]


def test_atomic_path_leaves_nothing_on_error(tmp_path: Path) -> None:
    target = tmp_path / "out.wav"
    with pytest.raises(RuntimeError), atomic_path(target) as tmp:
        assert tmp.suffix == ".wav"
        tmp.write_text("half")
        raise RuntimeError
    assert list(tmp_path.iterdir()) == []


def test_rewriting_identical_data_keeps_mtime(tmp_path: Path) -> None:
    import os

    path = tmp_path / "t.parquet"
    table = pa.table({"x": [1, 2]})
    write_parquet(table, path, stage="s", config_hash="a", schema_version=1)
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    write_parquet(table, path, stage="s", config_hash="b", schema_version=1)
    assert path.stat().st_mtime_ns == 1_000_000_000
    assert read_parquet_provenance(path)["config_hash"] == "b"
    write_parquet(pa.table({"x": [1, 3]}), path, stage="s", config_hash="b", schema_version=1)
    assert path.stat().st_mtime_ns > 1_000_000_000
