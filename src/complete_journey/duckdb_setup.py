"""DuckDB connection helpers for reproducible analytical table construction.

Bootstrap-stage only: this module opens connections and exposes helpers. It does
not create or transform any research tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from complete_journey.paths import DUCKDB_PATH, ensure_warehouse_dir


def connect(path: Path | str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection to the project warehouse.

    Args:
        path: Target database file. Defaults to the project warehouse.
        read_only: Open without write access. Not supported for in-memory DBs.
    """
    if path is None:
        ensure_warehouse_dir()
        return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)
    if str(path) == ":memory:":
        return duckdb.connect()
    return duckdb.connect(str(path), read_only=read_only)


def in_memory() -> duckdb.DuckDBPyConnection:
    """Open a throwaway in-memory connection (used by smoke tests)."""
    return duckdb.connect()


def table_count(con: duckdb.DuckDBPyConnection) -> int:
    """Number of user tables in the given connection (bootstrap sanity check)."""
    row: Any = con.execute("SELECT count(*) FROM information_schema.tables").fetchone()
    return int(row[0])
