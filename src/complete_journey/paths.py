"""Canonical project paths and the local DuckDB database location.

The DuckDB file lives under ``data/warehouse/`` and is gitignored. It is created
lazily; no research data is ingested at import time.
"""

from __future__ import annotations

from pathlib import Path

# Repository root = parents[2] for src/complete_journey/paths.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
WAREHOUSE_DIR: Path = DATA_DIR / "warehouse"

# Raw data is immutable: never write into RAW_DIR.
RAW_MANIFEST: Path = RAW_DIR / "manifest_raw.csv"

DUCKDB_PATH: Path = WAREHOUSE_DIR / "complete_journey.duckdb"
RAW_DATASET_DIR: Path = RAW_DIR / "dunnhumby_The-Complete-Journey"


def ensure_warehouse_dir() -> Path:
    """Create and return the gitignored warehouse directory."""
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    return WAREHOUSE_DIR
