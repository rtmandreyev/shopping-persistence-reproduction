"""Raw CSV discovery and DuckDB view registration for the Complete Journey audit.

Raw data is immutable. Nothing in this module writes to ``data/raw/``; it only
resolves paths and registers read-only views over the source CSVs.

Views are registered with **explicit column types** and ``strict_mode=True`` so
that a schema surprise (a decimal in an integer column, a malformed row, an
unexpected column count) raises an error instead of being silently coerced.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb

from complete_journey.paths import RAW_DIR

#: Subdirectory inside the raw dataset that holds the eight source CSVs.
CSV_SUBDIR: Final = "dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV"

#: The eight raw tables, keyed by the view name used throughout the audit SQL.
#:
#: Column order here is the project's *canonical* order for the audit views, not
#: necessarily the physical order in the CSV. Observed on 2026-09-30, several
#: files differ from the vendor guide's documented order (e.g. ``campaign_desc``
#: is physically ``DESCRIPTION, CAMPAIGN, START_DAY, END_DAY`` while the guide
#: lists ``CAMPAIGN`` first). Views therefore select and cast by name, so the
#: audit is unaffected by physical column order.
#:
#: Types are pinned from observed values and verified by
#: ``tests/test_audit_infrastructure.py``. They are *not* re-inferred per scan.
TABLE_SCHEMAS: Final[dict[str, dict[str, str]]] = {
    "transaction_data": {
        "household_key": "BIGINT",
        "BASKET_ID": "BIGINT",
        "DAY": "BIGINT",
        "PRODUCT_ID": "BIGINT",
        "QUANTITY": "BIGINT",
        "SALES_VALUE": "DOUBLE",
        "STORE_ID": "BIGINT",
        "RETAIL_DISC": "DOUBLE",
        "TRANS_TIME": "VARCHAR",
        "WEEK_NO": "BIGINT",
        "COUPON_DISC": "DOUBLE",
        "COUPON_MATCH_DISC": "DOUBLE",
    },
    "product": {
        "PRODUCT_ID": "BIGINT",
        "MANUFACTURER": "BIGINT",
        "DEPARTMENT": "VARCHAR",
        "BRAND": "VARCHAR",
        "COMMODITY_DESC": "VARCHAR",
        "SUB_COMMODITY_DESC": "VARCHAR",
        "CURR_SIZE_OF_PRODUCT": "VARCHAR",
    },
    "campaign_table": {
        "household_key": "BIGINT",
        "CAMPAIGN": "BIGINT",
        "DESCRIPTION": "VARCHAR",
    },
    "campaign_desc": {
        "CAMPAIGN": "BIGINT",
        "DESCRIPTION": "VARCHAR",
        "START_DAY": "BIGINT",
        "END_DAY": "BIGINT",
    },
    "coupon": {
        "CAMPAIGN": "BIGINT",
        "COUPON_UPC": "BIGINT",
        "PRODUCT_ID": "BIGINT",
    },
    "coupon_redempt": {
        "household_key": "BIGINT",
        "DAY": "BIGINT",
        "COUPON_UPC": "BIGINT",
        "CAMPAIGN": "BIGINT",
    },
    "hh_demographic": {
        "household_key": "BIGINT",
        "classification_1": "VARCHAR",
        "classification_2": "VARCHAR",
        "classification_3": "VARCHAR",
        "classification_4": "VARCHAR",
        "classification_5": "VARCHAR",
        "HOMEOWNER_DESC": "VARCHAR",
        "KID_CATEGORY_DESC": "VARCHAR",
    },
    "causal_data": {
        "PRODUCT_ID": "BIGINT",
        "STORE_ID": "BIGINT",
        "WEEK_NO": "BIGINT",
        "display": "VARCHAR",
        "mailer": "VARCHAR",
    },
}

#: Name of the vendor's data-dictionary PDF shipped inside the raw dataset.
USER_GUIDE_NAME: Final = "dunnhumby - The Complete Journey User Guide.pdf"

#: Columns that act as identifiers for distinct-count reporting.
IDENTIFIER_COLUMNS: Final[dict[str, list[str]]] = {
    "transaction_data": ["household_key", "BASKET_ID", "PRODUCT_ID", "STORE_ID"],
    "product": ["PRODUCT_ID", "MANUFACTURER"],
    "campaign_table": ["household_key", "CAMPAIGN"],
    "campaign_desc": ["CAMPAIGN"],
    "coupon": ["CAMPAIGN", "COUPON_UPC", "PRODUCT_ID"],
    "coupon_redempt": ["household_key", "COUPON_UPC", "CAMPAIGN"],
    "hh_demographic": ["household_key"],
    "causal_data": ["PRODUCT_ID", "STORE_ID", "WEEK_NO"],
}


class RawDataError(RuntimeError):
    """Raised when raw inputs are missing or fail the strict schema check."""


@dataclass(frozen=True)
class RawFile:
    """A raw source file with its on-disk size."""

    table: str
    path: Path
    size_bytes: int

    @property
    def filename(self) -> str:
        return self.path.name


def csv_dir(raw_dir: Path = RAW_DIR) -> Path:
    """Return the directory holding the source CSVs."""
    return raw_dir / CSV_SUBDIR


def expected_filenames() -> dict[str, str]:
    """Map table name -> expected CSV filename."""
    return {name: f"{name}.csv" for name in TABLE_SCHEMAS}


def discover_raw_files(raw_dir: Path = RAW_DIR) -> dict[str, RawFile]:
    """Resolve the eight raw CSVs. Raises :class:`RawDataError` if any is absent.

    Discovery is by exact filename; an unexpected or missing file is an error
    rather than a silently-ignored extra.
    """
    directory = csv_dir(raw_dir)
    if not directory.is_dir():
        raise RawDataError(f"raw CSV directory not found: {directory}")

    found: dict[str, RawFile] = {}
    missing: list[str] = []
    for table, filename in expected_filenames().items():
        path = directory / filename
        if path.is_file():
            found[table] = RawFile(table, path, path.stat().st_size)
        else:
            missing.append(filename)

    if missing:
        raise RawDataError(f"missing expected raw files in {directory}: {sorted(missing)}")

    unexpected = sorted(
        p.name for p in directory.glob("*.csv") if p.name not in set(expected_filenames().values())
    )
    if unexpected:
        raise RawDataError(
            f"unexpected CSV file(s) in raw directory: {unexpected}. "
            "The audit expects exactly the eight vendor tables; refusing to proceed."
        )
    return found


def user_guide_path(raw_dir: Path = RAW_DIR) -> Path:
    """Return the vendor user-guide PDF path."""
    return raw_dir / "dunnhumby_The-Complete-Journey" / USER_GUIDE_NAME


def read_csv_options() -> dict[str, object]:
    """Shared strict read options for every raw view.

    ``sample_size=-1`` scans the whole file for type inference so a late-appearing
    anomaly (a decimal in an integer column, a stray quote) is caught here rather
    than silently coerced downstream.
    """
    return {
        "header": True,
        "strict_mode": True,
        "ignore_errors": False,
        "sample_size": -1,
    }


def _cast_select(schema: dict[str, str]) -> str:
    """Build an explicit by-name, by-type projection.

    Selecting by name makes the view immune to physical column order, and the
    explicit cast makes a value that cannot hold the declared type an error
    rather than a silent NULL.
    """
    projections = ", ".join(
        f'CAST("{name}" AS {dtype}) AS "{name}"' for name, dtype in schema.items()
    )
    return f"SELECT {projections} FROM"


def sniff_header(path: Path) -> list[str]:
    """Read the header line of a CSV directly, with no data scan.

    The vendor files are plain comma-separated with no quoting or embedded
    newlines in any header, so a first-line split is exact here. Using DuckDB's
    CSV reader to do this is not possible without scanning, and reading the
    header by hand keeps :func:`register_views` cheap on a ~808 MiB dataset.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
    if not header:
        raise RawDataError(f"empty file, expected a CSV header: {path.name}")
    return [column.strip().strip('"') for column in header.rstrip("\r\n").split(",")]


def register_views(
    con: duckdb.DuckDBPyConnection,
    raw_dir: Path = RAW_DIR,
    verify_headers: bool = True,
) -> dict[str, RawFile]:
    """Register one read-only DuckDB view per raw table.

    The dataset is ~808 MiB and is never materialised into pandas; DuckDB
    projects only the columns each audit query needs.

    Headers are verified against :data:`TABLE_SCHEMAS` by name, so a renamed or
    added column raises :class:`RawDataError` instead of being ignored.
    """
    files = discover_raw_files(raw_dir)
    options = read_csv_options_as_sql()

    for table, raw_file in files.items():
        if verify_headers:
            actual = sniff_header(raw_file.path)
            expected = list(TABLE_SCHEMAS[table])
            if sorted(actual) != sorted(expected):
                raise RawDataError(
                    f"header mismatch for {raw_file.filename}:\n"
                    f"  expected (canonical): {sorted(expected)}\n"
                    f"  found in file:        {sorted(actual)}"
                )
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS "
            f"{_cast_select(TABLE_SCHEMAS[table])} "
            f"read_csv('{raw_file.path.as_posix()}', {options})"
        )
    return files


def read_csv_options_as_sql() -> str:
    """Render read options as SQL text for the view definition."""
    opts = read_csv_options()
    return ", ".join(f"{key} = {value!r}" for key, value in opts.items())


def snapshot_raw_files(raw_dir: Path = RAW_DIR) -> dict[str, dict[str, int | float | str]]:
    """Capture size and mtime for every raw file, for immutability assertions."""
    directory = csv_dir(raw_dir)
    snapshot: dict[str, dict[str, int | float | str]] = {}
    for path in sorted(directory.glob("*.csv")):
        stat = path.stat()
        snapshot[path.name] = {
            "size_bytes": stat.st_size,
            "mtime": dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.UTC).isoformat(),
        }
    return snapshot


def table_counts(con: duckdb.DuckDBPyConnection, tables: list[str] | None = None) -> dict[str, int]:
    """Row counts for the registered views. This is a full scan by design."""
    names = tables or list(TABLE_SCHEMAS)
    return {name: int(con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]) for name in names}
