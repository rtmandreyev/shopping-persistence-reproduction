"""Neutral department-level descriptive derivation (Phase 3, Question C support).

This module materialises the Phase 3 department derivation **as a temporary view in
a throwaway connection**. It is not a canonical Phase 2 table, it is never written
to ``data/processed/``, and it publishes no result: the module's output is a
*validation report* (grain, sources, join cardinality, fan-out, sentinel handling,
totality reconciliation), not a department finding.

Contract (declared here, asserted at run time):

* grain — ``household_key x department_desc x week_no`` (sparse)
* sources — ``transaction_data`` (raw, read-only) LEFT JOIN ``product`` (raw, read-only)
* join key — ``PRODUCT_ID``, expected 1:1
* the joined row count must equal the transaction-line count (fan-out detection)
* every transaction line must match a product (no silently unmatched department)
* ``product.PRODUCT_ID`` must be unique
* blank ``DEPARTMENT`` labels are preserved and flagged, never dropped
* unit and retailer-SALES_VALUE totals must equal the accepted anchors, because
  every line is assigned exactly one department and therefore the derivation
  exhausts the universe for those two measures
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb

from complete_journey.data.raw_tables import RawFile, register_views
from complete_journey.descriptive.contracts import (
    ACCEPTED_ANCHORS,
    DESCRIPTIVE_SQL_DIR,
    ReconciliationAnchors,
)

#: Generating SQL (one file per Phase 3 descriptive output).
DEPARTMENT_SQL_PATH: Final[Path] = DESCRIPTIVE_SQL_DIR / "01_department_week.sql"

#: Temporary view name. The ``descriptive_`` prefix marks it as a Phase 3
#: derivation so it can never be mistaken for an accepted canonical table.
DERIVATION_VIEW: Final[str] = "descriptive_department_week"

GRAIN: Final[tuple[str, ...]] = ("household_key", "department_desc", "week_no")
SOURCE_TABLES: Final[tuple[str, ...]] = ("transaction_data", "product")
JOIN_KEY: Final[str] = "PRODUCT_ID"
JOIN_CARDINALITY: Final[str] = "1:1 (product.PRODUCT_ID is unique)"

#: Declared result columns, in order. A schema change in the SQL must be reflected
#: here, so a silent column addition cannot slip into a later Phase 3 output.
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "household_key",
    "department_desc",
    "week_no",
    "department_sales_value",
    "department_units",
    "department_trips",
    "distinct_products",
    "department_national_sales_value",
    "department_private_sales_value",
    "department_national_units",
    "department_private_units",
    "distinct_national_products",
    "distinct_private_products",
    "department_is_unlabelled",
)

#: A blank vendor department label (15 products). Preserved, never filtered.
BLANK_DEPARTMENT_LABEL: Final[str] = ""


class DepartmentDerivationError(RuntimeError):
    """Raised when the department derivation violates its declared contract."""


@dataclass(frozen=True)
class DepartmentDerivationReport:
    """Structural validation evidence for the department derivation.

    Contains no department ranking, no interpretation and no finding.
    """

    grain: tuple[str, ...]
    source_tables: tuple[str, ...]
    join_key: str
    join_cardinality: str
    transaction_lines: int
    joined_rows: int
    unmatched_products: int
    product_rows: int
    product_distinct_ids: int
    output_rows: int
    output_key_unique: bool
    distinct_departments: int
    unlabelled_products: int
    unlabelled_rows: int
    units_total: int
    units_residual: int
    sales_value_total: float
    sales_value_residual: float

    @property
    def fan_out_detected(self) -> bool:
        return self.joined_rows != self.transaction_lines


def sql_text(path: Path = DEPARTMENT_SQL_PATH) -> str:
    """Return the committed derivation SQL (single source of truth)."""
    if not path.is_file():
        raise DepartmentDerivationError(f"department derivation SQL missing: {path}")
    return path.read_text(encoding="utf-8")


def register_sources(con: duckdb.DuckDBPyConnection) -> dict[str, RawFile]:
    """Register the accepted raw views (strict, by name, pinned types).

    Only the two views the derivation references (``transaction_data``,
    ``product``) are scanned by its queries; the other raw views stay unread.
    """
    return register_views(con)


def build_view(con: duckdb.DuckDBPyConnection, path: Path = DEPARTMENT_SQL_PATH) -> str:
    """Create the temporary derivation view and return its name."""
    con.execute(f"CREATE OR REPLACE VIEW {DERIVATION_VIEW} AS {sql_text(path)}")
    return DERIVATION_VIEW


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> float:
    row = con.execute(sql).fetchone()
    if row is None or row[0] is None:
        raise DepartmentDerivationError(f"query returned no value: {sql}")
    return float(row[0])


def _result_columns(con: duckdb.DuckDBPyConnection, view: str) -> tuple[str, ...]:
    rows = con.execute(f"DESCRIBE {view}").fetchall()
    return tuple(str(row[0]) for row in rows)


def validate(
    con: duckdb.DuckDBPyConnection,
    *,
    anchors: ReconciliationAnchors = ACCEPTED_ANCHORS,
    path: Path = DEPARTMENT_SQL_PATH,
    build: bool = True,
) -> DepartmentDerivationReport:
    """Build and validate the department derivation; raise on any contract breach.

    The connection must already have the raw views registered (see
    :func:`register_sources`); tests may instead register synthetic tables under the
    same names to exercise the semantics.

    Raises:
        DepartmentDerivationError: if the product join is not 1:1, if any
            transaction line is unmatched, if the join fans out, if the declared key
            is not unique, if the result schema differs from :data:`OUTPUT_COLUMNS`,
            or if unit / retailer-SALES_VALUE totals do not reconcile to the
            accepted anchors.
    """
    view = build_view(con, path) if build else DERIVATION_VIEW

    columns = _result_columns(con, view)
    if columns != OUTPUT_COLUMNS:
        raise DepartmentDerivationError(
            f"result schema mismatch: declared {list(OUTPUT_COLUMNS)} got {list(columns)}"
        )

    product_rows = int(_scalar(con, "SELECT count(*) FROM product"))
    product_distinct = int(_scalar(con, "SELECT count(DISTINCT PRODUCT_ID) FROM product"))
    if product_rows != product_distinct:
        raise DepartmentDerivationError(
            "product.PRODUCT_ID is not unique "
            f"({product_rows} rows vs {product_distinct} distinct ids): the department "
            "join would fan out, so the derivation refuses to run"
        )

    transaction_lines = int(_scalar(con, "SELECT count(*) FROM transaction_data"))
    joined = con.execute(
        "SELECT count(*) AS joined_rows, "
        "count(*) FILTER (WHERE p.PRODUCT_ID IS NULL) AS unmatched "
        "FROM transaction_data AS t LEFT JOIN product AS p ON p.PRODUCT_ID = t.PRODUCT_ID"
    ).fetchone()
    joined_rows = int(joined[0])
    unmatched = int(joined[1])
    if unmatched != 0:
        raise DepartmentDerivationError(
            f"{unmatched} transaction line(s) have no product row: department cannot be "
            "assigned for them, and silently dropping them would break totality"
        )
    if joined_rows != transaction_lines:
        raise DepartmentDerivationError(
            f"join fan-out detected: {transaction_lines} transaction lines produced "
            f"{joined_rows} joined rows"
        )

    output_rows = int(_scalar(con, f"SELECT count(*) FROM {view}"))
    duplicate_keys = int(
        _scalar(
            con,
            f"SELECT count(*) - count(DISTINCT (household_key, department_desc, week_no)) "
            f"FROM {view}",
        )
    )
    if duplicate_keys != 0:
        raise DepartmentDerivationError(
            f"declared key not unique: {duplicate_keys} duplicate "
            "(household_key, department_desc, week_no) row(s)"
        )

    units_total = int(_scalar(con, f"SELECT sum(department_units) FROM {view}"))
    sales_value_total = _scalar(con, f"SELECT sum(department_sales_value) FROM {view}")

    units_residual = units_total - anchors.units
    sales_value_residual = sales_value_total - anchors.retailer_sales_value
    if units_residual != 0:
        raise DepartmentDerivationError(
            f"department units do not exhaust the accepted universe: "
            f"{units_total} vs anchor {anchors.units} (residual {units_residual})"
        )
    money_tolerance = max(
        anchors.sales_value_abs_tolerance,
        anchors.sales_value_rel_tolerance * abs(anchors.retailer_sales_value),
    )
    if abs(sales_value_residual) > money_tolerance:
        raise DepartmentDerivationError(
            "department retailer SALES_VALUE does not exhaust the accepted universe: "
            f"{sales_value_total!r} vs anchor {anchors.retailer_sales_value!r} "
            f"(residual {sales_value_residual!r}, tolerance {money_tolerance:g})"
        )

    return DepartmentDerivationReport(
        grain=GRAIN,
        source_tables=SOURCE_TABLES,
        join_key=JOIN_KEY,
        join_cardinality=JOIN_CARDINALITY,
        transaction_lines=transaction_lines,
        joined_rows=joined_rows,
        unmatched_products=unmatched,
        product_rows=product_rows,
        product_distinct_ids=product_distinct,
        output_rows=output_rows,
        output_key_unique=True,
        distinct_departments=int(
            _scalar(con, f"SELECT count(DISTINCT department_desc) FROM {view}")
        ),
        unlabelled_products=int(
            _scalar(con, "SELECT count(*) FROM product WHERE trim(coalesce(DEPARTMENT, '')) = ''")
        ),
        unlabelled_rows=int(
            _scalar(con, f"SELECT count(*) FROM {view} WHERE department_is_unlabelled = 1")
        ),
        units_total=units_total,
        units_residual=units_residual,
        sales_value_total=sales_value_total,
        sales_value_residual=sales_value_residual,
    )


@dataclass(frozen=True)
class TotalityCheck:
    """Light-weight totality evidence for a derivation view that already exists.

    Used by Phase 3 runs that build the view for a composition check and do not
    need the full two-scan contract validation again.
    """

    rows: int
    distinct_departments: int
    units_residual: int
    sales_value_residual: float


def totality_residuals(
    con: duckdb.DuckDBPyConnection,
    *,
    anchors: ReconciliationAnchors = ACCEPTED_ANCHORS,
    view: str = DERIVATION_VIEW,
) -> TotalityCheck:
    """Check that an existing derivation view still exhausts the accepted anchors."""
    row = con.execute(
        f"SELECT count(*), count(DISTINCT department_desc), sum(department_units), "
        f"sum(department_sales_value) FROM {view}"
    ).fetchone()
    if row is None:
        raise DepartmentDerivationError(f"derivation view {view!r} returned no rows")
    rows, departments, units, sales_value = int(row[0]), int(row[1]), int(row[2]), float(row[3])
    return TotalityCheck(
        rows=rows,
        distinct_departments=departments,
        units_residual=units - anchors.units,
        sales_value_residual=sales_value - anchors.retailer_sales_value,
    )


def format_report(
    report: DepartmentDerivationReport, anchors: ReconciliationAnchors = ACCEPTED_ANCHORS
) -> str:
    """Render the structural validation evidence. No department result is reported."""
    lines: Sequence[str] = (
        f"department derivation (temporary view {DERIVATION_VIEW!r}; no artifact written)",
        f"  grain={report.grain}",
        f"  sources={report.source_tables} join={report.join_key} cardinality={report.join_cardinality}",
        f"  transaction lines={report.transaction_lines} joined rows={report.joined_rows} "
        f"unmatched={report.unmatched_products} fan_out={report.fan_out_detected}",
        f"  products={report.product_rows} distinct product ids={report.product_distinct_ids}",
        f"  output rows={report.output_rows} key unique={report.output_key_unique} "
        f"distinct department labels={report.distinct_departments}",
        f"  blank-label products preserved={report.unlabelled_products} "
        f"blank-label rows={report.unlabelled_rows}",
        f"  units total={report.units_total} residual vs anchor={report.units_residual}",
        f"  retailer SALES_VALUE total={report.sales_value_total!r} "
        f"residual vs anchor={report.sales_value_residual!r}",
        "  totals are accepted Phase 2 reconciliation anchors, not Phase 3 findings",
    )
    return "\n".join(lines)
