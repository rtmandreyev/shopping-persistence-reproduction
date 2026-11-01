"""Accepted-input verification and the Phase 3 reconciliation gate.

Two jobs, both fail-loudly:

1. :func:`verify_accepted_artifacts` — the accepted Phase 2 Parquet artifacts exist
   and their SHA-256 hashes and row counts still match the generated contracts.
2. :func:`run_reconciliation_gate` — every Phase 3 run recomputes the accepted
   headline totals **by more than one independent route** before any descriptive
   result may be published.

A single-route check can validate a table against itself; the whole point here is
that the same quantity is derived along genuinely different aggregation paths
(row grain vs basket grain vs commodity grain, or the grid vs the structural
product of households and weeks) and must agree. These are accepted Phase 2
anchors, not Phase 3 findings.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import duckdb

from complete_journey.descriptive.contracts import (
    ACCEPTED_ANCHORS,
    GENERATED_CONTRACTS_JSON,
    GENERATED_SAMPLE_FLAG_COUNTS,
    ReconciliationAnchors,
    accepted_parquet_path,
)
from complete_journey.paths import PROJECT_ROOT


class ReconciliationError(RuntimeError):
    """Raised when an accepted input is missing, altered, or does not reconcile."""


class ArtifactVerificationError(RuntimeError):
    """Raised when an accepted Phase 2 artifact no longer matches its contract."""


#: Tables the gate reads. Registering a view is lazy: only a table a query touches
#: is scanned, so this list is not a cost, it is a completeness statement.
GATE_TABLES: Final[tuple[str, ...]] = (
    "household_week_base",
    "basket_panel",
    "household_history",
    "household_category_week",
    "econometrics_household_week",
    "campaign_week_calendar",
    "campaign_household_week",
    "choice_line_price_candidates",
)

#: Reads a value from a connection.
Reading = Callable[[duckdb.DuckDBPyConnection], float]

Tolerance = Literal["exact", "money"]


def open_connection() -> duckdb.DuckDBPyConnection:
    """Open a throwaway in-memory connection with deterministic aggregation.

    Single-threaded aggregation mirrors the accepted build's
    ``DETERMINISTIC_SETTINGS`` rationale: ``SUM(DOUBLE)`` depends on summation
    order, so pinning it makes the gate's float comparisons reproducible
    run-to-run instead of drifting in the last bits.
    """
    con = duckdb.connect()
    con.execute("SET threads = 1")
    con.execute("SET preserve_insertion_order = true")
    return con


def register_accepted_views(
    con: duckdb.DuckDBPyConnection, tables: Sequence[str] | None = None
) -> dict[str, Path]:
    """Register read-only views over the accepted Phase 2 Parquet tables."""
    names = GATE_TABLES if tables is None else tuple(tables)
    registered: dict[str, Path] = {}
    for name in names:
        path = accepted_parquet_path(name)
        if not path.is_file():
            raise ReconciliationError(
                f"accepted input missing: {path} (table {name!r}); "
                "rebuild with scripts/build_canonical_layer.py before any Phase 3 run"
            )
        con.execute(
            f"CREATE OR REPLACE VIEW \"{name}\" AS SELECT * FROM read_parquet('{path.as_posix()}')"
        )
        registered[name] = path
    return registered


# --- accepted-artifact verification ------------------------------------------


@dataclass(frozen=True)
class ArtifactCheck:
    """Result of verifying one accepted artifact against its contract."""

    table: str
    path: str
    sha256_ok: bool
    row_count_ok: bool
    rows: int
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.sha256_ok and self.row_count_ok


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_accepted_artifacts(
    con: duckdb.DuckDBPyConnection | None = None,
    *,
    contracts_path: Path = GENERATED_CONTRACTS_JSON,
    root: Path = PROJECT_ROOT,
) -> tuple[ArtifactCheck, ...]:
    """Verify every accepted table's Parquet against the generated contracts.

    Checks that the declared tables are present, not Git-tracked, hash-identical to
    the generated contract, and still hold the contract's row count.

    Raises:
        ArtifactVerificationError: on the first mismatch, naming it precisely.
    """
    if not contracts_path.is_file():
        raise ArtifactVerificationError(
            f"generated contracts not found: {contracts_path}; run the supported "
            "Phase 2 build before any Phase 3 run"
        )
    document = json.loads(contracts_path.read_text(encoding="utf-8"))
    tables = document.get("tables")
    if not isinstance(tables, list) or not tables:
        raise ArtifactVerificationError(f"{contracts_path} declares no tables")

    owns_con = con is None
    connection = open_connection() if con is None else con
    checks: list[ArtifactCheck] = []
    try:
        for entry in tables:
            name = str(entry["table"])
            declared_rows = int(entry["actual_row_count"])
            rel = str(entry["parquet_path"])
            path = root / rel
            if entry.get("git_tracked"):
                raise ArtifactVerificationError(
                    f"accepted artifact {name!r} is declared Git-tracked; analytical "
                    "tables must never be tracked"
                )
            if not path.is_file():
                raise ArtifactVerificationError(f"accepted artifact missing: {path}")
            sha_ok = _sha256(path) == str(entry["parquet_sha256"])
            rows = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet('{path.as_posix()}')"
                ).fetchone()[0]
            )
            rows_ok = rows == declared_rows
            check = ArtifactCheck(
                table=name,
                path=rel,
                sha256_ok=sha_ok,
                row_count_ok=rows_ok,
                rows=rows,
                detail=(
                    ""
                    if sha_ok and rows_ok
                    else f"sha256_match={sha_ok} declared_rows={declared_rows} actual_rows={rows}"
                ),
            )
            checks.append(check)
            if not check.ok:
                raise ArtifactVerificationError(
                    f"accepted artifact {name!r} does not match its contract: {check.detail}"
                )
    finally:
        if owns_con:
            connection.close()
    return tuple(checks)


# --- the reconciliation gate -------------------------------------------------


@dataclass(frozen=True)
class RouteReading:
    """One independent derivation of a headline total."""

    route: str
    value: float


@dataclass(frozen=True)
class GateCheck:
    """One headline total, compared against its anchor along every route."""

    name: str
    anchor: float
    tolerance: Tolerance
    readings: tuple[RouteReading, ...]
    max_deviation: float
    passed: bool
    detail: str = ""
    abs_tolerance: float = 0.0
    rel_tolerance: float = 0.0

    def route_values(self) -> dict[str, float]:
        return {reading.route: reading.value for reading in self.readings}


def _sql_reader(sql: str) -> Reading:
    def read(con: duckdb.DuckDBPyConnection) -> float:
        row = con.execute(sql).fetchone()
        if row is None or row[0] is None:
            raise ReconciliationError(f"route returned no value: {sql}")
        return float(row[0])

    return read


def _csv_reader(path: Path, key_column: str, key_value: str, value_column: str) -> Reading:
    """Read an accepted tracked aggregate CSV as an independent second route."""

    def read(_con: duckdb.DuckDBPyConnection) -> float:
        if not path.is_file():
            raise ReconciliationError(f"accepted aggregate CSV missing: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get(key_column) == key_value:
                    return float(row[value_column])
        raise ReconciliationError(f"{path} has no row with {key_column}={key_value!r}")

    return read


@dataclass(frozen=True)
class _CheckSpec:
    name: str
    anchor: str
    tolerance: Tolerance
    readings: tuple[tuple[str, Reading], ...] = field(default_factory=tuple)


def _check_specs() -> tuple[_CheckSpec, ...]:
    """Build the gate's check list: one accepted anchor, several routes each."""
    sample_csv = GENERATED_SAMPLE_FLAG_COUNTS
    return (
        _CheckSpec(
            name="retailer_sales_value_total",
            anchor="retailer_sales_value",
            tolerance="money",
            readings=(
                (
                    "household_week_base",
                    _sql_reader("SELECT sum(weekly_sales_value) FROM household_week_base"),
                ),
                ("basket_panel", _sql_reader("SELECT sum(basket_sales_value) FROM basket_panel")),
                (
                    "household_category_week",
                    _sql_reader("SELECT sum(category_sales_value) FROM household_category_week"),
                ),
            ),
        ),
        _CheckSpec(
            name="units_total",
            anchor="units",
            tolerance="exact",
            readings=(
                ("household_week_base", _sql_reader("SELECT sum(units) FROM household_week_base")),
                ("basket_panel", _sql_reader("SELECT sum(basket_units) FROM basket_panel")),
                (
                    "household_category_week",
                    _sql_reader("SELECT sum(category_units) FROM household_category_week"),
                ),
                (
                    "choice_line_price_candidates",
                    _sql_reader("SELECT sum(quantity) FROM choice_line_price_candidates"),
                ),
            ),
        ),
        _CheckSpec(
            name="trips_total",
            anchor="baskets",
            tolerance="exact",
            readings=(
                ("household_week_base", _sql_reader("SELECT sum(trips) FROM household_week_base")),
                ("basket_panel", _sql_reader("SELECT count(*) FROM basket_panel")),
            ),
        ),
        _CheckSpec(
            name="households",
            anchor="households",
            tolerance="exact",
            readings=(
                (
                    "household_week_base",
                    _sql_reader("SELECT count(DISTINCT household_key) FROM household_week_base"),
                ),
                ("household_history", _sql_reader("SELECT count(*) FROM household_history")),
            ),
        ),
        _CheckSpec(
            name="household_week_grid",
            anchor="household_weeks",
            tolerance="exact",
            readings=(
                ("household_week_base", _sql_reader("SELECT count(*) FROM household_week_base")),
                (
                    "households_x_weeks_declared",
                    _sql_reader(
                        "SELECT (SELECT count(*) FROM household_history) * "
                        "(SELECT count(DISTINCT week_no) FROM basket_panel)"
                    ),
                ),
            ),
        ),
        _CheckSpec(
            name="active_household_weeks",
            anchor="active_household_weeks",
            tolerance="exact",
            readings=(
                (
                    "household_week_base",
                    _sql_reader(
                        "SELECT count(*) FROM household_week_base WHERE retailer_purchase = 1"
                    ),
                ),
                (
                    "basket_panel_distinct_household_week",
                    _sql_reader(
                        "SELECT count(DISTINCT (household_key, week_no)) FROM basket_panel"
                    ),
                ),
            ),
        ),
        _CheckSpec(
            name="weeks",
            anchor="weeks",
            tolerance="exact",
            readings=(
                (
                    "household_week_base",
                    _sql_reader("SELECT count(DISTINCT week_no) FROM household_week_base"),
                ),
                ("basket_panel", _sql_reader("SELECT count(DISTINCT week_no) FROM basket_panel")),
                (
                    "campaign_week_calendar",
                    _sql_reader("SELECT count(DISTINCT week_no) FROM campaign_week_calendar"),
                ),
            ),
        ),
        _CheckSpec(
            name="sample_full_households",
            anchor="sample_full_households",
            tolerance="exact",
            readings=(
                (
                    "econometrics_household_week",
                    _sql_reader(
                        "SELECT count(DISTINCT household_key) FROM econometrics_household_week "
                        "WHERE sample_full = 1"
                    ),
                ),
                (
                    "tracked_sample_flag_counts_csv",
                    _csv_reader(sample_csv, "sample", "sample_full", "households"),
                ),
            ),
        ),
        _CheckSpec(
            name="sample_sustained_support_households",
            anchor="sample_sustained_support_households",
            tolerance="exact",
            readings=(
                (
                    "econometrics_household_week",
                    _sql_reader(
                        "SELECT count(DISTINCT household_key) FROM econometrics_household_week "
                        "WHERE sample_sustained_support = 1"
                    ),
                ),
                (
                    "tracked_sample_flag_counts_csv",
                    _csv_reader(sample_csv, "sample", "sample_sustained_support", "households"),
                ),
            ),
        ),
        _CheckSpec(
            name="sample_higher_support_households",
            anchor="sample_higher_support_households",
            tolerance="exact",
            readings=(
                (
                    "econometrics_household_week",
                    _sql_reader(
                        "SELECT count(DISTINCT household_key) FROM econometrics_household_week "
                        "WHERE sample_higher_support = 1"
                    ),
                ),
                (
                    "tracked_sample_flag_counts_csv",
                    _csv_reader(sample_csv, "sample", "sample_higher_support", "households"),
                ),
            ),
        ),
        _CheckSpec(
            name="campaigns",
            anchor="campaigns",
            tolerance="exact",
            readings=(
                (
                    "campaign_week_calendar",
                    _sql_reader("SELECT count(DISTINCT campaign) FROM campaign_week_calendar"),
                ),
                (
                    "campaign_household_week",
                    _sql_reader("SELECT count(DISTINCT campaign) FROM campaign_household_week"),
                ),
            ),
        ),
        _CheckSpec(
            name="listed_pairs",
            anchor="listed_pairs",
            tolerance="exact",
            readings=(
                (
                    "campaign_household_week",
                    _sql_reader(
                        "SELECT count(DISTINCT (household_key, campaign)) FROM campaign_household_week"
                    ),
                ),
            ),
        ),
        _CheckSpec(
            name="listed_household_weeks",
            anchor="listed_household_weeks",
            tolerance="exact",
            readings=(
                (
                    "campaign_household_week",
                    _sql_reader("SELECT count(*) FROM campaign_household_week"),
                ),
                (
                    "listed_pairs_x_weeks",
                    _sql_reader(
                        "SELECT (SELECT count(DISTINCT (household_key, campaign)) "
                        "FROM campaign_household_week) * "
                        "(SELECT count(DISTINCT week_no) FROM campaign_week_calendar)"
                    ),
                ),
            ),
        ),
        _CheckSpec(
            name="listed_households",
            anchor="listed_households",
            tolerance="exact",
            readings=(
                (
                    "campaign_household_week",
                    _sql_reader(
                        "SELECT count(DISTINCT household_key) FROM campaign_household_week"
                    ),
                ),
                (
                    "household_history",
                    _sql_reader("SELECT sum(campaign_recipient_ever) FROM household_history"),
                ),
            ),
        ),
        _CheckSpec(
            name="recorded_redemption_rows",
            anchor="redemption_rows",
            tolerance="exact",
            readings=(
                (
                    "campaign_household_week",
                    _sql_reader("SELECT sum(coupon_redemption_rows) FROM campaign_household_week"),
                ),
                (
                    "household_history",
                    _sql_reader("SELECT sum(coupon_redemption_count) FROM household_history"),
                ),
            ),
        ),
        _CheckSpec(
            name="redemption_households",
            anchor="redemption_households",
            tolerance="exact",
            readings=(
                (
                    "household_history",
                    _sql_reader("SELECT sum(coupon_redeemer_ever) FROM household_history"),
                ),
            ),
        ),
    )


def _within(
    anchor: float, value: float, tolerance: Tolerance, anchors: ReconciliationAnchors
) -> bool:
    if tolerance == "exact":
        return abs(value - anchor) < 0.5
    return abs(value - anchor) <= max(
        anchors.sales_value_abs_tolerance, anchors.sales_value_rel_tolerance * abs(anchor)
    )


def run_reconciliation_gate(
    con: duckdb.DuckDBPyConnection | None = None,
    *,
    anchors: ReconciliationAnchors = ACCEPTED_ANCHORS,
    names: Sequence[str] | None = None,
    register_missing_views: bool = True,
) -> tuple[GateCheck, ...]:
    """Recompute every accepted headline total along independent routes.

    Args:
        con: Connection with the accepted tables registered. A fresh in-memory
            connection with accepted Parquet views is used when omitted.
        anchors: Accepted totals to reconcile against.
        names: Optional subset of check names (used by focused tests).
        register_missing_views: Register accepted views when ``con`` is omitted.

    Raises:
        ReconciliationError: if any route disagrees with another route or with the
            anchor. The message names the check, the anchor and every route value.
    """
    owns_con = con is None
    connection = open_connection() if con is None else con
    try:
        if owns_con and register_missing_views:
            register_accepted_views(connection)
        specs = _check_specs()
        if names is not None:
            wanted = set(names)
            specs = tuple(spec for spec in specs if spec.name in wanted)
            unknown = wanted - {spec.name for spec in specs}
            if unknown:
                raise ReconciliationError(f"unknown reconciliation check(s): {sorted(unknown)}")

        checks: list[GateCheck] = []
        failures: list[str] = []
        for spec in specs:
            anchor_value = float(getattr(anchors, spec.anchor))
            readings = tuple(
                RouteReading(route, reader(connection)) for route, reader in spec.readings
            )
            deviations = [abs(reading.value - anchor_value) for reading in readings]
            max_deviation = max(deviations) if deviations else float("inf")
            passed = all(
                _within(anchor_value, reading.value, spec.tolerance, anchors)
                for reading in readings
            )
            detail = ""
            if not passed:
                rendered = ", ".join(f"{reading.route}={reading.value!r}" for reading in readings)
                routes_disagree = any(
                    abs(a.value - b.value) > max(anchors.sales_value_abs_tolerance, 1e-9)
                    for index, a in enumerate(readings)
                    for b in readings[index + 1 :]
                )
                detail = (
                    "routes disagree; " if routes_disagree else ""
                ) + f"anchor={anchor_value!r} routes: {rendered}"
                failures.append(f"{spec.name}: {detail}")
            checks.append(
                GateCheck(
                    name=spec.name,
                    anchor=anchor_value,
                    tolerance=spec.tolerance,
                    readings=readings,
                    max_deviation=max_deviation,
                    passed=passed,
                    detail=detail,
                    abs_tolerance=anchors.sales_value_abs_tolerance,
                    rel_tolerance=anchors.sales_value_rel_tolerance,
                )
            )
        if failures:
            raise ReconciliationError(
                "reconciliation gate FAILED -- refusing to proceed:\n  " + "\n  ".join(failures)
            )
        return tuple(checks)
    finally:
        if owns_con:
            connection.close()


def format_gate_report(
    checks: Sequence[GateCheck], anchors: ReconciliationAnchors = ACCEPTED_ANCHORS
) -> str:
    """Render the gate result as concise, reviewable lines."""
    lines = [
        f"reconciliation gate: {len(checks)} accepted anchors, all routes agree",
        f"  ({anchors.sales_value_abs_tolerance:g} absolute / "
        f"{anchors.sales_value_rel_tolerance:g} relative tolerance on SUM(DOUBLE) totals)",
    ]
    for check in checks:
        routes = " | ".join(f"{reading.route}={reading.value!r}" for reading in check.readings)
        lines.append(f"  {check.name:<36} anchor={check.anchor!r:<14} routes: {routes}")
    return "\n".join(lines)
