"""Real-data point-in-time leakage verification for the Phase 2C feature store.

Rebuilds ``predictive_household_week_features`` from raw/interim sources that have
been truncated at a cutoff week and asserts that **every** feature value on every
row with ``as_of_week <= cutoff`` is unchanged. Any difference means information
from weeks after the as-of date reached a feature value at or before it.

This is the real-data counterpart of the synthetic fixture test in
``tests/test_phase2_canonical.py``; it is heavier (it re-runs the feature SQL on
truncated sources), so it lives here rather than in the default test run.

Usage:
    PYTHONPATH= python scripts/verify_predictive_leakage.py [cutoff_week]

Exit status 0 = no leakage, 1 = leakage detected or a prerequisite is missing.
"""

from __future__ import annotations

import sys

import duckdb

from complete_journey.data import contracts, raw_tables
from complete_journey.paths import PROJECT_ROOT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_SQL = PROJECT_ROOT / "sql" / "panels" / "12_predictive_household_week_features.sql"
DEFAULT_CUTOFF = 50

#: Local non-raw sources. ``coupon_redempt`` and ``hh_demographic`` are raw views
#: and are handled separately (they are filtered / re-pointed, not copied).
PARQUET_SOURCES = ("household_week_base", "household_category_week", "campaign_household_week")


def _parquet(table: str) -> str:
    return f"read_parquet('{PROCESSED_DIR / (table + '.parquet')}')"


def _values_equal(left, right) -> bool:  # type: ignore[no-untyped-def]
    """NaN-position-aware equality that tolerates int/float dtype differences."""
    import pandas as pd

    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        both_na = left.isna() & right.isna()
        close = (left.astype("float64") - right.astype("float64")).abs() <= 1e-9
        return bool((both_na | close.fillna(False)).all())
    return left.astype("str").equals(right.astype("str"))


def main(argv: list[str]) -> int:
    cutoff = int(argv[1]) if len(argv) > 1 else DEFAULT_CUTOFF
    feature_table = "predictive_household_week_features"
    if feature_table not in contracts.PHASE2C_TABLES:
        print("FAIL: predictive feature contract not found")
        return 1
    feature_parquet = PROCESSED_DIR / f"{feature_table}.parquet"
    for path in (
        feature_parquet,
        FEATURE_SQL,
        *(PROCESSED_DIR / f"{t}.parquet" for t in PARQUET_SOURCES),
    ):
        if not path.exists():
            print(f"FAIL: missing prerequisite {path.relative_to(PROJECT_ROOT)}")
            return 1

    full_con = duckdb.connect()
    truncated_con = duckdb.connect()
    try:
        full_con.execute("SET threads = 1")
        truncated_con.execute("SET threads = 1")
        for con in (full_con, truncated_con):
            raw_tables.register_views(con)

        # Full-population reference: the materialised table, restricted to the cutoff.
        full = full_con.execute(
            f"SELECT * FROM {_parquet(feature_table)} WHERE as_of_week <= {cutoff}"
        ).fetchdf()

        # Rebuild from sources whose post-cutoff information has been removed, so a
        # feature that silently read the future would change value.
        for table in PARQUET_SOURCES:
            truncated_con.execute(
                f"CREATE TABLE {table} AS SELECT * FROM {_parquet(table)} WHERE week_no <= {cutoff}"
            )
        truncated_con.execute(
            f"CREATE TABLE coupon_trunc AS SELECT * FROM coupon_redempt WHERE DAY <= {cutoff * 7};"
            "DROP VIEW coupon_redempt;"
            "ALTER TABLE coupon_trunc RENAME TO coupon_redempt;"
        )

        truncated = truncated_con.execute(FEATURE_SQL.read_text(encoding="utf-8")).fetchdf()
    finally:
        full_con.close()
        truncated_con.close()

    keys = ["household_key", "as_of_week"]
    truncated = (
        truncated[truncated["as_of_week"] <= cutoff].sort_values(keys).reset_index(drop=True)
    )
    full = full.sort_values(keys).reset_index(drop=True)
    expected_rows = contracts.PHASE2C_TABLES[feature_table]["expected_row_count"]
    expected = int(expected_rows) // 102 * cutoff  # type: ignore[arg-type]

    if len(full) != len(truncated) or len(full) != expected:
        print(f"FAIL: row mismatch full={len(full)} truncated={len(truncated)} expected={expected}")
        return 1

    mismatched: list[str] = []
    dtype_only: list[str] = []
    near_only: list[str] = []
    for column in full.columns:
        left, right = full[column], truncated[column]
        if _values_equal(left, right):
            if str(left.dtype) != str(right.dtype):
                dtype_only.append(f"{column} ({left.dtype} vs {right.dtype})")
            elif not left.equals(right):
                # Same dtype and equal within tolerance, but not bitwise identical:
                # report the magnitude so the artifact is quantified, not asserted.
                lf, rf = left.astype("float64"), right.astype("float64")
                near_only.append(f"{column} (max |delta| = {float((lf - rf).abs().max()):.3e})")
            continue
        mismatched.append(column)

    if dtype_only:
        print(f"INFO: {len(dtype_only)} column(s) differ only in pandas dtype, not in value:")
        for item in dtype_only:
            print(f"  {item}")
    if near_only:
        print(
            f"INFO: {len(near_only)} column(s) match within tolerance but not bitwise "
            "(float aggregation order):"
        )
        for item in near_only:
            print(f"  {item}")
    if mismatched:
        print(f"FAIL: future information leaked into {len(mismatched)} column(s): {mismatched}")
        for column in mismatched:
            left, right = full[column], truncated[column]
            lf, rf = left.astype("float64"), right.astype("float64")
            diff = ~((lf.isna() & rf.isna()) | ((lf - rf).abs() <= 1e-9).fillna(False))
            print(
                f"  {column}: {int(diff.sum())} differing rows; first at index {int(diff.idxmax())}"
            )
        return 1

    print(
        f"PASS: no leakage — all {len(full.columns)} feature columns on "
        f"{len(full):,} rows with as_of_week <= {cutoff} are unchanged when every "
        f"week > {cutoff} is removed from the sources"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
