"""Frozen Phase 4 sample construction; no outcome models or activity filtering."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from complete_journey.paths import PROJECT_ROOT


class PanelContractError(RuntimeError):
    """A structural failure, never an instruction to clean away observations."""


@dataclass(frozen=True)
class SampleSpec:
    name: str
    population: str
    flag: str
    first_response: int
    last_response: int
    households: int
    cells: int


SAMPLES = (
    SampleSpec("S0_baseline", "S0", "sample_full", 2, 102, 2500, 252500),
    SampleSpec("S0_full_week_pair", "S0", "sample_full", 3, 101, 2500, 247500),
    SampleSpec("S0_post_formation", "S0", "sample_full", 21, 101, 2500, 202500),
    SampleSpec("S1_baseline", "S1", "sample_sustained_support", 2, 102, 2115, 213615),
    SampleSpec("S2_baseline", "S2", "sample_higher_support", 2, 102, 1807, 182507),
)
LAGS = {
    "lag1_retailer_purchase": "retailer_purchase",
    "lag1_weekly_sales_value": "weekly_sales_value",
    "lag1_trips": "trips",
    "lag1_units": "units",
}
PHYSICAL_DOUBLE_INTEGER_FIELDS = (
    "units",
    "qa_quantity_zero_lines",
    "qa_retail_disc_positive_lines",
    "qa_implied_paid_negative_lines",
)
ACTIVITY_FIELDS = (
    "trips",
    "weekly_sales_value",
    "units",
    "distinct_products",
    "distinct_commodities",
    "active_store_count",
    "transaction_line_count",
    "weekly_retail_disc",
    "weekly_coupon_disc",
    "weekly_coupon_match_disc",
    *PHYSICAL_DOUBLE_INTEGER_FIELDS[1:],
)


def require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise PanelContractError(message)


def ordered_grid(frame: pd.DataFrame, households: int, weeks: tuple[int, ...]) -> pd.DataFrame:
    """Assert rectangular calendar support rather than invent missing observations."""
    require(not frame[["household_key", "week_no"]].isna().any().any(), "null panel key")
    require(not frame.duplicated(["household_key", "week_no"]).any(), "duplicate panel key")
    require(frame.household_key.nunique() == households, "household count mismatch")
    require(tuple(sorted(frame.week_no.unique())) == weeks, "calendar domain mismatch")
    require(len(frame) == households * len(weeks), "rectangular cell count mismatch")
    ordered = frame.sort_values(["household_key", "week_no"]).reset_index(drop=True)
    grid_weeks = ordered.week_no.to_numpy().reshape(households, len(weeks))
    require(
        np.array_equal(grid_weeks, np.broadcast_to(weeks, grid_weeks.shape)),
        "incomplete household calendar",
    )
    return ordered


def validate_lags(frame: pd.DataFrame, households: int = 2500, periods: int = 102) -> dict:
    """Independent calendar-neighbor check on the full grid, before sample selection."""
    ordered = ordered_grid(frame, households, tuple(range(1, periods + 1)))
    report = {}
    for lag, current in LAGS.items():
        a = ordered[lag].to_numpy(dtype=float).reshape(households, periods)
        b = ordered[current].to_numpy(dtype=float).reshape(households, periods)
        require(np.isnan(a[:, 0]).all(), f"{lag}: first-week history must be NULL")
        require(np.isfinite(a[:, 1:]).all(), f"{lag}: unavailable history after week 1")
        if current == "weekly_sales_value":
            discrepancy = float(np.max(np.abs(a[:, 1:] - b[:, :-1])))
            require(
                np.allclose(a[:, 1:], b[:, :-1], atol=1e-9, rtol=1e-12),
                f"{lag}: off-by-one or value mismatch",
            )
        else:
            discrepancy = float(np.max(np.abs(a[:, 1:] - b[:, :-1])))
            require(np.array_equal(a[:, 1:], b[:, :-1]), f"{lag}: off-by-one or value mismatch")
        report[lag] = {
            "pairs": households * (periods - 1),
            "first_week_nulls": households,
            "max_abs_difference": discrepancy,
        }
    return report


def validate_mart(frame: pd.DataFrame, base: pd.DataFrame) -> dict:
    ordered = ordered_grid(frame, 2500, tuple(range(1, 103)))
    reference = ordered_grid(base, 2500, tuple(range(1, 103)))
    require(frame.columns.is_unique and base.columns.is_unique, "duplicate column name")
    for column in reference.columns:
        require(column in ordered, f"base field missing: {column}")
        a, b = ordered[column].to_numpy(), reference[column].to_numpy()
        require(np.array_equal(a, b), f"mart/base reconciliation failure: {column}")
    allowed_missing = set(LAGS)
    for column in ordered:
        values = ordered[column].to_numpy(dtype=float)
        if column not in allowed_missing:
            require(np.isfinite(values).all(), f"nonfinite field: {column}")
    require(ordered.retailer_purchase.isin([0, 1]).all(), "nonbinary participation")
    purchase = ordered.retailer_purchase.to_numpy().reshape(2500, 102)
    counts = purchase.sum(axis=1)
    thirds = np.logical_and.reduce(
        [purchase[:, lo:hi].sum(axis=1) > 0 for lo, hi in ((0, 34), (34, 68), (68, 102))]
    )
    expected = {
        "sample_full": np.ones(2500, dtype=bool),
        "sample_sustained_support": (counts >= 20) & thirds,
        "sample_higher_support": (counts >= 30) & thirds,
    }
    populations = {}
    for flag, mask in expected.items():
        actual = ordered[flag].to_numpy().reshape(2500, 102)
        require(
            np.array_equal(actual, np.broadcast_to(mask[:, None], actual.shape)),
            f"sample flag definition or constancy failure: {flag}",
        )
        populations[flag] = int(mask.sum())
    require(tuple(populations.values()) == (2500, 2115, 1807), "frozen population counts differ")
    zeros = ordered.retailer_purchase == 0
    require(
        (ordered.loc[zeros, list(ACTIVITY_FIELDS)] == 0).all().all(),
        "nonpurchase row carries recorded activity",
    )
    require((ordered.loc[~zeros, "trips"] >= 1).all(), "purchase without basket")
    integral_fields = (*PHYSICAL_DOUBLE_INTEGER_FIELDS, "lag1_units", "trips", "lag1_trips")
    for column in integral_fields:
        values = ordered[column].dropna().to_numpy(dtype=float)
        require(
            np.isfinite(values).all() and np.equal(values, np.rint(values)).all(),
            f"logical integer values not integral: {column}",
        )
    return {
        "rows": len(ordered),
        "households": 2500,
        "weeks": 102,
        "populations": populations,
        "integral_values_verified_without_integer_cast": list(integral_fields),
        "lag_checks": validate_lags(ordered),
    }


def load_mart(root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, dict]:
    """Read accepted artifacts only; callers verify hashes before invoking this."""
    paths = {
        name: root / "data/processed" / f"{name}.parquet"
        for name in ("econometrics_household_week", "household_week_base")
    }
    declarations = json.loads((root / "outputs/tables/panels/contracts.json").read_text())
    declared = {table["table"]: table["required_columns"] for table in declarations["tables"]}
    for name, path in paths.items():
        metadata = pq.ParquetFile(path)
        require(
            metadata.schema_arrow.names == [column for column, _ in declared[name]],
            f"{name}: exact column order/schema mismatch",
        )
        fields = (
            (*PHYSICAL_DOUBLE_INTEGER_FIELDS, "lag1_units")
            if name == "econometrics_household_week"
            else PHYSICAL_DOUBLE_INTEGER_FIELDS
        )
        require(metadata.metadata.num_rows == 255000, f"{name}: footer row count")
        require(
            all(str(metadata.schema_arrow.field(c).type) == "double" for c in fields),
            f"{name}: accepted serialized type discrepancy changed",
        )
        for column, family in declared[name]:
            physical = metadata.schema_arrow.field(column).type
            if family == "integer" and column not in fields:
                require(
                    pa.types.is_integer(physical), f"{name}: unexpected noninteger storage {column}"
                )
            elif family == "double" or column in fields:
                require(
                    pa.types.is_floating(physical),
                    f"{name}: unexpected nonfloating storage {column}",
                )
    mart = pd.read_parquet(paths["econometrics_household_week"])
    base = pd.read_parquet(paths["household_week_base"])
    report = validate_mart(mart, base)
    return mart, report


def construct_sample(mart: pd.DataFrame, spec: SampleSpec) -> pd.DataFrame:
    """Select whole household flags and common response calendar, never activity."""
    chosen = mart.loc[
        (mart[spec.flag] == 1) & mart.week_no.between(spec.first_response, spec.last_response)
    ].copy()
    require(
        len(chosen) == spec.cells, f"{spec.name}: expected {spec.cells} cells, got {len(chosen)}"
    )
    chosen = ordered_grid(
        chosen, spec.households, tuple(range(spec.first_response, spec.last_response + 1))
    )
    require(chosen[list(LAGS)].notna().all().all(), f"{spec.name}: missing accepted history")
    return chosen


def sample_contract() -> list[dict]:
    return [asdict(spec) for spec in SAMPLES]
