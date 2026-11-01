"""Phase 3 Question A: retailer activity over time (descriptive only).

Question A asks what recorded retailer activity looks like over the 102-week
observation window and how household participation (the extensive margin) and
activity intensity (the intensive margins) contribute to its movement. Everything
here is descriptive: no regression, no time-series or seasonal model, no causal or
counterfactual statement, and no promoted finding — Question A produces *candidates*
that await a researcher promotion decision.

Frozen conventions applied (``docs/phases/PHASE_03_DESCRIPTIVE_ANALYTICS.md``):
S0 = all 2,500 households with no filtering to S1/S2; ``SALES_VALUE`` is retailer
revenue; a non-active household-week is no recorded purchase from this retailer, not
zero consumption; trips are baskets; weeks 1 and 102 are partial and are kept and
annotated rather than normalized; every rate carries its denominator; no normalized
discount share, no price measure and no counterfactual/reweighted series exists.

Every weekly headline total is computed twice — from the household-week grid and
from the atomic basket table — and must agree before anything is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot)

from complete_journey.descriptive import artifacts, contracts

SQL_DIR: Final[Path] = contracts.DESCRIPTIVE_SQL_DIR

SERIES_SQL: Final[Path] = SQL_DIR / "02_weekly_activity_series.sql"
BASKET_SQL: Final[Path] = SQL_DIR / "03_weekly_activity_series_basket_grain.sql"
COMPOSITION_SQL: Final[Path] = SQL_DIR / "04_weekly_composition.sql"
DEPARTMENT_COMPOSITION_SQL: Final[Path] = SQL_DIR / "05_weekly_department_composition.sql"
DIAGNOSTICS_SQL: Final[Path] = SQL_DIR / "06_weekly_diagnostics.sql"
IN_SUPPORT_SQL: Final[Path] = SQL_DIR / "07_weekly_in_support_and_calendar.sql"

#: Level series (the extensive margin and size measures).
LEVEL_SERIES: Final[tuple[str, ...]] = (
    "total_retailer_sales_value",
    "active_households",
    "trips",
    "units",
)

#: Intensive / rate-like series, each with an explicit denominator.
INTENSIVE_SERIES: Final[tuple[str, ...]] = (
    "retailer_sales_value_per_active_household_week",
    "trips_per_active_household",
    "units_per_active_household",
    "retailer_sales_value_per_trip",
    "units_per_trip",
)

#: Series that carry the full-week descriptive summary table.
SUMMARY_SERIES: Final[tuple[str, ...]] = (
    "total_retailer_sales_value",
    "active_households",
    "trips",
    "units",
    "retailer_sales_value_per_active_household_week",
    "trips_per_active_household",
    "retailer_sales_value_per_trip",
)

#: Series compared on a standardized scale (levels + the two margins).
STANDARDIZED_SERIES: Final[tuple[str, ...]] = (
    "total_retailer_sales_value",
    "active_households",
    "retailer_sales_value_per_active_household_week",
    "retailer_sales_value_per_trip",
)

#: A week is flagged as extreme when a standardized full-week series leaves
#: |z| >= this value. Declared, not tuned.
EXTREME_Z: Final[float] = 2.0

#: Bulk-line threshold for the extreme-quantity mechanical check. Declared, not
#: tuned; the Phase 1 audit reports a maximum line quantity of 89,638 units.
EXTREME_QUANTITY_THRESHOLD: Final[int] = 1000

#: Total retailer SALES_VALUE excluded from the line-grain route by construction
#: (the accepted line table holds positive-quantity lines only).
LINE_ROUTE_CAVEAT: Final[str] = (
    "line-grain route (positive-quantity lines only; excludes the 14,466 QUANTITY = 0 "
    "lines, which put its window total 19.28 below the accepted retailer SALES_VALUE total)"
)

#: Declared comparison windows for the participation-vs-intensity question. The
#: boundary at week 20 is data-derived, not tuned: 2,493 of 2,500 households have
#: first_purchase_week <= 20 in household_history, so the observed purchase-support
#: structure has essentially formed by then, and the last three weeks are excluded
#: separately because that structure is thinning again (trailing truncation of the
#: observation window) and week 102 is partial.
DECLARED_WINDOWS: Final[tuple[tuple[int, int, str], ...]] = (
    (2, 101, "all full weeks (baseline)"),
    (2, 19, "ramp: observed support still forming"),
    (20, 101, "post support formation"),
    (20, 99, "post support formation, excluding the tapering weeks 100-102"),
)

#: Week at which the observed purchase-support structure counts as formed
POOL_FORMATION_WEEK: Final[int] = 20

#: A day counts as near-empty when its retailer SALES_VALUE is below this
#: fraction of the window's median day. Declared, not tuned.
LOW_ACTIVITY_DAY_FRACTION: Final[float] = 0.25


def _sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _frame(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    return con.execute(_sql(path)).fetchdf().sort_values("week_no").reset_index(drop=True)


# --- series construction and reconciliation ----------------------------------


def weekly_series(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Row-grain weekly series from the accepted household-week grid."""
    return _frame(con, SERIES_SQL)


def basket_grain_series(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Independent weekly series from the atomic basket table."""
    return _frame(con, BASKET_SQL)


def series_cross_check(row_grain: pd.DataFrame, basket_grain: pd.DataFrame) -> pd.DataFrame:
    """Per-week comparison of the two independent routes for shared measures."""
    shared = (
        "total_retailer_sales_value",
        "active_households",
        "trips",
        "units",
    )
    merged = row_grain[["week_no", *shared]].merge(
        basket_grain[["week_no", *shared]],
        on="week_no",
        suffixes=("_row_grain", "_basket_grain"),
    )
    for measure in shared:
        merged[f"{measure}_abs_diff"] = (
            merged[f"{measure}_row_grain"] - merged[f"{measure}_basket_grain"]
        ).abs()
    merged["observed_days_in_week"] = basket_grain["observed_days_in_week"].to_numpy()
    merged["is_partial_week"] = merged["observed_days_in_week"] < 7
    return merged


def reconcile_series_totals(
    series: pd.DataFrame, anchors: contracts.ReconciliationAnchors = contracts.ACCEPTED_ANCHORS
) -> dict[str, float]:
    """Reconcile the weekly series against the accepted anchors.

    Returns residuals and deviations; the caller fails loudly on any miss.
    """
    return {
        "weeks": float(len(series)),
        "weeks_residual": float(len(series) - anchors.weeks),
        "retailer_sales_value_residual": float(
            series["total_retailer_sales_value"].sum() - anchors.retailer_sales_value
        ),
        "units_residual": float(series["units"].sum() - anchors.units),
        "trips_residual": float(series["trips"].sum() - anchors.baskets),
        "active_household_weeks_residual": float(
            series["active_households"].sum() - anchors.active_household_weeks
        ),
        "household_week_cells_residual": float(
            series["household_week_cells"].sum() - anchors.household_weeks
        ),
    }


# --- descriptive summaries ----------------------------------------------------


def full_weeks(series: pd.DataFrame) -> pd.DataFrame:
    """Weeks 2-101 only: the explicitly labelled full-week range."""
    low, high = contracts.FULL_WEEK_RANGE
    return series[(series["week_no"] >= low) & (series["week_no"] <= high)]


def summary_statistics(
    series: pd.DataFrame, columns: tuple[str, ...] = SUMMARY_SERIES
) -> pd.DataFrame:
    """Descriptive distribution of full-week values.

    Sample standard deviation (ddof = 1) and linear-interpolated quantiles, as
    computed by pandas. These are descriptive summaries, not inference.
    """
    frame = full_weeks(series)
    rows: list[dict[str, float | str]] = []
    for column in columns:
        values = frame[column]
        rows.append(
            {
                "series": column,
                "unit": _unit_of(column),
                "denominator": _denominator_of(column),
                "n_full_weeks": float(values.count()),
                "mean": float(values.mean()),
                "std_ddof1": float(values.std(ddof=1)),
                "min": float(values.min()),
                "p10": float(values.quantile(0.10)),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _unit_of(column: str) -> str:
    if "sales_value" in column:
        return "retailer SALES_VALUE (retailer revenue)"
    if column in ("active_households",):
        return "households"
    if column == "trips":
        return "baskets"
    return "units"


def _denominator_of(column: str) -> str:
    return {
        "total_retailer_sales_value": "whole 102-week window (sum of weekly values)",
        "active_households": "2,500 observed households",
        "trips": "whole window",
        "units": "whole window",
        "retailer_sales_value_per_active_household_week": "active household-weeks",
        "trips_per_active_household": "active household-weeks",
        "units_per_active_household": "active household-weeks",
        "retailer_sales_value_per_trip": "baskets (trips)",
        "units_per_trip": "baskets (trips)",
    }[column]


def partial_week_positions(
    series: pd.DataFrame, columns: tuple[str, ...] = SUMMARY_SERIES
) -> pd.DataFrame:
    """Where weeks 1 and 102 sit inside the full-week distribution.

    Reported as a position, never as an anomaly: both weeks cover fewer observed
    days by the accepted week mapping, so a low level there is a coverage fact.
    """
    frame = full_weeks(series)
    rows: list[dict[str, float | str | int]] = []
    for week in sorted(contracts.PARTIAL_WEEK_DAYS):
        row: dict[str, float | str | int] = {
            "week_no": week,
            "observed_days": contracts.PARTIAL_WEEK_DAYS[week],
        }
        for column in columns:
            value = float(series.loc[series["week_no"] == week, column].iloc[0])
            row[f"{column}"] = value
            row[f"{column}__full_week_percentile"] = float((frame[column] <= value).mean() * 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def standardized_series(
    series: pd.DataFrame, columns: tuple[str, ...] = STANDARDIZED_SERIES
) -> pd.DataFrame:
    """Z-scores against the FULL-WEEK mean and standard deviation.

    Standardization is documented exactly: ``z = (x - mean(weeks 2-101)) / sd(weeks
    2-101)`` with sample sd (ddof = 1). Weeks 1 and 102 are transformed with those
    full-week constants and remain visible as partial weeks; they never enter the
    mean or the standard deviation. Standardizing changes the scale only — every
    level series is also reported (and plotted) in its own units.
    """
    frame = full_weeks(series)
    out = pd.DataFrame({"week_no": series["week_no"].to_numpy()})
    for column in columns:
        mean = float(frame[column].mean())
        sd = float(frame[column].std(ddof=1))
        out[f"{column}__z"] = (series[column] - mean) / sd if sd else float("nan")
        out[f"{column}__full_week_mean"] = mean
        out[f"{column}__full_week_sd"] = sd
    out["is_partial_week"] = series["week_no"].isin(contracts.PARTIAL_WEEK_DAYS)
    return out


def correlation_table(
    series: pd.DataFrame, target: str = "total_retailer_sales_value"
) -> pd.DataFrame:
    """Descriptive Pearson correlations of weekly series with the weekly total.

    Pearson correlation of levels can be dominated by a shared trend, so the
    correlation of week-to-week changes is reported beside it. Neither is a causal
    or explanatory statement: they say which series moves alongside the total, not
    which one drives it.
    """
    others = (
        "active_households",
        "retailer_sales_value_per_active_household_week",
        "trips",
        "retailer_sales_value_per_trip",
        "trips_per_active_household",
    )
    full = full_weeks(series)
    rows: list[dict[str, float | str]] = []
    for column in others:
        levels_all = _corr(series[target], series[column])
        levels_full = _corr(full[target], full[column])
        changes_all = _corr(series[target].diff(), series[column].diff())
        changes_full = _corr(full[target].diff(), full[column].diff())
        rows.append(
            {
                "series": column,
                "pearson_levels_all_weeks": float(levels_all),
                "pearson_levels_full_weeks": float(levels_full),
                "pearson_week_changes_all_weeks": float(changes_all),
                "pearson_week_changes_full_weeks": float(changes_full),
            }
        )
    return pd.DataFrame(rows)


def flag_extreme_weeks(
    series: pd.DataFrame, columns: tuple[str, ...] = LEVEL_SERIES + INTENSIVE_SERIES
) -> pd.DataFrame:
    """Weeks whose standardized full-week value leaves |z| >= EXTREME_Z."""
    z = standardized_series(series, columns)
    flagged = z[["week_no", "is_partial_week"]].copy()
    z_columns = [f"{column}__z" for column in columns]
    flagged["max_abs_z"] = z[z_columns].abs().max(axis=1)
    flagged["extreme_columns"] = [
        ",".join(column for column in columns if abs(z.loc[index, f"{column}__z"]) >= EXTREME_Z)
        for index in z.index
    ]
    flagged = flagged[flagged["max_abs_z"] >= EXTREME_Z].reset_index(drop=True)
    return flagged


# --- composition and mechanical diagnostics ----------------------------------


def weekly_composition(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Weekly commodity-mix descriptors (mechanical explanation check)."""
    return _frame(con, COMPOSITION_SQL)


def department_composition(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Weekly department-mix descriptors from the validated Phase 3 derivation."""
    return _frame(con, DEPARTMENT_COMPOSITION_SQL)


def week_diagnostics(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-week concentration, bulk-line and anomaly diagnostics."""
    return _frame(con, DIAGNOSTICS_SQL)


def in_support_and_calendar(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-week observed purchase support, participation share and day-level coverage."""
    return _frame(con, IN_SUPPORT_SQL)


#: Pool / calendar columns merged into the weekly analysis frame.
IN_SUPPORT_COLUMNS: Final[tuple[str, ...]] = (
    "in_support_households",
    "active_share_of_in_support",
    "min_day_sales_value",
    "max_day_sales_value",
    "lowest_day_number",
    "highest_day_number",
    "week_sales_value_excluding_lowest_day",
    "has_low_activity_day",
    "median_day_sales_value",
    "local_median_day_sales_value",
    "local_median_week_sales_value",
    "week_share_of_local_median",
)


def window_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Declared-window correlations of weekly total value with its components.

    These are descriptive associations, not a decomposition and not causal: the
    weekly total is arithmetically ``active households x retailer SALES_VALUE per
    active household-week``, so the two components are related to each other and to
    the total by construction. The declared windows exist because the first weeks
    are dominated by households entering the observation window.
    """
    rows: list[dict[str, object]] = []
    for low, high, label in DECLARED_WINDOWS:
        window = frame[(frame["week_no"] >= low) & (frame["week_no"] <= high)]
        total = window["total_retailer_sales_value"]
        rows.append(
            {
                "window": label,
                "week_from": low,
                "week_to": high,
                "n_weeks": len(window),
                "corr_levels_active_households": _corr(total, window["active_households"]),
                "corr_levels_active_share_of_in_support": _corr(
                    total, window["active_share_of_in_support"]
                ),
                "corr_levels_value_per_active_household_week": _corr(
                    total, window["retailer_sales_value_per_active_household_week"]
                ),
                "corr_levels_value_per_trip": _corr(total, window["retailer_sales_value_per_trip"]),
                "corr_changes_active_households": _corr(
                    total.diff(), window["active_households"].diff()
                ),
                "corr_changes_value_per_active_household_week": _corr(
                    total.diff(),
                    window["retailer_sales_value_per_active_household_week"].diff(),
                ),
                "corr_changes_value_per_trip": _corr(
                    total.diff(), window["retailer_sales_value_per_trip"].diff()
                ),
            }
        )
    return pd.DataFrame(rows)


def window_correlation_cross_check(frame: pd.DataFrame, basket_grain: pd.DataFrame) -> pd.DataFrame:
    """Recompute the headline window correlations from the independent basket grain.

    The participation and intensity series are rebuilt from the atomic basket table
    (a different aggregation path) and the same declared-window correlations are
    recomputed, so a reported correlation is not resting on a single derivation.
    """
    measures = ("total_retailer_sales_value", "active_households", "trips")
    merged = frame[["week_no", *measures]].merge(
        basket_grain[["week_no", *measures]], on="week_no", suffixes=("_row_grain", "_basket_grain")
    )
    for suffix in ("row_grain", "basket_grain"):
        merged[f"value_per_active_{suffix}"] = (
            merged[f"total_retailer_sales_value_{suffix}"] / merged[f"active_households_{suffix}"]
        )
        merged[f"value_per_trip_{suffix}"] = (
            merged[f"total_retailer_sales_value_{suffix}"] / merged[f"trips_{suffix}"]
        )
    rows: list[dict[str, object]] = []
    for low, high, label in DECLARED_WINDOWS:
        window = merged[(merged["week_no"] >= low) & (merged["week_no"] <= high)]
        row: dict[str, object] = {
            "window": label,
            "week_from": low,
            "week_to": high,
            "n_weeks": len(window),
        }
        for suffix in ("row_grain", "basket_grain"):
            row[f"corr_levels_active_{suffix}"] = _corr(
                window[f"total_retailer_sales_value_{suffix}"],
                window[f"active_households_{suffix}"],
            )
            row[f"corr_levels_value_per_active_{suffix}"] = _corr(
                window[f"total_retailer_sales_value_{suffix}"], window[f"value_per_active_{suffix}"]
            )
            row[f"corr_levels_value_per_trip_{suffix}"] = _corr(
                window[f"total_retailer_sales_value_{suffix}"], window[f"value_per_trip_{suffix}"]
            )
            row[f"corr_changes_value_per_active_{suffix}"] = _corr(
                window[f"total_retailer_sales_value_{suffix}"].diff(),
                window[f"value_per_active_{suffix}"].diff(),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def thirds_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Descriptive means by sample third (weeks 2-34, 35-68, 69-102)."""
    rows: list[dict[str, object]] = []
    for low, high, label in (
        (2, 34, "third 1 (weeks 2-34)"),
        (35, 68, "third 2 (35-68)"),
        (69, 102, "third 3 (69-102)"),
    ):
        window = frame[(frame["week_no"] >= low) & (frame["week_no"] <= high)]
        rows.append(
            {
                "third": label,
                "n_weeks": len(window),
                "mean_total_retailer_sales_value": float(
                    window["total_retailer_sales_value"].mean()
                ),
                "mean_active_households": float(window["active_households"].mean()),
                "mean_in_support_households": float(window["in_support_households"].mean()),
                "mean_active_share_of_in_support": float(
                    window["active_share_of_in_support"].mean()
                ),
                "mean_trips": float(window["trips"].mean()),
                "mean_units": float(window["units"].mean()),
                "mean_value_per_active_household_week": float(
                    window["retailer_sales_value_per_active_household_week"].mean()
                ),
                "mean_trips_per_active_household": float(
                    window["trips_per_active_household"].mean()
                ),
                "mean_value_per_trip": float(window["retailer_sales_value_per_trip"].mean()),
            }
        )
    return pd.DataFrame(rows)


def campaign_context_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Descriptive co-movement with campaign timing (context only, never an effect).

    Campaign listings cluster in the later part of the window while the household
    in-support set also changes, so a co-movement here has at least two mechanical candidates
    before any campaign reading. Nothing in Phase 3 attributes a movement to a
    campaign, tests a response, or defines exposure.
    """
    rows: list[dict[str, object]] = []
    for low, high, label in DECLARED_WINDOWS:
        window = frame[(frame["week_no"] >= low) & (frame["week_no"] <= high)]
        campaigns = window["active_listed_campaigns"]
        no_variance = bool(campaigns.nunique() <= 1)
        rows.append(
            {
                "window": label,
                "n_weeks": len(window),
                "weeks_with_no_listed_campaign_in_span": int((campaigns == 0).sum()),
                "corr_levels_active_listed_campaigns_vs_total": _corr(
                    campaigns, window["total_retailer_sales_value"]
                ),
                "corr_levels_active_listed_campaigns_vs_active_households": _corr(
                    campaigns, window["active_households"]
                ),
                "corr_levels_active_listed_campaigns_vs_value_per_trip": _corr(
                    campaigns, window["retailer_sales_value_per_trip"]
                ),
                "note": (
                    "correlation undefined: every week in this window has the same "
                    "number of listed campaigns in span (zero variance)"
                    if no_variance
                    else "listed-campaign timing context only; not exposure, not an effect"
                ),
            }
        )
    return pd.DataFrame(rows)


def bulk_line_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """How much of weekly units comes from bulk lines (QUANTITY >= threshold).

    Units are SUM(QUANTITY) and QUANTITY is recorded in units; the Phase 1 audit
    shows a maximum line quantity of 89,638, i.e. bulk lines rather than baskets.
    This summary states how concentrated that is, so a units-based reading is not
    mistaken for household consumption volume.
    """
    share = diagnostics["extreme_quantity_unit_share"].dropna()
    rows = [
        ("threshold_units", float(EXTREME_QUANTITY_THRESHOLD)),
        ("weeks_observed", float(share.count())),
        ("median_share_of_units_from_bulk_lines", float(share.median())),
        ("p10_share_of_units_from_bulk_lines", float(share.quantile(0.10))),
        ("min_share_of_units_from_bulk_lines", float(share.min())),
        ("median_bulk_lines_per_week", float(diagnostics["extreme_quantity_lines"].median())),
        ("min_bulk_lines_per_week", float(diagnostics["extreme_quantity_lines"].min())),
        ("max_bulk_lines_per_week", float(diagnostics["extreme_quantity_lines"].max())),
        ("max_line_quantity_in_window", float(diagnostics["max_line_quantity"].max())),
        (
            "weeks_where_bulk_lines_carry_over_95pct_of_units",
            float((share > 0.95).sum()),
        ),
        (
            "median_share_of_weekly_value_from_bulk_lines",
            float(
                (
                    diagnostics["extreme_quantity_sales_value"] / diagnostics["week_sales_value"]
                ).median()
            ),
        ),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value"])


def near_empty_day_facts(frame: pd.DataFrame) -> pd.DataFrame:
    """Weeks containing a near-empty trading day, with a labelled sensitivity.

    The as-is week total is reported first; the sensitivity replaces the week's
    lowest day with the window's median day so a reader can see how much of the
    week's level that single day accounts for. The baseline is never replaced.
    """
    flagged = frame[frame["has_low_activity_day"] == 1]
    rows: list[dict[str, object]] = []
    for record in flagged.itertuples(index=False):
        median_day = float(record.local_median_day_sales_value)
        sensitivity = float(record.week_sales_value_excluding_lowest_day) + median_day
        rows.append(
            {
                "week_no": int(record.week_no),
                "lowest_day_number": int(record.lowest_day_number),
                "lowest_day_sales_value": float(record.min_day_sales_value),
                "local_median_day_sales_value": median_day,
                "window_median_day_sales_value": float(record.median_day_sales_value),
                "week_total_as_is": float(record.total_retailer_sales_value),
                "week_share_of_local_median": float(record.week_share_of_local_median),
                "labelled_sensitivity_week_total_if_lowest_day_were_typical": sensitivity,
                "sensitivity_share_of_local_median": sensitivity
                / float(record.local_median_week_sales_value),
                "active_households": int(record.active_households),
                "lowest_day_share_of_week_total": float(
                    record.min_day_sales_value / record.total_retailer_sales_value
                ),
            }
        )
    return pd.DataFrame(rows)


def build_weekly_frame(
    con: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assemble the joined weekly analysis frame (levels + composition + context)."""
    series = weekly_series(con)
    basket = basket_grain_series(con)
    composition = weekly_composition(con)
    departments = department_composition(con)
    frame = series.merge(
        basket[["week_no", "observed_days_in_week", "first_observed_day", "last_observed_day"]],
        on="week_no",
        how="left",
    )
    frame = frame.merge(composition, on="week_no", how="left")
    frame = frame.merge(
        departments[
            [
                "week_no",
                "active_departments",
                "largest_department_desc",
                "largest_department_sales_value_share",
                "top3_department_sales_value_share",
            ]
        ],
        on="week_no",
        how="left",
    )
    in_support = in_support_and_calendar(con)
    frame = frame.merge(in_support[["week_no", *IN_SUPPORT_COLUMNS]], on="week_no", how="left")
    frame["is_partial_week"] = frame["observed_days_in_week"] < 7
    return frame, basket, in_support


# --- figures ------------------------------------------------------------------


def _mark_partial_weeks(ax: plt.Axes, frame: pd.DataFrame) -> None:
    partial = frame[frame["is_partial_week"]]
    for week in partial["week_no"]:
        ax.axvspan(week - 0.5, week + 0.5, color="0.85", zorder=0)
    ax.annotate(
        "weeks 1 and 102 shaded: partial coverage (5 and 6 observed days)",
        xy=(0.01, 0.02),
        xycoords="axes fraction",
        fontsize=7,
        color="0.35",
    )


def figure_levels(frame: pd.DataFrame, path: Path) -> Path:
    """F1 candidate (exploratory): weekly levels plus campaign-timing context."""
    panels = (
        ("total_retailer_sales_value", "retailer SALES_VALUE"),
        ("active_households", "active households"),
        ("trips", "trips (baskets)"),
        ("units", "units"),
    )
    figure, axes = plt.subplots(len(panels) + 1, 1, figsize=(11, 12), sharex=True)
    for ax, (column, label) in zip(axes[:-1], panels, strict=True):
        ax.plot(frame["week_no"], frame[column], linewidth=1.2, color="#1f4e79")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.25, linewidth=0.5)
        _mark_partial_weeks(ax, frame)
    context = axes[-1]
    context.bar(
        frame["week_no"],
        frame["active_listed_campaigns"],
        color="0.6",
        width=0.8,
    )
    context.set_ylabel("campaigns in span\n(context only)", fontsize=8)
    context.set_xlabel("retailer week (1-102)")
    context.grid(alpha=0.25, linewidth=0.5)
    context.annotate(
        "listed-campaign timing context only: overlap is not exposure and not an effect",
        xy=(0.01, 0.86),
        xycoords="axes fraction",
        fontsize=7,
        color="0.35",
    )
    figure.suptitle("Weekly retailer activity (S0: 2,500 households, weeks 1-102)", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    result = artifacts.write_figure_png(figure, path)
    plt.close(figure)
    return result


def figure_intensive(frame: pd.DataFrame, path: Path) -> Path:
    """F2 candidate (exploratory): the intensive margins in their own units."""
    panels = (
        (
            "retailer_sales_value_per_active_household_week",
            "retailer SALES_VALUE per\nactive household-week",
        ),
        ("trips_per_active_household", "trips per\nactive household"),
        ("retailer_sales_value_per_trip", "retailer SALES_VALUE\nper trip"),
    )
    figure, axes = plt.subplots(len(panels), 1, figsize=(11, 9), sharex=True)
    for ax, (column, label) in zip(axes, panels, strict=True):
        ax.plot(frame["week_no"], frame[column], linewidth=1.2, color="#7f3f00")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.25, linewidth=0.5)
        _mark_partial_weeks(ax, frame)
    axes[-1].set_xlabel("retailer week (1-102)")
    figure.suptitle("Intensive margins per week (own units; denominators stated)", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    result = artifacts.write_figure_png(figure, path)
    plt.close(figure)
    return result


def figure_standardized(frame: pd.DataFrame, path: Path) -> Path:
    """Exploratory standardized comparison (scale-only; levels ship separately)."""
    z = standardized_series(frame)
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for column in ("total_retailer_sales_value", "active_households"):
        axes[0].plot(z["week_no"], z[f"{column}__z"], linewidth=1.2, label=column)
    axes[0].axhline(0, color="0.4", linewidth=0.8)
    axes[0].set_ylabel("z (full-week scale)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].grid(alpha=0.25, linewidth=0.5)
    _mark_partial_weeks(axes[0], frame)
    for column in (
        "retailer_sales_value_per_active_household_week",
        "retailer_sales_value_per_trip",
    ):
        axes[1].plot(z["week_no"], z[f"{column}__z"], linewidth=1.2, label=column)
    axes[1].axhline(0, color="0.4", linewidth=0.8)
    axes[1].set_ylabel("z (full-week scale)", fontsize=8)
    axes[1].set_xlabel("retailer week (1-102)")
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].grid(alpha=0.25, linewidth=0.5)
    _mark_partial_weeks(axes[1], frame)
    figure.suptitle(
        "Standardized comparison: z-scores use full-week (2-101) mean and sd; levels are reported separately",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    result = artifacts.write_figure_png(figure, path)
    plt.close(figure)
    return result


def figure_diagnostics(frame: pd.DataFrame, diagnostics: pd.DataFrame, path: Path) -> Path:
    """Exploratory diagnostics: observed support, participation share, concentration, bulk lines."""
    merged = frame[["week_no", "in_support_households", "active_share_of_in_support"]].merge(
        diagnostics, on="week_no", how="left"
    )
    figure, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    axes[0].plot(merged["week_no"], merged["in_support_households"], linewidth=1.2)
    axes[0].set_ylabel("households inside their own\nobserved purchase-support window", fontsize=8)
    axes[1].plot(merged["week_no"], 100 * merged["active_share_of_in_support"], linewidth=1.2)
    axes[1].set_ylabel("active share of\nin-support households (%)", fontsize=8)
    axes[2].plot(
        merged["week_no"], 100 * merged["top10_household_sales_value_share"], linewidth=1.0
    )
    axes[2].set_ylabel("top-10 household\nshare of week (%)", fontsize=8)
    axes[3].plot(merged["week_no"], 100 * merged["extreme_quantity_unit_share"], linewidth=1.0)
    axes[3].set_ylabel("unit share of lines with\nQUANTITY >= 1000 (%)", fontsize=8)
    axes[4].plot(merged["week_no"], merged["discount_exceedance_lines"], linewidth=1.0)
    axes[4].set_ylabel("lines where recorded\ndiscounts exceed SALES_VALUE", fontsize=8)
    axes[4].set_xlabel("retailer week (1-102)")
    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.5)
        _mark_partial_weeks(ax, frame)
    figure.suptitle(
        "Mechanical diagnostics for the Question A challenges (as-is data; no trimming)",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    result = artifacts.write_figure_png(figure, path)
    plt.close(figure)
    return result


# --- orchestration ------------------------------------------------------------


@dataclass(frozen=True)
class QuestionAOutputs:
    """Locations of the exploratory Question A artifacts (all Git-ignored)."""

    tables: tuple[Path, ...]
    figures: tuple[Path, ...]


@dataclass(frozen=True)
class QuestionAResult:
    """Everything the entry point needs to report Question A."""

    frame: pd.DataFrame
    basket_grain: pd.DataFrame
    cross_check: pd.DataFrame
    totals: dict[str, float]
    summaries: pd.DataFrame
    partial_weeks: pd.DataFrame
    standardized: pd.DataFrame
    correlations: pd.DataFrame
    extreme_weeks: pd.DataFrame
    diagnostics: pd.DataFrame
    in_support: pd.DataFrame
    window_correlations: pd.DataFrame
    window_correlation_cross_check: pd.DataFrame
    thirds: pd.DataFrame
    campaign_context: pd.DataFrame
    bulk_lines: pd.DataFrame
    near_empty_days: pd.DataFrame
    summary_table: pd.DataFrame
    outputs: QuestionAOutputs


TABLE_OUTPUTS: Final[tuple[tuple[str, str], ...]] = (
    ("q_a_weekly_series.csv", "frame"),
    ("q_a_weekly_series_basket_grain.csv", "basket_grain"),
    ("q_a_series_cross_check.csv", "cross_check"),
    ("q_a_in_support_and_calendar.csv", "in_support"),
    ("q_a_full_week_summaries.csv", "summaries"),
    ("q_a_partial_week_positions.csv", "partial_weeks"),
    ("q_a_standardized_series.csv", "standardized"),
    ("q_a_correlations.csv", "correlations"),
    ("q_a_window_correlations.csv", "window_correlations"),
    ("q_a_window_correlation_cross_check.csv", "window_correlation_cross_check"),
    ("q_a_thirds_summary.csv", "thirds"),
    ("q_a_campaign_context.csv", "campaign_context"),
    ("q_a_bulk_line_summary.csv", "bulk_lines"),
    ("q_a_near_empty_days.csv", "near_empty_days"),
    ("q_a_extreme_weeks.csv", "extreme_weeks"),
    ("q_a_week_diagnostics.csv", "diagnostics"),
    ("q_a_summary_table.csv", "summary_table"),
)


def write_table(path: Path, frame: pd.DataFrame) -> Path:
    """Write an exploratory CSV, converting non-string cells to strings."""
    columns = [str(column) for column in frame.columns]
    rows = [
        tuple("" if pd.isna(value) else value for value in record)
        for record in frame.itertuples(index=False, name=None)
    ]
    return artifacts.write_table_csv(path, columns, rows, scope="exploratory")


def build_question_a(con: duckdb.DuckDBPyConnection) -> QuestionAResult:
    """Run the whole Question A pipeline (no gate; the caller gates first)."""
    frame, basket_grain, in_support = build_weekly_frame(con)
    cross_check = series_cross_check(frame, basket_grain)
    totals = reconcile_series_totals(frame)
    summaries = summary_statistics(frame)
    partial = partial_week_positions(frame)
    standardized = standardized_series(frame)
    correlations = correlation_table(frame)
    extremes = flag_extreme_weeks(frame)
    diagnostics = week_diagnostics(con)
    window_corr = window_correlations(frame)
    window_cross = window_correlation_cross_check(frame, basket_grain)
    thirds = thirds_summary(frame)
    campaign = campaign_context_summary(frame)
    bulk = bulk_line_summary(diagnostics)
    near_empty = near_empty_day_facts(frame)

    summary_table = build_summary_table(
        frame,
        summaries,
        partial,
        correlations,
        diagnostics,
        window_corr,
        thirds,
        bulk,
        campaign,
        near_empty,
    )

    tables: list[Path] = []
    frames = {
        "frame": frame,
        "basket_grain": basket_grain,
        "cross_check": cross_check,
        "in_support": in_support,
        "summaries": summaries,
        "partial_weeks": partial,
        "standardized": standardized,
        "correlations": correlations,
        "window_correlations": window_corr,
        "window_correlation_cross_check": window_cross,
        "thirds": thirds,
        "campaign_context": campaign,
        "bulk_lines": bulk,
        "near_empty_days": near_empty,
        "extreme_weeks": extremes,
        "diagnostics": diagnostics,
        "summary_table": summary_table,
    }
    for filename, key in TABLE_OUTPUTS:
        tables.append(write_table(artifacts.exploratory_path("table", filename), frames[key]))

    figures = (
        figure_levels(frame, artifacts.exploratory_path("figure", "q_a_weekly_levels.png")),
        figure_intensive(frame, artifacts.exploratory_path("figure", "q_a_intensive_margins.png")),
        figure_standardized(
            frame, artifacts.exploratory_path("figure", "q_a_standardized_comparison.png")
        ),
        figure_diagnostics(
            frame, diagnostics, artifacts.exploratory_path("figure", "q_a_week_diagnostics.png")
        ),
    )

    return QuestionAResult(
        frame=frame,
        basket_grain=basket_grain,
        cross_check=cross_check,
        totals=totals,
        summaries=summaries,
        partial_weeks=partial,
        standardized=standardized,
        correlations=correlations,
        extreme_weeks=extremes,
        diagnostics=diagnostics,
        in_support=in_support,
        window_correlations=window_corr,
        window_correlation_cross_check=window_cross,
        thirds=thirds,
        campaign_context=campaign,
        bulk_lines=bulk,
        near_empty_days=near_empty,
        summary_table=summary_table,
        outputs=QuestionAOutputs(tables=tuple(tables), figures=figures),
    )


def build_summary_table(
    frame: pd.DataFrame,
    summaries: pd.DataFrame,
    partial_weeks: pd.DataFrame,
    correlations: pd.DataFrame,
    diagnostics: pd.DataFrame,
    window_correlations: pd.DataFrame,
    thirds: pd.DataFrame,
    bulk_lines: pd.DataFrame,
    campaign_context: pd.DataFrame,
    near_empty_days: pd.DataFrame,
) -> pd.DataFrame:
    """The Step 11 exploratory Question A summary table (not T1)."""
    rows: list[dict[str, object]] = []
    for record in summaries.itertuples(index=False):
        rows.append(
            {
                "block": "full_week_summary",
                "series": record.series,
                "statistic": "mean/median/min/max",
                "value": f"{record.mean:.4f} / {record.median:.4f} / {record.min:.4f} / {record.max:.4f}",
                "note": f"weeks {contracts.FULL_WEEK_RANGE[0]}-{contracts.FULL_WEEK_RANGE[1]}, n={int(record.n_full_weeks)}",
            }
        )
    for record in partial_weeks.itertuples(index=False):
        for column in SUMMARY_SERIES:
            rows.append(
                {
                    "block": "partial_week",
                    "series": column,
                    "statistic": f"week {record.week_no} value / full-week percentile",
                    "value": f"{getattr(record, column):.4f} / {getattr(record, column + '__full_week_percentile'):.2f}",
                    "note": f"{record.observed_days} observed days",
                }
            )
    for record in correlations.itertuples(index=False):
        rows.append(
            {
                "block": "correlation_with_weekly_total",
                "series": record.series,
                "statistic": "pearson levels (full weeks) / week changes (full weeks)",
                "value": f"{record.pearson_levels_full_weeks:.4f} / {record.pearson_week_changes_full_weeks:.4f}",
                "note": "descriptive association only; not causal and not explanatory",
            }
        )
    for record in window_correlations.itertuples(index=False):
        rows.append(
            {
                "block": "declared_window_correlations",
                "series": f"weeks {record.week_from}-{record.week_to}",
                "statistic": "levels: active / value per active household-week",
                "value": (
                    f"{record.corr_levels_active_households:.4f} / "
                    f"{record.corr_levels_value_per_active_household_week:.4f}"
                ),
                "note": f"{record.window}; changes: {record.corr_changes_active_households:.4f} / "
                f"{record.corr_changes_value_per_active_household_week:.4f}",
            }
        )
    for record in thirds.itertuples(index=False):
        rows.append(
            {
                "block": "thirds_summary",
                "series": record.third,
                "statistic": "mean total / mean active households / mean households in window",
                "value": (
                    f"{record.mean_total_retailer_sales_value:.2f} / "
                    f"{record.mean_active_households:.2f} / {record.mean_in_support_households:.2f}"
                ),
                "note": f"mean value per trip {record.mean_value_per_trip:.4f}; "
                f"mean active share of in-support households {record.mean_active_share_of_in_support:.4f}",
            }
        )
    for record in bulk_lines.itertuples(index=False):
        rows.append(
            {
                "block": "bulk_line_summary",
                "series": "units composition",
                "statistic": record.statistic,
                "value": f"{record.value:.6f}",
                "note": f"bulk line = QUANTITY >= {EXTREME_QUANTITY_THRESHOLD}",
            }
        )
    for record in campaign_context.itertuples(index=False):
        rows.append(
            {
                "block": "campaign_timing_context",
                "series": record.window,
                "statistic": "corr(active listed campaigns, weekly total)",
                "value": f"{record.corr_levels_active_listed_campaigns_vs_total:.4f}",
                "note": f"{record.weeks_with_no_listed_campaign_in_span} weeks with no listed campaign in span; "
                "timing context only, never an effect",
            }
        )
    for record in near_empty_days.itertuples(index=False):
        rows.append(
            {
                "block": "near_empty_day_diagnostic",
                "series": f"week {record.week_no} (day {record.lowest_day_number})",
                "statistic": "lowest day value / week total as-is / week share of local median",
                "value": (
                    f"{record.lowest_day_sales_value:.2f} / {record.week_total_as_is:.2f} / "
                    f"{record.week_share_of_local_median:.4f}"
                ),
                "note": "labelled sensitivity if the lowest day were typical: "
                f"{record.labelled_sensitivity_week_total_if_lowest_day_were_typical:.2f} "
                f"(share {record.sensitivity_share_of_local_median:.4f}); as-is baseline retained",
            }
        )
    for record in diagnostics.itertuples(index=False):
        rows.append(
            {
                "block": "week_concentration_diagnostics",
                "series": "all weeks",
                "statistic": f"week {record.week_no} top1/top10 household share",
                "value": f"{record.top1_household_sales_value_share:.4f} / {record.top10_household_sales_value_share:.4f}",
                "note": f"max basket share {record.max_basket_sales_value_share:.4f}",
            }
        )
    return pd.DataFrame(rows)


def _corr(left: pd.Series, right: pd.Series) -> float:
    """Pearson correlation, NaN when either side has no variance.

    Guards ``pandas.Series.corr`` so a constant series reports an undefined
    correlation explicitly instead of emitting a divide-by-zero warning.
    """
    if left.nunique(dropna=True) <= 1 or right.nunique(dropna=True) <= 1:
        return float("nan")
    return float(left.corr(right))


def _format_corr(value: float) -> str:
    """Render a correlation, stating explicitly when it is undefined."""
    if pd.isna(value):
        return "n/a (no variance in this window)"
    return f"{value:+.3f}"


def format_report(result: QuestionAResult) -> str:
    """Render the Question A descriptive report."""
    lines: list[str] = []
    lines.append("Question A -- retailer activity over time (S0: 2,500 households, weeks 1-102)")
    lines.append("  series totals vs accepted anchors (accepted Phase 2 anchors, not findings):")
    for key, value in result.totals.items():
        lines.append(f"    {key:<36} {value!r}")
    lines.append("  independent route agreement (row grain vs basket grain), max abs difference:")
    for measure in ("total_retailer_sales_value", "active_households", "trips", "units"):
        lines.append(f"    {measure:<36} {result.cross_check[f'{measure}_abs_diff'].max()!r}")
    lines.append("")
    lines.append("  full-week descriptive summaries (weeks 2-101; descriptive, not inference):")
    header = f"    {'series':<48}{'mean':>14}{'sd':>14}{'min':>12}{'p25':>12}{'median':>12}{'p75':>12}{'max':>12}"
    lines.append(header)
    for record in result.summaries.itertuples(index=False):
        lines.append(
            f"    {record.series:<48}{record.mean:>14.2f}{record.std_ddof1:>14.2f}"
            f"{record.min:>12.2f}{record.p25:>12.2f}{record.median:>12.2f}{record.p75:>12.2f}{record.max:>12.2f}"
        )
    lines.append("")
    lines.append("  partial weeks (kept and annotated; positions are coverage facts):")
    for record in result.partial_weeks.itertuples(index=False):
        positions = ", ".join(
            f"{column}={getattr(record, column + '__full_week_percentile'):.0f}th pct"
            for column in SUMMARY_SERIES
        )
        lines.append(
            f"    week {record.week_no} ({record.observed_days} observed days): {positions}"
        )
    lines.append("")
    lines.append("  descriptive correlations with weekly retailer SALES_VALUE:")
    for record in result.correlations.itertuples(index=False):
        lines.append(
            f"    {record.series:<48} levels(full)={record.pearson_levels_full_weeks:+.3f}"
            f"  changes(full)={record.pearson_week_changes_full_weeks:+.3f}"
            f"  levels(all)={record.pearson_levels_all_weeks:+.3f}"
        )
    lines.append("")
    lines.append(f"  weeks flagged at |z| >= {EXTREME_Z} on any weekly series:")
    if result.extreme_weeks.empty:
        lines.append("    none")
    for record in result.extreme_weeks.itertuples(index=False):
        lines.append(
            f"    week {int(record.week_no):>3}  max|z|={record.max_abs_z:.2f}"
            f"  partial={bool(record.is_partial_week)}  series={record.extreme_columns}"
        )
    lines.append("")
    lines.append(
        "  observed purchase support (households with first_purchase_week <= w <= last_purchase_week):"
    )
    frame = result.frame
    for _, row in frame.iterrows():
        if int(row["week_no"]) in (1, 2, 5, 10, 15, 20, 50, 90, 100, 101, 102):
            lines.append(
                f"    week {int(row['week_no']):>3}  in_support={int(row['in_support_households']):>5}"
                f"  active={int(row['active_households']):>5}"
                f"  active share of in-support={row['active_share_of_in_support']:.4f}"
            )
    lines.append("")
    lines.append("  declared-window correlations with weekly retailer SALES_VALUE:")
    for record in result.window_correlations.itertuples(index=False):
        lines.append(
            f"    weeks {record.week_from:>3}-{record.week_to:<3} n={record.n_weeks:>3}"
            f"  levels: active={record.corr_levels_active_households:+.3f}"
            f" active/in-support={record.corr_levels_active_share_of_in_support:+.3f}"
            f" value/active={record.corr_levels_value_per_active_household_week:+.3f}"
            f"  changes: active={record.corr_changes_active_households:+.3f}"
            f" value/active={record.corr_changes_value_per_active_household_week:+.3f}"
        )
        lines.append(f"      ({record.window})")
    lines.append("  window-correlation cross-check (independent basket-grain route):")
    for record in result.window_correlation_cross_check.itertuples(index=False):
        lines.append(
            f"    weeks {record.week_from:>3}-{record.week_to:<3}"
            f"  levels active: row={record.corr_levels_active_row_grain:+.3f}"
            f" basket={record.corr_levels_active_basket_grain:+.3f}"
            f"  levels value/active: row={record.corr_levels_value_per_active_row_grain:+.3f}"
            f" basket={record.corr_levels_value_per_active_basket_grain:+.3f}"
            f"  changes value/active: row={record.corr_changes_value_per_active_row_grain:+.3f}"
            f" basket={record.corr_changes_value_per_active_basket_grain:+.3f}"
        )
    lines.append("")
    lines.append("  descriptive means by sample third:")
    for record in result.thirds.itertuples(index=False):
        lines.append(
            f"    {record.third:<22} total={record.mean_total_retailer_sales_value:>12,.0f}"
            f"  active={record.mean_active_households:>7,.0f}"
            f"  in_support={record.mean_in_support_households:>7,.0f}"
            f"  active/in-support={record.mean_active_share_of_in_support:.3f}"
            f"  value/trip={record.mean_value_per_trip:.2f}"
        )
    lines.append("")
    lines.append(f"  bulk-line composition of units (QUANTITY >= {EXTREME_QUANTITY_THRESHOLD}):")
    for record in result.bulk_lines.itertuples(index=False):
        lines.append(f"    {record.statistic:<52} {record.value!r}")
    lines.append("")
    lines.append("  weeks containing a near-empty trading day (as-is baseline retained):")
    if result.near_empty_days.empty:
        lines.append("    none")
    for record in result.near_empty_days.itertuples(index=False):
        lines.append(
            f"    week {record.week_no:>3} day {record.lowest_day_number:>3}"
            f" value={record.lowest_day_sales_value:,.2f}"
            f"  week as-is={record.week_total_as_is:,.2f}"
            f"  vs local median={record.week_share_of_local_median:.3f}"
            f"  sensitivity={record.labelled_sensitivity_week_total_if_lowest_day_were_typical:,.2f}"
            f" (share {record.sensitivity_share_of_local_median:.3f})"
        )
    lines.append("")
    lines.append("  campaign timing context (never an effect):")
    for record in result.campaign_context.itertuples(index=False):
        lines.append(
            f"    {record.window:<52} corr(campaigns,total)={_format_corr(record.corr_levels_active_listed_campaigns_vs_total)}"
            f"  weeks_without_listed_campaign={record.weeks_with_no_listed_campaign_in_span}"
        )
    lines.append("")
    lines.append("  exploratory artifacts (Git-ignored; not published, no allowlist entry):")
    for path in result.outputs.tables + result.outputs.figures:
        lines.append(f"    {path.relative_to(artifacts.PROJECT_ROOT).as_posix()}")
    return "\n".join(lines)
