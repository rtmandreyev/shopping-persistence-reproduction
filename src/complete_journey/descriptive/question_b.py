"""Phase 3 Question B: household heterogeneity, concentration and persistence.

Exploratory pass. This module builds the household-level analytical object, the
household-week measures for within/between variation, and the outlier/anomaly trace from
accepted Phase 2 tables, then calculates descriptive statistics and candidate figures.

What it deliberately does not do:

* no regression, no panel model, no fixed-effects estimate, no cluster or segment
  construction, no imputation and no trimming;
* nothing is promoted: candidates enter the challenge register as CHALLENGED or FAILED
  only, and F3/F4 are exploratory candidates, not published figures;
* no canonical table is created -- the household-level object is a Phase 3 exploratory
  artefact written to an ignored path (aggregate tables) and to the local, untracked
  state directory (the row-level object that carries ``household_key``).

Semantics that must survive every use of a number produced here:

* sales-value columns are retailer SALES_VALUE / retailer revenue, never household
  grocery expenditure and never a customer-paid price;
* an inactive household-week means "no recorded purchase from this retailer", never
  zero grocery consumption;
* trips are baskets, and unit totals are a QA-03 measurement caveat rather than a
  household consumption measure;
* the population is S0 (all 2,500 households). S1/S2 are activity-selected sensitivity
  populations, never a filter for a headline result and never "more valid".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from complete_journey.descriptive import artifacts, contracts, metrics
from complete_journey.paths import PROJECT_ROOT

# --- declared conventions ----------------------------------------------------

#: First/second half of the observation window (declared, not tuned).
HALVES: Final[tuple[tuple[int, int, str], ...]] = (
    (1, 51, "first half (weeks 1-51)"),
    (52, 102, "second half (weeks 52-102)"),
)
HALF_BOUNDARY_WEEK: Final[int] = 51

#: Quantiles reported for every distribution (Step 2 of the brief).
QUANTILE_PROBS: Final[tuple[float, ...]] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
QUANTILE_LABELS: Final[tuple[str, ...]] = (
    "p05",
    "p10",
    "p25",
    "median",
    "p75",
    "p90",
    "p95",
    "p99",
)

#: Household-level distribution measures: (column, what it is, its denominator).
QUANTILE_MEASURES: Final[tuple[tuple[str, str, str], ...]] = (
    ("active_weeks", "weeks with a recorded purchase", "the 102-week observation grid"),
    (
        "total_sales_value",
        "retailer SALES_VALUE (retailer revenue)",
        "household total over the window",
    ),
    ("total_baskets", "trips (baskets)", "household total over the window"),
    ("distinct_products", "distinct products purchased", "household total over the window"),
    ("distinct_commodities", "distinct commodities purchased", "household total over the window"),
    ("distinct_stores", "distinct stores used", "household total over the window"),
    (
        "mean_basket_sales_value",
        "mean basket retailer SALES_VALUE",
        "that household's own recorded baskets",
    ),
    (
        "sales_value_per_active_week",
        "retailer SALES_VALUE per active household-week",
        "that household's own active weeks",
    ),
    ("trips_per_active_week", "trips per active week", "that household's own active weeks"),
    (
        "sales_value_per_trip",
        "retailer SALES_VALUE per trip",
        "that household's own recorded baskets",
    ),
)

#: Concentration targets: (column, label, what the total is).
CONCENTRATION_TARGETS: Final[tuple[tuple[str, str, str], ...]] = (
    ("total_sales_value", "retailer SALES_VALUE", "total recorded retailer SALES_VALUE"),
    ("total_baskets", "trips (baskets)", "total recorded trips"),
    ("active_weeks", "active weeks", "total household-weeks with a recorded purchase"),
)

#: Top-x% shares reported for each concentration target.
CONCENTRATION_FRACTIONS: Final[tuple[float, ...]] = (0.01, 0.05, 0.10, 0.20)

#: Extensive x intensive association pairs for Step 3.
EXTENSIVE_MEASURES: Final[tuple[str, ...]] = (
    "active_weeks",
    "active_share_102",
    "trips_per_active_week",
)
INTENSIVE_MEASURES: Final[tuple[str, ...]] = (
    "sales_value_per_active_week",
    "sales_value_per_trip",
    "mean_basket_sales_value",
)
ASSOCIATION_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("active_weeks", "total_sales_value"),
    ("active_weeks", "sales_value_per_active_week"),
    ("active_weeks", "sales_value_per_trip"),
    ("active_weeks", "mean_basket_sales_value"),
    ("active_share_102", "sales_value_per_active_week"),
    ("trips_per_active_week", "sales_value_per_trip"),
    ("distinct_commodities", "sales_value_per_active_week"),
)

#: Persistence measures compared across the two halves.
PERSISTENCE_MEASURES: Final[tuple[tuple[str, str, str], ...]] = (
    ("active_weeks", "active_weeks_first_half", "active_weeks_second_half"),
    ("trips", "trips_first_half", "trips_second_half"),
    ("retailer SALES_VALUE", "sales_value_first_half", "sales_value_second_half"),
)

#: Household-week measures for the within/between decomposition.
WITHIN_BETWEEN_MEASURES: Final[tuple[tuple[str, str], ...]] = (
    ("retailer_purchase", "0/1 recorded-purchase indicator (extensive participation)"),
    ("weekly_sales_value", "weekly retailer SALES_VALUE (retailer revenue)"),
    ("trips", "weekly trips (baskets)"),
    ("distinct_commodities", "distinct commodities in the household-week"),
)

#: Declared absolute screens, mirrored in sql/descriptive/10_household_anomalies.sql.
EXTREME_QUANTITY_THRESHOLD: Final[int] = 1000
EXTREME_BASKET_SALES_VALUE: Final[float] = 500.0

#: The largest 1% of households (25 of 2,500) are dominated-tail comparisons.
TOP_SHARE_FRACTION: Final[float] = 0.01

#: Sample-flag names come from the shared vocabulary registry (contracts.py) so the
#: literals live in exactly one place and this module never re-states a population
#: definition of its own.
SAMPLE_FLAGS: Final[tuple[str, str, str]] = (
    contracts.HEADLINE_SAMPLE_FLAG,
    *contracts.SENSITIVITY_SAMPLE_FLAGS,
)
POPULATION_FLAG: Final[dict[str, str]] = dict(zip(("S0", "S1", "S2"), SAMPLE_FLAGS, strict=True))

#: Populations for the labelled Step 9 sensitivity (S0 is the headline population).
POPULATIONS: Final[tuple[tuple[str, str], ...]] = (
    ("S0", "all 2,500 households (headline population)"),
    ("S1", ">=20 active weeks and activity in all three thirds (sensitivity)"),
    ("S2", ">=30 active weeks and activity in all three thirds (sensitivity)"),
)

HOUSEHOLD_WEEK_NOTE: Final[str] = (
    "An inactive household-week is no recorded purchase from this retailer in that "
    "household-week; it is never zero grocery consumption."
)
UNITS_CAVEAT: Final[str] = (
    "Unit totals are dominated by very large-quantity transaction lines (Question A "
    "finding QA-03) and are not used as a headline household-heterogeneity measure."
)
SALES_VALUE_NOTE_FOR_FIGURES: Final[str] = (
    "Retailer SALES_VALUE is retailer revenue: the sales value the retailer recorded on "
    "these transactions. It is not a measure of household spending and not an item-level "
    "amount."
)
CONCENTRATION_INTERPRETATION: Final[str] = (
    "Concentration of recorded retailer activity across households. Not income "
    "inequality, not welfare inequality, and not a normative statement about customer value."
)
LOCAL_HOUSEHOLD_OBJECT: Final[Path] = artifacts.LOCAL_STATE_DIR / "q_b_household_level.csv"


class QuestionBError(RuntimeError):
    """Raised when the Question B object does not reconcile to accepted totals."""


# --- loading -----------------------------------------------------------------


def _sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _frame(con, filename: str, sort_by: str | None = None) -> pd.DataFrame:
    frame = con.execute(_sql(contracts.DESCRIPTIVE_SQL_DIR / filename)).fetchdf()
    if sort_by is not None and sort_by in frame.columns:
        frame = frame.sort_values(sort_by).reset_index(drop=True)
    return frame


def load_household_level(con) -> pd.DataFrame:
    """One row per household: the Question B household-level analytical object."""
    return _frame(con, "08_household_level.sql", sort_by="household_key")


def load_household_week(con) -> pd.DataFrame:
    """Household-week measures on the full 2,500 x 102 grid."""
    return _frame(con, "09_household_week_measures.sql", sort_by=None)


def load_anomaly_trace(con) -> pd.DataFrame:
    """Per-household outlier and anomaly trace (nothing is trimmed)."""
    return _frame(con, "10_household_anomalies.sql", sort_by="household_key")


def reconcile_household_object(household: pd.DataFrame) -> list[tuple[str, float, float]]:
    """Compare the object's independent recounts against the accepted anchors.

    Each row carries recounts computed from a different accepted table than the
    ``household_history`` column it mirrors, so agreement is evidence that the joins did
    not drop or duplicate a household.
    """
    checks: list[tuple[str, float, float]] = []

    def add(name: str, got: float, want: float) -> None:
        checks.append((name, float(got), float(want)))

    add("households", len(household), contracts.HOUSEHOLD_UNIVERSE)
    anchor = contracts.ACCEPTED_ANCHORS
    add(
        "retailer SALES_VALUE (household totals)",
        household["total_sales_value"].sum(),
        anchor.retailer_sales_value,
    )
    add(
        "retailer SALES_VALUE (week-base recount)",
        household["sales_value_recount"].sum(),
        anchor.retailer_sales_value,
    )
    add("baskets (household_history)", household["total_baskets"].sum(), anchor.baskets)
    add("baskets (basket_panel recount)", household["baskets_recount"].sum(), anchor.baskets)
    add("baskets (week-base recount)", household["trips_recount"].sum(), anchor.baskets)
    add("units (household_history)", household["total_units"].sum(), anchor.units)
    add("units (week-base recount)", household["units_recount"].sum(), anchor.units)
    add("active household-weeks", household["active_weeks"].sum(), anchor.active_household_weeks)
    add(
        "active household-weeks (recount)",
        household["active_weeks_recount"].sum(),
        anchor.active_household_weeks,
    )

    failures = [
        (name, got, want)
        for name, got, want in checks
        if not _close(got, want, tolerance=max(anchor.sales_value_abs_tolerance, 1e-4))
    ]
    if failures:
        detail = "; ".join(f"{name}: got {got!r} want {want!r}" for name, got, want in failures)
        raise QuestionBError(f"Question B object does not reconcile: {detail}")
    return checks


def _close(got: float, want: float, tolerance: float = 1e-4) -> bool:
    return abs(got - want) <= tolerance


def assert_half_split_is_complete(household: pd.DataFrame) -> None:
    """The two halves must partition the window exactly (no household-week lost)."""
    first = household["active_weeks_first_half"] + household["active_weeks_second_half"]
    if not (first == household["active_weeks"]).all():
        raise QuestionBError("first/second-half active weeks do not sum to active weeks")
    value = household["sales_value_first_half"] + household["sales_value_second_half"]
    if not np.allclose(value, household["total_sales_value"], rtol=0, atol=1e-6):
        raise QuestionBError("first/second-half retailer SALES_VALUE does not sum to the total")


def household_sample_flags(household_week: pd.DataFrame) -> pd.DataFrame:
    """Household-grain S0/S1/S2 flags taken from the accepted household-week flags.

    The accepted flags are constant within a household (verified: no household varies),
    so collapsing them with ``max`` reproduces the accepted household populations exactly
    and avoids re-implementing the activity/thirds rule.
    """
    flags = (
        household_week.groupby("household_key", observed=True)[list(SAMPLE_FLAGS)]
        .max()
        .reset_index()
    )
    return flags


def population_masks(household: pd.DataFrame, flags: pd.DataFrame) -> dict[str, pd.Series]:
    """S0/S1/S2 household masks from the accepted sample flags (S0 is the headline)."""
    merged = household[["household_key"]].merge(flags, on="household_key", how="left")
    anchor = contracts.ACCEPTED_ANCHORS
    masks = {name: merged[flag].fillna(0).astype(bool) for name, flag in POPULATION_FLAG.items()}
    observed = {name: int(mask.sum()) for name, mask in masks.items()}
    expected = {
        name: int(getattr(anchor, f"{flag}_households")) for name, flag in POPULATION_FLAG.items()
    }
    if observed != expected:
        raise QuestionBError(
            f"sample populations do not match the accepted flags: got {observed}, want {expected}"
        )
    return {name: mask.to_numpy() for name, mask in masks.items()}


# --- distributions -----------------------------------------------------------


def quantile_table(
    frame: pd.DataFrame,
    measures: tuple[tuple[str, str, str], ...] = QUANTILE_MEASURES,
) -> pd.DataFrame:
    """Raw-scale distribution summary. Logs are never substituted for raw values here."""
    rows = []
    for column, label, denominator in measures:
        values = frame[column]
        row = {
            "measure": column,
            "what_it_is": label,
            "denominator": denominator,
            "n_households": int(values.notna().sum()),
            "min": float(values.min()),
            "mean": float(values.mean()),
            "sd_ddof1": float(values.std(ddof=1)),
            "max": float(values.max()),
            "n_zero": int((values == 0).sum()),
            "share_zero": float((values == 0).mean()),
        }
        for prob, name in zip(QUANTILE_PROBS, QUANTILE_LABELS, strict=True):
            row[name] = float(values.quantile(prob))
        row["p99_over_median"] = (
            float(values.quantile(0.99) / values.quantile(0.50))
            if values.quantile(0.50)
            else np.nan
        )
        row["skew"] = float(values.skew())
        rows.append(row)
    return pd.DataFrame(rows)


def log1p_series(values: pd.Series) -> pd.Series:
    """Log1p transform for *visualization only*; reported tables stay in raw units."""
    return np.log1p(values.clip(lower=0))


def association_table(
    frame: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...] = ASSOCIATION_PAIRS,
) -> pd.DataFrame:
    """Pearson and Spearman association between extensive and intensive measures."""
    rows = []
    for left, right in pairs:
        sub = frame[[left, right]].dropna()
        rows.append(
            {
                "measure_a": left,
                "measure_b": right,
                "n_households": len(sub),
                "pearson": float(sub[left].corr(sub[right])) if len(sub) > 2 else np.nan,
                "spearman": float(sub[left].corr(sub[right], method="spearman"))
                if len(sub) > 2
                else np.nan,
                "note": "rank correlation is reported beside Pearson because both "
                "measures are skewed",
            }
        )
    return pd.DataFrame(rows)


def quantile_group_table(
    frame: pd.DataFrame,
    by: str,
    metric: str,
    groups: int = 5,
) -> pd.DataFrame:
    """Median (and mean) of ``metric`` inside quantile groups of ``by``.

    Quantile groups are a descriptive device, not customer segments: they are cut from
    ``by`` itself with equal counts and no interpretation is attached to the labels.
    """
    working = frame[[by, metric]].dropna()
    labels = [f"Q{i + 1}" for i in range(groups)]
    working = working.assign(group=pd.qcut(working[by], q=groups, labels=labels, duplicates="drop"))
    table = (
        working.groupby("group", observed=True)
        .agg(
            n_households=(metric, "size"),
            group_min=(by, "min"),
            group_max=(by, "max"),
            metric_median=(metric, "median"),
            metric_mean=(metric, "mean"),
        )
        .reset_index()
    )
    table["by_measure"] = by
    table["metric"] = metric
    return table


# --- concentration -----------------------------------------------------------


def concentration_table(
    frame: pd.DataFrame,
    targets: tuple[tuple[str, str, str], ...] = CONCENTRATION_TARGETS,
    fractions: tuple[float, ...] = CONCENTRATION_FRACTIONS,
) -> pd.DataFrame:
    """Lorenz-consistent concentration statistics for each target.

    Shares are computed with :mod:`complete_journey.descriptive.metrics`, which selects
    exactly ``ceil(n * fraction)`` households (25 / 125 / 250 of 2,500). The Gini is
    supplementary: the curves and top shares are the primary evidence.
    """
    rows = []
    for column, label, total_description in targets:
        values = frame[column].to_numpy(dtype=float)
        row = {
            "target": column,
            "what_it_is": label,
            "total_description": total_description,
            "n_households": len(values),
            "total": float(values.sum()),
            "gini_supplementary": metrics.gini(values),
        }
        for fraction in fractions:
            row[f"top_{int(fraction * 100)}pct_share"] = metrics.top_activity_share(
                values, fraction
            )
        row["households_for_top_1pct"] = int(np.ceil(len(values) * 0.01))
        rows.append(row)
    return pd.DataFrame(rows)


def lorenz_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Lorenz coordinates for one target (population share, activity share)."""
    population, activity = metrics.lorenz_coordinates(frame[column].to_numpy(dtype=float))
    return pd.DataFrame({"population_share": population, "activity_share": activity})


def concentration_component_table(
    frame: pd.DataFrame, fraction: float = TOP_SHARE_FRACTION
) -> pd.DataFrame:
    """Is value concentration the same households as trip and activity concentration?

    Reports the overlap between the top households by retailer SALES_VALUE and the top
    households by trips and by active weeks, plus the share of all retailer SALES_VALUE
    those same households hold. Overlap is an association between two rankings of the
    same households; it is not a decomposition of inequality.
    """
    k = int(np.ceil(len(frame) * fraction))
    by_value = set(frame.nlargest(k, "total_sales_value")["household_key"])
    by_trips = set(frame.nlargest(k, "total_baskets")["household_key"])
    by_weeks = set(frame.nlargest(k, "active_weeks")["household_key"])
    value_total = frame["total_sales_value"].sum()
    rows = [
        ("households_selected", k, k),
        (
            "share_of_trips_held",
            frame.nlargest(k, "total_baskets")["total_baskets"].sum()
            / frame["total_baskets"].sum(),
            frame["total_baskets"].sum(),
        ),
        ("top_value_households_also_top_trips", len(by_value & by_trips) / k, k),
        ("top_value_households_also_top_active_weeks", len(by_value & by_weeks) / k, k),
        (
            "sales_value_share_of_top_value_households",
            frame.nlargest(k, "total_sales_value")["total_sales_value"].sum() / value_total,
            1.0,
        ),
        (
            "sales_value_share_of_top_trip_households",
            frame[frame["household_key"].isin(by_trips)]["total_sales_value"].sum() / value_total,
            1.0,
        ),
        (
            "sales_value_share_of_top_active_week_households",
            frame[frame["household_key"].isin(by_weeks)]["total_sales_value"].sum() / value_total,
            1.0,
        ),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value", "reference_denominator"])


# --- persistence -------------------------------------------------------------


def persistence_table(
    frame: pd.DataFrame,
    measures: tuple[tuple[str, str, str], ...] = PERSISTENCE_MEASURES,
    groups: int = 5,
) -> pd.DataFrame:
    """First-half versus second-half persistence (Pearson, Spearman, group stability).

    A low second-half value is a low recorded second-half activity level. It is never
    described as churn: Question A established that the observed purchase support is an
    observational envelope, not verified entry or exit.
    """
    rows = []
    for label, first_column, second_column in measures:
        sub = frame[[first_column, second_column]].dropna()
        first_labels = pd.qcut(sub[first_column].rank(method="first"), q=groups, labels=False)
        second_labels = pd.qcut(sub[second_column].rank(method="first"), q=groups, labels=False)
        same_group = float((first_labels == second_labels).mean())
        adjacent = float((abs(first_labels - second_labels) <= 1).mean())
        rows.append(
            {
                "measure": label,
                "first_half_column": first_column,
                "second_half_column": second_column,
                "n_households": len(sub),
                "pearson": float(sub[first_column].corr(sub[second_column])),
                "spearman": float(sub[first_column].corr(sub[second_column], method="spearman")),
                "median_first_half": float(sub[first_column].median()),
                "median_second_half": float(sub[second_column].median()),
                "share_same_quintile": same_group,
                "share_same_or_adjacent_quintile": adjacent,
                "share_second_half_below_half_of_first_half_median": float(
                    (sub[second_column] < 0.5 * sub[first_column].median()).mean()
                ),
                "note": "halves are weeks 1-51 and 52-102; a low second-half value is a "
                "low recorded second-half activity level, never churn",
            }
        )
    return pd.DataFrame(rows)


def quintile_transition_table(frame: pd.DataFrame, measure: str, groups: int = 5) -> pd.DataFrame:
    """Row-normalised transitions between first-half and second-half quintiles."""
    first_column = {
        "active_weeks": "active_weeks_first_half",
        "trips": "trips_first_half",
        "retailer SALES_VALUE": "sales_value_first_half",
    }[measure]
    second_column = first_column.replace("first", "second")
    working = frame[[first_column, second_column]].dropna()
    labels = [f"Q{i + 1}" for i in range(groups)]
    working = working.assign(
        first_group=pd.qcut(working[first_column].rank(method="first"), q=groups, labels=labels),
        second_group=pd.qcut(working[second_column].rank(method="first"), q=groups, labels=labels),
    )
    table = pd.crosstab(working["first_group"], working["second_group"], normalize="index")
    table.insert(0, "measure", measure)
    return table.reset_index().rename(columns={"first_group": "first_half_group"})


# --- within / between variation ---------------------------------------------


def variance_decomposition(
    frame: pd.DataFrame,
    measures: tuple[tuple[str, str], ...] = WITHIN_BETWEEN_MEASURES,
    weight_column: str | None = None,
    subset_column: str | None = None,
) -> pd.DataFrame:
    """Descriptive within/between variation for household-week measures.

    Convention (stated, not implied). For measure ``y`` over household-week cells with
    household groups ``i`` and group sizes ``n_i``:

    * ``overall_ss`` = sum of (y - grand mean)^2 over all cells (no ddof),
    * ``between_ss`` = sum over households of n_i * (household mean - grand mean)^2,
    * ``within_ss``  = sum of (y - household mean)^2 over all cells.

    ``between_ss + within_ss == overall_ss`` holds exactly by construction, and the
    residual is reported as proof rather than assumed. The *shares* reported below are
    therefore shares of total variation (sums of squares), not R-squared from any model
    and not a fixed-effects estimate.

    The variance components are also reported with an explicit ddof: ``overall_var`` uses
    ddof=1 over cells, ``between_var`` uses ddof=1 over household means, ``within_var``
    uses ddof=1 over cells within groups pooled. Those three variances do **not** add to
    ``overall_var`` unless every household has the same number of cells, so the addition
    is never asserted; the sums-of-squares identity above is the exact statement.

    ``weight_column`` applies a non-negative weight to each cell (used to restrict
    attention to in-support weeks); ``subset_column`` restricts the cells included.
    """
    working = frame.copy()
    if subset_column is not None:
        working = working[working[subset_column] == 1]
    rows = []
    for column, label in measures:
        sub = working[[column, "household_key"]].dropna()
        if weight_column is not None:
            sub = working[[column, "household_key", weight_column]].dropna()
            weights = sub[weight_column].to_numpy(dtype=float)
        else:
            weights = np.ones(len(sub))
        values = sub[column].to_numpy(dtype=float)
        households = sub["household_key"].to_numpy()
        total_weight = weights.sum()
        grand_mean = float((weights * values).sum() / total_weight)
        group = pd.DataFrame({"h": households, "y": values, "w": weights})
        grouped = group.groupby("h").apply(
            lambda part: pd.Series(
                {
                    "n": part["w"].sum(),
                    "mean": (part["w"] * part["y"]).sum() / part["w"].sum(),
                    "variance": part["y"].var(ddof=1) if len(part) > 1 else np.nan,
                }
            ),
            include_groups=False,
        )
        between_ss = float((grouped["n"] * (grouped["mean"] - grand_mean) ** 2).sum())
        within_ss = float(
            (weights * (values - group["h"].map(grouped["mean"]).to_numpy()) ** 2).sum()
        )
        overall_ss = float((weights * (values - grand_mean) ** 2).sum())
        rows.append(
            {
                "measure": column,
                "what_it_is": label,
                "cells": len(values),
                "households": grouped.shape[0],
                "restriction": "all grid weeks"
                if subset_column is None
                else f"{subset_column} == 1",
                "overall_mean": grand_mean,
                "overall_var_ddof1": float(np.var(values, ddof=1)),
                "between_var_ddof1_over_household_means": float(grouped["mean"].var(ddof=1)),
                "within_var_ddof1_pooled": float(
                    (grouped["variance"] * (grouped["n"] - 1)).sum()
                    / (grouped["n"].sum() - grouped.shape[0])
                ),
                "overall_ss": overall_ss,
                "between_ss": between_ss,
                "within_ss": within_ss,
                "ss_identity_residual": overall_ss - between_ss - within_ss,
                "between_share_of_total_variation": between_ss / overall_ss
                if overall_ss
                else np.nan,
                "within_share_of_total_variation": within_ss / overall_ss if overall_ss else np.nan,
                "note": "shares of total variation (sums of squares); not an R-squared, not "
                "a fixed-effects estimate. Variances do not add unless group sizes are equal.",
            }
        )
    return pd.DataFrame(rows)


def household_week_variation_table(household_week: pd.DataFrame) -> pd.DataFrame:
    """Within/between on both denominators: all grid weeks and in-support weeks only.

    The two panels answer different questions: the grid panel includes the weeks outside a
    household's observed purchase-support window, the in-support panel does not.
    """
    all_weeks = variance_decomposition(household_week)
    all_weeks.insert(0, "panel", "all 102 grid weeks")
    in_support = variance_decomposition(household_week, subset_column="in_support")
    in_support.insert(0, "panel", "in-support weeks only")
    active = variance_decomposition(household_week[household_week["retailer_purchase"] == 1])
    active.insert(0, "panel", "active household-weeks only (purchase-conditional)")
    return pd.concat([all_weeks, in_support, active], ignore_index=True)


# --- anomalies / outliers ----------------------------------------------------


def outlier_trace_table(
    household: pd.DataFrame,
    anomaly: pd.DataFrame,
    fraction: float = TOP_SHARE_FRACTION,
) -> pd.DataFrame:
    """Compare the dominated tail with the rest, without trimming anything.

    Answers: are the largest households ordinary households with more weeks, or are they
    carried by extreme baskets, extreme quantity lines, flagged lines, or short support
    windows?
    """
    k = int(np.ceil(len(household) * fraction))
    # Only the anomaly-specific columns come from the trace: ``support_window_weeks`` and
    # ``bulk_quantity_lines`` already exist in the household object, and merging them
    # twice would silently create suffixed columns.
    merged = household.merge(
        anomaly[
            [
                "household_key",
                "top_basket_share_of_household_sales_value",
                "share_household_sales_value_in_extreme_baskets",
                "share_household_sales_value_in_bulk_lines",
                "baskets_over_declared_screen",
                "weeks_with_quantity_zero_lines",
                "weeks_with_positive_retail_disc",
                "weeks_with_negative_implied_paid",
            ]
        ],
        on="household_key",
        how="left",
        validate="one_to_one",
    )
    top_keys = set(household.nlargest(k, "total_sales_value")["household_key"])
    merged["group"] = np.where(
        merged["household_key"].isin(top_keys),
        f"top {int(fraction * 100)}% by retailer SALES_VALUE",
        "all other households",
    )
    grouped = merged.groupby("group", observed=True).agg(
        households=("household_key", "size"),
        mean_active_weeks=("active_weeks", "mean"),
        mean_support_window_weeks=("support_window_weeks", "mean"),
        median_top_basket_share=("top_basket_share_of_household_sales_value", "median"),
        mean_share_value_in_extreme_baskets=(
            "share_household_sales_value_in_extreme_baskets",
            "mean",
        ),
        mean_share_value_in_bulk_lines=("share_household_sales_value_in_bulk_lines", "mean"),
        mean_baskets_over_declared_screen=("baskets_over_declared_screen", "mean"),
        mean_bulk_quantity_lines=("bulk_quantity_lines", "mean"),
        share_with_any_quantity_zero_week=(
            "weeks_with_quantity_zero_lines",
            lambda s: float((s > 0).mean()),
        ),
        share_with_any_flagged_discount_week=(
            "weeks_with_positive_retail_disc",
            lambda s: float((s > 0).mean()),
        ),
        share_with_any_negative_implied_paid_week=(
            "weeks_with_negative_implied_paid",
            lambda s: float((s > 0).mean()),
        ),
    )
    return grouped.reset_index()


def anomaly_sensitivity_table(
    household: pd.DataFrame,
    anomaly: pd.DataFrame,
    top_household_removed: bool = True,
) -> pd.DataFrame:
    """Labelled sensitivities beside the baseline: the baseline is never replaced.

    Screens applied one at a time: drop the single largest household; drop the top 1%;
    drop households whose largest basket is more than half of their own total retailer
    SALES_VALUE; drop households with an unusually short support window. Each row states
    what it removed and how many households remain.
    """
    # Only the anomaly-specific column comes from the trace: ``support_window_weeks``
    # already exists in the household object, and merging it twice would silently create
    # a suffixed column.
    merged = household.merge(
        anomaly[["household_key", "top_basket_share_of_household_sales_value"]],
        on="household_key",
        how="left",
        validate="one_to_one",
    )
    k = int(np.ceil(len(merged) * TOP_SHARE_FRACTION))
    screens: list[tuple[str, pd.DataFrame]] = [("baseline (all households)", merged)]
    if top_household_removed:
        largest = merged.nlargest(1, "total_sales_value")["household_key"]
        screens.append(
            (
                "excluding the single largest household",
                merged[~merged["household_key"].isin(largest)],
            )
        )
    screens.append(
        (
            f"excluding the top {int(TOP_SHARE_FRACTION * 100)}% by retailer SALES_VALUE",
            merged[
                ~merged["household_key"].isin(
                    merged.nlargest(k, "total_sales_value")["household_key"]
                )
            ],
        )
    )
    screens.append(
        (
            "excluding households whose largest basket exceeds half of their total",
            merged[merged["top_basket_share_of_household_sales_value"] <= 0.5],
        )
    )
    short_cut = merged["support_window_weeks"].quantile(0.10)
    screens.append(
        (
            f"excluding households in the shortest decile of support windows (<= {short_cut:.0f} weeks)",
            merged[merged["support_window_weeks"] > short_cut],
        )
    )

    rows = []
    for label, frame in screens:
        values = frame["total_sales_value"].to_numpy(dtype=float)
        rows.append(
            {
                "screen": label,
                "households": len(frame),
                "retailer_sales_value_total": float(values.sum()),
                "share_of_baseline_value": float(values.sum() / merged["total_sales_value"].sum()),
                "top_1pct_share": metrics.top_activity_share(values, 0.01),
                "top_10pct_share": metrics.top_activity_share(values, 0.10),
                "gini_supplementary": metrics.gini(values),
                "median_active_weeks": float(frame["active_weeks"].median()),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_table(
    household: pd.DataFrame,
    masks: dict[str, pd.Series],
    measures: tuple[tuple[str, str], ...] = (
        ("households", ""),
        ("active_weeks", "median active weeks"),
        ("total_sales_value", "median household retailer SALES_VALUE"),
        ("sales_value_per_active_week", "median retailer SALES_VALUE per active week"),
        ("sales_value_per_trip", "median retailer SALES_VALUE per trip"),
    ),
    fraction: float = 0.01,
) -> pd.DataFrame:
    """The same descriptive statistics on S0, S1 and S2, labelled as sensitivity."""
    rows = []
    for column, label in measures:
        row: dict[str, object] = {
            "measure": column,
            "statistic": label or "households in population",
        }
        for name, _description in POPULATIONS:
            sub = household[masks[name]]
            row[name] = float(len(sub)) if column == "households" else float(sub[column].median())
            row[f"{name}_top_1pct_share"] = metrics.top_activity_share(
                sub["total_sales_value"].to_numpy(dtype=float), fraction
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    table["note"] = (
        "S1/S2 are activity-selected sensitivity populations, not less-biased or more "
        "valid samples; S0 is the headline population."
    )
    return table


# --- candidate figures (exploratory, never published in this run) ------------


def _ecdf(ax, values: pd.Series, label: str, *, log_x: bool = False) -> None:
    series = values.dropna().sort_values()
    if log_x:
        series = log1p_series(series)
    y = np.arange(1, len(series) + 1) / len(series)
    ax.plot(series.to_numpy(), y, lw=1.4, color="#1f4e79")
    ax.set_ylabel("cumulative share\nof households", fontsize=7.5)
    ax.set_title(label, fontsize=8.5, loc="left")
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7)
    if log_x:
        ax.set_xlabel("log1p scale (raw values in the summary table)", fontsize=7)


def _finish(fig, title: str, caption: str) -> None:
    fig.suptitle(title, fontsize=10.5, y=0.985)
    fig.text(0.006, 0.004, caption, fontsize=6.2, va="bottom")


def figure_heterogeneity_candidates(household: pd.DataFrame) -> object:
    """F4 candidate: distributions of household-level measures (population S0)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.13, hspace=0.42, wspace=0.24)
    _ecdf(axes[0, 0], household["active_weeks"], "A. Active weeks (of the 102-week grid)")
    _ecdf(
        axes[0, 1],
        household["total_sales_value"],
        "B. Total retailer SALES_VALUE",
        log_x=True,
    )
    _ecdf(axes[0, 2], household["total_baskets"], "C. Trips (baskets)", log_x=True)
    _ecdf(axes[1, 0], household["distinct_commodities"], "D. Distinct commodities purchased")
    _ecdf(
        axes[1, 1],
        household["sales_value_per_active_week"],
        "E. Retailer SALES_VALUE per active household-week",
    )
    _ecdf(axes[1, 2], household["sales_value_per_trip"], "F. Retailer SALES_VALUE per trip")
    _finish(
        fig,
        "Candidate F3 - household heterogeneity (population S0, all 2,500 households)",
        "Empirical CDFs of household-level measures; population S0, no filtering to S1/S2. "
        "Panels B and C use a log1p x-axis for readability only - the reported table stays in raw "
        f"units. {SALES_VALUE_NOTE_FOR_FIGURES} An inactive household-week is no recorded purchase "
        "from this retailer, not zero grocery consumption.",
    )
    return fig


def figure_concentration_candidates(household: pd.DataFrame) -> object:
    """F4 candidate: Lorenz-style concentration curves for three activity measures."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.19, wspace=0.22)
    colours = {
        "total_sales_value": "#1f4e79",
        "total_baskets": "#2e7d32",
        "active_weeks": "#b26500",
    }
    for column, label, _total in CONCENTRATION_TARGETS:
        coordinates = lorenz_table(household, column)
        axes[0].plot(
            coordinates["population_share"] * 100,
            coordinates["activity_share"] * 100,
            lw=1.6,
            color=colours[column],
            label=label,
        )
    axes[0].plot([0, 100], [0, 100], color="0.5", lw=0.9, ls="--", label="equal distribution")
    axes[0].set_xlabel("cumulative share of households (%), sorted smallest to largest", fontsize=8)
    axes[0].set_ylabel("cumulative share of recorded activity (%)", fontsize=8)
    axes[0].set_title("A. Lorenz-style concentration curves", fontsize=9, loc="left")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].grid(alpha=0.25, lw=0.5)

    table = concentration_table(household)
    labels = [f"top {int(fraction * 100)}%" for fraction in CONCENTRATION_FRACTIONS]
    positions = np.arange(len(CONCENTRATION_FRACTIONS))
    width = 0.26
    for offset, (column, label, _total) in enumerate(CONCENTRATION_TARGETS):
        row = table.loc[table["target"] == column].iloc[0]
        axes[1].bar(
            positions + (offset - 1) * width,
            [row[f"top_{int(f * 100)}pct_share"] * 100 for f in CONCENTRATION_FRACTIONS],
            width=width,
            color=colours[column],
            label=label,
        )
    axes[1].set_xticks(positions, labels, fontsize=8)
    axes[1].set_ylabel("share of the measure's total (%)", fontsize=8)
    axes[1].set_title("B. Top-x% shares of recorded activity", fontsize=9, loc="left")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25, lw=0.5, axis="y")
    for ax in axes:
        ax.tick_params(labelsize=7.5)
    _finish(
        fig,
        "Candidate F4 - concentration of recorded retailer activity across households",
        "Curves are cumulative shares of the three measures over households sorted smallest to largest "
        "(Lorenz-style, with an equal-distribution reference). Shares select exactly ceil(n x fraction) "
        f"households (25 / 125 / 250). {CONCENTRATION_INTERPRETATION} The Gini is a supplementary number "
        "and is not shown here.",
    )
    return fig


def figure_extensive_intensive(household: pd.DataFrame) -> object:
    """Exploratory diagnostic: extensive activity against intensive rates."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.22, wspace=0.26)
    panels = (
        ("total_sales_value", "A. Active weeks vs total retailer SALES_VALUE", True),
        ("sales_value_per_active_week", "B. Active weeks vs SALES_VALUE per active week", False),
        ("sales_value_per_trip", "C. Active weeks vs SALES_VALUE per trip", False),
    )
    for ax, (column, title, log_y) in zip(axes, panels, strict=True):
        y = log1p_series(household[column]) if log_y else household[column]
        ax.hexbin(household["active_weeks"], y, gridsize=28, cmap="Blues", mincnt=1)
        ax.set_xlabel("active weeks", fontsize=8)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.grid(alpha=0.2, lw=0.5)
        ax.tick_params(labelsize=7)
        rho = household[["active_weeks", column]].corr(method="spearman").iloc[0, 1]
        ax.text(
            0.03,
            0.94,
            f"Spearman = {rho:+.3f}",
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
        )
        ax.set_ylabel(("log1p " if log_y else "") + "measure", fontsize=7.5)
    _finish(
        fig,
        "Exploratory diagnostic - extensive activity against intensive rates (population S0)",
        "Hexbin counts of households. Rank correlations are reported beside each panel. An association "
        "between an extensive measure (how many weeks a household buys) and an intensive measure (how much "
        "it records when it buys) is descriptive only; neither measure is adjusted for the other.",
    )
    return fig


def figure_persistence(household: pd.DataFrame) -> object:
    """Exploratory diagnostic: first-half against second-half household activity."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.22, wspace=0.26)
    panels = (
        (
            "active_weeks_first_half",
            "active_weeks_second_half",
            "A. Active weeks, first vs second half",
            False,
        ),
        ("trips_first_half", "trips_second_half", "B. Trips, first vs second half", True),
        (
            "sales_value_first_half",
            "sales_value_second_half",
            "C. Retailer SALES_VALUE, first vs second half",
            True,
        ),
    )
    for ax, (x_column, y_column, title, log_scale) in zip(axes, panels, strict=True):
        x = log1p_series(household[x_column]) if log_scale else household[x_column]
        y = log1p_series(household[y_column]) if log_scale else household[y_column]
        both = pd.DataFrame({"x": x, "y": y}).dropna()
        ax.scatter(both["x"], both["y"], s=5, alpha=0.35, color="#1f4e79", edgecolors="none")
        limit = max(both["x"].max(), both["y"].max())
        ax.plot([0, limit], [0, limit], color="0.45", lw=0.9, ls="--")
        rho = both["x"].corr(both["y"], method="spearman")
        ax.text(
            0.03, 0.95, f"Spearman = {rho:+.3f}", transform=ax.transAxes, fontsize=7.5, va="top"
        )
        ax.set_xlabel(("log1p " if log_scale else "") + "first half", fontsize=8)
        ax.set_ylabel(("log1p " if log_scale else "") + "second half", fontsize=8)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.grid(alpha=0.2, lw=0.5)
        ax.tick_params(labelsize=7)
    _finish(
        fig,
        "Exploratory diagnostic - first-half against second-half household activity",
        "Halves are weeks 1-51 and 52-102. A household low in the second half is a household with low "
        "recorded second-half activity; this is never described as churn, because the observed purchase "
        "support is an observational envelope, not verified entry or exit.",
    )
    return fig


def figure_within_between(within_between: pd.DataFrame) -> object:
    """Exploratory diagnostic: between- and within-household shares of variation."""
    import matplotlib.pyplot as plt

    grid = within_between[within_between["panel"] == "all 102 grid weeks"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.24, wspace=0.26)
    positions = np.arange(len(grid))
    axes[0].bar(
        positions,
        grid["between_share_of_total_variation"] * 100,
        width=0.6,
        color="#1f4e79",
        label="between households",
    )
    axes[0].bar(
        positions,
        grid["within_share_of_total_variation"] * 100,
        width=0.6,
        bottom=grid["between_share_of_total_variation"] * 100,
        color="#b26500",
        label="within households",
    )
    axes[0].set_xticks(positions, grid["measure"], rotation=20, ha="right", fontsize=7.5)
    axes[0].set_ylabel("share of total variation (%)", fontsize=8)
    axes[0].set_title("A. Shares of total variation (sums of squares)", fontsize=9, loc="left")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25, lw=0.5, axis="y")

    for panel, colour in (("all 102 grid weeks", "#1f4e79"), ("in-support weeks only", "#2e7d32")):
        row = within_between[
            (within_between["panel"] == panel) & (within_between["measure"] == "weekly_sales_value")
        ].iloc[0]
        axes[1].plot(
            [0, row["between_share_of_total_variation"] * 100],
            [panel, panel],
            marker="|",
            lw=0,
            color=colour,
        )
        axes[1].barh(
            panel,
            row["between_share_of_total_variation"] * 100,
            color=colour,
            height=0.5,
        )
        axes[1].text(
            row["between_share_of_total_variation"] * 100 + 1.5,
            panel,
            f"{row['between_share_of_total_variation'] * 100:.1f}% between",
            fontsize=7.5,
            va="center",
        )
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("share of total variation in weekly retailer SALES_VALUE (%)", fontsize=8)
    axes[1].set_title("B. Weekly retailer SALES_VALUE, two denominators", fontsize=9, loc="left")
    axes[1].grid(alpha=0.25, lw=0.5, axis="x")
    axes[1].tick_params(labelsize=7.5)
    _finish(
        fig,
        "Exploratory diagnostic - between-household and within-household variation",
        "Shares are shares of total variation (sums of squares): between + within = total exactly, and the "
        "residual is reported in the table. This is not an R-squared, not a fixed-effects estimate and not "
        "an 'explained by household fixed effects' statement; the variance components do not add unless every "
        "household has the same number of cells.",
    )
    return fig


# --- build -------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionBResult:
    """Everything the Question B exploratory pass produced, with its evidence."""

    household: pd.DataFrame
    household_week: pd.DataFrame
    anomaly: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    figures: dict[str, Path]
    reconciliation: list[tuple[str, float, float]]
    populations: dict[str, int]
    summary_path: Path
    local_object_path: Path


def _write_table(name: str, frame: pd.DataFrame) -> Path:
    """Write an exploratory (Git-ignored) aggregate table; no identifier may appear."""
    path = artifacts.exploratory_path("table", name)
    return artifacts.write_table_csv(
        path,
        list(frame.columns),
        frame.itertuples(index=False, name=None),
        scope="exploratory",
    )


def _write_figure(name: str, fig) -> Path:
    path = artifacts.exploratory_path("figure", name)
    return artifacts.write_figure_png(fig, path)


def _write_local_object(frame: pd.DataFrame, path: Path) -> Path:
    """Write the row-level household object locally, where ``household_key`` may live."""
    artifacts.assert_local_only(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def summary_table(tables: dict[str, pd.DataFrame], populations: dict[str, int]) -> pd.DataFrame:
    """Long-format Question B summary (Step 12). Exploratory; not the final Table 1."""
    rows: list[dict[str, object]] = []

    def add(section: str, statistic: str, value: object, note: str = "") -> None:
        rows.append({"section": section, "statistic": statistic, "value": value, "note": note})

    for name, count in populations.items():
        add("population", f"households in {name}", count, "accepted Phase 2 sample flag")

    quantiles = tables["quantiles"]
    for record in quantiles.itertuples(index=False):
        add(
            "household distribution",
            f"{record.measure}: median",
            record.median,
            f"{record.what_it_is}; denominator: {record.denominator}",
        )
        add("household distribution", f"{record.measure}: p90", record.p90, record.denominator)
        add("household distribution", f"{record.measure}: p99", record.p99, record.denominator)
        add(
            "household distribution",
            f"{record.measure}: share zero",
            record.share_zero,
            record.denominator,
        )

    for record in tables["concentration"].itertuples(index=False):
        for fraction in CONCENTRATION_FRACTIONS:
            add(
                "concentration",
                f"{record.what_it_is}: top {int(fraction * 100)}% share",
                getattr(record, f"top_{int(fraction * 100)}pct_share"),
                record.total_description,
            )
        add(
            "concentration",
            f"{record.what_it_is}: Gini (supplementary)",
            record.gini_supplementary,
            CONCENTRATION_INTERPRETATION,
        )

    for record in tables["persistence"].itertuples(index=False):
        add(
            "persistence",
            f"{record.measure}: Pearson first vs second half",
            record.pearson,
            "weeks 1-51 vs 52-102",
        )
        add(
            "persistence",
            f"{record.measure}: Spearman first vs second half",
            record.spearman,
            "weeks 1-51 vs 52-102",
        )
        add(
            "persistence",
            f"{record.measure}: share in the same quintile",
            record.share_same_quintile,
            "quintiles of each half, equal-count cut",
        )

    for record in tables["within_between"].itertuples(index=False):
        add(
            "within/between variation",
            f"{record.measure} ({record.panel}): between share of total variation",
            record.between_share_of_total_variation,
            "sums-of-squares share; not an R-squared and not a fixed-effects estimate",
        )
        add(
            "within/between variation",
            f"{record.measure} ({record.panel}): within share of total variation",
            record.within_share_of_total_variation,
            "sums-of-squares share",
        )

    for record in tables["sensitivity"].itertuples(index=False):
        for name in populations:
            add(
                "S0/S1/S2 sensitivity",
                f"{record.statistic or record.measure}: {name}",
                getattr(record, name),
                "S1/S2 are activity-selected sensitivity populations, not more valid samples",
            )

    for record in tables["anomaly_sensitivity"].itertuples(index=False):
        add(
            "anomaly sensitivity (baseline first)",
            f"{record.screen}: households",
            record.households,
            "baseline is never replaced by a screen",
        )
        add(
            "anomaly sensitivity (baseline first)",
            f"{record.screen}: top 1% share",
            record.top_1pct_share,
            "unscreened baseline is the first row",
        )

    return pd.DataFrame(rows)


def build_question_b(con) -> QuestionBResult:
    """Run the Question B exploratory pass and write only ignored/local artefacts."""
    household = load_household_level(con)
    household_week = load_household_week(con)
    anomaly = load_anomaly_trace(con)

    reconciliation = reconcile_household_object(household)
    assert_half_split_is_complete(household)
    flags = household_sample_flags(household_week)
    masks = population_masks(household, flags)
    populations = {name: int(mask.sum()) for name, mask in masks.items()}

    within_between = household_week_variation_table(household_week)
    tables: dict[str, pd.DataFrame] = {
        "quantiles": quantile_table(household),
        "associations": association_table(household),
        "quantile_groups_value_by_active_weeks": quantile_group_table(
            household, by="active_weeks", metric="total_sales_value"
        ),
        "quantile_groups_rate_by_active_weeks": quantile_group_table(
            household, by="active_weeks", metric="sales_value_per_active_week"
        ),
        "concentration": concentration_table(household),
        "concentration_components": concentration_component_table(household),
        "persistence": persistence_table(household),
        "quintile_transitions": pd.concat(
            [
                quintile_transition_table(household, measure)
                for measure in ("active_weeks", "trips", "retailer SALES_VALUE")
            ],
            ignore_index=True,
        ),
        "within_between": within_between,
        "outlier_trace": outlier_trace_table(household, anomaly),
        "anomaly_sensitivity": anomaly_sensitivity_table(household, anomaly),
        "sensitivity": sensitivity_table(household, masks),
    }
    tables["lorenz_points"] = pd.concat(
        [
            lorenz_table(household, column).assign(target=column, what_it_is=label)
            for column, label, _total in CONCENTRATION_TARGETS
        ],
        ignore_index=True,
    )

    figures: dict[str, Path] = {}
    from matplotlib import pyplot as plt

    builders = (
        ("q_b_heterogeneity_candidates.png", figure_heterogeneity_candidates),
        ("q_b_concentration_candidates.png", figure_concentration_candidates),
        ("q_b_extensive_intensive.png", figure_extensive_intensive),
        ("q_b_persistence.png", figure_persistence),
    )
    for filename, builder in builders:
        fig = builder(household)
        try:
            figures[filename] = _write_figure(filename, fig)
        finally:
            plt.close(fig)
    fig = figure_within_between(within_between)
    try:
        figures["q_b_within_between.png"] = _write_figure("q_b_within_between.png", fig)
    finally:
        plt.close(fig)

    tables["summary"] = summary_table(tables, populations)
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        paths[name] = _write_table(f"q_b_{name}.csv", frame)

    local_object = _write_local_object(household, LOCAL_HOUSEHOLD_OBJECT)
    return QuestionBResult(
        household=household,
        household_week=household_week,
        anomaly=anomaly,
        tables=tables,
        figures=figures,
        reconciliation=reconciliation,
        populations=populations,
        summary_path=paths["summary"],
        local_object_path=local_object,
    )


def format_report(result: QuestionBResult) -> str:
    """Human-readable summary of the exploratory pass (no finding is promoted here)."""
    lines: list[str] = []
    lines.append("  reconciliation (independent recounts against accepted anchors):")
    for name, got, want in result.reconciliation:
        lines.append(f"    {name:<44} {got:>16,.4f}  (accepted {want:>16,.4f})")

    lines.append("")
    lines.append("  household distribution, S0 (median / p90 / p99 / share zero):")
    quantiles = result.tables["quantiles"]
    for record in quantiles.itertuples(index=False):
        lines.append(
            f"    {record.measure:<30} {record.median:>10,.3f} {record.p90:>12,.3f} "
            f"{record.p99:>14,.3f} {record.share_zero:>7.3f}"
        )

    lines.append("")
    lines.append("  extensive x intensive association (Pearson / Spearman):")
    for record in result.tables["associations"].itertuples(index=False):
        lines.append(
            f"    {record.measure_a:<24} x {record.measure_b:<26} "
            f"{record.pearson:>+8.3f} / {record.spearman:>+8.3f}"
        )

    lines.append("")
    lines.append("  concentration of recorded retailer activity (top 1% / 5% / 10% / 20%):")
    for record in result.tables["concentration"].itertuples(index=False):
        lines.append(
            f"    {record.what_it_is:<24} "
            f"{record.top_1pct_share:>7.4f} {record.top_5pct_share:>8.4f} "
            f"{record.top_10pct_share:>8.4f} {record.top_20pct_share:>8.4f} "
            f" (Gini {record.gini_supplementary:.4f}, supplementary)"
        )

    lines.append("")
    lines.append("  first-half vs second-half persistence (Pearson / Spearman / same quintile):")
    for record in result.tables["persistence"].itertuples(index=False):
        lines.append(
            f"    {record.measure:<22} {record.pearson:>+8.3f} / {record.spearman:>+8.3f} / "
            f"{record.share_same_quintile:>7.4f}"
        )

    lines.append("")
    lines.append("  within/between variation (between share of total variation):")
    for record in result.tables["within_between"].itertuples(index=False):
        lines.append(
            f"    {record.panel:<46} {record.measure:<22} "
            f"{record.between_share_of_total_variation:>7.4f} between / "
            f"{record.within_share_of_total_variation:>7.4f} within "
            f"(identity residual {record.ss_identity_residual:+.3e})"
        )

    lines.append("")
    lines.append("  dominated tail vs the rest (anomaly trace):")
    for record in result.tables["outlier_trace"].itertuples(index=False):
        lines.append(
            f"    {record.group:<38} households={record.households:<5} "
            f"active weeks {record.mean_active_weeks:>6.1f} "
            f"support window {record.mean_support_window_weeks:>6.1f} "
            f"bulk-line value share {record.mean_share_value_in_bulk_lines:>7.4f}"
        )

    lines.append("")
    lines.append("  labelled anomaly sensitivities (baseline first):")
    for record in result.tables["anomaly_sensitivity"].itertuples(index=False):
        lines.append(
            f"    {record.screen:<68} households={record.households:<5} "
            f"top1% {record.top_1pct_share:>7.4f} / top10% {record.top_10pct_share:>7.4f}"
        )

    lines.append("")
    lines.append("  S0 / S1 / S2 sensitivity (median, and top-1% share of retailer SALES_VALUE):")
    for record in result.tables["sensitivity"].itertuples(index=False):
        lines.append(
            f"    {record.measure:<30} "
            f"S0 {record.S0:>10,.3f} (top1% {record.S0_top_1pct_share:.4f})  "
            f"S1 {record.S1:>10,.3f} ({record.S1_top_1pct_share:.4f})  "
            f"S2 {record.S2:>10,.3f} ({record.S2_top_1pct_share:.4f})"
        )

    lines.append("")
    lines.append("  outputs (exploratory, Git-ignored; no household identifier in any of them):")
    for name, frame in sorted(result.tables.items()):
        lines.append(f"    q_b_{name}.csv  ({len(frame)} rows)")
    for name, path in sorted(result.figures.items()):
        lines.append(f"    {name}  ->  {path.relative_to(PROJECT_ROOT)}")
    lines.append(
        f"    local row-level object (keeps household_key, never tracked): {result.local_object_path.relative_to(PROJECT_ROOT)}"
    )
    return "\n".join(lines)
