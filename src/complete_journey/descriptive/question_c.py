"""Phase 3 Question C: product/commodity structure and Phase 5 shortlist diagnostics.

Exploratory/reconnaissance pass. This module builds commodity-, department- and
occasion-level descriptive objects from accepted Phase 2 tables (plus the validated Phase 3
department derivation and the accepted choice-layer inputs, used descriptively only), then
calculates the eight frozen Phase 5 shortlist diagnostics and prepares the evidence the
researcher needs to choose a commodity.

What this module deliberately does not do:

* it does NOT choose a commodity, rank a "best" commodity, or build a composite score;
* it does NOT define a choice occasion, a choice set, a price concept or a price variable;
* it does NOT infer shelf availability from purchased product-store-week presence (and never
  uses ``causal_data`` as an availability measure);
* it does NOT create a generic brand share, a weekly price or a commodity price index;
* it does NOT trim anything: bulk-line and outlier evidence is reported as a labelled
  sensitivity beside the as-is baseline.

Semantics that must survive every use of a number produced here:

* sales-value columns are retailer SALES_VALUE / retailer revenue, never customer spend,
  grocery expenditure or a price;
* ``household_category_week`` is sparse and purchase-conditional, so the absence of a row
  does not imply zero household consumption of that commodity;
* penetration is *observed-household penetration* = distinct buying households / 2,500, and
  is never called market penetration;
* recorded units are dominated by very large-quantity lines (Question A finding QA-03) and
  are never a headline ranking;
* national/private composition is reported in both retailer SALES_VALUE and recorded units,
  each labelled, never as one generic brand share;
* the accepted 34-week thirds are weeks 1-34, 35-68 and 69-102, and Question A finding QA-01
  means early-third levels are affected by observed support formation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from complete_journey.descriptive import artifacts, contracts, department, metrics, reconcile

# --- declared conventions ----------------------------------------------------

#: Extra accepted tables Question C reads, over and above the reconciliation gate tables.
#: The choice-layer inputs are used descriptively only: no choice model, no choice_long.
EXTRA_ACCEPTED_TABLES: Final[tuple[str, ...]] = (
    "choice_line_price_candidates",
    "choice_household_product_week",
)

#: Accepted 34-week thirds (see docs/SAMPLE_DEFINITIONS.md).
THIRDS: Final[tuple[tuple[int, int, str], ...]] = (
    (1, 34, "early third (weeks 1-34)"),
    (35, 68, "middle third (weeks 35-68)"),
    (69, 102, "late third (weeks 69-102)"),
)

#: Penetration denominator: the observed transaction-universe households.
PENETRATION_DENOMINATOR: Final[int] = contracts.HOUSEHOLD_UNIVERSE

#: Bulk-line screen (same declared screen as Question A's QA-03 check).
EXTREME_QUANTITY_THRESHOLD: Final[int] = 1000

#: Commodity labels that are not purchasable product categories: the vendor's coupon /
#: miscellaneous label and the missing-label sentinel. They are RETAINED in every table
#: (the exploratory T2 precursor keeps all 308 labels) and excluded only from the
#: Phase 5 shortlist reconnaissance, with that exclusion stated as a decision.
NON_PRODUCT_COMMODITY_LABELS: Final[tuple[str, ...]] = (
    "COUPON/MISC ITEMS",
    "NO COMMODITY DESCRIPTION",
)

#: The eight frozen Phase 5 shortlist dimensions, in the brief's order.
SHORTLIST_DIMENSIONS: Final[tuple[str, ...]] = (
    "1 breadth",
    "2 repeat purchasing",
    "3 purchase-occasion diagnostics",
    "4 assortment complexity",
    "5 candidate-price usability",
    "6 temporal stability",
    "7 outlier/bulk sensitivity",
    "8 availability limitation",
)

#: Availability caveat carried as a constant column in the diagnostic matrix, because the
#: accepted data cannot measure shelf availability at all.
AVAILABILITY_CAVEAT: Final[str] = (
    "Purchased product-store-week presence is not shelf availability and absence is not "
    "'no placement'; the accepted data cannot measure shelf availability, and causal_data "
    "records are never used as an availability measure."
)

#: Declared thresholds for the shortlist reconnaissance. These are transparent screens on
#: INDIVIDUAL diagnostic columns (not a composite score), stated so the researcher can see
#: exactly why each candidate entered the list, and reversible by changing this block.
SHORTLIST_BREADTH_RANK: Final[int] = 40
SHORTLIST_MIN_SHARE_3PLUS_WEEKS: Final[float] = 0.50
SHORTLIST_MIN_PRICE_USABILITY: Final[float] = 0.95
SHORTLIST_MIN_DISTINCT_PRODUCTS: Final[int] = 20
SHORTLIST_MAX_DISTINCT_PRODUCTS: Final[int] = 600

PRICE_USABILITY_QUESTION: Final[str] = (
    "If Phase 5 selected this commodity, how much of its transaction-line evidence has the "
    "already-accepted candidate-price formula available?"
)
UNITS_CAVEAT: Final[str] = (
    "Recorded units are dominated by very large-quantity lines (Question A finding QA-03): "
    "unit-based columns are reported for challenge and are never the headline ranking."
)


class QuestionCError(RuntimeError):
    """Raised when a Question C object does not reconcile to the accepted totals."""


# --- loading -----------------------------------------------------------------


def register_question_c_views(con) -> dict[str, Path]:
    """Register the accepted views Question C reads, including the choice-layer inputs."""
    registered = reconcile.register_accepted_views(con)
    registered.update(reconcile.register_accepted_views(con, tables=EXTRA_ACCEPTED_TABLES))
    return registered


def _frame(con, filename: str) -> pd.DataFrame:
    sql = (contracts.DESCRIPTIVE_SQL_DIR / filename).read_text(encoding="utf-8")
    return con.execute(sql).fetchdf()


def load_department_view(con) -> str:
    """Build the validated Phase 3 department derivation view and return its name."""
    department.register_sources(con)
    return department.build_view(con)


def load_commodity_level(con) -> pd.DataFrame:
    return _frame(con, "11_commodity_level.sql")


def load_commodity_occasion_and_price(con) -> pd.DataFrame:
    return _frame(con, "12_commodity_occasion_and_price.sql")


def load_department_level(con) -> pd.DataFrame:
    return _frame(con, "13_department_level.sql")


def load_carry_forward(con) -> pd.DataFrame:
    return _frame(con, "14_question_b_carry_forward.sql")


# --- reconciliation ----------------------------------------------------------


def reconcile_commodity_object(
    commodity: pd.DataFrame, occasion: pd.DataFrame, department_level: pd.DataFrame
) -> list[tuple[str, float, float]]:
    """Compare the Question C objects against the accepted anchors."""
    anchor = contracts.ACCEPTED_ANCHORS
    checks: list[tuple[str, float, float]] = [
        ("commodity rows (all labels)", len(commodity), 308.0),
        (
            "commodity retailer SALES_VALUE",
            commodity["retailer_sales_value"].sum(),
            anchor.retailer_sales_value,
        ),
        ("commodity recorded units", commodity["recorded_units"].sum(), anchor.units),
        (
            "national + private vs total (max absolute residual)",
            float(
                (
                    commodity["national_sales_value"]
                    + commodity["private_sales_value"]
                    - commodity["retailer_sales_value"]
                )
                .abs()
                .max()
            ),
            0.0,
        ),
        ("line-route transaction lines", occasion["transaction_lines"].sum(), 2581266.0),
        ("line-route recorded units", occasion["recorded_units"].sum(), anchor.units),
        ("department rows (all labels)", len(department_level), 44.0),
        (
            "department retailer SALES_VALUE",
            department_level["retailer_sales_value"].sum(),
            anchor.retailer_sales_value,
        ),
    ]
    failures = [
        (name, got, want) for name, got, want in checks if not _close(got, want, tolerance=1.0)
    ]
    if failures:
        detail = "; ".join(f"{name}: got {got!r} want {want!r}" for name, got, want in failures)
        raise QuestionCError(f"Question C objects do not reconcile: {detail}")
    return checks


def _close(got: float, want: float, tolerance: float) -> bool:
    """Absolute tolerance, floored at the anchor's own sales-value tolerance."""
    return abs(got - want) <= max(tolerance, contracts.ACCEPTED_ANCHORS.sales_value_abs_tolerance)


# --- descriptive helpers -----------------------------------------------------


def concentration_summary(frame: pd.DataFrame, column: str) -> dict[str, float]:
    """Concentration of one measure across the rows of ``frame`` (departments or commodities)."""
    values = frame[column].to_numpy(dtype=float)
    row = {
        "measure": column,
        "rows": float(len(frame)),
        "total": float(values.sum()),
        "gini_supplementary": metrics.gini(values),
    }
    for fraction, label in (
        (0.01, "top_1pct"),
        (0.05, "top_5pct"),
        (0.10, "top_10pct"),
        (0.20, "top_20pct"),
    ):
        row[f"{label}_share"] = metrics.top_activity_share(values, fraction)
    return row


def composition_columns(frame: pd.DataFrame, label_column: str = "commodity_desc") -> pd.DataFrame:
    """Stack the labelled national/private compositions into a tidy table.

    Both metrics are reported, each with its own numerator and denominator named, so neither
    can be read as "the" private-label share.
    """
    rows = []
    for record in frame.itertuples(index=False):
        national_share_value = record.national_share_of_sales_value
        private_share_value = record.private_share_of_sales_value
        national_share_units = record.national_share_of_units
        private_share_units = record.private_share_of_units
        label = getattr(record, label_column)
        rows.append(
            {
                "label": label,
                "metric": "retailer SALES_VALUE",
                "numerator": record.private_sales_value,
                "denominator": record.national_sales_value + record.private_sales_value,
                "private_share": private_share_value,
                "national_share": national_share_value,
                "unit_caveat": "not applicable: this metric is retailer SALES_VALUE based",
            }
        )
        rows.append(
            {
                "label": label,
                "metric": "recorded units",
                "numerator": record.private_units,
                "denominator": record.national_units + record.private_units,
                "private_share": private_share_units,
                "national_share": national_share_units,
                "unit_caveat": UNITS_CAVEAT,
            }
        )
    return pd.DataFrame(rows)


def stability_table(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Transparent per-label temporal stability diagnostics (components, not one score).

    Reported side by side because they answer slightly different questions: the
    minimum-to-maximum third ratio is sensitive to a small third, the coefficient of
    variation summarises relative dispersion, and presence in all three thirds is a
    binary support statement. Question A finding QA-01 means the early third is affected by
    observed support formation, so an early-versus-late difference is not product growth.
    """
    rows = []
    for record in frame.itertuples(index=False):
        values = np.array(
            [record.sales_value_early, record.sales_value_middle, record.sales_value_late],
            dtype=float,
        )
        households = np.array(
            [
                record.buying_households_early,
                record.buying_households_middle,
                record.buying_households_late,
            ],
            dtype=float,
        )
        maximum = values.max()
        rows.append(
            {
                "label": getattr(record, label_column),
                "sales_value_early": values[0],
                "sales_value_middle": values[1],
                "sales_value_late": values[2],
                "buying_households_early": households[0],
                "buying_households_middle": households[1],
                "buying_households_late": households[2],
                "min_third_to_max_third_ratio": float(values.min() / maximum)
                if maximum > 0
                else np.nan,
                "cv_across_thirds": float(values.std(ddof=1) / values.mean())
                if values.mean() > 0
                else np.nan,
                "present_in_all_three_thirds": int(record.present_in_all_three_thirds),
                "households_min_to_max_ratio": float(households.min() / households.max())
                if households.max() > 0
                else np.nan,
                "note": "components reported directly; the early third is affected by "
                "observed support formation (Question A finding QA-01)",
            }
        )
    return pd.DataFrame(rows)


def composition_similarity(
    frame: pd.DataFrame,
    label_column: str,
    group_column: str = "household_group",
    value_column: str = "retailer_sales_value",
) -> dict[str, float]:
    """Overlap of two groups' composition shares (overlap coefficient plus top-5 overlap).

    The overlap coefficient is sum over labels of min(share_A, share_B): 1.0 means the two
    groups spend their retailer SALES_VALUE in exactly the same proportions, 0.0 means they
    never share a label. It is a descriptive similarity of composition, not a behavioural
    statement, and it is never interpreted as a preference or motive.
    """
    groups = sorted(frame[group_column].unique())
    if len(groups) != 2:
        raise QuestionCError(f"composition_similarity expects two groups, got {groups}")
    left, right = groups
    left_shares = (
        frame[frame[group_column] == left].set_index(label_column)[value_column]
        / frame[frame[group_column] == left][value_column].sum()
    )
    right_shares = (
        frame[frame[group_column] == right].set_index(label_column)[value_column]
        / frame[frame[group_column] == right][value_column].sum()
    )
    joined = pd.concat([left_shares.rename("left"), right_shares.rename("right")], axis=1).fillna(
        0.0
    )
    top_left = list(left_shares.nlargest(5).index)
    top_right = list(right_shares.nlargest(5).index)
    return {
        "labels_left": float(len(left_shares)),
        "labels_right": float(len(right_shares)),
        "overlap_coefficient": float(np.minimum(joined["left"], joined["right"]).sum()),
        "top5_shared_labels": float(len(set(top_left) & set(top_right))),
        "top5_left": float(len(top_left)),
        "top5_right": float(len(top_right)),
    }


def breadth_intensity_table(commodity: pd.DataFrame) -> pd.DataFrame:
    """Breadth versus intensity rank positions for every commodity label.

    Large commodities can be large through breadth (many buying households), through repeat
    purchasing, through a high per-occasion amount, or through a combination. This table
    reports the rank position on each dimension so the mix can be read off directly rather
    than inferred from the value ranking.
    """
    frame = commodity.copy()
    frame["value_rank"] = frame["retailer_sales_value"].rank(ascending=False, method="min")
    frame["breadth_rank"] = frame["buying_households"].rank(ascending=False, method="min")
    frame["repeat_rank"] = frame["share_households_3plus_weeks"].rank(ascending=False, method="min")
    frame["intensity_rank"] = frame["sales_value_per_buying_household_week"].rank(
        ascending=False, method="min"
    )
    frame["rank_spread_breadth_to_intensity"] = frame["intensity_rank"] - frame["breadth_rank"]
    return frame[
        [
            "commodity_desc",
            "retailer_sales_value",
            "value_rank",
            "buying_households",
            "breadth_rank",
            "share_households_3plus_weeks",
            "repeat_rank",
            "sales_value_per_buying_household_week",
            "intensity_rank",
            "rank_spread_breadth_to_intensity",
        ]
    ]


def bulk_sensitivity_table(commodity: pd.DataFrame, occasion: pd.DataFrame) -> pd.DataFrame:
    """Unit-rank versus value-rank divergence once bulk lines are separated (QA-03).

    The as-is ranking is reported first, then the same ranking with bulk lines excluded, so
    the reader can see whether a commodity's apparent size in units survives the challenge.
    Nothing is trimmed: both columns are present.
    """
    merged = commodity.merge(
        occasion[
            [
                "commodity_desc",
                "bulk_units",
                "bulk_unit_share",
                "bulk_line_sales_value_share",
                "bulk_lines",
                "recorded_units",
                "line_sales_value",
                "top_basket_share_of_commodity_line_sales_value",
                "max_line_quantity",
                "transaction_lines",
            ]
        ],
        on="commodity_desc",
        how="left",
        suffixes=("", "_lines"),
        validate="one_to_one",
    )
    merged["units_excluding_bulk_lines"] = merged["recorded_units_lines"] - merged[
        "bulk_units"
    ].fillna(0)
    merged["unit_rank_as_is"] = merged["recorded_units_lines"].rank(ascending=False, method="min")
    merged["unit_rank_excluding_bulk"] = merged["units_excluding_bulk_lines"].rank(
        ascending=False, method="min"
    )
    merged["value_rank"] = merged["retailer_sales_value"].rank(ascending=False, method="min")
    merged["unit_rank_shift_excluding_bulk"] = (
        merged["unit_rank_as_is"] - merged["unit_rank_excluding_bulk"]
    )
    return merged[
        [
            "commodity_desc",
            "recorded_units_lines",
            "bulk_lines",
            "bulk_units",
            "bulk_unit_share",
            "bulk_line_sales_value_share",
            "units_excluding_bulk_lines",
            "unit_rank_as_is",
            "unit_rank_excluding_bulk",
            "unit_rank_shift_excluding_bulk",
            "value_rank",
            "max_line_quantity",
            "top_basket_share_of_commodity_line_sales_value",
            "transaction_lines",
        ]
    ]


# --- Phase 5 shortlist diagnostic matrix -------------------------------------


def shortlist_matrix(
    commodity: pd.DataFrame,
    occasion: pd.DataFrame,
    stability: pd.DataFrame,
    bulk: pd.DataFrame,
) -> pd.DataFrame:
    """One row per commodity covering the eight frozen shortlist dimensions.

    The eight dimensions are columns, never collapsed into a score: there is no overall
    score, no rank and no "best" column, and the availability dimension is a documented
    constant because the accepted data cannot measure shelf availability at all.
    """
    matrix = (
        commodity.merge(
            occasion[
                [
                    "commodity_desc",
                    "transaction_lines",
                    "positive_quantity_lines",
                    "loyal_candidate_usable_share",
                    "nonloyal_candidate_usable_share",
                    "package_size_completeness_share",
                    "baskets_with_commodity",
                    "baskets_single_commodity_product",
                    "baskets_multi_commodity_product",
                    "share_baskets_multi_commodity_product",
                    "median_products_from_commodity_per_basket",
                    "households_with_repeated_commodity_baskets",
                    "share_households_with_repeated_commodity_baskets",
                    "median_baskets_per_buying_household",
                    "p90_baskets_per_buying_household",
                ]
            ],
            on="commodity_desc",
            how="left",
            validate="one_to_one",
        )
        .merge(
            # ``present_in_all_three_thirds`` is already in the commodity object (the
            # stability table repeats it), so only the stability-specific columns are
            # taken here; merging the duplicate would silently suffix the column name.
            stability[
                [
                    "label",
                    "min_third_to_max_third_ratio",
                    "cv_across_thirds",
                    "households_min_to_max_ratio",
                ]
            ].rename(columns={"label": "commodity_desc"}),
            on="commodity_desc",
            how="left",
            validate="one_to_one",
        )
        .merge(
            bulk[
                [
                    "commodity_desc",
                    "bulk_unit_share",
                    "bulk_line_sales_value_share",
                    "unit_rank_shift_excluding_bulk",
                    "top_basket_share_of_commodity_line_sales_value",
                ]
            ],
            on="commodity_desc",
            how="left",
            validate="one_to_one",
        )
    )
    matrix["dimension_1_breadth"] = matrix["buying_households"]
    matrix["dimension_2_repeat"] = matrix["share_households_3plus_weeks"]
    matrix["dimension_3_occasion"] = matrix["share_baskets_multi_commodity_product"]
    matrix["dimension_4_assortment"] = matrix["distinct_products"]
    matrix["dimension_5_price_usability"] = matrix[
        ["loyal_candidate_usable_share", "nonloyal_candidate_usable_share"]
    ].min(axis=1)
    matrix["dimension_6_stability"] = matrix["min_third_to_max_third_ratio"]
    matrix["dimension_7_bulk_sensitivity"] = matrix["bulk_unit_share"]
    matrix["dimension_8_availability_limitation"] = AVAILABILITY_CAVEAT
    matrix["is_purchasable_product_label"] = ~matrix["commodity_desc"].isin(
        NON_PRODUCT_COMMODITY_LABELS
    )
    return matrix


def shortlist_screen_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-criterion pass/fail columns for the declared shortlist screens.

    Every screen is on one diagnostic column, so a reader can see exactly which criterion a
    commodity fails and re-run the screen with different thresholds. The screens are a
    reconnaissance device, not a selection rule.
    """
    frame = matrix.copy()
    frame["screen_breadth_top40"] = (
        frame["buying_households"].rank(ascending=False, method="min") <= SHORTLIST_BREADTH_RANK
    )
    frame["screen_repeat_3plus_weeks"] = (
        frame["share_households_3plus_weeks"] >= SHORTLIST_MIN_SHARE_3PLUS_WEEKS
    )
    frame["screen_price_usability"] = (
        frame["dimension_5_price_usability"] >= SHORTLIST_MIN_PRICE_USABILITY
    )
    frame["screen_present_in_all_thirds"] = frame["present_in_all_three_thirds"] == 1
    frame["screen_assortment_band"] = frame["distinct_products"].between(
        SHORTLIST_MIN_DISTINCT_PRODUCTS, SHORTLIST_MAX_DISTINCT_PRODUCTS
    )
    frame["screen_purchasable_label"] = frame["is_purchasable_product_label"]
    screen_columns = [column for column in frame.columns if column.startswith("screen_")]
    frame["screens_passed"] = frame[screen_columns].sum(axis=1)
    frame["passes_all_declared_screens"] = frame[screen_columns].all(axis=1)
    return frame


def shortlist_candidates(matrix: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """The reconnaissance shortlist: strengths, weaknesses and open Phase 5 decisions.

    Ordered by the number of declared screens passed and then by retailer SALES_VALUE, purely
    so the table has a stable order. No candidate is called best, no winner is labelled, and
    trade-offs between candidates are preserved rather than resolved.
    """
    screened = shortlist_screen_matrix(matrix)
    selected = screened[screened["passes_all_declared_screens"]].nlargest(
        limit, "retailer_sales_value"
    )
    rows = []
    for record in selected.itertuples(index=False):
        strengths = [
            f"breadth: {record.buying_households:,.0f} buying households "
            f"({record.observed_household_penetration:.1%} observed-household penetration)",
            f"repeat: {record.share_households_3plus_weeks:.1%} of buying households purchase in "
            "3 or more distinct weeks",
            f"candidate-price coverage: {record.dimension_5_price_usability:.1%} of positive-quantity "
            "lines carry both accepted candidate formulas",
            f"stability: minimum-to-maximum third ratio {record.min_third_to_max_third_ratio:.3f}, "
            "present in all three thirds",
            f"occasion evidence: {record.baskets_with_commodity:,.0f} commodity-containing baskets, "
            f"{record.share_baskets_multi_commodity_product:.1%} containing 2+ products from the commodity",
        ]
        weaknesses = [
            f"assortment: {record.distinct_products:,.0f} distinct purchased products "
            f"(median {record.median_products_per_household:,.0f} per buying household) - choice-set "
            "construction is not trivial",
            f"bulk-line exposure: {record.bulk_unit_share:.1%} of recorded units come from "
            "QUANTITY >= 1,000 lines (QA-03 caveat)",
            f"value concentration: the largest single basket is "
            f"{record.top_basket_share_of_commodity_line_sales_value:.2%} of the commodity's line "
            "sales value",
            f"private-label share of retailer SALES_VALUE "
            f"{record.private_share_of_sales_value:.1%} (national/private mix is not neutral)",
        ]
        rows.append(
            {
                "commodity_desc": record.commodity_desc,
                "why_it_entered": f"passes all {int(record.screens_passed)} declared screens",
                "retailer_sales_value": record.retailer_sales_value,
                "value_rank": record.retailer_sales_value
                and float((matrix["retailer_sales_value"] > record.retailer_sales_value).sum() + 1),
                "buying_households": record.buying_households,
                "observed_household_penetration": record.observed_household_penetration,
                "share_households_3plus_weeks": record.share_households_3plus_weeks,
                "distinct_products": record.distinct_products,
                "median_products_per_household": record.median_products_per_household,
                "price_usability": record.dimension_5_price_usability,
                "min_third_to_max_third_ratio": record.min_third_to_max_third_ratio,
                "bulk_unit_share": record.bulk_unit_share,
                "basis_of_baskets": record.baskets_with_commodity,
                "share_baskets_multi_commodity_product": record.share_baskets_multi_commodity_product,
                "strengths": " | ".join(strengths),
                "weaknesses_and_caveats": " | ".join(weaknesses),
                "what_phase_5_would_still_decide": (
                    "the choice-occasion definition (week, basket or store-week), the choice set "
                    "(product vs product-store-week) and its construction from this assortment, "
                    "which accepted candidate price is the economic price, and how the "
                    "national/private mix is handled"
                ),
            }
        )
    return pd.DataFrame(rows)


# --- candidate figures (exploratory; never published in this run) ------------


def _finish(fig, title: str, caption: str) -> None:
    fig.suptitle(title, fontsize=10.5, y=0.985)
    fig.text(0.006, 0.004, caption, fontsize=6.2, va="bottom")


def figure_structure_candidates(commodity: pd.DataFrame, department_level: pd.DataFrame):
    """F5 candidate: major departments and major commodities by retailer SALES_VALUE."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.87, bottom=0.24, wspace=0.42)

    departments = department_level[
        ~department_level["department_is_unlabelled"].fillna(False)
    ].nlargest(15, "retailer_sales_value")
    axes[0].barh(
        departments["department_desc"][::-1],
        departments["retailer_sales_value"][::-1],
        color="#1f4e79",
    )
    for position, (value, penetration) in enumerate(
        zip(
            departments["retailer_sales_value"][::-1],
            departments["observed_household_penetration"][::-1],
            strict=True,
        )
    ):
        axes[0].text(value, position, f"  {penetration:.0%}", va="center", fontsize=6.6)
    axes[0].set_xlabel("retailer SALES_VALUE (retailer revenue)", fontsize=8)
    axes[0].set_title(
        "A. Departments by retailer SALES_VALUE (penetration labelled)", fontsize=9, loc="left"
    )
    axes[0].tick_params(labelsize=7)
    axes[0].grid(alpha=0.25, lw=0.5, axis="x")

    commodities = commodity.nlargest(20, "retailer_sales_value")
    axes[1].barh(
        commodities["commodity_desc"][::-1],
        commodities["retailer_sales_value"][::-1],
        color="#2e7d32",
    )
    for position, penetration in enumerate(commodities["observed_household_penetration"][::-1]):
        axes[1].text(
            commodities["retailer_sales_value"][::-1].iloc[position],
            position,
            f"  {penetration:.0%}",
            va="center",
            fontsize=6.6,
        )
    axes[1].set_xlabel("retailer SALES_VALUE (retailer revenue)", fontsize=8)
    axes[1].set_title(
        "B. Commodities by retailer SALES_VALUE (penetration labelled)", fontsize=9, loc="left"
    )
    axes[1].tick_params(labelsize=6.6)
    axes[1].grid(alpha=0.25, lw=0.5, axis="x")
    _finish(
        fig,
        "Candidate F5 - product and commodity structure (population S0)",
        "Bar values are retailer SALES_VALUE (retailer revenue) over the 102-week window; the "
        "labels give observed-household penetration = distinct buying households / 2,500, never "
        "market penetration. The unlabelled sentinel department is retained in the table (it "
        "carries no recorded value) and is not shown here. Unit totals are excluded because "
        "recorded units are dominated by very large-quantity lines (Question A finding QA-03). "
        "COUPON/MISC ITEMS is the vendor's coupon/miscellaneous label, not a purchasable product "
        f"category. {AVAILABILITY_CAVEAT}",
    )
    return fig


def figure_penetration_vs_intensity(commodity: pd.DataFrame):
    """F6 candidate: breadth against purchase-conditional intensity (and repeat purchasing)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.2, wspace=0.22)
    frame = commodity[~commodity["commodity_desc"].isin(NON_PRODUCT_COMMODITY_LABELS)]

    size = 6 + 60 * (frame["distinct_products"] / frame["distinct_products"].max())
    axes[0].scatter(
        frame["observed_household_penetration"] * 100,
        frame["sales_value_per_buying_household_week"],
        s=size,
        alpha=0.5,
        color="#1f4e79",
        edgecolors="none",
    )
    for record in frame.nlargest(8, "retailer_sales_value").itertuples(index=False):
        axes[0].annotate(
            record.commodity_desc,
            (
                record.observed_household_penetration * 100,
                record.sales_value_per_buying_household_week,
            ),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    axes[0].set_xlabel("observed-household penetration (%)", fontsize=8)
    axes[0].set_ylabel("retailer SALES_VALUE per buying household-week", fontsize=8)
    axes[0].set_title("A. Breadth against purchase-conditional intensity", fontsize=9, loc="left")
    axes[0].grid(alpha=0.25, lw=0.5)

    axes[1].scatter(
        frame["observed_household_penetration"] * 100,
        frame["share_households_3plus_weeks"] * 100,
        s=size,
        alpha=0.5,
        color="#6a1b9a",
        edgecolors="none",
    )
    for record in frame.nlargest(8, "retailer_sales_value").itertuples(index=False):
        axes[1].annotate(
            record.commodity_desc,
            (
                record.observed_household_penetration * 100,
                record.share_households_3plus_weeks * 100,
            ),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    axes[1].set_xlabel("observed-household penetration (%)", fontsize=8)
    axes[1].set_ylabel("share of buying households with 3+ purchase weeks (%)", fontsize=8)
    axes[1].set_title("B. Breadth against repeated purchasing", fontsize=9, loc="left")
    axes[1].grid(alpha=0.25, lw=0.5)
    _finish(
        fig,
        "Candidate F6 - commodity penetration against intensity and repeat purchasing",
        "Point size encodes the number of distinct purchased products in the commodity. Both axes "
        "are descriptive; the vertical axis of panel A is retailer SALES_VALUE per buying "
        "household-week (denominator: that commodity's own buying household-weeks), and panel B "
        "uses the commodity's buying households as the denominator. COUPON/MISC ITEMS and the "
        "missing-label sentinel are excluded here because they are not purchasable product "
        "categories; they remain in the tables.",
    )
    return fig


def figure_national_private(commodity: pd.DataFrame, top_n: int = 20):
    """F7 candidate: national/private composition in both metrics, side by side."""
    import matplotlib.pyplot as plt

    frame = commodity[~commodity["commodity_desc"].isin(NON_PRODUCT_COMMODITY_LABELS)].nlargest(
        top_n, "retailer_sales_value"
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0))
    fig.subplots_adjust(left=0.16, right=0.985, top=0.87, bottom=0.24, wspace=0.5)
    for ax, (value_column, title, metric) in zip(
        axes,
        (
            (
                "private_share_of_sales_value",
                "A. Composition in retailer SALES_VALUE",
                "retailer SALES_VALUE",
            ),
            ("private_share_of_units", "B. Composition in recorded units", "recorded units"),
        ),
        strict=True,
    ):
        ax.barh(
            frame["commodity_desc"][::-1],
            (frame[value_column] * 100)[::-1],
            color="#b26500",
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"private-label share of the commodity's {metric} (%)", fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=6.8)
        ax.grid(alpha=0.25, lw=0.5, axis="x")
    _f7_caption = (
        "Panel A: numerator = recorded private-label retailer SALES_VALUE in the commodity, "
        "denominator = that commodity's national plus private retailer SALES_VALUE. Panel B: "
        "numerator = recorded private-label units, denominator = national plus private units. "
        "They are different metrics and neither is 'the' private-label share. "
        f"{UNITS_CAVEAT} Restricted to the top {top_n} commodities by retailer SALES_VALUE for "
        "readability; the exploratory table retains all 308 labels."
    )
    _finish(fig, "Candidate F7 - national/private-label composition", _f7_caption)
    return fig


def figure_repeat_and_stability(matrix: pd.DataFrame, department_level: pd.DataFrame):
    """Extra diagnostic: repeat purchasing and temporal stability, components kept separate.

    Takes the assembled shortlist matrix, because the stability diagnostic is computed there
    rather than in the commodity object.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.86, bottom=0.24, wspace=0.3)
    frame = matrix[~matrix["commodity_desc"].isin(NON_PRODUCT_COMMODITY_LABELS)]

    axes[0].scatter(
        frame["buying_households"],
        frame["share_households_3plus_weeks"] * 100,
        s=6,
        alpha=0.45,
        color="#2e7d32",
        edgecolors="none",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("buying households (log scale)", fontsize=8)
    axes[0].set_ylabel("share of buying households with 3+ purchase weeks (%)", fontsize=8)
    axes[0].set_title("A. Is repeat purchasing just breadth?", fontsize=9, loc="left")

    axes[1].scatter(
        frame["retailer_sales_value"],
        frame["min_third_to_max_third_ratio"],
        s=6,
        alpha=0.45,
        color="#6a1b9a",
        edgecolors="none",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("retailer SALES_VALUE (log scale)", fontsize=8)
    axes[1].set_ylabel("minimum-to-maximum third ratio", fontsize=8)
    axes[1].set_title("B. Temporal stability against size", fontsize=9, loc="left")

    departments = department_level[
        ~department_level["department_is_unlabelled"].fillna(False)
    ].nlargest(12, "retailer_sales_value")
    axes[2].barh(
        departments["department_desc"][::-1],
        (departments["private_share_of_sales_value"] * 100)[::-1],
        color="#b26500",
        label="retailer SALES_VALUE based",
    )
    axes[2].barh(
        departments["department_desc"][::-1],
        (departments["private_share_of_units"] * 100)[::-1],
        color="#1f4e79",
        alpha=0.45,
        height=0.45,
        label="recorded-unit based",
    )
    axes[2].set_xlabel("private-label share (%)", fontsize=8)
    axes[2].set_title(
        "C. Private-label share, two metrics (top departments)", fontsize=9, loc="left"
    )
    axes[2].legend(fontsize=6.5)
    axes[2].tick_params(labelsize=6.5)
    for ax in axes[:2]:
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
    _finish(
        fig,
        "Exploratory diagnostics - repeat purchasing, stability, label composition",
        "Panel A tests whether repeat purchasing is mechanically a breadth story: a commodity "
        "with many buying households does not automatically have a high share of repeat "
        "households, and the scatter is reported rather than summarised into one score. Panel B "
        "keeps the components of the stability diagnostic visible; the early third is affected by "
        "observed support formation (Question A finding QA-01). Panel C compares the two label "
        "composition metrics for the same departments, where the unit-based metric carries the "
        f"QA-03 caveat. {AVAILABILITY_CAVEAT}",
    )
    return fig


def figure_carry_forward(carry_forward: pd.DataFrame, top_n: int = 10):
    """Question B carry-forward diagnostic: broad composition of the two extreme groups."""
    import matplotlib.pyplot as plt

    departments = carry_forward[carry_forward["dimension"] == "department"]
    groups = sorted(departments["household_group"].unique())
    reference = groups[0]
    labels = (
        departments[departments["household_group"] == reference]
        .nlargest(top_n, "retailer_sales_value")["label"]
        .tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    fig.subplots_adjust(left=0.2, right=0.985, top=0.87, bottom=0.22, wspace=0.5)
    positions = np.arange(len(labels))[::-1]
    width = 0.38
    for offset, group in enumerate(groups):
        subset = departments[departments["household_group"] == group].set_index("label")
        shares = [
            float(subset.loc[label, "share_of_group_retailer_sales_value"]) * 100
            if label in subset.index
            else 0.0
            for label in labels
        ]
        axes[0].barh(
            positions + (offset - 0.5) * width,
            shares,
            height=width,
            label=group,
            color="#1f4e79" if offset == 0 else "#b26500",
        )
    axes[0].set_yticks(positions, labels, fontsize=7)
    axes[0].set_xlabel("share of the group's recorded retailer SALES_VALUE (%)", fontsize=8)
    axes[0].set_title("A. Department composition of the two extreme groups", fontsize=9, loc="left")
    axes[0].legend(fontsize=6.8)
    axes[0].grid(alpha=0.25, lw=0.5, axis="x")

    for offset, group in enumerate(groups):
        subset = departments[departments["household_group"] == group].nlargest(
            3, "retailer_sales_value"
        )
        axes[1].bar(
            np.arange(len(subset)) + (offset - 0.5) * 0.38,
            subset["share_of_group_retailer_sales_value"] * 100,
            width=0.38,
            label=group,
            color="#1f4e79" if offset == 0 else "#b26500",
        )
    axes[1].set_xticks(np.arange(3), [f"largest {i + 1}" for i in range(3)], fontsize=7.5)
    axes[1].set_ylabel("share of group value (%)", fontsize=8)
    axes[1].set_title("B. Concentration in the largest departments", fontsize=9, loc="left")
    axes[1].grid(alpha=0.25, lw=0.5, axis="y")
    axes[1].tick_params(labelsize=7)
    _finish(
        fig,
        "Question B carry-forward - broad composition of the extreme household groups",
        "Groups are the top 1% of households by whole-sample retailer SALES_VALUE and the top 1% "
        "by whole-sample trips (25 households each, defined by a declared rank rule). Only group "
        "aggregates are shown: no household identifier appears anywhere. Composition is "
        "descriptive and is never read as preferences, motives, demographics or causality.",
    )
    return fig


# --- build -------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionCResult:
    """Everything the Question C reconnaissance pass produced, with its evidence."""

    commodity: pd.DataFrame
    department_level: pd.DataFrame
    occasion: pd.DataFrame
    carry_forward: pd.DataFrame
    matrix: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    figures: dict[str, Path]
    reconciliation: list[tuple[str, float, float]]
    summary_path: Path


def _write_table(name: str, frame: pd.DataFrame) -> Path:
    path = artifacts.exploratory_path("table", name)
    return artifacts.write_table_csv(
        path,
        list(frame.columns),
        frame.itertuples(index=False, name=None),
        scope="exploratory",
    )


def _write_figure(name: str, fig) -> Path:
    return artifacts.write_figure_png(fig, artifacts.exploratory_path("figure", name))


def summary_table(
    tables: dict[str, pd.DataFrame], reconciliation: list[tuple[str, float, float]]
) -> pd.DataFrame:
    """Long-format Question C summary (exploratory; not the final T2)."""
    rows: list[dict[str, object]] = []

    def add(section: str, statistic: str, value: object, note: str = "") -> None:
        rows.append({"section": section, "statistic": statistic, "value": value, "note": note})

    for name, got, want in reconciliation:
        add("reconciliation", name, got, f"accepted {want}")

    for record in tables["concentration"].itertuples(index=False):
        add("concentration", f"{record.measure}: rows", record.rows, "labels in the dimension")
        add("concentration", f"{record.measure}: top 1% share", record.top_1pct_share, "")
        add("concentration", f"{record.measure}: top 10% share", record.top_10pct_share, "")
        add(
            "concentration",
            f"{record.measure}: Gini (supplementary)",
            record.gini_supplementary,
            "supplementary only",
        )

    commodity = tables["commodity_level"]
    for column in (
        "buying_households",
        "observed_household_penetration",
        "retailer_sales_value",
        "share_households_3plus_weeks",
        "share_households_5plus_weeks",
        "sales_value_per_buying_household_week",
        "distinct_products",
    ):
        add(
            "commodity distribution",
            f"{column}: median",
            commodity[column].median(),
            "all 308 labels",
        )
        add(
            "commodity distribution",
            f"{column}: p90",
            commodity[column].quantile(0.90),
            "all 308 labels",
        )
        add("commodity distribution", f"{column}: max", commodity[column].max(), "all 308 labels")

    matrix = tables["shortlist_matrix"]
    for column in (
        "dimension_5_price_usability",
        "bulk_unit_share",
        "bulk_line_sales_value_share",
        "share_baskets_multi_commodity_product",
        "share_households_with_repeated_commodity_baskets",
        "package_size_completeness_share",
    ):
        add("shortlist diagnostics", f"{column}: median", matrix[column].median(), "per-commodity")
        add("shortlist diagnostics", f"{column}: min", matrix[column].min(), "per-commodity")

    candidates = tables["shortlist_candidates"]
    add(
        "shortlist",
        "candidates passing all declared screens",
        len(candidates),
        "reconnaissance only",
    )
    for record in candidates.itertuples(index=False):
        add(
            "shortlist",
            f"{record.commodity_desc}: retailer SALES_VALUE",
            record.retailer_sales_value,
            f"{record.buying_households:,.0f} buying households; "
            f"{record.share_households_3plus_weeks:.1%} with 3+ purchase weeks; "
            f"price usability {record.price_usability:.1%}",
        )

    for record in tables["carry_forward_similarity"].itertuples(index=False):
        add(
            "carry-forward",
            f"{record.dimension}: composition overlap coefficient",
            record.overlap_coefficient,
            "sum of min(share) between the two extreme groups; descriptive similarity only",
        )
        add(
            "carry-forward",
            f"{record.dimension}: shared labels in each group's top 5",
            record.top5_shared_labels,
            "of 5",
        )
    return pd.DataFrame(rows)


def build_t2_matrix(con) -> pd.DataFrame:
    """Build the Phase 5 handoff matrix (one row per commodity, all 308 labels).

    The final T2 table and the exploratory shortlist matrix are built by this same
    function, so the published artifact and the reviewed cross-tabs cannot drift apart.
    """
    load_department_view(con)
    commodity = load_commodity_level(con)
    occasion = load_commodity_occasion_and_price(con)
    stability = stability_table(commodity, "commodity_desc")
    bulk = bulk_sensitivity_table(commodity, occasion)
    return shortlist_matrix(commodity, occasion, stability, bulk)


def build_question_c(con) -> QuestionCResult:
    """Run the Question C reconnaissance pass and write only ignored exploratory artefacts."""
    load_department_view(con)
    commodity = load_commodity_level(con)
    occasion = load_commodity_occasion_and_price(con)
    department_level = load_department_level(con)
    carry_forward = load_carry_forward(con)

    reconciliation = reconcile_commodity_object(commodity, occasion, department_level)
    stability = stability_table(commodity, "commodity_desc")
    department_stability = stability_table(department_level, "department_desc")
    composition = composition_columns(commodity)
    bulk = bulk_sensitivity_table(commodity, occasion)
    breadth_intensity = breadth_intensity_table(commodity)
    matrix = shortlist_matrix(commodity, occasion, stability, bulk)
    screened = shortlist_screen_matrix(matrix)
    candidates = shortlist_candidates(matrix)

    concentration = pd.DataFrame(
        [
            {
                "dimension": "commodity",
                **concentration_summary(commodity, "retailer_sales_value"),
                "what": "retailer SALES_VALUE",
            },
            {
                "dimension": "commodity",
                **concentration_summary(commodity, "buying_households"),
                "what": "buying households",
            },
            {
                "dimension": "commodity",
                **concentration_summary(commodity, "household_weeks"),
                "what": "buying household-weeks",
            },
            {
                "dimension": "department",
                **concentration_summary(department_level, "retailer_sales_value"),
                "what": "retailer SALES_VALUE",
            },
            {
                "dimension": "department",
                **concentration_summary(department_level, "buying_households"),
                "what": "buying households",
            },
        ]
    )

    similarity_rows = []
    for dimension in ("commodity", "department"):
        subset = carry_forward[carry_forward["dimension"] == dimension]
        similarity = composition_similarity(subset, "label")
        similarity_rows.append({"dimension": dimension, **similarity})
    carry_similarity = pd.DataFrame(similarity_rows)

    tables: dict[str, pd.DataFrame] = {
        "commodity_level": commodity,
        "department_level": department_level,
        "department_stability": department_stability,
        "commodity_occasion_price": occasion,
        "commodity_stability": stability,
        "composition_national_private": composition,
        "composition_national_private_department": composition_columns(
            department_level, "department_desc"
        ),
        "bulk_sensitivity": bulk,
        "breadth_intensity": breadth_intensity,
        "concentration": concentration,
        "shortlist_matrix": matrix,
        "shortlist_screens": screened,
        "shortlist_candidates": candidates,
        "carry_forward": carry_forward,
        "carry_forward_similarity": carry_similarity,
    }

    figures: dict[str, Path] = {}
    from matplotlib import pyplot as plt

    builders = (
        (
            "q_c_f5_structure_candidates.png",
            lambda: figure_structure_candidates(commodity, department_level),
        ),
        ("q_c_f6_penetration_vs_intensity.png", lambda: figure_penetration_vs_intensity(commodity)),
        ("q_c_f7_national_private.png", lambda: figure_national_private(commodity)),
        (
            "q_c_repeat_stability_diagnostics.png",
            lambda: figure_repeat_and_stability(matrix, department_level),
        ),
        ("q_c_carry_forward_composition.png", lambda: figure_carry_forward(carry_forward)),
    )
    for filename, builder in builders:
        fig = builder()
        try:
            figures[filename] = _write_figure(filename, fig)
        finally:
            plt.close(fig)

    tables["summary"] = summary_table(tables, reconciliation)
    for name, frame in tables.items():
        _write_table(f"q_c_{name}.csv", frame)

    return QuestionCResult(
        commodity=commodity,
        department_level=department_level,
        occasion=occasion,
        carry_forward=carry_forward,
        matrix=matrix,
        tables=tables,
        figures=figures,
        reconciliation=reconciliation,
        summary_path=artifacts.exploratory_path("table", "q_c_summary.csv"),
    )


def format_report(result: QuestionCResult) -> str:
    """Human-readable summary of the reconnaissance pass (no finding is promoted here)."""
    lines: list[str] = []
    lines.append("  reconciliation:")
    for name, got, want in result.reconciliation:
        lines.append(f"    {name:<52} {got:>16,.4f}  (accepted {want:>16,.4f})")

    lines.append("")
    lines.append("  department structure (top 10 by retailer SALES_VALUE):")
    for record in result.department_level.nlargest(10, "retailer_sales_value").itertuples(
        index=False
    ):
        lines.append(
            f"    {str(record.department_desc)[:26]:<28} households={record.buying_households:>5} "
            f"pen={record.observed_household_penetration:>6.3f} value={record.retailer_sales_value:>14,.2f} "
            f"priv(val)={record.private_share_of_sales_value:>6.3f} priv(units)={record.private_share_of_units:>6.3f}"
        )
    unlabelled = result.department_level[
        result.department_level["department_is_unlabelled"].fillna(False)
    ]
    if len(unlabelled):
        lines.append(
            f"    [sentinel department retained] labels={len(unlabelled)} "
            f"households={int(unlabelled['buying_households'].iloc[0])} "
            f"value={float(unlabelled['retailer_sales_value'].iloc[0]):,.2f}"
        )

    lines.append("")
    lines.append("  commodity structure (top 12 by retailer SALES_VALUE):")
    for record in result.commodity.nlargest(12, "retailer_sales_value").itertuples(index=False):
        lines.append(
            f"    {str(record.commodity_desc)[:28]:<30} households={record.buying_households:>5} "
            f"pen={record.observed_household_penetration:>6.3f} weeks={record.household_weeks:>7} "
            f"value={record.retailer_sales_value:>13,.2f} 3+wk={record.share_households_3plus_weeks:>6.3f} "
            f"prod={record.distinct_products:>5} priv(val)={record.private_share_of_sales_value:>6.3f}"
        )

    lines.append("")
    lines.append("  concentration across labels:")
    for record in result.tables["concentration"].itertuples(index=False):
        lines.append(
            f"    {record.dimension:<11} {record.what:<24} rows={record.rows:>4.0f} "
            f"top1%={record.top_1pct_share:>6.3f} top10%={record.top_10pct_share:>6.3f} "
            f"gini={record.gini_supplementary:>6.3f}"
        )

    lines.append("")
    lines.append("  commodity distribution across all 308 labels (median / p90 / max):")
    for column in (
        "buying_households",
        "observed_household_penetration",
        "household_weeks",
        "retailer_sales_value",
        "share_households_2plus_weeks",
        "share_households_3plus_weeks",
        "share_households_5plus_weeks",
        "distinct_products",
        "sales_value_per_buying_household_week",
        "trips_per_buying_household_week",
    ):
        series = result.commodity[column]
        lines.append(
            f"    {column:<38} {series.median():>12,.4f} {series.quantile(0.90):>14,.4f} {series.max():>16,.4f}"
        )

    lines.append("")
    lines.append("  repeat purchasing (all 308 labels, median / p90):")
    for column in (
        "share_households_2plus_weeks",
        "share_households_3plus_weeks",
        "share_households_5plus_weeks",
    ):
        lines.append(
            f"    {column:<38} {result.commodity[column].median():>12.4f} {result.commodity[column].quantile(0.90):>14.4f}"
        )

    lines.append("")
    lines.append("  purchase-occasion diagnostics (all 308 labels, median / p90):")
    occasion_columns = (
        "commodity_baskets",
        "baskets_multi_commodity_product",
        "share_baskets_multi_commodity_product",
        "share_households_with_repeated_commodity_baskets",
        "median_products_from_commodity_per_basket",
        "median_baskets_per_buying_household",
    )
    for column in occasion_columns:
        series = result.occasion[column]
        lines.append(f"    {column:<46} {series.median():>12.4f} {series.quantile(0.90):>14.4f}")

    lines.append("")
    lines.append("  assortment complexity (all 308 labels, median / p90 / max):")
    for column in (
        "distinct_products",
        "median_products_per_household",
        "p90_products_per_household",
        "max_products_per_household",
    ):
        series = result.commodity[column]
        lines.append(
            f"    {column:<38} {series.median():>12,.2f} {series.quantile(0.90):>14,.2f} {series.max():>16,.2f}"
        )

    lines.append("")
    lines.append("  candidate-price usability (per commodity):")
    matrix = result.matrix
    lines.append(
        f"    loyal usable share: min={matrix['loyal_candidate_usable_share'].min():.4f} "
        f"median={matrix['loyal_candidate_usable_share'].median():.4f} "
        f"| nonloyal: min={matrix['nonloyal_candidate_usable_share'].min():.4f} "
        f"median={matrix['nonloyal_candidate_usable_share'].median():.4f}"
    )
    lines.append(
        f"    package-size completeness: median={matrix['package_size_completeness_share'].median():.4f} "
        f"min={matrix['package_size_completeness_share'].min():.4f}"
    )
    lines.append("    commodities with the lowest joint candidate coverage:")
    for record in matrix.nsmallest(5, "dimension_5_price_usability")[
        [
            "commodity_desc",
            "transaction_lines",
            "loyal_candidate_usable_share",
            "nonloyal_candidate_usable_share",
        ]
    ].itertuples(index=False):
        lines.append(
            f"      {str(record.commodity_desc)[:34]:<36} lines={record.transaction_lines:>8,.0f} "
            f"loyal={record.loyal_candidate_usable_share:.4f} nonloyal={record.nonloyal_candidate_usable_share:.4f}"
        )

    lines.append("")
    lines.append("  temporal stability (commodity):")
    stability = result.tables["commodity_stability"]
    lines.append(
        f"    min-to-max third ratio: median={stability['min_third_to_max_third_ratio'].median():.4f} "
        f"p10={stability['min_third_to_max_third_ratio'].quantile(0.10):.4f} "
        f"| present in all three thirds: {int(stability['present_in_all_three_thirds'].sum())}/{len(stability)} labels"
    )
    lines.append(
        f"    CV across thirds: median={stability['cv_across_thirds'].median():.4f} "
        f"p90={stability['cv_across_thirds'].quantile(0.90):.4f} "
        f"| labels below a 0.5 ratio: {int((stability['min_third_to_max_third_ratio'] < 0.5).sum())} "
        f"| labels below 0.2: {int((stability['min_third_to_max_third_ratio'] < 0.2).sum())}"
    )

    lines.append("")
    lines.append("  bulk / outlier sensitivity (QA-03):")
    bulk = result.tables["bulk_sensitivity"]
    lines.append(
        f"    bulk unit share: median={bulk['bulk_unit_share'].median():.4f} "
        f"p90={bulk['bulk_unit_share'].quantile(0.90):.4f} max={bulk['bulk_unit_share'].max():.4f}"
    )
    lines.append(
        f"    value in bulk lines: median={bulk['bulk_line_sales_value_share'].median():.4f} "
        f"max={bulk['bulk_line_sales_value_share'].max():.4f} "
        f"| labels whose unit rank moves 20+ places once bulk lines are removed: "
        f"{int((bulk['unit_rank_shift_excluding_bulk'].abs() >= 20).sum())}"
    )
    lines.append(
        f"    top single basket share of line sales value: median={bulk['top_basket_share_of_commodity_line_sales_value'].median():.4f} "
        f"max={bulk['top_basket_share_of_commodity_line_sales_value'].max():.4f}"
    )

    lines.append("")
    lines.append("  shortlist screens (declared, one column each):")
    screened = result.tables["shortlist_screens"]
    for column in [c for c in screened.columns if c.startswith("screen_")]:
        lines.append(f"    {column:<34} passes={int(screened[column].sum()):>4}")
    lines.append(
        f"    commodities passing every declared screen: {int(screened['passes_all_declared_screens'].sum())}"
    )

    lines.append("")
    lines.append("  reconnaissance shortlist (no winner, no ranking claim):")
    candidates = result.tables["shortlist_candidates"]
    if not len(candidates):
        lines.append("    NO commodity passes every declared screen: report the trade-offs instead")
    for record in candidates.itertuples(index=False):
        lines.append(
            f"    {str(record.commodity_desc)[:30]:<32} households={record.buying_households:>5} "
            f"pen={record.observed_household_penetration:>6.3f} 3+wk={record.share_households_3plus_weeks:>6.3f} "
            f"prod={record.distinct_products:>5} price={record.price_usability:>6.3f} "
            f"stab={record.min_third_to_max_third_ratio:>6.3f} bulk_units={record.bulk_unit_share:>6.3f}"
        )

    lines.append("")
    lines.append("  Question B carry-forward (broad composition of the two extreme groups):")
    for record in result.tables["carry_forward_similarity"].itertuples(index=False):
        lines.append(
            f"    {record.dimension:<11} labels {record.labels_left:.0f}/{record.labels_right:.0f} "
            f"overlap coefficient={record.overlap_coefficient:.4f} "
            f"shared top-5 labels={record.top5_shared_labels:.0f}/5"
        )

    lines.append("")
    lines.append("  outputs (exploratory, Git-ignored; no household identifier in any of them):")
    for name, frame in sorted(result.tables.items()):
        lines.append(f"    q_c_{name}.csv  ({len(frame)} rows)")
    for name, path in sorted(result.figures.items()):
        lines.append(f"    {name}  ->  {path.name}")
    return "\n".join(lines)
