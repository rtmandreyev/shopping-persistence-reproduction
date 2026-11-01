"""Question D: campaign/promotion structure and Phase 6 identification reconnaissance.

EXPLORATORY / CHALLENGE run. This module produces descriptive evidence about how campaigns,
listed households and recorded redemptions are distributed, how much listing overlap exists, and
which campaign design features constrain a later identification investigation. It estimates NO
campaign effect, defines NO comparison group, chooses NO event window or horizon, selects
NO campaign, and creates NO instrument.

Vocabulary: this module uses *campaign*, *campaign type*, *listed household*, *listed pair*,
*listed household-week*, *campaign calendar span*, *recorded redemption*, *overlapping listing* and
*descriptive pre/campaign/post support*. The campaign vocabulary this project refuses to use is
registered as data in ``contracts.CAMPAIGN_FORBIDDEN_VOCABULARY`` and none of those terms appears
anywhere in this module: a listed household is factual database membership rather than an assigned
unit, a household not listed for a campaign is not a comparison group, and no campaign effect is
estimated here. The narrative section that explains why those terms are not justified is
``docs/phases/PHASE_03_DESCRIPTIVE_ANALYTICS.md`` section 21.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from complete_journey.data import raw_tables
from complete_journey.descriptive import artifacts, contracts, reconcile

# --- declared constants -------------------------------------------------------

HOUSEHOLD_POPULATION: Final[int] = 2500
ACCEPTED_LISTED_PAIRS: Final[int] = 7208
ACCEPTED_LISTED_HOUSEHOLD_WEEKS: Final[int] = 735216
ACCEPTED_REDEMPTION_ROWS: Final[int] = 2318
ACCEPTED_CAMPAIGNS: Final[int] = 30
ACCEPTED_PHASE1_OVERLAP_MEDIAN_PCT: Final[float] = 85.77

#: Reconnaissance windows only. These are NOT the Phase 6 event window, anticipation window or
#: final pre/post horizon; they exist so campaigns can be described on a common descriptive
#: basis. Each is truncated by the observed weeks 1..102.
RECONNAISSANCE_WEEKS_BEFORE: Final[int] = 4
RECONNAISSANCE_WEEKS_AFTER: Final[int] = 4

#: Declared one-column screens for the Phase 6 investigation shortlist. They ask "is this
#: campaign worth a later identification review", never "is this campaign causal". A stricter
#: secondary cut on overlap is declared here as well so no cut is chosen after seeing results.
SHORTLIST_MIN_OBSERVED_WEEKS_BEFORE: Final[int] = 4
SHORTLIST_MIN_OBSERVED_WEEKS_AFTER: Final[int] = 4
SHORTLIST_MIN_LISTED_HOUSEHOLDS: Final[int] = 50
SHORTLIST_MAX_ACCEPTED_OVERLAP_PCT: Final[float] = 50.0
SHORTLIST_STRICT_MAX_ACCEPTED_OVERLAP_PCT: Final[float] = 25.0
SHORTLIST_TARGET_MIN: Final[int] = 3
SHORTLIST_TARGET_MAX: Final[int] = 6

#: Trajectory diagnostics require a minimum listed group so a "trajectory" is not drawn from a
#: handful of households (the same threshold Step 13 sets for any campaign-relative diagnostic).
TRAJECTORY_MIN_LISTED_HOUSEHOLDS: Final[int] = 50

#: Dimensions of the Phase 6 reconnaissance matrix. Separate columns, never a composite score.
RECONNAISSANCE_DIMENSIONS: Final[tuple[str, ...]] = (
    "dimension_1_timing_support",
    "dimension_2_listing_breadth",
    "dimension_3_overlap",
    "dimension_4_pre_campaign_observable_selection",
    "dimension_5_pre_campaign_trajectory_comparability",
    "dimension_6_redemption_information",
    "dimension_7_edge_of_sample_risk",
    "dimension_8_assignment_caveat",
)

#: Vocabulary this module does use. The vocabulary it must NOT use is registered once, as data,
#: in ``contracts.CAMPAIGN_FORBIDDEN_VOCABULARY``; the descriptive package is lint-checked against
#: that registry, so the terms never appear in this module even in prose.
VOCABULARY: Final[tuple[str, ...]] = (
    "campaign",
    "campaign type",
    "listed household",
    "listed pair",
    "listed household-week",
    "campaign calendar span",
    "recorded redemption",
    "overlapping listing",
    "descriptive pre/campaign/post support",
)

ASSIGNMENT_CAVEAT: Final[str] = (
    "Campaign receipt is not documented as randomised in this dataset; Type A household coupon "
    "selection is unobserved and based on prior purchasing behaviour (vendor limitation A), so "
    "for Type A the coupons actually received are outside the database. A listed household is "
    "factual database membership rather than an assigned unit, and a household not listed for a "
    "campaign is not a comparison group."
)
REDEMPTION_CAVEAT: Final[str] = (
    "Recorded redemption is a fact about the redemption table. Coupon assignment and receipt are "
    "not fully observed (vendor limitation A: for Type A the household-specific coupons actually "
    "received are outside the database), and the denominator available here is listed households, "
    "which is not the population that received an offer. Every redemption column is therefore a "
    "recorded count or a share of listed households, never a rate of assignment or receipt."
)
OVERLAP_CAVEAT: Final[str] = (
    "Two accepted overlap definitions are reported side by side and neither is replaced: the "
    "Phase 1 campaign-level definition (share of a campaign's listed recipients also listed for "
    "a different campaign whose nominal day interval overlaps inclusively) and the Phase 2 "
    "household-week definition (overlapping_listed_campaign_count). Overlap is listed-campaign "
    "overlap, never realized exposure."
)
SUPPORT_CAVEAT: Final[str] = (
    "Pre/campaign/post figures use declared reconnaissance windows (up to 4 observed weeks "
    "before, the clipped campaign span, up to 4 observed weeks after), truncated by the observed "
    "window. They are NOT the Phase 6 event window, anticipation window or final pre/post design."
)
TRAJECTORY_CAVEAT: Final[str] = (
    "Pre-campaign descriptive trajectories only: calendar weeks before the campaign start, "
    "listed versus not-listed group means. This is NOT a parallel-trends test, NOT a pre-trend "
    "test, NOT an event study, and no post-campaign quantity is estimated."
)


class QuestionDError(RuntimeError):
    """Raised when a Question D object does not reconcile to the accepted campaign facts."""


# --- loading ------------------------------------------------------------------


def register_question_d_views(con) -> dict[str, Path]:
    """Register the accepted views Question D reads, plus the accepted raw source interface.

    The raw campaign/coupon tables are consulted through ``raw_tables.register_views`` (the
    accepted by-name, strictly-typed source interface) because campaign description and coupon
    redemption facts are not carried in the Phase 2 mart.
    """
    registered: dict[str, Path] = {
        name: raw_file.path for name, raw_file in raw_tables.register_views(con).items()
    }
    registered.update(reconcile.register_accepted_views(con))
    return registered


def _frame(con, filename: str) -> pd.DataFrame:
    sql = (contracts.DESCRIPTIVE_SQL_DIR / filename).read_text(encoding="utf-8")
    return con.execute(sql).fetchdf()


def load_campaign_level(con) -> pd.DataFrame:
    return _frame(con, "15_campaign_level.sql")


def load_campaign_calendar(con) -> pd.DataFrame:
    return _frame(con, "16_campaign_calendar.sql")


def load_campaign_support_windows(con) -> pd.DataFrame:
    return _frame(con, "17_campaign_support_windows.sql")


def load_campaign_selection_diagnostics(con) -> pd.DataFrame:
    return _frame(con, "18_campaign_selection_diagnostics.sql")


def load_campaign_trajectory(con) -> pd.DataFrame:
    return _frame(con, "19_campaign_trajectory.sql")


# --- reconciliation -----------------------------------------------------------


def check_reconciliation(campaign_level: pd.DataFrame) -> list[str]:
    """Verify the campaign object against the accepted Phase 1/2 facts."""
    checks: list[str] = []

    def record(label: str, observed, expected) -> None:
        ok = observed == expected
        checks.append(
            f"{'ok  ' if ok else 'FAIL'} {label}: observed={observed} accepted={expected}"
        )
        if not ok:
            raise QuestionDError(
                f"reconciliation failed: {label} observed={observed} expected={expected}"
            )

    record("campaigns", len(campaign_level), ACCEPTED_CAMPAIGNS)
    record("listed pairs", int(campaign_level["listed_households"].sum()), ACCEPTED_LISTED_PAIRS)
    record(
        "listed household-weeks",
        int(campaign_level["listed_household_weeks_total"].sum()),
        ACCEPTED_LISTED_HOUSEHOLD_WEEKS,
    )
    record(
        "recorded redemption rows",
        int(campaign_level["recorded_redemption_rows"].sum()),
        ACCEPTED_REDEMPTION_ROWS,
    )
    median_overlap = round(
        float(campaign_level["accepted_phase1_recipient_overlap_pct"].median()), 2
    )
    record(
        "Phase 1 recipient-overlap median (%)", median_overlap, ACCEPTED_PHASE1_OVERLAP_MEDIAN_PCT
    )
    record(
        "campaigns extending beyond transaction support",
        int(campaign_level["campaign_extends_beyond_transaction_support"].sum()),
        1,
    )
    return checks


# --- descriptive building blocks ---------------------------------------------


def calendar_summary(calendar: pd.DataFrame) -> dict[str, float]:
    """Calendar density and concurrency summary (descriptive only)."""
    spans = calendar["campaign_spans_touching_week"]
    return {
        "weeks_with_no_campaign_span": int((spans == 0).sum()),
        "weeks_with_at_least_one_campaign_span": int((spans > 0).sum()),
        "max_concurrent_campaign_spans": int(spans.max()),
        "median_concurrent_campaign_spans": float(spans.median()),
        "mean_concurrent_campaign_spans": round(float(spans.mean()), 3),
        "weeks_at_max_concurrency": int((spans == spans.max()).sum()),
        "median_campaign_span_weeks": float(
            calendar.loc[
                calendar["campaign_spans_touching_week"] > 0, "campaign_spans_touching_week"
            ].median()
        ),
    }


def max_concurrency_weeks(calendar: pd.DataFrame) -> list[int]:
    peak = calendar["campaign_spans_touching_week"].max()
    return sorted(
        int(week)
        for week in calendar.loc[calendar["campaign_spans_touching_week"] == peak, "week_no"]
    )


def listing_breadth(campaign_level: pd.DataFrame) -> dict[str, float]:
    """Listing breadth across campaigns."""
    listed = campaign_level["listed_households"]
    return {
        "campaigns": int(len(campaign_level)),
        "min_listed_households": int(listed.min()),
        "median_listed_households": float(listed.median()),
        "max_listed_households": int(listed.max()),
        "campaigns_with_at_least_100_listed": int((listed >= 100).sum()),
        "campaigns_with_fewer_than_50_listed": int((listed < 50).sum()),
        "mean_listed_share_of_2500_pct": round(
            float(campaign_level["listed_share_of_2500_pct"].mean()), 2
        ),
    }


def overlap_summary(campaign_level: pd.DataFrame) -> pd.DataFrame:
    """Both accepted overlap definitions per campaign, plus the concurrency count."""
    return campaign_level[
        [
            "campaign",
            "campaign_type",
            "listed_households",
            "accepted_phase1_recipient_overlap_pct",
            "listed_household_overlap_pct",
            "listed_household_week_overlap_pct",
            "max_overlapping_listed_campaigns",
            "overlapping_campaign_spans",
            "concurrent_campaign_spans_other",
        ]
    ].copy()


def redemption_summary(campaign_level: pd.DataFrame) -> dict[str, float]:
    """Recorded redemption facts: counts and shares of listed households, never rates."""
    rows = campaign_level["recorded_redemption_rows"]
    return {
        "campaigns_with_any_recorded_redemption": int((rows > 0).sum()),
        "campaigns_with_no_recorded_redemption": int((rows == 0).sum()),
        "campaigns_with_a_single_recorded_redemption_row": int((rows == 1).sum()),
        "total_recorded_redemption_rows": int(rows.sum()),
        "total_distinct_redeeming_households": int(
            campaign_level["distinct_redeeming_households"].sum()
        ),
        "campaign_with_most_redemption_rows": int(
            campaign_level.loc[rows.idxmax(), "campaign"] if rows.max() > 0 else 0
        ),
        "max_redemption_rows_in_a_campaign": int(rows.max()),
        "share_of_all_redemption_rows_in_top_3_campaigns_pct": round(
            100.0 * float(rows.nlargest(3).sum()) / float(rows.sum()), 2
        )
        if rows.sum() > 0
        else 0.0,
        "redeemers_not_listed_for_that_campaign": int(
            (
                campaign_level["distinct_redeeming_households"]
                - campaign_level["distinct_redeeming_households_also_listed"]
            ).sum()
        ),
    }


def redemption_concentration(campaign_level: pd.DataFrame) -> pd.DataFrame:
    frame = campaign_level[
        [
            "campaign",
            "campaign_type",
            "listed_households",
            "recorded_redemption_rows",
            "distinct_redeeming_households",
            "distinct_redeeming_households_also_listed",
            "recorded_redemption_household_share_of_listed_pct",
        ]
    ].copy()
    return frame.sort_values("recorded_redemption_rows", ascending=False).reset_index(drop=True)


def support_summary(support: pd.DataFrame) -> pd.DataFrame:
    """Reconnaissance-window support per campaign, both groups side by side."""
    return support[
        [
            "campaign",
            "campaign_type",
            "observed_weeks_in_pre_window",
            "observed_weeks_in_campaign_span",
            "observed_weeks_in_post_window",
            "listed_households",
            "not_listed_households",
            "listed_active_share_pre_pct",
            "not_listed_active_share_pre_pct",
            "listed_active_share_campaign_pct",
            "not_listed_active_share_campaign_pct",
            "listed_active_share_post_pct",
            "not_listed_active_share_post_pct",
        ]
    ].copy()


def support_summary_facts(support: pd.DataFrame) -> dict[str, object]:
    truncated = support[support["observed_weeks_in_post_window"] == 0]
    pre = support["listed_active_share_pre_pct"]
    not_pre = support["not_listed_active_share_pre_pct"]
    return {
        "campaigns_with_truncated_post_window": len(truncated),
        "truncated_campaigns": sorted(int(value) for value in truncated["campaign"]),
        "campaigns_with_full_4_week_pre_and_post": int(
            (
                (support["observed_weeks_in_pre_window"] >= RECONNAISSANCE_WEEKS_BEFORE)
                & (support["observed_weeks_in_post_window"] >= RECONNAISSANCE_WEEKS_AFTER)
            ).sum()
        ),
        "median_listed_active_share_pre_pct": round(float(pre.median()), 4),
        "median_not_listed_active_share_pre_pct": round(float(not_pre.median()), 4),
        "campaigns_where_listed_pre_share_exceeds_not_listed": int((pre > not_pre).sum()),
        "min_level_gap_pp": round(float((pre - not_pre).min()), 4),
        "max_level_gap_pp": round(float((pre - not_pre).max()), 4),
    }


def selection_diagnostics(selection: pd.DataFrame) -> pd.DataFrame:
    """Listed versus not-listed strictly pre-campaign characteristics, with a descriptive SMD.

    The standardized mean difference is reported as a DESCRIPTIVE magnitude only. Nothing is
    adjusted, matched or weighted, and this is not an estimate of any campaign effect.
    """
    listed = selection[selection["listed"] == 1].set_index("campaign")
    other = selection[selection["listed"] == 0].set_index("campaign")
    metrics = (
        ("prior_active_weeks", "mean_prior_active_weeks"),
        ("prior_retailer_sales_value", "mean_prior_retailer_sales_value"),
        ("prior_trips", "mean_prior_trips"),
        ("prior_distinct_commodities", "mean_prior_distinct_commodities"),
        ("prior_active_share", "mean_prior_active_share"),
    )
    frame = pd.DataFrame(
        {
            "campaign": listed.index,
            "campaign_type": listed["campaign_type"].to_numpy(),
            "listed_households": listed["households"].to_numpy(),
        }
    )
    for label, column in metrics:
        frame[f"listed_{label}"] = listed[column].to_numpy()
        frame[f"not_listed_{label}"] = other[column].to_numpy()
        frame[f"ratio_{label}"] = listed[column].to_numpy() / np.where(
            other[column].to_numpy() == 0, np.nan, other[column].to_numpy()
        )
    frame["max_ratio_prior_retailer_sales_value"] = frame["ratio_prior_retailer_sales_value"]
    # Descriptive standardized difference on the four principal pre-campaign measures.
    for label, _ in metrics[:4]:
        pooled = (frame[f"listed_{label}"] + frame[f"not_listed_{label}"]) / 2.0
        frame[f"descriptive_gap_{label}_pct_of_pooled"] = (
            100.0
            * (frame[f"listed_{label}"] - frame[f"not_listed_{label}"])
            / pooled.replace(0.0, np.nan)
        )
    return frame


def selection_summary_facts(selection_diagnostics_frame: pd.DataFrame) -> dict[str, float]:
    frame = selection_diagnostics_frame
    value_gap = frame["descriptive_gap_prior_retailer_sales_value_pct_of_pooled"]
    weeks_gap = frame["descriptive_gap_prior_active_weeks_pct_of_pooled"]
    return {
        "campaigns_where_listed_prior_value_exceeds_not_listed": int((value_gap > 0).sum()),
        "min_listed_prior_value_gap_pct": round(float(value_gap.min()), 2),
        "median_listed_prior_value_gap_pct": round(float(value_gap.median()), 2),
        "max_listed_prior_value_gap_pct": round(float(value_gap.max()), 2),
        "min_listed_prior_active_weeks_gap_pct": round(float(weeks_gap.min()), 2),
        "median_listed_prior_active_weeks_gap_pct": round(float(weeks_gap.median()), 2),
        "campaigns_with_prior_value_ratio_above_2": int(
            (frame["ratio_prior_retailer_sales_value"] > 2.0).sum()
        ),
        "min_prior_value_ratio": round(float(frame["ratio_prior_retailer_sales_value"].min()), 3),
        "max_prior_value_ratio": round(float(frame["ratio_prior_retailer_sales_value"].max()), 3),
    }


def trajectory_diagnostics(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Pre-campaign descriptive trajectory comparison for campaigns with enough listed households."""
    rows: list[dict[str, object]] = []
    for campaign, group in trajectory.groupby("campaign", sort=True):
        listed = group[group["listed"] == 1].sort_values("week_no")
        other = group[group["listed"] == 0].sort_values("week_no")
        if listed.empty or other.empty:
            continue
        listed_households = int(listed["group_households"].max())
        if listed_households < TRAJECTORY_MIN_LISTED_HOUSEHOLDS:
            continue
        merged = listed.merge(other, on="week_no", suffixes=("_listed", "_not_listed"))
        if len(merged) < 4:
            continue
        listed_series = merged["mean_household_retailer_sales_value_listed"]
        other_series = merged["mean_household_retailer_sales_value_not_listed"]
        correlation = (
            float(np.corrcoef(listed_series, other_series)[0, 1]) if len(merged) > 2 else np.nan
        )
        rows.append(
            {
                "campaign": int(campaign),
                "campaign_type": str(listed["campaign_type"].iloc[0]),
                "listed_households": listed_households,
                "pre_campaign_weeks_compared": int(len(merged)),
                "mean_weekly_sales_value_listed": round(float(listed_series.mean()), 4),
                "mean_weekly_sales_value_not_listed": round(float(other_series.mean()), 4),
                "level_ratio_listed_over_not_listed": round(
                    float(listed_series.mean()) / float(other_series.mean()), 4
                )
                if float(other_series.mean()) > 0
                else np.nan,
                "correlation_of_weekly_levels": round(correlation, 4)
                if np.isfinite(correlation)
                else np.nan,
                "listed_series_coefficient_of_variation": round(
                    float(listed_series.std() / listed_series.mean()), 4
                )
                if float(listed_series.mean()) > 0
                else np.nan,
                "not_listed_series_coefficient_of_variation": round(
                    float(other_series.std() / other_series.mean()), 4
                )
                if float(other_series.mean()) > 0
                else np.nan,
                "reported_level_gap_pp": round(
                    float(listed_series.mean() - other_series.mean()), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def trajectory_summary_facts(trajectory_frame: pd.DataFrame) -> dict[str, float]:
    if trajectory_frame.empty:
        return {"campaigns_with_trajectory_diagnostic": 0}
    ratios = trajectory_frame["level_ratio_listed_over_not_listed"]
    correlations = trajectory_frame["correlation_of_weekly_levels"]
    return {
        "campaigns_with_trajectory_diagnostic": int(len(trajectory_frame)),
        "campaigns_excluded_for_thin_listed_group": ACCEPTED_CAMPAIGNS - int(len(trajectory_frame)),
        "min_level_ratio": round(float(ratios.min()), 4),
        "median_level_ratio": round(float(ratios.median()), 4),
        "max_level_ratio": round(float(ratios.max()), 4),
        "min_correlation_of_levels": round(float(correlations.min()), 4),
        "median_correlation_of_levels": round(float(correlations.median()), 4),
    }


# --- Phase 6 reconnaissance matrix -------------------------------------------


def reconnaissance_matrix(
    campaign_level: pd.DataFrame,
    support: pd.DataFrame,
    selection_diag: pd.DataFrame,
    trajectory_frame: pd.DataFrame,
) -> pd.DataFrame:
    """One row per campaign with the eight dimensions as SEPARATE columns.

    No composite score, no weighting, no overall rank and no "most causal" ordering is produced.
    """
    frame = campaign_level.merge(
        support[
            [
                "campaign",
                "observed_weeks_in_pre_window",
                "observed_weeks_in_post_window",
                "listed_active_share_pre_pct",
                "not_listed_active_share_pre_pct",
            ]
        ],
        on="campaign",
        how="left",
        validate="one_to_one",
    ).merge(
        selection_diag[
            [
                "campaign",
                "listed_prior_active_weeks",
                "not_listed_prior_active_weeks",
                "listed_prior_retailer_sales_value",
                "not_listed_prior_retailer_sales_value",
                "ratio_prior_retailer_sales_value",
                "descriptive_gap_prior_retailer_sales_value_pct_of_pooled",
                "descriptive_gap_prior_active_weeks_pct_of_pooled",
            ]
        ],
        on="campaign",
        how="left",
        validate="one_to_one",
    )
    trajectory_columns = (
        trajectory_frame[
            [
                "campaign",
                "pre_campaign_weeks_compared",
                "level_ratio_listed_over_not_listed",
                "correlation_of_weekly_levels",
            ]
        ]
        if not trajectory_frame.empty
        else pd.DataFrame(columns=["campaign"])
    )
    frame = frame.merge(trajectory_columns, on="campaign", how="left", validate="one_to_one")

    frame["dimension_1_timing_support"] = np.where(
        (frame["observed_weeks_in_pre_window"] >= SHORTLIST_MIN_OBSERVED_WEEKS_BEFORE)
        & (frame["observed_weeks_in_post_window"] >= SHORTLIST_MIN_OBSERVED_WEEKS_AFTER),
        f"at least {SHORTLIST_MIN_OBSERVED_WEEKS_BEFORE} observed weeks before and "
        f"{SHORTLIST_MIN_OBSERVED_WEEKS_AFTER} after",
        "truncated reconnaissance windows "
        f"(before={frame['observed_weeks_in_pre_window'].astype(int)}, "
        f"after={frame['observed_weeks_in_post_window'].astype(int)} weeks)",
    )
    frame["dimension_2_listing_breadth"] = (
        frame["listed_households"].astype(int).astype(str)
        + " listed households ("
        + frame["listed_share_of_2500_pct"].round(2).astype(str)
        + "% of 2,500)"
    )
    frame["dimension_3_overlap"] = (
        "Phase 1 recipient overlap "
        + frame["accepted_phase1_recipient_overlap_pct"].round(2).astype(str)
        + "%; Phase 2 household-week overlap "
        + frame["listed_household_week_overlap_pct"].round(2).astype(str)
        + "%; "
        + frame["overlapping_campaign_spans"].astype(int).astype(str)
        + " overlapping campaign spans"
    )
    frame["dimension_4_pre_campaign_observable_selection"] = (
        "listed minus not-listed prior retailer SALES_VALUE "
        + frame["descriptive_gap_prior_retailer_sales_value_pct_of_pooled"].round(1).astype(str)
        + "% of pooled mean (ratio "
        + frame["ratio_prior_retailer_sales_value"].round(2).astype(str)
        + "x); prior active weeks "
        + frame["descriptive_gap_prior_active_weeks_pct_of_pooled"].round(1).astype(str)
        + "% of pooled mean"
    )
    frame["dimension_5_pre_campaign_trajectory_comparability"] = np.where(
        frame["correlation_of_weekly_levels"].isna(),
        f"not reported: fewer than {TRAJECTORY_MIN_LISTED_HOUSEHOLDS} listed households or too few "
        "pre-campaign weeks",
        "pre-campaign descriptive trajectories over "
        + frame["pre_campaign_weeks_compared"].astype("Int64").astype(str)
        + " weeks; level ratio "
        + frame["level_ratio_listed_over_not_listed"].round(2).astype(str)
        + "x; correlation of weekly levels "
        + frame["correlation_of_weekly_levels"].round(2).astype(str),
    )
    frame["dimension_6_redemption_information"] = (
        frame["recorded_redemption_rows"].astype(int).astype(str)
        + " recorded redemption rows, "
        + frame["distinct_redeeming_households"].astype(int).astype(str)
        + " distinct redeeming households, "
        + frame["distinct_redeeming_households_also_listed"].astype(int).astype(str)
        + " of them listed for this campaign (counts of recorded redemption, not a rate of "
        "assignment or receipt)"
    )
    frame["dimension_7_edge_of_sample_risk"] = np.where(
        (frame["observed_weeks_in_post_window"] == 0)
        | (frame["campaign_extends_beyond_transaction_support"] == 1),
        "high: the campaign span runs into the end of observed data "
        f"(post weeks={frame['observed_weeks_in_post_window'].astype(int)}, "
        f"extends beyond support={frame['campaign_extends_beyond_transaction_support'].astype(int)})",
        f"lower: {frame['observed_weeks_in_post_window'].astype(int)} observed weeks after the span",
    )
    frame["dimension_8_assignment_caveat"] = (
        frame["campaign_type"].astype(str) + ": " + ASSIGNMENT_CAVEAT
    )
    return frame


def shortlist_screens(matrix: pd.DataFrame) -> pd.DataFrame:
    """Apply the DECLARED screens. A pass means "worth later identification review", nothing more."""
    frame = matrix.copy()
    frame["passes_timing_screen"] = (
        frame["observed_weeks_in_pre_window"] >= SHORTLIST_MIN_OBSERVED_WEEKS_BEFORE
    ) & (frame["observed_weeks_in_post_window"] >= SHORTLIST_MIN_OBSERVED_WEEKS_AFTER)
    frame["passes_breadth_screen"] = frame["listed_households"] >= SHORTLIST_MIN_LISTED_HOUSEHOLDS
    frame["passes_overlap_screen"] = (
        frame["accepted_phase1_recipient_overlap_pct"] <= SHORTLIST_MAX_ACCEPTED_OVERLAP_PCT
    )
    frame["passes_strict_overlap_screen"] = (
        frame["accepted_phase1_recipient_overlap_pct"] <= SHORTLIST_STRICT_MAX_ACCEPTED_OVERLAP_PCT
    )
    frame["passes_edge_screen"] = frame["campaign_extends_beyond_transaction_support"] == 0
    frame["passes_all_declared_screens"] = (
        frame["passes_timing_screen"]
        & frame["passes_breadth_screen"]
        & frame["passes_overlap_screen"]
        & frame["passes_edge_screen"]
    )
    frame["passes_strict_overlap_variant"] = (
        frame["passes_all_declared_screens"] & frame["passes_strict_overlap_screen"]
    )
    return frame


def shortlist_members(screened: pd.DataFrame, strict: bool = False) -> list[int]:
    column = "passes_strict_overlap_variant" if strict else "passes_all_declared_screens"
    return sorted(int(value) for value in screened.loc[screened[column], "campaign"])


def shortlist_report(screened: pd.DataFrame, candidate_ids: list[int]) -> pd.DataFrame:
    frame = screened[screened["campaign"].isin(candidate_ids)].copy()
    rows: list[dict[str, object]] = []
    for record in frame.itertuples(index=False):
        strengths = []
        weaknesses = []
        if record.passes_timing_screen:
            strengths.append(
                f"{int(record.observed_weeks_in_pre_window)} observed weeks before and "
                f"{int(record.observed_weeks_in_post_window)} after the span"
            )
        if record.passes_breadth_screen:
            strengths.append(f"{int(record.listed_households)} listed households")
        if record.recorded_redemption_rows > 0:
            strengths.append(
                f"{int(record.recorded_redemption_rows)} recorded redemption rows from "
                f"{int(record.distinct_redeeming_households)} distinct redeeming households"
            )
        weaknesses.append(
            f"accepted Phase 1 recipient overlap {record.accepted_phase1_recipient_overlap_pct:.2f}% "
            f"across {int(record.overlapping_campaign_spans)} overlapping campaign spans"
        )
        weaknesses.append(
            "listed households differ observably before the campaign (prior retailer SALES_VALUE "
            f"{record.ratio_prior_retailer_sales_value:.2f}x the not-listed mean)"
        )
        if not record.passes_overlap_screen:
            weaknesses.append(
                f"overlap exceeds the declared {SHORTLIST_MAX_ACCEPTED_OVERLAP_PCT:.0f}% screen"
            )
        rows.append(
            {
                "campaign": int(record.campaign),
                "campaign_type": record.campaign_type,
                "start_week": int(record.start_week),
                "end_week": int(record.end_week),
                "listed_households": int(record.listed_households),
                "accepted_phase1_recipient_overlap_pct": record.accepted_phase1_recipient_overlap_pct,
                "overlapping_campaign_spans": int(record.overlapping_campaign_spans),
                "observed_weeks_before": int(record.observed_weeks_in_pre_window),
                "observed_weeks_after": int(record.observed_weeks_in_post_window),
                "recorded_redemption_rows": int(record.recorded_redemption_rows),
                "distinct_redeeming_households": int(record.distinct_redeeming_households),
                "prior_value_ratio_listed_over_not_listed": record.ratio_prior_retailer_sales_value,
                "pre_campaign_trajectory_level_ratio": record.level_ratio_listed_over_not_listed,
                "strengths": "; ".join(strengths),
                "weaknesses": "; ".join(weaknesses),
                "assignment_caveat": record.dimension_8_assignment_caveat,
            }
        )
    return pd.DataFrame(rows).sort_values("campaign").reset_index(drop=True)


# --- output construction ------------------------------------------------------


@dataclass
class QuestionDResult:
    """Everything the exploratory Question D pass produces, before any promotion decision."""

    campaign_level: pd.DataFrame
    calendar: pd.DataFrame
    support: pd.DataFrame
    selection_diagnostics: pd.DataFrame
    trajectory: pd.DataFrame
    matrix: pd.DataFrame
    screened: pd.DataFrame
    shortlist: pd.DataFrame
    summary: pd.DataFrame
    context: dict[str, object] = field(default_factory=dict)


def _summary_table(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_question_d(con) -> QuestionDResult:
    register_question_d_views(con)
    campaign_level = load_campaign_level(con)
    calendar = load_campaign_calendar(con)
    support = load_campaign_support_windows(con)
    selection = load_campaign_selection_diagnostics(con)
    trajectory = load_campaign_trajectory(con)

    reconciliation = check_reconciliation(campaign_level)
    diagnostics = selection_diagnostics(selection)
    trajectory_frame = trajectory_diagnostics(trajectory)
    matrix = reconnaissance_matrix(campaign_level, support, diagnostics, trajectory_frame)
    screened = shortlist_screens(matrix)
    candidates = shortlist_members(screened)
    strict_candidates = shortlist_members(screened, strict=True)
    shortlist = shortlist_report(screened, candidates)

    calendar_facts = calendar_summary(calendar)
    breadth_facts = listing_breadth(campaign_level)
    redemption_facts = redemption_summary(campaign_level)
    support_facts = support_summary_facts(support)
    selection_facts = selection_summary_facts(diagnostics)
    trajectory_facts = trajectory_summary_facts(trajectory_frame)

    household_burden = (
        con.execute(
            """
        WITH per_household AS (
            SELECT household_key, count(DISTINCT CAMPAIGN) AS campaigns_listed
            FROM campaign_table
            GROUP BY household_key
        )
        SELECT count(*) AS listed_households,
               min(campaigns_listed) AS min_campaigns,
               median(campaigns_listed) AS median_campaigns,
               max(campaigns_listed) AS max_campaigns,
               sum(campaigns_listed) AS listed_pairs
        FROM per_household
        """
        )
        .fetchdf()
        .iloc[0]
    )

    rows: list[tuple[str, str]] = [("campaigns", str(len(campaign_level)))]
    rows.append(
        (
            "campaign types",
            ", ".join(
                f"{k}={v}"
                for k, v in campaign_level["campaign_type"].value_counts().sort_index().items()
            ),
        )
    )
    rows.append(("listed pairs", f"{int(campaign_level['listed_households'].sum()):,}"))
    rows.append(
        (
            "listed households (distinct)",
            f"{int(household_burden['listed_households']):,} of {HOUSEHOLD_POPULATION:,}",
        )
    )
    rows.append(
        (
            "campaigns listed per household",
            (
                f"min {int(household_burden['min_campaigns'])}, median {household_burden['median_campaigns']:.0f}, "
                f"max {int(household_burden['max_campaigns'])}"
            ),
        )
    )
    for key, value in calendar_facts.items():
        rows.append((key, str(value)))
    rows.append(
        ("weeks at maximum concurrency", ", ".join(str(w) for w in max_concurrency_weeks(calendar)))
    )
    for key, value in breadth_facts.items():
        rows.append((key, str(value)))
    for key, value in redemption_facts.items():
        rows.append((key, str(value)))
    for key, value in support_facts.items():
        rows.append((key, str(value)))
    for key, value in selection_facts.items():
        rows.append((key, str(value)))
    for key, value in trajectory_facts.items():
        rows.append((key, str(value)))
    rows.append(("declared shortlist passers", ", ".join(str(c) for c in candidates) or "none"))
    rows.append(
        ("strict-overlap shortlist passers", ", ".join(str(c) for c in strict_candidates) or "none")
    )

    return QuestionDResult(
        campaign_level=campaign_level,
        calendar=calendar,
        support=support,
        selection_diagnostics=diagnostics,
        trajectory=trajectory_frame,
        matrix=matrix,
        screened=screened,
        shortlist=shortlist,
        summary=_summary_table(rows),
        context={
            "reconciliation": reconciliation,
            "campaign_level": campaign_level,
            "calendar": calendar,
            "support": support,
            "selection": diagnostics,
            "trajectory": trajectory_frame,
            "trajectory_raw": trajectory,
            "matrix": matrix,
            "screened": screened,
            "shortlist": shortlist,
            "shortlist_candidates": candidates,
            "shortlist_candidates_strict": strict_candidates,
            "summary": _summary_table(rows),
        },
    )


def format_report(result: QuestionDResult) -> str:
    lines = ["Question D campaign structure and Phase 6 reconnaissance (exploratory)", ""]
    lines.append("Reconciliation against accepted campaign facts:")
    reconciliation = result.context["reconciliation"]
    assert isinstance(reconciliation, list)
    lines.extend(f"  {line}" for line in reconciliation)
    lines.append("")
    matrix = result.matrix
    lines.append(f"Campaign snapshot ({len(matrix)} campaigns, all retained):")
    preview = matrix[
        [
            "campaign",
            "campaign_type",
            "start_week",
            "end_week",
            "listed_households",
            "accepted_phase1_recipient_overlap_pct",
            "recorded_redemption_rows",
            "ratio_prior_retailer_sales_value",
        ]
    ]
    lines.append(preview.to_string(index=False))
    lines.append("")
    lines.append("Summary (descriptive only; nothing promoted, no campaign selected):")
    lines.extend(f"  {row.metric}: {row.value}" for row in result.summary.itertuples(index=False))
    lines.append("")
    lines.append(f"Shortlist for later identification review ({len(result.shortlist)} campaigns):")
    if result.shortlist.empty:
        lines.append("  none pass the declared screens")
    else:
        lines.append(
            result.shortlist[
                [
                    "campaign",
                    "campaign_type",
                    "listed_households",
                    "accepted_phase1_recipient_overlap_pct",
                    "observed_weeks_before",
                    "observed_weeks_after",
                    "recorded_redemption_rows",
                ]
            ].to_string(index=False)
        )
    lines.append("")
    lines.append(ASSIGNMENT_CAVEAT)
    lines.append(REDEMPTION_CAVEAT)
    lines.append(OVERLAP_CAVEAT)
    lines.append(SUPPORT_CAVEAT)
    lines.append(TRAJECTORY_CAVEAT)
    return "\n".join(lines)


# --- exploratory outputs (candidate F8, trajectory diagnostic, T3 precursor) ---

CAMPAIGN_TYPE_COLOURS: Final[dict[str, str]] = {
    "TypeA": "#1f4e79",
    "TypeB": "#b26500",
    "TypeC": "#2e7d32",
}

F8_CANDIDATE_CAPTION: Final[str] = (
    "Panel A: every campaign's calendar span across the 102 observed weeks (one row per campaign, "
    "ordered by start week; colour encodes campaign type; week membership uses the accepted Phase 1 "
    "mapping with campaign start/end days preserved in the table). Panel B: listed households and "
    "the accepted Phase 1 recipient-overlap share per campaign. This is a CAMPAIGN-STRUCTURE "
    "figure. It deliberately shows no outcome variable and no campaign-relative time axis, so it "
    "cannot be read as a campaign effect; recorded retailer SALES_VALUE is not overlaid. Overlap is "
    "listed-campaign overlap among a campaign's own listed recipients, never realized exposure."
)

TRAJECTORY_FIGURE_CAPTION: Final[str] = (
    "Pre-campaign descriptive trajectories only, on CALENDAR time: mean weekly retailer "
    "SALES_VALUE per household for households listed for the campaign versus households not listed "
    "for it, over the observed weeks strictly before the campaign start. No campaign-week outcome "
    "appears in these panels. This is NOT a parallel-trends test, NOT a pre-trend test and NOT an "
    "event study; the not-listed group is not a comparison group. Only campaigns with at least "
    f"{TRAJECTORY_MIN_LISTED_HOUSEHOLDS} listed households are drawn."
)


def figure_campaign_landscape(campaign_level: pd.DataFrame) -> object:
    """Candidate F8: campaign calendar, listing breadth and overlap (structure only)."""
    import matplotlib.pyplot as plt

    ordered = campaign_level.sort_values(["start_week", "campaign"]).reset_index(drop=True)
    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 9.0), gridspec_kw={"width_ratios": [2.6, 1.0, 1.0]}
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.24, wspace=0.75)

    axis = axes[0]
    for position, record in enumerate(ordered.itertuples(index=False)):
        colour = CAMPAIGN_TYPE_COLOURS.get(str(record.campaign_type), "#555555")
        axis.barh(
            position,
            record.end_week - record.start_week + 1,
            left=record.start_week,
            height=0.72,
            color=colour,
            alpha=0.9,
        )
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels(
        [
            f"{int(record.campaign)}  ({record.campaign_type[-1]})"
            for record in ordered.itertuples(index=False)
        ],
        fontsize=6.6,
    )
    axis.invert_yaxis()
    axis.set_xlim(0.5, 102.5)
    axis.set_xlabel("observed week (accepted Phase 1 week mapping)", fontsize=8)
    axis.set_title("A. Campaign calendar spans (all 30 campaigns)", fontsize=9, loc="left")
    axis.grid(alpha=0.25, lw=0.5, axis="x")
    handles = [
        plt.Line2D([0], [0], color=colour, lw=6, label=campaign_type)
        for campaign_type, colour in sorted(CAMPAIGN_TYPE_COLOURS.items())
    ]
    axis.legend(handles=handles, fontsize=6.5, loc="lower right", framealpha=0.9)

    listing_axis = axes[1]
    listing_axis.barh(range(len(ordered)), ordered["listed_households"], color="#1f4e79")
    listing_axis.set_yticks(range(len(ordered)))
    listing_axis.set_yticklabels([])
    listing_axis.invert_yaxis()
    listing_axis.set_xlabel("listed households", fontsize=8)
    listing_axis.set_title("B. Listing breadth", fontsize=9, loc="left")
    listing_axis.grid(alpha=0.25, lw=0.5, axis="x")

    overlap_axis = axes[2]
    overlap_axis.barh(
        range(len(ordered)),
        ordered["accepted_phase1_recipient_overlap_pct"],
        color="#b26500",
    )
    overlap_axis.set_yticks(range(len(ordered)))
    overlap_axis.set_yticklabels([])
    overlap_axis.invert_yaxis()
    overlap_axis.set_xlim(0, 100)
    overlap_axis.set_xlabel("accepted Phase 1 recipient overlap (%)", fontsize=7.6)
    overlap_axis.set_title("C. Listed-recipient overlap", fontsize=9, loc="left")
    overlap_axis.grid(alpha=0.25, lw=0.5, axis="x")

    fig.suptitle(
        "CANDIDATE F8 - campaign landscape (structure only; no outcome variable, no campaign effect)",
        fontsize=11,
    )
    fig.text(
        0.07,
        0.005,
        F8_CANDIDATE_CAPTION,
        fontsize=6.2,
        va="bottom",
        wrap=True,
    )
    return fig


def figure_pre_campaign_trajectories(trajectory: pd.DataFrame) -> object:
    """Exploratory pre-campaign descriptive trajectory diagnostic (calendar time only)."""
    import matplotlib.pyplot as plt

    diagnostic = trajectory_diagnostics(trajectory)
    if diagnostic.empty:
        raise QuestionDError("no campaign supports the trajectory diagnostic")
    selected = diagnostic.nlargest(6, "listed_households")["campaign"].tolist()
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.3)
    colours = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for index, campaign in enumerate(selected):
        group = trajectory[trajectory["campaign"] == campaign]
        listed = group[group["listed"] == 1].sort_values("week_no")
        other = group[group["listed"] == 0].sort_values("week_no")
        colour = colours[index % len(colours)]  # type: ignore[index]
        ax.plot(
            listed["week_no"],
            listed["mean_household_retailer_sales_value"],
            color=colour,
            lw=1.1,
            label=f"campaign {campaign} listed",
        )
        ax.plot(
            other["week_no"],
            other["mean_household_retailer_sales_value"],
            color=colour,
            lw=0.9,
            ls="--",
            alpha=0.75,
            label=f"campaign {campaign} not listed",
        )
    ax.set_xlabel(
        "observed week (calendar time, weeks strictly before each campaign's start)", fontsize=8
    )
    ax.set_ylabel("mean weekly retailer SALES_VALUE per household", fontsize=8)
    ax.set_title(
        "Exploratory: pre-campaign descriptive trajectories (listed solid, not listed dashed)",
        fontsize=9.5,
        loc="left",
    )
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=5.6, ncols=3, loc="upper left")
    fig.text(0.09, 0.005, TRAJECTORY_FIGURE_CAPTION, fontsize=6.2, va="bottom", wrap=True)
    return fig


#: Caveat columns the published T3 carries explicitly, in addition to the dimensions that state
#: their own caveat inline. Research handoff text, never a claim about a campaign effect.
T3_CAVEAT_COLUMNS: Final[tuple[str, ...]] = (
    "assignment_caveat",
    "redemption_caveat",
    "overlap_caveat",
    "support_caveat",
    "trajectory_caveat",
    "edge_of_sample_caveat",
)


def build_t3(matrix: pd.DataFrame) -> pd.DataFrame:
    """Assemble the published T3 campaign reconnaissance table from the reconnaissance matrix.

    One row per campaign, all 30 retained, the eight dimensions separate, and the standing caveats
    carried as their own columns. No treatment indicator, no comparison-group definition, no
    estimate, no causal score, no rank and no selected or recommended campaign.
    """
    frame = matrix.copy()
    frame["assignment_caveat"] = ASSIGNMENT_CAVEAT
    frame["redemption_caveat"] = REDEMPTION_CAVEAT
    frame["overlap_caveat"] = OVERLAP_CAVEAT
    frame["support_caveat"] = SUPPORT_CAVEAT
    frame["trajectory_caveat"] = TRAJECTORY_CAVEAT
    frame["edge_of_sample_caveat"] = (
        "Reconnaissance windows are truncated by the observed weeks 1-102: campaigns 15 and 24 "
        "have no observed weeks after their span, campaign 24 extends beyond the observed "
        "transaction window, and campaigns 26-30 have fewer observed pre-campaign weeks than the "
        "late campaigns. Weeks 1 and 102 are partial observed weeks."
    )
    banned = ("score", "rank", "winner", "recommended", "selected_campaign", "effect_estimate")
    offenders = [column for column in frame.columns if any(token in column for token in banned)]
    if offenders:
        raise QuestionDError(f"T3 must not carry ranking or selection columns: {offenders}")
    columns = " ".join(frame.columns).lower()
    refused = sorted(term for term in contracts.CAMPAIGN_FORBIDDEN_VOCABULARY if term in columns)
    if refused:
        raise QuestionDError(f"T3 column names must not use the refused vocabulary: {refused}")
    return frame


def write_exploratory_outputs(result: QuestionDResult) -> dict[str, Path]:
    """Write the ignored exploratory tables (T3 precursor) and candidate figures."""
    from matplotlib import pyplot as plt

    tables: dict[str, pd.DataFrame] = {
        "campaign_level": result.campaign_level,
        "campaign_calendar": result.calendar,
        "campaign_support_windows": result.support,
        "campaign_selection_diagnostics": result.selection_diagnostics,
        "campaign_trajectory": result.trajectory,
        "campaign_reconnaissance_matrix": result.matrix,
        "campaign_screens": result.screened,
        "campaign_shortlist": result.shortlist,
        "summary": result.summary,
    }
    written: dict[str, Path] = {}
    for name, frame in tables.items():
        path = artifacts.exploratory_path("table", f"q_d_{name}.csv")
        written[path.name] = artifacts.write_table_csv(
            path,
            list(frame.columns),
            frame.itertuples(index=False, name=None),
            scope="exploratory",
        )
    builders = (
        (
            "q_d_f8_campaign_landscape.png",
            lambda: figure_campaign_landscape(result.campaign_level),
        ),
        (
            "q_d_pre_campaign_trajectories.png",
            lambda: figure_pre_campaign_trajectories(result.context["trajectory_raw"]),
        ),
    )
    for filename, builder in builders:
        fig = builder()
        try:
            path = artifacts.write_figure_png(fig, artifacts.exploratory_path("figure", filename))
            written[filename] = path
        finally:
            plt.close(fig)
    return written
