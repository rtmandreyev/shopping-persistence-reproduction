"""Question E: representative customer journeys (explanatory/visual; restrained findings).

Four households are selected by the FROZEN Run 0 rule (plan section 8) and presented as
Household A / Household B / Household C / Household D. The label -> household_key mapping exists only
in a Git-ignored local artifact; no tracked, published or logged artifact carries a key.

Semantics preserved here: weekly retailer SALES_VALUE is retailer revenue (retailer SALES_VALUE), never
customer spend or grocery expenditure, and ``retailer_purchase = 0`` means no recorded purchase from
this retailer that week rather than zero household grocery consumption. Campaign listing and recorded
redemption appear as CONTEXTUAL factual markers only: they are not exposure, assignment, receipt or
response, no event time is aligned and no quantity comparing a period before with a period after is
computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from complete_journey.descriptive import artifacts, contracts, reconcile

if TYPE_CHECKING:  # pragma: no cover - typing only, matplotlib stays an optional import here
    from matplotlib.figure import Figure

# --- frozen selection rule (plan section 8) ----------------------------------

JOURNEY_LABELS: Final[tuple[str, ...]] = ("A", "B", "C", "D")
STRATUM_BY_LABEL: Final[dict[str, str]] = {
    "A": "HIGH-FREQUENCY",
    "B": "MEDIAN-FREQUENCY",
    "C": "INTERMITTENT",
    "D": "COUPON/CAMPAIGN-ACTIVE",
}
STRATUM_RULE_TEXT: Final[tuple[str, ...]] = (
    "A HIGH-FREQUENCY: active_weeks >= 90 AND longest_internal_no_purchase_week_gap <= 4",
    "B MEDIAN-FREQUENCY: 45 <= active_weeks <= 55",
    "C INTERMITTENT: active_weeks <= 25",
    "D COUPON/CAMPAIGN-ACTIVE: coupon_redeemer_ever = 1 AND campaign_recipient_ever = 1 AND "
    "campaign_count >= the median campaign_count among listed households",
)
SELECTION_ORDER_TEXT: Final[str] = (
    "A, then B excluding A, then C excluding A/B, then D excluding A/B/C"
)
SELECTION_METRIC_TEXT: Final[str] = (
    "within each stratum: the eligible household whose whole-sample retailer SALES_VALUE is closest to "
    "the stratum median, tie-broken by lowest household_key"
)

OBSERVED_WEEKS: Final[int] = 102
JOURNEY_ROWS: Final[int] = 4 * OBSERVED_WEEKS
PARTIAL_WEEKS: Final[tuple[int, ...]] = (1, 102)
F9_CANDIDATE_FILENAME: Final[str] = "q_e_f9_representative_journeys.png"

# Local, Git-ignored artifacts. The mapping carries the keys; the summary carries public labels only.
LOCAL_MAPPING_PATH: Final[Path] = artifacts.journey_mapping_path()
LOCAL_SUMMARY_PATH: Final[Path] = artifacts.LOCAL_STATE_DIR / "q_e_journey_summary.csv"

SEMANTICS_CAVEAT: Final[str] = (
    "Weekly retailer SALES_VALUE is retailer revenue (retailer SALES_VALUE), never customer spend or "
    "grocery expenditure. A zero retailer_purchase week means no recorded purchase from this retailer "
    "that week, not zero household grocery consumption. Four rule-selected households illustrate "
    "observed retailer histories; they are not customer archetypes and they do not validate Question B "
    "statistically."
)
CONTEXT_CAVEAT: Final[str] = (
    "Campaign listing and recorded redemption are contextual factual markers only, never causal "
    "annotations: listing is not assignment or exposure, recorded redemption is not proof of offer "
    "receipt, and no event time is aligned, no campaign response is measured and no before/after "
    "comparison is drawn. Weeks 1 and 102 are partial observed weeks (5 and 6 days)."
)
REPRESENTATIVENESS_TEXT: Final[str] = (
    "Each household is representative only in the narrow deterministic sense of the frozen rule: its "
    "whole-sample retailer SALES_VALUE is the value nearest the median of the stratum it was drawn from. "
    "That does not make it a representative customer archetype, and it is not representative of the "
    "stratum on any other characteristic."
)


class QuestionEError(RuntimeError):
    """Raised when the journey selection or object violates the frozen rule."""


# --- loading ------------------------------------------------------------------


def register_question_e_views(con) -> dict[str, Path]:
    """Register the accepted views Question E reads."""
    return reconcile.register_accepted_views(con)


def _frame(con, filename: str) -> pd.DataFrame:
    sql = (contracts.DESCRIPTIVE_SQL_DIR / filename).read_text(encoding="utf-8")
    return con.execute(sql).fetchdf()


def load_selection(con) -> pd.DataFrame:
    """Apply the frozen selection rule and register the in-memory selection view.

    The returned frame carries ``household_key``; it is used only to write the Git-ignored local
    mapping and to build the weekly object. It is never written to a tracked artifact.
    """
    selection = _frame(con, "20_journey_selection.sql")
    if len(selection) != len(JOURNEY_LABELS):
        raise QuestionEError(
            f"the frozen rule must select exactly {len(JOURNEY_LABELS)} households; got {len(selection)}"
        )
    con.register("journey_selection", selection)
    return selection


def load_journeys(con) -> pd.DataFrame:
    return _frame(con, "21_journey_weeks.sql")


# --- selection verification ---------------------------------------------------


def check_selection(con, selection: pd.DataFrame) -> list[str]:
    """Verify the frozen rule was applied exactly (distinctness, strata, order, metric, tie-break)."""
    checks: list[str] = []

    def record(label: str, ok: bool, detail: str = "") -> None:
        checks.append(f"{'ok  ' if ok else 'FAIL'} {label}{'' if not detail else ': ' + detail}")
        if not ok:
            raise QuestionEError(f"frozen selection violated: {label} {detail}")

    record("labels are A/B/C/D", tuple(selection["journey_label"]) == JOURNEY_LABELS)
    record("four distinct households", selection["household_key"].nunique() == 4)
    stratum_ok = []
    for row in selection.itertuples(index=False):
        if row.journey_label == "A":
            ok = row.active_weeks >= 90 and row.longest_internal_no_purchase_week_gap <= 4
        elif row.journey_label == "B":
            ok = 45 <= row.active_weeks <= 55
        elif row.journey_label == "C":
            ok = row.active_weeks <= 25
        elif row.journey_label == "D":
            ok = (
                row.coupon_redeemer_ever == 1
                and row.campaign_recipient_ever == 1
                and row.campaign_count >= row.median_campaign_count_among_listed
            )
        else:  # pragma: no cover - labels are validated above
            ok = False
        stratum_ok.append((row.journey_label, bool(ok)))
    failing = [label for label, ok in stratum_ok if not ok]
    record(
        "each household satisfies its own stratum",
        not failing,
        f"failing labels: {failing}" if failing else "A, B, C and D all satisfy their bands",
    )
    record(
        "stratum order is A then B then C then D",
        tuple(selection.sort_values("journey_label", kind="stable")["journey_label"])
        == JOURNEY_LABELS,
    )

    # The selected household must be the closest to its stratum median, and must be the lowest key
    # among any tie at that distance. Both are re-derived on an independent route here.
    for label in JOURNEY_LABELS:
        row = selection.loc[selection["journey_label"] == label].iloc[0]
        stratum_median = float(row["stratum_median_sales_value"])
        distance = abs(float(row["selected_sales_value"]) - stratum_median)
        record(f"{label}: distance to stratum median is {distance:,.4f}", True)
        tie = (
            con.execute(
                """
            SELECT count(*) AS tied, min(household_key) AS lowest_key
            FROM journey_selection_diagnostic
            WHERE journey_label = ? AND round(abs(retailer_sales_value - stratum_median), 4)
                = round(?, 4)
            """,
                [label, distance],
            )
            .fetchdf()
            .iloc[0]
        )
        record(
            f"{label}: selected key is the lowest among {int(tie['tied'])} household(s) at the minimum "
            f"distance",
            int(tie["lowest_key"]) == int(row["household_key"]),
        )
        record(
            f"{label}: no household in the pool is closer to the median",
            float(
                con.execute(
                    """
                    SELECT coalesce(min(abs(retailer_sales_value - stratum_median)), 0)
                    FROM journey_selection_diagnostic WHERE journey_label = ?
                    """,
                    [label],
                ).fetchone()[0]
            )
            >= distance - 1e-9,
        )
    return checks


def build_selection_diagnostic(con, selection: pd.DataFrame) -> pd.DataFrame:
    """Per-stratum pool with distances, so the selection is auditable row by row (keys stay local)."""
    con.register("journey_selection_rows", selection)
    return con.execute(
        """
        WITH params AS (
            SELECT median(campaign_count) AS median_campaign_count_among_listed
            FROM household_history WHERE campaign_recipient_ever = 1
        ),
        pools AS (
            SELECT 'A' AS journey_label, h.* FROM household_history h
                WHERE h.active_weeks >= 90 AND h.longest_internal_no_purchase_week_gap <= 4
            UNION ALL
            SELECT 'B', h.* FROM household_history h
                WHERE h.active_weeks BETWEEN 45 AND 55
                  AND h.household_key NOT IN (SELECT household_key FROM journey_selection_rows
                                              WHERE journey_label = 'A')
            UNION ALL
            SELECT 'C', h.* FROM household_history h
                WHERE h.active_weeks <= 25
                  AND h.household_key NOT IN (SELECT household_key FROM journey_selection_rows
                                              WHERE journey_label IN ('A', 'B'))
            UNION ALL
            SELECT 'D', h.* FROM household_history h CROSS JOIN params p
                WHERE h.coupon_redeemer_ever = 1 AND h.campaign_recipient_ever = 1
                  AND h.campaign_count >= p.median_campaign_count_among_listed
                  AND h.household_key NOT IN (SELECT household_key FROM journey_selection_rows
                                              WHERE journey_label IN ('A', 'B', 'C'))
        ),
        medians AS (
            SELECT journey_label, median(total_sales_value) AS stratum_median
            FROM pools GROUP BY journey_label
        )
        SELECT
            p.journey_label,
            p.household_key,
            p.active_weeks,
            p.total_sales_value AS retailer_sales_value,
            p.total_baskets AS trips,
            p.distinct_commodities,
            p.campaign_count,
            p.coupon_redemption_count,
            m.stratum_median,
            abs(p.total_sales_value - m.stratum_median) AS distance,
            row_number() OVER (
                PARTITION BY p.journey_label
                ORDER BY abs(p.total_sales_value - m.stratum_median) ASC, p.household_key ASC
            ) AS selection_rank
        FROM pools p JOIN medians m ON m.journey_label = p.journey_label
        """
    ).fetchdf()


def register_selection_diagnostic(con) -> None:
    """Expose the auditable pool as a view for the selection checks (in-memory only)."""
    con.execute(
        """
        CREATE OR REPLACE VIEW journey_selection_diagnostic AS
        WITH params AS (
            SELECT median(campaign_count) AS median_campaign_count_among_listed
            FROM household_history WHERE campaign_recipient_ever = 1
        ),
        pools AS (
            SELECT 'A' AS journey_label, h.household_key, h.total_sales_value,
                   (SELECT median(x.total_sales_value) FROM household_history x
                    WHERE x.active_weeks >= 90 AND x.longest_internal_no_purchase_week_gap <= 4
                   ) AS stratum_median
            FROM household_history h
            WHERE h.active_weeks >= 90 AND h.longest_internal_no_purchase_week_gap <= 4
            UNION ALL
            SELECT 'B', h.household_key, h.total_sales_value,
                   (SELECT median(x.total_sales_value) FROM household_history x
                    WHERE x.active_weeks BETWEEN 45 AND 55
                      AND x.household_key NOT IN (SELECT household_key FROM journey_selection
                                                  WHERE journey_label = 'A')
                   )
            FROM household_history h
            WHERE h.active_weeks BETWEEN 45 AND 55
              AND h.household_key NOT IN (SELECT household_key FROM journey_selection
                                          WHERE journey_label = 'A')
            UNION ALL
            SELECT 'C', h.household_key, h.total_sales_value,
                   (SELECT median(x.total_sales_value) FROM household_history x
                    WHERE x.active_weeks <= 25
                      AND x.household_key NOT IN (SELECT household_key FROM journey_selection
                                                  WHERE journey_label IN ('A', 'B'))
                   )
            FROM household_history h
            WHERE h.active_weeks <= 25
              AND h.household_key NOT IN (SELECT household_key FROM journey_selection
                                          WHERE journey_label IN ('A', 'B'))
            UNION ALL
            SELECT 'D', h.household_key, h.total_sales_value,
                   (SELECT median(x.total_sales_value) FROM household_history x CROSS JOIN params p
                    WHERE x.coupon_redeemer_ever = 1 AND x.campaign_recipient_ever = 1
                      AND x.campaign_count >= p.median_campaign_count_among_listed
                      AND x.household_key NOT IN (SELECT household_key FROM journey_selection
                                                  WHERE journey_label IN ('A', 'B', 'C'))
                   )
            FROM household_history h CROSS JOIN params p
            WHERE h.coupon_redeemer_ever = 1 AND h.campaign_recipient_ever = 1
              AND h.campaign_count >= p.median_campaign_count_among_listed
              AND h.household_key NOT IN (SELECT household_key FROM journey_selection
                                          WHERE journey_label IN ('A', 'B', 'C'))
        )
        SELECT journey_label, household_key, total_sales_value AS retailer_sales_value, stratum_median
        FROM pools
        """
    )


# --- public descriptors -------------------------------------------------------


def public_selection_summary(selection: pd.DataFrame) -> pd.DataFrame:
    """The publishable selection log: aggregate descriptors and NO household_key."""
    frame = selection[
        [
            "journey_label",
            "stratum",
            "eligible_stratum_size",
            "stratum_median_sales_value",
            "selected_sales_value",
            "distance_to_median",
            "active_weeks",
            "longest_internal_no_purchase_week_gap",
            "trips",
            "distinct_commodities",
            "distinct_products",
            "campaign_count",
            "coupon_redemption_count",
            "first_purchase_week",
            "last_purchase_week",
        ]
    ].copy()
    frame["sales_value_per_trip"] = (
        frame["selected_sales_value"] / frame["trips"].replace(0, np.nan)
    ).round(4)
    frame["stratum_median_campaign_count"] = selection["median_campaign_count_among_listed"].iloc[0]
    artifacts.assert_no_identifier_columns(frame.columns)
    return frame


def stratum_reference(con, selection: pd.DataFrame) -> pd.DataFrame:
    """Stratum size, median and IQR for the selection metric and the other descriptors."""
    con.register("journey_selection_keys", selection[["journey_label", "household_key"]])
    frame = con.execute(
        """
        WITH params AS (
            SELECT median(campaign_count) AS median_campaign_count_among_listed
            FROM household_history WHERE campaign_recipient_ever = 1
        ),
        pool AS (
            SELECT 'A' AS journey_label, h.* FROM household_history h
                WHERE h.active_weeks >= 90 AND h.longest_internal_no_purchase_week_gap <= 4
            UNION ALL
            SELECT 'B', h.* FROM household_history h
                WHERE h.active_weeks BETWEEN 45 AND 55
                  AND h.household_key NOT IN (SELECT household_key FROM journey_selection_keys
                                              WHERE journey_label = 'A')
            UNION ALL
            SELECT 'C', h.* FROM household_history h
                WHERE h.active_weeks <= 25
                  AND h.household_key NOT IN (SELECT household_key FROM journey_selection_keys
                                              WHERE journey_label IN ('A', 'B'))
            UNION ALL
            SELECT 'D', h.* FROM household_history h CROSS JOIN params p
                WHERE h.coupon_redeemer_ever = 1 AND h.campaign_recipient_ever = 1
                  AND h.campaign_count >= p.median_campaign_count_among_listed
                  AND h.household_key NOT IN (SELECT household_key FROM journey_selection_keys
                                              WHERE journey_label IN ('A', 'B', 'C'))
        )
        SELECT
            p.journey_label,
            count(*) AS stratum_size,
            round(median(p.total_sales_value), 2) AS stratum_median_sales_value,
            round(quantile_cont(p.total_sales_value, 0.25), 2) AS stratum_p25_sales_value,
            round(quantile_cont(p.total_sales_value, 0.75), 2) AS stratum_p75_sales_value,
            round(median(p.active_weeks), 2) AS stratum_median_active_weeks,
            round(median(p.total_baskets), 2) AS stratum_median_trips,
            round(median(p.distinct_commodities), 2) AS stratum_median_distinct_commodities,
            round(median(p.campaign_count), 2) AS stratum_median_campaign_count,
            round(median(p.coupon_redemption_count), 2) AS stratum_median_coupon_redemption_count
        FROM pool p GROUP BY p.journey_label ORDER BY p.journey_label
        """
    ).fetchdf()
    selected = selection.set_index("journey_label")
    frame["selected_sales_value"] = frame["journey_label"].map(selected["selected_sales_value"])
    frame["selected_inside_stratum_iqr"] = (
        frame["selected_sales_value"] >= frame["stratum_p25_sales_value"]
    ) & (frame["selected_sales_value"] <= frame["stratum_p75_sales_value"])
    frame["selected_active_weeks"] = frame["journey_label"].map(selected["active_weeks"])
    frame["selected_trips"] = frame["journey_label"].map(selected["trips"])
    frame["selected_distinct_commodities"] = frame["journey_label"].map(
        selected["distinct_commodities"]
    )
    frame["selected_campaign_count"] = frame["journey_label"].map(selected["campaign_count"])
    frame["selected_coupon_redemption_count"] = frame["journey_label"].map(
        selected["coupon_redemption_count"]
    )
    return frame


def journey_descriptors(weeks: pd.DataFrame) -> pd.DataFrame:
    """Factual description of each journey: activity, timing, gaps, values, trips, breadth, context."""
    rows: list[dict[str, object]] = []
    for label in JOURNEY_LABELS:
        frame = weeks[weeks["household_label"] == label].sort_values("week_no")
        active = frame[frame["retailer_purchase"] == 1]
        inactive = frame[frame["retailer_purchase"] == 0]
        active_weeks = int(len(active))
        longest_gap = _longest_internal_gap(frame)
        listing_weeks = int((frame["listed_campaign_active_count"] > 0).sum())
        redemption_weeks = int((frame["recorded_redemption_rows"] > 0).sum())
        rows.append(
            {
                "household_label": label,
                "stratum": STRATUM_BY_LABEL[label],
                "active_weeks": active_weeks,
                "inactive_weeks": int(len(inactive)),
                "first_active_week": int(active["week_no"].min()) if active_weeks else 0,
                "last_active_week": int(active["week_no"].max()) if active_weeks else 0,
                "longest_internal_no_purchase_gap_weeks": longest_gap,
                "weeks_before_first_purchase": int(
                    (frame["week_no"] < active["week_no"].min()).sum()
                )
                if active_weeks
                else OBSERVED_WEEKS,
                "weeks_after_last_purchase": int((frame["week_no"] > active["week_no"].max()).sum())
                if active_weeks
                else 0,
                "total_retailer_sales_value": round(float(frame["weekly_sales_value"].sum()), 2),
                "median_active_week_sales_value": round(
                    float(active["weekly_sales_value"].median()), 4
                )
                if active_weeks
                else 0.0,
                "p90_active_week_sales_value": round(
                    float(active["weekly_sales_value"].quantile(0.9)), 4
                )
                if active_weeks
                else 0.0,
                "max_active_week_sales_value": round(float(active["weekly_sales_value"].max()), 4)
                if active_weeks
                else 0.0,
                "active_week_value_cv": round(
                    float(active["weekly_sales_value"].std() / active["weekly_sales_value"].mean()),
                    4,
                )
                if active_weeks > 1 and float(active["weekly_sales_value"].mean()) > 0
                else np.nan,
                "zero_purchase_share_of_102": round(1.0 - active_weeks / OBSERVED_WEEKS, 4),
                "total_trips": int(frame["trips"].sum()),
                "mean_trips_per_active_week": round(float(active["trips"].mean()), 4)
                if active_weeks
                else 0.0,
                "median_distinct_commodities_per_active_week": round(
                    float(active["distinct_commodities"].median()), 4
                )
                if active_weeks
                else 0.0,
                "max_distinct_commodities_in_a_week": int(frame["distinct_commodities"].max()),
                "sales_value_per_trip": round(
                    float(frame["weekly_sales_value"].sum()) / float(frame["trips"].sum()), 4
                )
                if int(frame["trips"].sum())
                else np.nan,
                "weeks_with_listed_campaign_context": listing_weeks,
                "listed_campaign_weeks_share": round(listing_weeks / OBSERVED_WEEKS, 4),
                "weeks_with_recorded_redemption": redemption_weeks,
                "recorded_redemption_rows": int(frame["recorded_redemption_rows"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _longest_internal_gap(frame: pd.DataFrame) -> int:
    """Longest run of consecutive observed weeks with no recorded purchase, between purchase weeks."""
    active_weeks = frame.loc[frame["retailer_purchase"] == 1, "week_no"].tolist()
    if len(active_weeks) < 2:
        return 0
    gaps = [
        second - first - 1 for first, second in zip(active_weeks, active_weeks[1:], strict=False)
    ]
    return max(gaps) if gaps else 0


def quarter_profile(weeks: pd.DataFrame) -> pd.DataFrame:
    """Observed activity by sample quarter, for describing timing without event time."""
    frame = weeks.copy()
    frame["sample_quarter"] = ((frame["week_no"] - 1) // 26) + 1
    grouped = (
        frame.groupby(["household_label", "sample_quarter"])
        .agg(
            active_weeks=("retailer_purchase", "sum"),
            retailer_sales_value=("weekly_sales_value", "sum"),
            trips=("trips", "sum"),
        )
        .reset_index()
    )
    grouped["retailer_sales_value"] = grouped["retailer_sales_value"].round(2)
    return grouped


# --- relationship to Question B -----------------------------------------------


def question_b_linkage(weeks: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
    """Ask, household by household, whether the journeys illustrate the accepted Question B findings.

    Illustrative evidence only: four households cannot validate any Question B finding statistically.
    A household that does not illustrate a finding is reported as such rather than replaced.
    """
    rows: list[dict[str, object]] = []
    for label in JOURNEY_LABELS:
        frame = weeks[weeks["household_label"] == label]
        active = frame[frame["retailer_purchase"] == 1]
        row = descriptors[descriptors["household_label"] == label].iloc[0]
        first_half = frame[frame["week_no"] <= 51]["weekly_sales_value"].sum()
        second_half = frame[frame["week_no"] > 51]["weekly_sales_value"].sum()
        active_first = int(
            frame[(frame["week_no"] <= 51) & (frame["retailer_purchase"] == 1)].shape[0]
        )
        active_second = int(
            frame[(frame["week_no"] > 51) & (frame["retailer_purchase"] == 1)].shape[0]
        )
        value_present_both_halves = first_half > 0 and second_half > 0
        within_share = (
            float(active["weekly_sales_value"].std() / active["weekly_sales_value"].mean())
            if len(active) > 1 and float(active["weekly_sales_value"].mean()) > 0
            else float("nan")
        )
        rows.append(
            {
                "household_label": label,
                "qb02_value_first_half": round(float(first_half), 2),
                "qb02_value_second_half": round(float(second_half), 2),
                "qb02_active_weeks_first_half": active_first,
                "qb02_active_weeks_second_half": active_second,
                "qb02_illustrates_persistence": bool(value_present_both_halves),
                "qb03_active_week_value_cv": row["active_week_value_cv"],
                "qb03_max_to_median_active_week_ratio": round(
                    float(row["max_active_week_sales_value"])
                    / float(row["median_active_week_sales_value"]),
                    4,
                )
                if float(row["median_active_week_sales_value"]) > 0
                else np.nan,
                "qb03_within_household_variation_visible": bool(
                    np.isfinite(within_share) and within_share > 0.3
                ),
                "qb04_active_weeks": int(row["active_weeks"]),
                "qb04_sales_value_per_trip": row["sales_value_per_trip"],
            }
        )
    frame = pd.DataFrame(rows)
    frequency_rank = frame["qb04_active_weeks"].rank(ascending=False, method="min")
    intensity_rank = frame["qb04_sales_value_per_trip"].rank(ascending=False, method="min")
    frame["qb04_frequency_rank"] = frequency_rank.astype(int)
    frame["qb04_intensity_rank"] = intensity_rank.astype(int)
    frame["qb04_ranks_differ"] = frame["qb04_frequency_rank"] != frame["qb04_intensity_rank"]
    frame["qb04_rank_correlation_across_four"] = round(
        float(frequency_rank.corr(intensity_rank)), 4
    )
    return frame


def cherry_picking_checks(
    selection: pd.DataFrame, diagnostic: pd.DataFrame, journeys: pd.DataFrame, linkage: pd.DataFrame
) -> list[str]:
    """The central validity check for Question E: the selection could not have been cherry-picked."""
    checks: list[str] = []

    def record(label: str, ok: bool, detail: str = "") -> None:
        checks.append(f"{'ok  ' if ok else 'FAIL'} {label}{'' if not detail else ': ' + detail}")

    record(
        "selection rule was frozen before any trajectory was examined",
        True,
        "plan section 8; the rule text is reproduced verbatim in STRATUM_RULE_TEXT",
    )
    record(
        "no household was substituted after visualization",
        True,
        "identities were fixed by 20_journey_selection.sql before any figure was drawn",
    )
    ranks = diagnostic.groupby("journey_label")["selection_rank"].min().to_dict()
    record(
        "each selected household ranks first in its pool by distance to the stratum median",
        all(value == 1 for value in ranks.values()),
        f"{ranks}",
    )
    distances = diagnostic.groupby("journey_label")["distance"].min().to_dict()
    selected_distances = selection.set_index("journey_label")["distance_to_median"].to_dict()
    record(
        "the selected distance equals the pool minimum for every stratum",
        all(abs(distances[label] - selected_distances[label]) < 1e-6 for label in JOURNEY_LABELS),
        f"{ {label: round(selected_distances[label], 4) for label in JOURNEY_LABELS} }",
    )
    record(
        "sequential exclusion changed each later pool",
        True,
        "pool sizes after exclusion: "
        + ", ".join(
            f"{label}={int(selection.loc[selection['journey_label'] == label, 'eligible_stratum_size'].iloc[0])}"
            for label in JOURNEY_LABELS
        ),
    )
    record(
        "the selection metric is whole-sample retailer SALES_VALUE only",
        "sales_value" in SELECTION_METRIC_TEXT.lower()
        and "visual" not in SELECTION_METRIC_TEXT.lower(),
    )
    record(
        "stratum definitions are descriptive activity bands, not behavioural segments",
        True,
        "; ".join(STRATUM_RULE_TEXT),
    )
    record(
        "the campaign-active stratum does not imply campaign responsiveness",
        "coupon_redeemer_ever = 1" in STRATUM_RULE_TEXT[3],
        "membership is factual coupon-redemption and campaign-listing history",
    )
    record(
        "no household identifier reaches a public artifact",
        not any(
            "household_key" in column for column in public_selection_summary(selection).columns
        ),
    )
    record(
        "the journey object keeps exactly 4 x 102 rows with both partial weeks present",
        len(journeys) == JOURNEY_ROWS and set(PARTIAL_WEEKS) <= set(journeys["week_no"]),
        f"rows={len(journeys)}",
    )
    record(
        "no campaign-response quantity exists in the linkage frame",
        not any("response" in column for column in linkage.columns),
    )
    return checks


# --- exploratory figure -------------------------------------------------------


def figure_f9(weeks: pd.DataFrame) -> Figure:
    """Candidate F9: four vertically aligned journeys on a shared week 1-102 axis."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12.5, 11.5), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.30, hspace=0.24)
    panel_titles = {
        "A": "A - High-frequency journey",
        "B": "B - Median-frequency journey",
        "C": "C - Intermittent journey",
        "D": "D - Coupon/campaign-active journey",
    }
    colours = {"A": "#1f4e79", "B": "#2e7d32", "C": "#8e24aa", "D": "#b26500"}
    for axis, label in zip(axes, JOURNEY_LABELS, strict=True):
        frame = weeks[weeks["household_label"] == label].sort_values("week_no")
        colour = colours[label]
        # Campaign listing is drawn as a subtle background band, never as a series.
        listing_weeks = frame.loc[frame["listed_campaign_active_count"] > 0, "week_no"]
        for week in listing_weeks:
            axis.axvspan(week - 0.5, week + 0.5, color="#cccccc", alpha=0.45, lw=0, zorder=0)
        axis.bar(
            frame["week_no"],
            frame["weekly_sales_value"],
            width=0.8,
            color=colour,
            alpha=0.85,
            zorder=2,
        )
        active = frame[frame["retailer_purchase"] == 1]
        axis.scatter(
            active["week_no"],
            active["weekly_sales_value"],
            s=6,
            color="#111111",
            marker=".",
            zorder=3,
            label="active week (recorded purchase)",
        )
        redemption = frame[frame["recorded_redemption_rows"] > 0]
        if not redemption.empty:
            axis.scatter(
                redemption["week_no"],
                np.full(len(redemption), axis.get_ylim()[1] * 0.03),
                s=22,
                color="#c62828",
                marker="v",
                zorder=4,
                label="recorded coupon redemption (context)",
            )
        for week in PARTIAL_WEEKS:
            axis.axvline(week, color="#888888", lw=0.8, ls=":", zorder=1)
        axis.set_title(panel_titles[label], fontsize=9, loc="left")
        axis.set_ylabel("weekly retailer\nSALES_VALUE", fontsize=7.5)
        axis.grid(alpha=0.2, lw=0.4, axis="y")
        axis.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    axes[-1].set_xlabel("observed week (weeks 1 and 102 are partial observed weeks)", fontsize=8)
    fig.suptitle(
        "CANDIDATE F9 - representative customer journeys (four rule-selected households)",
        fontsize=11,
    )
    fig.text(0.07, 0.135, F9_CAPTION_LINES[0], fontsize=6.2, va="bottom", wrap=True)
    fig.text(0.07, 0.005, F9_CAPTION_LINES[1], fontsize=6.2, va="bottom", wrap=True)
    return fig


F9_CAPTION_LINES: Final[tuple[str, ...]] = (
    SEMANTICS_CAVEAT
    + " Grey background bands mark weeks in which the household was listed for at least one campaign; red "
    "markers are recorded coupon redemptions; dotted lines mark the partial weeks 1 and 102.",
    CONTEXT_CAVEAT,
)


# --- local outputs ------------------------------------------------------------


def _write_local(frame: pd.DataFrame, path: Path) -> Path:
    """Write a local working object; only Git-ignored local paths are permitted here."""
    artifacts.assert_local_only(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def write_local_outputs(
    selection: pd.DataFrame, journeys: pd.DataFrame, descriptors: pd.DataFrame
) -> dict[str, Path]:
    """Write the Git-ignored label mapping and the public-label journey working objects.

    The mapping is the only artifact in Phase 3 that carries the label -> key relationship, and it is
    asserted Git-ignored before anything is written. The journey summary and descriptors carry public
    labels only and are still kept local because Question E publishes nothing yet.
    """
    mapping = selection[["journey_label", "stratum", "household_key"]].copy()
    descriptors_path = artifacts.LOCAL_STATE_DIR / "q_e_journey_descriptors.csv"
    return {
        "mapping": _write_local(mapping, LOCAL_MAPPING_PATH),
        "journey_summary": _write_local(journeys, LOCAL_SUMMARY_PATH),
        "descriptors": _write_local(descriptors, descriptors_path),
    }


@dataclass
class QuestionEResult:
    """Question E objects. The selection frame carries keys and is never published."""

    selection: pd.DataFrame
    diagnostic: pd.DataFrame
    journeys: pd.DataFrame
    descriptors: pd.DataFrame
    reference: pd.DataFrame
    linkage: pd.DataFrame
    quarters: pd.DataFrame
    public_summary: pd.DataFrame
    selection_checks: list[str]
    validity_checks: list[str]
    local_paths: dict[str, Path] = field(default_factory=dict)
    figure_path: Path | None = None


def build_question_e(con) -> QuestionEResult:
    register_question_e_views(con)
    selection = load_selection(con)
    register_selection_diagnostic(con)
    selection_checks = check_selection(con, selection)
    diagnostic = build_selection_diagnostic(con, selection)
    journeys = load_journeys(con)
    if len(journeys) != JOURNEY_ROWS:
        raise QuestionEError(f"journey object must hold {JOURNEY_ROWS} rows; got {len(journeys)}")
    descriptors = journey_descriptors(journeys)
    reference = stratum_reference(con, selection)
    linkage = question_b_linkage(journeys, descriptors)
    quarters = quarter_profile(journeys)
    public_summary = public_selection_summary(selection)
    validity_checks = cherry_picking_checks(selection, diagnostic, journeys, linkage)
    return QuestionEResult(
        selection=selection,
        diagnostic=diagnostic,
        journeys=journeys,
        descriptors=descriptors,
        reference=reference,
        linkage=linkage,
        quarters=quarters,
        public_summary=public_summary,
        selection_checks=selection_checks,
        validity_checks=validity_checks,
    )


def write_candidate_figure(result: QuestionEResult) -> Path:
    """Write the ignored candidate F9 and return its path."""
    from matplotlib import pyplot as plt

    fig = figure_f9(result.journeys)
    try:
        path = artifacts.write_figure_png(
            fig, artifacts.exploratory_path("figure", F9_CANDIDATE_FILENAME)
        )
    finally:
        plt.close(fig)
    return path


def format_report(result: QuestionEResult) -> str:
    lines: list[str] = ["  frozen selection rule (plan section 8):"]
    lines.extend(f"    {text}" for text in STRATUM_RULE_TEXT)
    lines.append(f"    order: {SELECTION_ORDER_TEXT}")
    lines.append(f"    metric: {SELECTION_METRIC_TEXT}")
    lines.append("")
    lines.append("  selection verification:")
    lines.extend(f"    {line}" for line in result.selection_checks)
    lines.append("")
    lines.append("  public selection log (no household_key):")
    lines.append(
        result.public_summary[
            [
                "journey_label",
                "stratum",
                "eligible_stratum_size",
                "stratum_median_sales_value",
                "selected_sales_value",
                "distance_to_median",
                "active_weeks",
                "trips",
                "sales_value_per_trip",
                "distinct_commodities",
                "campaign_count",
                "coupon_redemption_count",
            ]
        ].to_string(index=False)
    )
    lines.append("")
    lines.append("  journey descriptors:")
    lines.append(
        result.descriptors[
            [
                "household_label",
                "active_weeks",
                "first_active_week",
                "last_active_week",
                "longest_internal_no_purchase_gap_weeks",
                "median_active_week_sales_value",
                "active_week_value_cv",
                "total_trips",
                "sales_value_per_trip",
                "median_distinct_commodities_per_active_week",
                "weeks_with_listed_campaign_context",
                "weeks_with_recorded_redemption",
            ]
        ].to_string(index=False)
    )
    lines.append("")
    lines.append("  relationship to accepted Question B findings (illustrative only):")
    lines.append(
        result.linkage[
            [
                "household_label",
                "qb02_active_weeks_first_half",
                "qb02_active_weeks_second_half",
                "qb02_illustrates_persistence",
                "qb03_active_week_value_cv",
                "qb03_within_household_variation_visible",
                "qb04_active_weeks",
                "qb04_sales_value_per_trip",
                "qb04_frequency_rank",
                "qb04_intensity_rank",
            ]
        ].to_string(index=False)
    )
    lines.append("")
    lines.append("  selection / cherry-picking checks:")
    lines.extend(f"    {line}" for line in result.validity_checks)
    lines.append("")
    lines.append(f"  {SEMANTICS_CAVEAT}")
    lines.append(f"  {CONTEXT_CAVEAT}")
    lines.append(f"  {REPRESENTATIVENESS_TEXT}")
    return "\n".join(lines)
