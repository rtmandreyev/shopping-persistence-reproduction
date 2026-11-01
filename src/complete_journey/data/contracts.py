"""Phase 2A output contracts for the canonical analytical data layer.

A contract states, for every canonical table, its grain, key, required columns,
structural population, generating SQL, and source lineage. The contracts are the
single source of truth used by:

* the builder (:mod:`complete_journey.data.canonical`) to validate what it wrote;
* ``tests/test_phase2_contracts.py`` to assert the layer is documented;
* the generated ``outputs/tables/panels/contracts.json`` and ``docs/data_lineage.csv``.

Row-count expectations are stated ONLY where the population is structurally
required by the approved design (the full household x week grid, and the
one-row-per-household tables). Tables whose size is an empirical property of the
data carry ``expected_row_count = None`` and are reconciled instead of pinned.
"""

from __future__ import annotations

import re
from typing import Final

#: DuckDB type-name families used for family-level validation. Pinning exact
#: integer widths would be brittle (DuckDB widens ``sum(BIGINT)`` to HUGEINT and
#: ``count`` to BIGINT); the family is what carries meaning.
INTEGER_FAMILY: Final[frozenset[str]] = frozenset(
    {
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "INT",
        "INT2",
        "INT4",
        "INT8",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "UHUGEINT",
    }
)
DOUBLE_FAMILY: Final[frozenset[str]] = frozenset({"DOUBLE", "FLOAT", "REAL", "DECIMAL"})
TEXT_FAMILY: Final[frozenset[str]] = frozenset({"VARCHAR", "TEXT", "STRING"})

#: Column families a contract may declare.
FAMILIES: Final[frozenset[str]] = frozenset({"integer", "double", "text"})


def column_family(duckdb_type: str) -> str:
    """Map a concrete DuckDB type name to a coarse family.

    Raises:
        ValueError: if the type is not recognised, so an unexpected type cannot
            silently pass validation.
    """
    base = duckdb_type.upper().split("(")[0].strip()
    if base in INTEGER_FAMILY:
        return "integer"
    if base in DOUBLE_FAMILY:
        return "double"
    if base in TEXT_FAMILY:
        return "text"
    raise ValueError(f"unrecognised DuckDB type for family mapping: {duckdb_type!r}")


#: Canonical (durable, reusable) tables written as Parquet under data/processed/.
CANONICAL_TABLES: Final[dict[str, dict[str, object]]] = {
    "basket_panel": {
        "title": "Basket-level canonical panel",
        "grain": "household_key x basket_id",
        "key": ("household_key", "basket_id"),
        "columns": (
            ("household_key", "integer"),
            ("basket_id", "integer"),
            ("day", "integer"),
            ("week_no", "integer"),
            ("store_id", "integer"),
            ("basket_sales_value", "double"),
            ("basket_units", "integer"),
            ("distinct_products", "integer"),
            ("distinct_commodities", "integer"),
            ("transaction_line_count", "integer"),
            ("basket_retail_disc", "double"),
            ("basket_coupon_disc", "double"),
            ("basket_coupon_match_disc", "double"),
            ("qa_quantity_zero_lines", "integer"),
            ("qa_retail_disc_positive_lines", "integer"),
            ("qa_implied_paid_negative_lines", "integer"),
        ),
        "population": "Every distinct shopping basket in transaction_data. No basket "
        "is dropped for carrying a flagged anomaly.",
        "time_support": "Days 1-711 / weeks 1-102 (basket day is uniquely determined).",
        "zero_semantics": "Not applicable: a basket exists only because it has "
        "transaction lines. basket_units may be 0 when every line has QUANTITY = 0 "
        "(flagged by qa_quantity_zero_lines).",
        "missing_semantics": "No nulls: every basket has a household, day, week, and store.",
        "expected_row_count": None,
        "sql": "01_basket_panel.sql",
        "scope": "processed",
        "sources": ("transaction_data", "product"),
        "limitations": "Sales values are retailer revenue, not customer-paid prices. "
        "Discount sums keep source sign conventions.",
    },
    "household_week_base": {
        "title": "Complete household x week retailer-observation grid",
        "grain": "household_key x week_no",
        "key": ("household_key", "week_no"),
        "columns": (
            ("household_key", "integer"),
            ("week_no", "integer"),
            ("retailer_purchase", "integer"),
            ("trips", "integer"),
            ("weekly_sales_value", "double"),
            ("units", "integer"),
            ("distinct_products", "integer"),
            ("distinct_commodities", "integer"),
            ("active_store_count", "integer"),
            ("transaction_line_count", "integer"),
            ("weekly_retail_disc", "double"),
            ("weekly_coupon_disc", "double"),
            ("weekly_coupon_match_disc", "double"),
            ("qa_quantity_zero_lines", "integer"),
            ("qa_retail_disc_positive_lines", "integer"),
            ("qa_implied_paid_negative_lines", "integer"),
        ),
        "population": "Every transacting household x every week in the source week "
        "domain (structurally 2,500 x 102 = 255,000 rows).",
        "time_support": "Weeks 1-102 (derived from transaction_data, not hardcoded).",
        "zero_semantics": "retailer_purchase = 0 and all activity metrics = 0 means "
        "'no recorded purchase from THIS retailer that week'. It does NOT mean the "
        "household consumed zero groceries.",
        "missing_semantics": "No nulls by construction.",
        "expected_row_count": 255000,
        "sql": "02_household_week_base.sql",
        "scope": "processed",
        "sources": ("transaction_data", "product"),
        "limitations": "Non-purchase weeks are observed zeros of recorded retailer "
        "activity, not zero grocery consumption. First/last purchase boundaries "
        "are not verified enrollment/exit or observation eligibility. Logical "
        "integer-family units and qa_*_lines may serialize as double; see "
        "docs/VARIABLE_DICTIONARY.md (P4-001).",
    },
    "household_category_week": {
        "title": "Sparse household x commodity x week factual table",
        "grain": "household_key x commodity_desc x week_no",
        "key": ("household_key", "commodity_desc", "week_no"),
        "columns": (
            ("household_key", "integer"),
            ("commodity_desc", "text"),
            ("week_no", "integer"),
            ("category_sales_value", "double"),
            ("category_units", "integer"),
            ("category_trips", "integer"),
            ("distinct_products", "integer"),
            ("category_national_sales_value", "double"),
            ("category_private_sales_value", "double"),
            ("category_national_units", "integer"),
            ("category_private_units", "integer"),
            ("distinct_national_products", "integer"),
            ("distinct_private_products", "integer"),
            ("category_retail_disc", "double"),
            ("category_coupon_disc", "double"),
            ("category_coupon_match_disc", "double"),
        ),
        "population": "SPARSE: one row per observed household x commodity x week "
        "purchase. Deliberately not the dense 2,500 x 308 x 102 grid.",
        "time_support": "Weeks 1-102; commodity from product.COMMODITY_DESC.",
        "zero_semantics": "An absent row is an unobserved commodity-week, not a zero.",
        "missing_semantics": "No nulls: commodity_desc is always present; product "
        "labels including vendor sentinels are preserved, never dropped.",
        "expected_row_count": None,
        "sql": "03_household_category_week.sql",
        "scope": "processed",
        "sources": ("transaction_data", "product"),
        "limitations": "No product availability inference, no chosen discrete-choice "
        "commodity. Sentinel commodity labels are kept as ordinary categories.",
    },
    "household_history": {
        "title": "Household-level full-sample activity summary",
        "grain": "household_key",
        "key": ("household_key",),
        "columns": (
            ("household_key", "integer"),
            ("first_purchase_day", "integer"),
            ("last_purchase_day", "integer"),
            ("first_purchase_week", "integer"),
            ("last_purchase_week", "integer"),
            ("active_weeks", "integer"),
            ("inactive_grid_weeks", "integer"),
            ("active_weeks_first_half", "integer"),
            ("active_weeks_second_half", "integer"),
            ("total_baskets", "integer"),
            ("total_sales_value", "double"),
            ("total_units", "integer"),
            ("distinct_products", "integer"),
            ("distinct_commodities", "integer"),
            ("distinct_stores", "integer"),
            ("longest_internal_no_purchase_week_gap", "integer"),
            ("leading_no_purchase_weeks", "integer"),
            ("trailing_no_purchase_weeks", "integer"),
            ("has_demographics", "integer"),
            ("campaign_recipient_ever", "integer"),
            ("campaign_count", "integer"),
            ("coupon_redeemer_ever", "integer"),
            ("coupon_redemption_count", "integer"),
        ),
        "population": "Every transacting household (structurally 2,500 rows).",
        "time_support": "Whole-sample summary over weeks 1-102.",
        "zero_semantics": "'inactive' = no recorded retailer purchase that week. "
        "Leading/trailing inactivity is calendar truncation, not attrition.",
        "missing_semantics": "No nulls; campaign/coupon/demographic indicators are 0 "
        "when the household does not appear in the relevant table.",
        "expected_row_count": 2500,
        "sql": "04_household_history.sql",
        "scope": "processed",
        "sources": (
            "transaction_data",
            "product",
            "hh_demographic",
            "campaign_table",
            "coupon_redempt",
        ),
        "limitations": "campaign_* and coupon_* are descriptive existence/count "
        "summaries, not treatment variables.",
    },
}

#: Local-only, customer-level diagnostic. NEVER Git-tracked.
PRIVATE_TABLES: Final[dict[str, dict[str, object]]] = {
    "panel_eligibility": {
        "title": "Household eligibility diagnostics (local-only, evidence only)",
        "grain": "household_key",
        "key": ("household_key",),
        "columns": (
            ("household_key", "integer"),
            ("active_weeks", "integer"),
            ("active_share_102", "double"),
            ("first_purchase_week", "integer"),
            ("last_purchase_week", "integer"),
            ("active_weeks_first_half", "integer"),
            ("active_weeks_second_half", "integer"),
            ("active_weeks_block_1", "integer"),
            ("active_weeks_block_2", "integer"),
            ("active_weeks_block_3", "integer"),
            ("active_weeks_block_4", "integer"),
            ("active_weeks_early", "integer"),
            ("active_weeks_middle", "integer"),
            ("active_weeks_late", "integer"),
            ("longest_internal_no_purchase_week_gap", "integer"),
            ("leading_no_purchase_weeks", "integer"),
            ("trailing_no_purchase_weeks", "integer"),
            ("total_trips", "integer"),
            ("total_sales_value", "double"),
            ("has_demographics", "integer"),
            ("campaign_recipient_ever", "integer"),
            ("campaign_count", "integer"),
            ("coupon_redeemer_ever", "integer"),
        ),
        "population": "Every transacting household (structurally 2,500 rows). "
        "LOCAL-ONLY and customer-level: never Git-tracked.",
        "time_support": "Whole-sample diagnostics over weeks 1-102.",
        "zero_semantics": "'inactive' = no recorded retailer purchase that week.",
        "missing_semantics": "No nulls.",
        "expected_row_count": 2500,
        "sql": "05_panel_eligibility.sql",
        "scope": "private",
        "sources": (
            "transaction_data",
            "hh_demographic",
            "campaign_table",
            "coupon_redempt",
        ),
        "limitations": "Does NOT determine eligibility: no flag, no threshold, no "
        "ranking, no exclusion. Evidence for a researcher decision made after 2A.",
    },
}

#: Phase 2B preparation tables: the econometric mart, neutral campaign
#: infrastructure, and discrete-choice ingredients. All are observation/household
#: level and therefore LOCAL (Parquet under data/processed/, Git-ignored).
PHASE2B_TABLES: Final[dict[str, dict[str, object]]] = {
    "econometrics_household_week": {
        "title": "Econometric household-week mart (S0 baseline + S1/S2 flags)",
        "grain": "household_key x week_no",
        "key": ("household_key", "week_no"),
        "columns": (
            ("household_key", "integer"),
            ("week_no", "integer"),
            ("retailer_purchase", "integer"),
            ("weekly_sales_value", "double"),
            ("trips", "integer"),
            ("units", "integer"),
            ("distinct_products", "integer"),
            ("distinct_commodities", "integer"),
            ("active_store_count", "integer"),
            ("transaction_line_count", "integer"),
            ("weekly_retail_disc", "double"),
            ("weekly_coupon_disc", "double"),
            ("weekly_coupon_match_disc", "double"),
            ("retail_disc_magnitude", "double"),
            ("coupon_disc_magnitude", "double"),
            ("coupon_match_disc_magnitude", "double"),
            ("total_discount_magnitude", "double"),
            ("qa_quantity_zero_lines", "integer"),
            ("qa_retail_disc_positive_lines", "integer"),
            ("qa_implied_paid_negative_lines", "integer"),
            ("lag1_retailer_purchase", "integer"),
            ("lag1_weekly_sales_value", "double"),
            ("lag1_trips", "integer"),
            ("lag1_units", "integer"),
            ("listed_campaign_active_count", "integer"),
            ("listed_campaign_recipient_any", "integer"),
            ("sample_full", "integer"),
            ("sample_sustained_support", "integer"),
            ("sample_higher_support", "integer"),
        ),
        "population": "All 2,500 transaction-universe households x all 102 weeks "
        "= 255,000 rows. S0 conditions on nothing; S1/S2 are sensitivity flags only.",
        "time_support": "Weeks 1-102.",
        "zero_semantics": "Inherited from household_week_base: retailer_purchase = 0 "
        "means no recorded purchase from this retailer, not zero consumption.",
        "missing_semantics": "Retailer-nonpurchase outcomes are observed recorded-activity "
        "zeros, not missing values. lag1_* are NULL in a household's first week "
        "(no invented zero history). Price-share variables are deliberately absent.",
        "expected_row_count": 255000,
        "sql": "06_econometrics_household_week.sql",
        "scope": "processed",
        "sources": (
            "household_week_base",
            "panel_eligibility",
            "campaign_table",
            "campaign_desc",
        ),
        "limitations": "Data preparation, not a model. S1/S2 condition on full-period "
        "realized activity and are not causal or predictive eligibility rules. "
        "Logical integer-family units, qa_*_lines and lag1_units may serialize as "
        "double; see docs/VARIABLE_DICTIONARY.md (P4-001).",
    },
    "campaign_week_calendar": {
        "title": "Neutral campaign week calendar",
        "grain": "campaign x week_no",
        "key": ("campaign", "week_no"),
        "columns": (
            ("campaign", "integer"),
            ("campaign_type", "text"),
            ("start_day", "integer"),
            ("end_day", "integer"),
            ("duration_days", "integer"),
            ("nominal_start_week", "integer"),
            ("nominal_end_week", "integer"),
            ("start_week", "integer"),
            ("end_week", "integer"),
            ("ends_after_transaction_window", "integer"),
            ("week_no", "integer"),
            ("relative_week", "integer"),
            ("period_label", "text"),
            ("is_campaign_week", "integer"),
        ),
        "population": "30 campaigns x 102 observed weeks = 3,060 rows.",
        "time_support": "Weeks 1-102; period labels use the accepted P1-009 convention.",
        "zero_semantics": "is_campaign_week = 0 means the week is outside the clipped "
        "campaign span (pre/post); it is not an exposure value.",
        "missing_semantics": "No nulls.",
        "expected_row_count": 3060,
        "sql": "07_campaign_week_calendar.sql",
        "scope": "processed",
        "sources": ("campaign_desc", "transaction_data"),
        "limitations": "Infrastructure only: no treatment/control, no window, no horizon, "
        "no estimand.",
    },
    "campaign_household_week": {
        "title": "Listed-recipient campaign household-week timeline",
        "grain": "household_key x campaign x week_no",
        "key": ("household_key", "campaign", "week_no"),
        "columns": (
            ("household_key", "integer"),
            ("campaign", "integer"),
            ("campaign_type", "text"),
            ("week_no", "integer"),
            ("relative_week", "integer"),
            ("period_label", "text"),
            ("is_campaign_week", "integer"),
            ("listed_recipient", "integer"),
            ("retailer_purchase", "integer"),
            ("trips", "integer"),
            ("weekly_sales_value", "double"),
            ("units", "integer"),
            ("overlapping_listed_campaign_count", "integer"),
            ("coupon_redemption_rows", "integer"),
        ),
        "population": "Every (household, campaign) pair listed in campaign_table "
        "(7,208) x 102 weeks = 735,216 rows. Listed recipients only; non-listed "
        "households are reachable via campaign_week_calendar, NOT enumerated here.",
        "time_support": "Weeks 1-102 per listed pair.",
        "zero_semantics": "coupon_redemption_rows = 0 means no recorded redemption that "
        "week. overlapping_listed_campaign_count counts listings, not realized exposure.",
        "missing_semantics": "No nulls.",
        "expected_row_count": 735216,
        "sql": "08_campaign_household_week.sql",
        "scope": "processed",
        "sources": (
            "campaign_table",
            "campaign_week_calendar",
            "household_week_base",
            "coupon_redempt",
        ),
        "limitations": "Infrastructure only. 'Listed' is never 'treated'; no control group "
        "is defined anywhere in this pipeline.",
    },
    "choice_product_store_week": {
        "title": "Purchased product-store-week facts with causal_data metadata",
        "grain": "product_id x store_id x week_no",
        "key": ("product_id", "store_id", "week_no"),
        "columns": (
            ("product_id", "integer"),
            ("store_id", "integer"),
            ("week_no", "integer"),
            ("transaction_lines", "integer"),
            ("transaction_units", "integer"),
            ("transaction_sales_value", "double"),
            ("distinct_households", "integer"),
            ("causal_record_present", "integer"),
            ("causal_record_rows", "integer"),
            ("causal_display_codes", "text"),
            ("causal_mailer_codes", "text"),
            ("display_conflict", "integer"),
            ("mailer_conflict", "integer"),
            ("any_display_descriptive", "integer"),
            ("any_mailer_descriptive", "integer"),
        ),
        "population": "The purchased PSW universe (2,370,784 keys). causal_data keys that "
        "were never purchased are not materialised (they remain in raw causal_data).",
        "time_support": "Weeks 1-102 (purchases); causal_data weeks 9-101.",
        "zero_semantics": "causal_record_present = 0 means NO recorded causal row: its code "
        "columns are NULL, never '0', and never 'no placement'.",
        "missing_semantics": "display_conflict / mailer_conflict / any_* are NULL when no "
        "record exists (undefined without a record), else 0/1.",
        "expected_row_count": 2370784,
        "sql": "09_choice_product_store_week.sql",
        "scope": "processed",
        "sources": ("transaction_data", "causal_data"),
        "limitations": "Placement is not a treatment and not randomised. Raw code sets are "
        "preserved as sorted lists; no canonical exposure is chosen.",
    },
    "choice_household_product_week": {
        "title": "Household-product-week purchases, price candidates and prior history",
        "grain": "household_key x product_id x week_no",
        "key": ("household_key", "product_id", "week_no"),
        "columns": (
            ("household_key", "integer"),
            ("product_id", "integer"),
            ("week_no", "integer"),
            ("brand", "text"),
            ("commodity_desc", "text"),
            ("purchase_lines", "integer"),
            ("units", "integer"),
            ("sales_value", "double"),
            ("retail_disc", "double"),
            ("coupon_disc", "double"),
            ("coupon_match_disc", "double"),
            ("distinct_stores", "integer"),
            ("prior_purchase_weeks", "integer"),
            ("prior_purchase_lines", "integer"),
            ("prior_units", "integer"),
            ("weeks_since_prior_product_purchase", "integer"),
            ("prior_brand_purchase_lines", "integer"),
            ("weeks_since_prior_brand_purchase", "integer"),
            ("prior_commodity_purchase_lines", "integer"),
            ("weeks_since_prior_commodity_purchase", "integer"),
            ("previously_purchased_product", "integer"),
        ),
        "population": "SPARSE: observed household x product x week purchase cells "
        "(2,519,013). No dense household x product-grid is built.",
        "time_support": "Weeks 1-102.",
        "zero_semantics": "prior_* counters are 0 when there is no prior purchase; "
        "weeks_since_prior_* is NULL there.",
        "missing_semantics": "weeks_since_prior_* NULL = no prior purchase. Price "
        "candidates are NOT here (see choice_line_price_candidates).",
        "expected_row_count": 2519013,
        "sql": "10_choice_household_product_week.sql",
        "scope": "processed",
        "sources": ("transaction_data", "product"),
        "limitations": "No price columns: the vendor formulas are exact only at line "
        "grain. No commodity is selected and no choice set is built.",
    },
    "choice_line_price_candidates": {
        "title": "Transaction-line vendor unit-price candidates (exact Phase 1 formulas)",
        "grain": "household_key x basket_id x product_id",
        "key": ("household_key", "basket_id", "product_id"),
        "columns": (
            ("household_key", "integer"),
            ("basket_id", "integer"),
            ("product_id", "integer"),
            ("store_id", "integer"),
            ("day", "integer"),
            ("week_no", "integer"),
            ("quantity", "integer"),
            ("sales_value", "double"),
            ("retail_disc", "double"),
            ("coupon_disc", "double"),
            ("coupon_match_disc", "double"),
            ("vendor_loyalty_unit_price_candidate", "double"),
            ("vendor_nonloyalty_unit_price_candidate", "double"),
        ),
        "population": "Positive-quantity transaction lines only (2,581,266). The "
        "14,466 QUANTITY = 0 lines are excluded because the formula is undefined "
        "there; they remain in the canonical tables.",
        "time_support": "Days 1-711 / weeks 1-102.",
        "zero_semantics": "Not applicable: QUANTITY > 0 by construction, so no "
        "division by zero and no NULL candidates.",
        "missing_semantics": "No nulls.",
        "expected_row_count": 2581266,
        "sql": "11_choice_line_price_candidates.sql",
        "scope": "processed",
        "sources": ("transaction_data",),
        "limitations": "CANDIDATE measures only, never the chosen economic price; no "
        "column is named 'price'. Not aggregated to any occasion grain.",
    },
}


#: Phase 2C predictive feature infrastructure. Local, observation-level.
PHASE2C_TABLES: Final[dict[str, dict[str, object]]] = {
    "predictive_household_week_features": {
        "title": "Point-in-time household-week predictive feature store",
        "grain": "household_key x as_of_week",
        "key": ("household_key", "as_of_week"),
        "columns": (
            ("household_key", "integer"),
            ("as_of_week", "integer"),
            ("history_weeks_available", "integer"),
            ("current_retailer_purchase", "integer"),
            ("current_weekly_sales_value", "double"),
            ("current_trips", "integer"),
            ("current_units", "integer"),
            ("current_distinct_products", "integer"),
            ("current_distinct_commodities", "integer"),
            ("current_active_store_count", "integer"),
            ("lag1_retailer_purchase", "integer"),
            ("lag2_retailer_purchase", "integer"),
            ("lag4_retailer_purchase", "integer"),
            ("lag1_weekly_sales_value", "double"),
            ("lag2_weekly_sales_value", "double"),
            ("lag4_weekly_sales_value", "double"),
            ("lag1_trips", "integer"),
            ("lag2_trips", "integer"),
            ("lag4_trips", "integer"),
            ("lag1_units", "integer"),
            ("lag2_units", "integer"),
            ("lag4_units", "integer"),
            ("active_weeks_last4", "integer"),
            ("active_weeks_last8", "integer"),
            ("active_weeks_last12", "integer"),
            ("active_share_last4", "double"),
            ("active_share_last8", "double"),
            ("active_share_last12", "double"),
            ("sales_value_last4", "double"),
            ("sales_value_last8", "double"),
            ("sales_value_last12", "double"),
            ("mean_weekly_sales_value_last4", "double"),
            ("mean_weekly_sales_value_last8", "double"),
            ("mean_weekly_sales_value_last12", "double"),
            ("trips_last4", "integer"),
            ("trips_last8", "integer"),
            ("trips_last12", "integer"),
            ("mean_trips_last4", "double"),
            ("mean_trips_last8", "double"),
            ("mean_trips_last12", "double"),
            ("units_last4", "integer"),
            ("units_last8", "integer"),
            ("units_last12", "integer"),
            ("mean_units_last4", "double"),
            ("mean_units_last8", "double"),
            ("mean_units_last12", "double"),
            ("distinct_commodities_last4", "integer"),
            ("distinct_commodities_last8", "integer"),
            ("distinct_commodities_last12", "integer"),
            ("distinct_commodities_to_date", "integer"),
            ("ever_purchased_as_of_week", "integer"),
            ("weeks_since_last_retailer_purchase", "integer"),
            ("active_weeks_to_date", "integer"),
            ("active_share_to_date", "double"),
            ("trips_to_date", "integer"),
            ("sales_value_to_date", "double"),
            ("units_to_date", "integer"),
            ("retail_disc_magnitude_last4", "double"),
            ("retail_disc_magnitude_last8", "double"),
            ("retail_disc_magnitude_last12", "double"),
            ("coupon_disc_magnitude_last4", "double"),
            ("coupon_disc_magnitude_last8", "double"),
            ("coupon_disc_magnitude_last12", "double"),
            ("coupon_match_disc_magnitude_last4", "double"),
            ("coupon_match_disc_magnitude_last8", "double"),
            ("coupon_match_disc_magnitude_last12", "double"),
            ("total_discount_magnitude_last4", "double"),
            ("total_discount_magnitude_last8", "double"),
            ("total_discount_magnitude_last12", "double"),
            ("listed_campaign_active_count_current", "integer"),
            ("listed_campaign_weeks_to_date", "integer"),
            ("distinct_listed_campaigns_to_date", "integer"),
            ("coupon_redemption_rows_current_week", "integer"),
            ("coupon_redemption_rows_prior_4w", "integer"),
            ("coupon_redemption_rows_prior_8w", "integer"),
            ("coupon_redemption_rows_prior_12w", "integer"),
            ("coupon_redemption_rows_to_date", "integer"),
            ("has_demographics", "integer"),
            ("demographic_classification_1", "text"),
            ("demographic_classification_2", "text"),
            ("demographic_classification_3", "text"),
            ("demographic_classification_4", "text"),
            ("demographic_classification_5", "text"),
            ("demographic_homeowner_desc", "text"),
            ("demographic_kid_category_desc", "text"),
        ),
        "population": "All 2,500 transaction-universe households x all 102 observed "
        "weeks = 255,000 rows. Not restricted to S1/S2 and not restricted to the "
        "801 demographic households.",
        "time_support": "as_of_week = t means information through the END of week t. "
        "Features use weeks <= t only; an eventual target must be strictly after t.",
        "zero_semantics": "A historical retailer-nonpurchase week is a genuine 0 in "
        "the retailer-observation grid and enters rolling windows as a zero.",
        "missing_semantics": "NULL is family-specific: lag unavailable before week 1; "
        "weeks_since_last_retailer_purchase NULL = never purchased yet (see "
        "ever_purchased_as_of_week); demographic fields NULL = unavailable. NULLs are "
        "never filled with zero in this table.",
        "expected_row_count": 255000,
        "sql": "12_predictive_household_week_features.sql",
        "scope": "processed",
        "sources": (
            "household_week_base",
            "household_category_week",
            "campaign_household_week",
            "coupon_redempt",
            "hh_demographic",
        ),
        "limitations": "Feature infrastructure only: no target, no split, no model, "
        "no eligibility rule. causal_data placement is deliberately excluded.",
    },
}

#: Family metadata for the Phase 2C feature inventory (docs + machine-readable).
PREDICTIVE_FAMILY_METADATA: Final[dict[str, dict[str, str]]] = {
    "identity": {
        "source_table": "-",
        "available_through": "week <= t",
        "zero_semantics": "n/a",
        "null_semantics": "never NULL",
        "future_target_safe": "yes",
        "notes": "keys and time index",
    },
    "current_week_state": {
        "source_table": "household_week_base",
        "available_through": "week <= t (t included)",
        "zero_semantics": "0 = no recorded purchase that week",
        "null_semantics": "never NULL",
        "future_target_safe": "yes",
        "notes": "target must begin at t+1",
    },
    "lag": {
        "source_table": "household_week_base",
        "available_through": "week t-k, k>=1",
        "zero_semantics": "0 = a genuine retailer-nonpurchase week at t-k",
        "null_semantics": "NULL when t-k < 1 (unavailable before sample start)",
        "future_target_safe": "yes",
        "notes": "not backfilled",
    },
    "rolling_frequency": {
        "source_table": "household_week_base",
        "available_through": "weeks max(1,t-W+1)..t",
        "zero_semantics": "over the FULL window including non-purchase weeks",
        "null_semantics": "never NULL; early weeks use the shorter available window",
        "future_target_safe": "yes",
        "notes": "mean/share denominator = weeks available in window",
    },
    "rolling_monetary": {
        "source_table": "household_week_base",
        "available_through": "weeks max(1,t-W+1)..t",
        "zero_semantics": "non-purchase weeks contribute 0 sales",
        "null_semantics": "never NULL; early weeks use the shorter available window",
        "future_target_safe": "yes",
        "notes": "SALES_VALUE = retailer revenue",
    },
    "rolling_diversity": {
        "source_table": "household_category_week",
        "available_through": "weeks max(1,t-W+1)..t",
        "zero_semantics": "0 distinct commodities if no purchase in window",
        "null_semantics": "never NULL",
        "future_target_safe": "yes",
        "notes": "distinct commodity_desc, sparse events only",
    },
    "cumulative_to_date": {
        "source_table": "household_week_base",
        "available_through": "weeks <= t",
        "zero_semantics": "0 before any purchase",
        "null_semantics": "never NULL",
        "future_target_safe": "yes",
        "notes": "to_date = through the as-of week inclusive",
    },
    "recency": {
        "source_table": "household_week_base",
        "available_through": "weeks <= t",
        "zero_semantics": "n/a for recency; ever_purchased_as_of_week is 0/1",
        "null_semantics": "weeks_since_last_retailer_purchase NULL = never purchased yet",
        "future_target_safe": "yes",
        "notes": "'never' is NULL plus an explicit indicator, never a sentinel number",
    },
    "category_history": {
        "source_table": "household_category_week",
        "available_through": "weeks <= t",
        "zero_semantics": "0 = no commodity purchased in the window/to date",
        "null_semantics": "never NULL",
        "future_target_safe": "yes",
        "notes": "no target commodity chosen; no 308-way dummies",
    },
    "discount_history": {
        "source_table": "household_week_base",
        "available_through": "weeks max(1,t-W+1)..t",
        "zero_semantics": "0 = no recorded discount in the window",
        "null_semantics": "never NULL",
        "future_target_safe": "yes",
        "notes": "magnitudes only; NO normalized share (P2-003); not exogenous treatment",
    },
    "campaign_listing_history": {
        "source_table": "campaign_household_week",
        "available_through": "weeks <= t",
        "zero_semantics": "0 = no listed campaign",
        "null_semantics": "never NULL",
        "future_target_safe": "yes",
        "notes": "FACTUAL listing only; never treatment/exposure/control",
    },
    "coupon_redemption_history": {
        "source_table": "coupon_redempt",
        "available_through": "weeks <= t",
        "zero_semantics": "0 = no recorded redemption",
        "null_semantics": "never NULL",
        "future_target_safe": "yes",
        "notes": "actual redemptions only; no pool-based assignment or take-up rate",
    },
    "demographics": {
        "source_table": "hh_demographic",
        "available_through": "time-invariant attribute",
        "zero_semantics": "n/a",
        "null_semantics": "NULL = demographics unavailable for this household (optional attribute, not missing data)",
        "future_target_safe": "yes",
        "notes": "original categories preserved as text; no ordinal/one-hot encoding; rows never restricted",
    },
}

#: Ordered feature-inventory columns.
PREDICTIVE_INVENTORY_COLUMNS: Final[tuple[str, ...]] = (
    "feature_name",
    "feature_family",
    "source_table",
    "window_or_lag",
    "available_through",
    "zero_semantics",
    "null_semantics",
    "future_target_safe",
    "notes",
)

_ROLLING_WINDOWS: Final[tuple[tuple[str, str], ...]] = (
    ("_last4", "weeks max(1,t-3)..t"),
    ("_last8", "weeks max(1,t-7)..t"),
    ("_last12", "weeks max(1,t-11)..t"),
)
_ROLLING_FAMILY_BY_STEM: Final[dict[str, str]] = {
    "active_weeks": "rolling_frequency",
    "active_share": "rolling_frequency",
    "trips": "rolling_frequency",
    "mean_trips": "rolling_frequency",
    "units": "rolling_frequency",
    "mean_units": "rolling_frequency",
    "sales_value": "rolling_monetary",
    "mean_weekly_sales_value": "rolling_monetary",
    "distinct_commodities": "rolling_diversity",
    "retail_disc_magnitude": "discount_history",
    "coupon_disc_magnitude": "discount_history",
    "coupon_match_disc_magnitude": "discount_history",
    "total_discount_magnitude": "discount_history",
}
_CUMULATIVE_FAMILY_BY_STEM: Final[dict[str, str]] = {
    "active_weeks": "cumulative_to_date",
    "active_share": "cumulative_to_date",
    "trips": "cumulative_to_date",
    "sales_value": "cumulative_to_date",
    "units": "cumulative_to_date",
    "distinct_commodities": "category_history",
    "listed_campaign_weeks": "campaign_listing_history",
    "distinct_listed_campaigns": "campaign_listing_history",
    "coupon_redemption_rows": "coupon_redemption_history",
}


def classify_predictive_feature(column: str) -> tuple[str, str]:
    """Return ``(feature_family, window_or_lag)`` for a Phase 2C feature column.

    Raises:
        ValueError: for an unrecognised column, so a new feature cannot silently
            enter the table without being classified in the inventory.
    """
    if column in ("household_key",):
        return "identity", "none"
    if column in ("as_of_week", "history_weeks_available"):
        return "identity", "as_of t"
    # explicit prefixes that would otherwise collide with the generic rules
    if column.startswith("coupon_redemption_rows_"):
        if column.endswith("_current_week"):
            return "coupon_redemption_history", "current week t"
        if column.endswith("_to_date"):
            return "coupon_redemption_history", "weeks <= t"
        m = re.match(r"coupon_redemption_rows_prior_(\d+)w$", column)
        if m:
            k = int(m.group(1))
            return "coupon_redemption_history", f"weeks [t-{k}, t-1] (strictly prior)"
    if column == "listed_campaign_active_count_current":
        return "campaign_listing_history", "current week t"
    if column == "has_demographics" or column.startswith("demographic_"):
        return "demographics", "time-invariant"
    if column in ("ever_purchased_as_of_week", "weeks_since_last_retailer_purchase"):
        return "recency", "weeks <= t"
    if column.startswith("current_"):
        return "current_week_state", "current week t"
    m = re.match(r"lag(\d+)_(.+)$", column)
    if m:
        return "lag", f"lag {m.group(1)}"
    for suffix, window in _ROLLING_WINDOWS:
        if column.endswith(suffix):
            stem = column[: -len(suffix)]
            if stem in _ROLLING_FAMILY_BY_STEM:
                return _ROLLING_FAMILY_BY_STEM[stem], window
    if column.endswith("_to_date"):
        stem = column[: -len("_to_date")]
        if stem in _CUMULATIVE_FAMILY_BY_STEM:
            return _CUMULATIVE_FAMILY_BY_STEM[stem], "weeks <= t"
    raise ValueError(f"unclassified Phase 2C feature: {column!r}")


def predictive_feature_inventory() -> tuple[tuple[str, ...], ...]:
    """Deterministic machine-readable inventory of the Phase 2C feature columns."""
    spec = PHASE2C_TABLES["predictive_household_week_features"]
    rows: list[tuple[str, ...]] = []
    for name, _family in spec["columns"]:  # type: ignore[union-attr]
        family, window = classify_predictive_feature(str(name))
        meta = PREDICTIVE_FAMILY_METADATA[family]
        rows.append(
            (
                str(name),
                family,
                meta["source_table"],
                window,
                meta["available_through"],
                meta["zero_semantics"],
                meta["null_semantics"],
                meta["future_target_safe"],
                meta["notes"],
            )
        )
    return tuple(rows)


#: QA check SQL files in sql/qa/ (executed after all tables exist).
#: Each emits the uniform columns in :data:`CHECK_COLUMNS`.
CHECK_SQL: Final[tuple[str, ...]] = (
    "01_basket_panel_checks",
    "02_household_week_base_checks",
    "03_household_category_week_checks",
    "04_household_history_checks",
    "05_panel_eligibility_checks",
    "15_econometrics_household_week_checks",
    "16_campaign_week_calendar_checks",
    "17_campaign_household_week_checks",
    "18_choice_product_store_week_checks",
    "19_choice_household_product_week_checks",
    "21_choice_line_price_candidates_checks",
    "22_predictive_household_week_features_checks",
)

#: Uniform QA check output schema.
CHECK_COLUMNS: Final[tuple[str, ...]] = (
    "table_name",
    "check_name",
    "check_class",
    "status",
    "observed",
    "expected",
)

#: Tracked, non-identifying aggregate summaries: sql stem -> CSV filename.
SUMMARY_SQL: Final[dict[str, str]] = {
    "06_eligibility_active_weeks_distribution": "panel_eligibility_active_weeks_distribution.csv",
    "07_eligibility_active_weeks_thresholds": "panel_eligibility_active_weeks_thresholds.csv",
    "08_eligibility_support_bucket_by_early_late": "panel_eligibility_support_bucket_by_early_late.csv",
    "09_eligibility_support_halves_crosstab": "panel_eligibility_support_halves_crosstab.csv",
    "10_eligibility_support_distributions": "panel_eligibility_support_distributions.csv",
    "11_eligibility_halves_by_active_weeks_thresholds": (
        "panel_eligibility_halves_by_active_weeks_thresholds.csv"
    ),
    "12_eligibility_thirds_by_active_weeks_thresholds": (
        "panel_eligibility_thirds_by_active_weeks_thresholds.csv"
    ),
    "13_eligibility_observed_span_by_active_weeks": (
        "panel_eligibility_observed_span_by_active_weeks.csv"
    ),
    "14_eligibility_gap_thresholds": "panel_eligibility_gap_thresholds.csv",
    "20_sample_flag_counts": "sample_flag_counts.csv",
}

#: Machine-readable lineage: (output_table, source_table, relationship, stage).
# fmt: off
LINEAGE: Final[tuple[tuple[str, str, str, str], ...]] = (
    (
        "basket_panel",
        "transaction_data",
        "aggregates receipt lines to household x basket",
        "derive",
    ),
    ("basket_panel", "product", "left-join for distinct_commodities (COMMODITY_DESC)", "enrich"),
    (
        "household_week_base",
        "transaction_data",
        "weekly aggregates over the full household x week grid",
        "derive",
    ),
    ("household_week_base", "product", "left-join for distinct_commodities", "enrich"),
    (
        "household_category_week",
        "transaction_data",
        "aggregates receipt lines by commodity x week",
        "derive",
    ),
    (
        "household_category_week",
        "product",
        "left-join for commodity_desc and BRAND splits",
        "enrich",
    ),
    ("household_history", "transaction_data", "full-sample household activity summary", "derive"),
    ("household_history", "product", "left-join for distinct_commodities", "enrich"),
    ("household_history", "hh_demographic", "has_demographics existence indicator", "enrich"),
    ("household_history", "campaign_table", "descriptive campaign recipient/count", "enrich"),
    ("household_history", "coupon_redempt", "descriptive coupon redeemer/count", "enrich"),
    (
        "panel_eligibility",
        "household_history",
        "reuses household-level activity diagnostics",
        "derive",
    ),
    ("panel_eligibility", "transaction_data", "trips and sales totals", "derive"),
    ("panel_eligibility", "hh_demographic", "demographics availability indicator", "enrich"),
    ("panel_eligibility", "campaign_table", "campaign coverage indicators", "enrich"),
    ("panel_eligibility", "coupon_redempt", "coupon redeemer indicator", "enrich"),
    ("econometrics_household_week", "household_week_base", "weekly outcomes and QA flags preserved", "derive"),
    ("econometrics_household_week", "panel_eligibility", "supplies S1/S2 sample flags (S0 flags all rows)", "flags"),
    ("econometrics_household_week", "campaign_desc", "nominal campaign week windows (P1-009 convention)", "enrich"),
    ("econometrics_household_week", "campaign_table", "factual listed-campaign weekly state", "enrich"),
    ("campaign_week_calendar", "campaign_desc", "vendor start/end day mapped to nominal and clipped weeks", "derive"),
    ("campaign_week_calendar", "transaction_data", "observed week domain (weeks 1-102)", "scope"),
    ("campaign_household_week", "campaign_table", "listed (household, campaign) pairs", "scope"),
    ("campaign_household_week", "campaign_week_calendar", "week span, relative week and period label", "derive"),
    ("campaign_household_week", "household_week_base", "household weekly retailer activity", "enrich"),
    ("campaign_household_week", "coupon_redempt", "factual coupon redemption rows per campaign week", "enrich"),
    ("choice_product_store_week", "transaction_data", "purchased product-store-week facts", "derive"),
    ("choice_product_store_week", "causal_data", "recorded placement code metadata and conflict flags", "enrich"),
    ("choice_household_product_week", "transaction_data", "observed household x product x week purchase facts", "derive"),
    ("choice_household_product_week", "product", "brand, commodity and price-candidate inputs", "enrich"),
    ("choice_line_price_candidates", "transaction_data", "exact Phase 1 vendor unit-price candidates at line grain", "derive"),
    ("predictive_household_week_features", "household_week_base", "current week state, lags, rolling, recency, frequency and monetary features through t", "derive"),
    ("predictive_household_week_features", "household_category_week", "distinct-commodity rolling and cumulative history through t", "derive"),
    ("predictive_household_week_features", "campaign_household_week", "factual listed-campaign history through t", "enrich"),
    ("predictive_household_week_features", "coupon_redempt", "factual redemption history through t", "enrich"),
    ("predictive_household_week_features", "hh_demographic", "time-invariant demographic attributes (never restricts rows)", "enrich"),
)
# fmt: on
