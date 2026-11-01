-- Phase 2A canonical table: household_history
-- Grain : household_key  (one row per household)
-- Key   : household_key -- unique
-- Population: every household that transacts (2,500). No household is excluded,
--   ranked, or filtered.
--
-- Factual full-sample activity summaries for understanding support. This is a
-- DESCRIPTIVE support summary, not an eligibility rule: it contains no
-- eligible/exclude flag and no threshold.
--
-- "Inactive" throughout means "no recorded purchase from THIS retailer in that
-- week". Leading/trailing inactivity is calendar truncation, NOT entry, exit, or
-- attrition, and must not be read as churn.
--
-- campaign_recipient_ever / campaign_count and coupon_* are descriptive
-- existence/count summaries only. They are NOT treatment variables and carry no
-- causal interpretation (Phase 1: campaign receipt is not documented as random).
--
-- Weeks are the source week domain (1..102), derived, not hardcoded. Halves split
-- the 102-week grid into weeks 1-51 and 52-102. product is LEFT-joined so an
-- unmatched product cannot drop a transaction line; QA asserts zero unmatched.

WITH
grid AS (
    SELECT DISTINCT WEEK_NO AS week_no FROM transaction_data
),
grid_stats AS (
    SELECT count(*) AS total_weeks, max(week_no) AS last_week FROM grid
),
active AS (
    SELECT DISTINCT household_key, WEEK_NO AS week_no FROM transaction_data
),
hh_activity AS (
    SELECT
        household_key,
        count(*)                                                    AS active_weeks,
        min(week_no)                                                AS first_purchase_week,
        max(week_no)                                                AS last_purchase_week,
        sum(CASE WHEN week_no <= 51 THEN 1 ELSE 0 END)              AS active_weeks_first_half,
        sum(CASE WHEN week_no >= 52 THEN 1 ELSE 0 END)              AS active_weeks_second_half
    FROM active
    GROUP BY household_key
),
gaps AS (
    SELECT household_key, week_no - lag(week_no) OVER (PARTITION BY household_key ORDER BY week_no) - 1 AS gap
    FROM active
),
max_gap AS (
    SELECT household_key, coalesce(max(gap), 0) AS longest_gap FROM gaps GROUP BY household_key
),
txn AS (
    SELECT
        t.household_key,
        min(t.DAY)                        AS first_purchase_day,
        max(t.DAY)                        AS last_purchase_day,
        count(DISTINCT t.BASKET_ID)       AS total_baskets,
        sum(t.SALES_VALUE)                AS total_sales_value,
        sum(t.QUANTITY)                   AS total_units,
        count(DISTINCT t.PRODUCT_ID)      AS distinct_products,
        count(DISTINCT p.COMMODITY_DESC)  AS distinct_commodities,
        count(DISTINCT t.STORE_ID)        AS distinct_stores
    FROM transaction_data AS t
    LEFT JOIN product AS p ON p.PRODUCT_ID = t.PRODUCT_ID
    GROUP BY t.household_key
),
demo AS (
    SELECT DISTINCT household_key FROM hh_demographic
),
camp AS (
    SELECT household_key, count(*) AS campaign_count FROM campaign_table GROUP BY household_key
),
redeem AS (
    SELECT household_key, count(*) AS coupon_redemption_count FROM coupon_redempt GROUP BY household_key
)
SELECT
    h.household_key,
    txn.first_purchase_day,
    txn.last_purchase_day,
    h.first_purchase_week,
    h.last_purchase_week,
    h.active_weeks,
    gs.total_weeks - h.active_weeks                        AS inactive_grid_weeks,
    h.active_weeks_first_half,
    h.active_weeks_second_half,
    txn.total_baskets,
    txn.total_sales_value,
    txn.total_units,
    txn.distinct_products,
    txn.distinct_commodities,
    txn.distinct_stores,
    mg.longest_gap                                         AS longest_internal_no_purchase_week_gap,
    h.first_purchase_week - 1                              AS leading_no_purchase_weeks,
    gs.last_week - h.last_purchase_week                    AS trailing_no_purchase_weeks,
    CASE WHEN demo.household_key IS NULL THEN 0 ELSE 1 END AS has_demographics,
    CASE WHEN camp.household_key IS NULL THEN 0 ELSE 1 END AS campaign_recipient_ever,
    COALESCE(camp.campaign_count, 0)                       AS campaign_count,
    CASE WHEN redeem.household_key IS NULL THEN 0 ELSE 1 END AS coupon_redeemer_ever,
    COALESCE(redeem.coupon_redemption_count, 0)            AS coupon_redemption_count
FROM hh_activity AS h
JOIN txn      ON txn.household_key = h.household_key
JOIN max_gap AS mg ON mg.household_key = h.household_key
CROSS JOIN grid_stats AS gs
LEFT JOIN demo   ON demo.household_key = h.household_key
LEFT JOIN camp   ON camp.household_key = h.household_key
LEFT JOIN redeem ON redeem.household_key = h.household_key
ORDER BY h.household_key
