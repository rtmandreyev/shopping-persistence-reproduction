-- Phase 2A LOCAL-ONLY household-level diagnostic: panel_eligibility
-- Grain : household_key  (one row per household)
-- Key   : household_key -- unique
-- OUTPUT: data/processed/private/panel_eligibility.parquet  (GIT-IGNORED, customer-level)
--
-- THIS TABLE DOES NOT DETERMINE ELIGIBILITY. It supplies evidence for the project
-- owner to choose a primary econometric population AFTER Phase 2A. It contains:
--   * NO eligible_primary_panel flag
--   * NO thresholds
--   * NO ranking of households as good/bad
--   * NO exclusions
--
-- "Inactive" means "no recorded purchase from THIS retailer in that week":
-- unobserved, not zero consumption and not attrition. Leading/trailing inactivity
-- is calendar truncation.
--
-- Block definitions are explicit grid partitions of the 102-week domain and are
-- diagnostic buckets only, not analysis windows:
--   halves  : weeks 1-51 / 52-102
--   blocks  : 1-26 / 27-51 / 52-77 / 78-102   (four contiguous grid blocks)
--   thirds  : 1-34 / 35-68 / 69-102           (early / middle / late)
--
-- campaign_* / coupon_* are descriptive coverage indicators only, never treatment.

WITH
grid AS (
    SELECT DISTINCT WEEK_NO AS week_no FROM transaction_data
),
grid_stats AS (
    SELECT count(*) AS total_weeks FROM grid
),
active AS (
    SELECT DISTINCT household_key, WEEK_NO AS week_no FROM transaction_data
),
hh_activity AS (
    SELECT
        household_key,
        count(*)                                                          AS active_weeks,
        min(week_no)                                                      AS first_purchase_week,
        max(week_no)                                                      AS last_purchase_week,
        sum(CASE WHEN week_no <= 51 THEN 1 ELSE 0 END)                    AS active_weeks_first_half,
        sum(CASE WHEN week_no >= 52 THEN 1 ELSE 0 END)                    AS active_weeks_second_half,
        sum(CASE WHEN week_no BETWEEN 1 AND 26 THEN 1 ELSE 0 END)         AS active_weeks_block_1,
        sum(CASE WHEN week_no BETWEEN 27 AND 51 THEN 1 ELSE 0 END)        AS active_weeks_block_2,
        sum(CASE WHEN week_no BETWEEN 52 AND 77 THEN 1 ELSE 0 END)        AS active_weeks_block_3,
        sum(CASE WHEN week_no BETWEEN 78 AND 102 THEN 1 ELSE 0 END)       AS active_weeks_block_4,
        sum(CASE WHEN week_no <= 34 THEN 1 ELSE 0 END)                    AS active_weeks_early,
        sum(CASE WHEN week_no BETWEEN 35 AND 68 THEN 1 ELSE 0 END)        AS active_weeks_middle,
        sum(CASE WHEN week_no >= 69 THEN 1 ELSE 0 END)                    AS active_weeks_late
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
        household_key,
        count(DISTINCT BASKET_ID) AS total_trips,
        sum(SALES_VALUE)          AS total_sales_value
    FROM transaction_data
    GROUP BY household_key
),
demo AS (
    SELECT DISTINCT household_key FROM hh_demographic
),
camp AS (
    SELECT household_key, count(*) AS campaign_count FROM campaign_table GROUP BY household_key
),
redeem AS (
    SELECT DISTINCT household_key FROM coupon_redempt
)
SELECT
    h.household_key,
    h.active_weeks,
    CAST(h.active_weeks AS DOUBLE) / gs.total_weeks      AS active_share_102,
    h.first_purchase_week,
    h.last_purchase_week,
    h.active_weeks_first_half,
    h.active_weeks_second_half,
    h.active_weeks_block_1,
    h.active_weeks_block_2,
    h.active_weeks_block_3,
    h.active_weeks_block_4,
    h.active_weeks_early,
    h.active_weeks_middle,
    h.active_weeks_late,
    mg.longest_gap                                       AS longest_internal_no_purchase_week_gap,
    h.first_purchase_week - 1                            AS leading_no_purchase_weeks,
    gs.total_weeks - h.last_purchase_week                AS trailing_no_purchase_weeks,
    txn.total_trips,
    txn.total_sales_value,
    CASE WHEN demo.household_key IS NULL THEN 0 ELSE 1 END   AS has_demographics,
    CASE WHEN camp.household_key IS NULL THEN 0 ELSE 1 END   AS campaign_recipient_ever,
    COALESCE(camp.campaign_count, 0)                         AS campaign_count,
    CASE WHEN redeem.household_key IS NULL THEN 0 ELSE 1 END AS coupon_redeemer_ever
FROM hh_activity AS h
JOIN txn     ON txn.household_key = h.household_key
JOIN max_gap AS mg ON mg.household_key = h.household_key
CROSS JOIN grid_stats AS gs
LEFT JOIN demo   ON demo.household_key = h.household_key
LEFT JOIN camp   ON camp.household_key = h.household_key
LEFT JOIN redeem ON redeem.household_key = h.household_key
ORDER BY h.household_key
