-- Phase 2A QA: household_history
-- Uniform check output: table_name, check_name, check_class, status, observed, expected

WITH
hh AS (
    SELECT
        count(*)                                          AS n_rows,
        count(DISTINCT household_key)                     AS n_keys,
        sum(active_weeks)                                 AS active_sum,
        sum(total_baskets)                                AS baskets_sum,
        sum(total_sales_value)                            AS sales_sum,
        sum(total_units)                                  AS units_sum,
        sum(CASE WHEN has_demographics = 1 THEN 1 ELSE 0 END)      AS n_demo,
        sum(CASE WHEN campaign_recipient_ever = 1 THEN 1 ELSE 0 END) AS n_camp,
        sum(CASE WHEN coupon_redeemer_ever = 1 THEN 1 ELSE 0 END)  AS n_redeem
    FROM household_history
),
hwb_active AS (
    SELECT household_key, min(week_no) AS first_w, max(week_no) AS last_w, count(*) AS n_active
    FROM household_week_base WHERE retailer_purchase = 1 GROUP BY household_key
),
grid_stats AS (
    SELECT count(DISTINCT week_no) AS total_weeks FROM household_week_base
),
support_mismatch AS (
    SELECT count(*) AS n
    FROM household_history h JOIN hwb_active a ON a.household_key = h.household_key
    WHERE h.first_purchase_week <> a.first_w
       OR h.last_purchase_week <> a.last_w
       OR h.active_weeks <> a.n_active
),
gap_violations AS (
    SELECT count(*) AS n
    FROM household_history, grid_stats
    WHERE leading_no_purchase_weeks <> first_purchase_week - 1
       OR trailing_no_purchase_weeks <> grid_stats.total_weeks - last_purchase_week
       OR inactive_grid_weeks <> grid_stats.total_weeks - active_weeks
       OR active_weeks_first_half + active_weeks_second_half <> active_weeks
),
src AS (
    SELECT
        count(DISTINCT household_key)              AS n_households,
        count(DISTINCT (household_key, BASKET_ID)) AS baskets,
        sum(SALES_VALUE)                           AS sales_sum,
        sum(QUANTITY)                              AS units_sum
    FROM transaction_data
),
src_demo AS (SELECT count(*) AS n FROM hh_demographic),
src_camp AS (SELECT count(DISTINCT household_key) AS n FROM campaign_table),
src_redeem AS (SELECT count(DISTINCT household_key) AS n FROM coupon_redempt)

SELECT 'household_history' AS table_name, 'row_count_is_household_universe' AS check_name, 'HARD_INVARIANT' AS check_class,
       CASE WHEN hh.n_rows = src.n_households THEN 'PASS' ELSE 'FAIL' END AS status,
       CAST(hh.n_rows AS VARCHAR) AS observed, CAST(src.n_households AS VARCHAR) AS expected FROM hh, src
UNION ALL SELECT 'household_history','declared_key_unique','HARD_INVARIANT',
       CASE WHEN hh.n_rows = hh.n_keys THEN 'PASS' ELSE 'FAIL' END,
       CAST(hh.n_rows AS VARCHAR), CAST(hh.n_keys AS VARCHAR) FROM hh
UNION ALL SELECT 'household_history','active_weeks_reconcile_to_household_week_base','HARD_INVARIANT',
       CASE WHEN hh.active_sum = (SELECT count(*) FROM household_week_base WHERE retailer_purchase = 1)
            THEN 'PASS' ELSE 'FAIL' END,
       CAST(hh.active_sum AS VARCHAR),
       CAST((SELECT count(*) FROM household_week_base WHERE retailer_purchase = 1) AS VARCHAR) FROM hh
UNION ALL SELECT 'household_history','first_last_support_consistent','HARD_INVARIANT',
       CASE WHEN support_mismatch.n = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(support_mismatch.n AS VARCHAR), '0' FROM support_mismatch
UNION ALL SELECT 'household_history','gap_arithmetic_coherent','HARD_INVARIANT',
       CASE WHEN gap_violations.n = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(gap_violations.n AS VARCHAR), '0' FROM gap_violations
UNION ALL SELECT 'household_history','total_baskets_reconcile','HARD_INVARIANT',
       CASE WHEN hh.baskets_sum = src.baskets THEN 'PASS' ELSE 'FAIL' END,
       CAST(hh.baskets_sum AS VARCHAR), CAST(src.baskets AS VARCHAR) FROM hh, src
UNION ALL SELECT 'household_history','total_sales_value_reconciles','HARD_INVARIANT',
       CASE WHEN abs(hh.sales_sum - src.sales_sum) <= greatest(1e-6, abs(src.sales_sum) * 1e-9) THEN 'PASS' ELSE 'FAIL' END,
       CAST(hh.sales_sum AS VARCHAR), CAST(src.sales_sum AS VARCHAR) FROM hh, src
UNION ALL SELECT 'household_history','total_units_reconcile','HARD_INVARIANT',
       CASE WHEN hh.units_sum = src.units_sum THEN 'PASS' ELSE 'FAIL' END,
       CAST(hh.units_sum AS VARCHAR), CAST(src.units_sum AS VARCHAR) FROM hh, src
UNION ALL SELECT 'household_history','demographics_flag_matches_source','HARD_INVARIANT',
       CASE WHEN hh.n_demo = src_demo.n THEN 'PASS' ELSE 'FAIL' END,
       CAST(hh.n_demo AS VARCHAR), CAST(src_demo.n AS VARCHAR) FROM hh, src_demo
UNION ALL SELECT 'household_history','campaign_recipient_flag_matches_source','HARD_INVARIANT',
       CASE WHEN hh.n_camp = src_camp.n THEN 'PASS' ELSE 'FAIL' END,
       CAST(hh.n_camp AS VARCHAR), CAST(src_camp.n AS VARCHAR) FROM hh, src_camp
UNION ALL SELECT 'household_history','coupon_redeemer_flag_matches_source','HARD_INVARIANT',
       CASE WHEN hh.n_redeem = src_redeem.n THEN 'PASS' ELSE 'FAIL' END,
       CAST(hh.n_redeem AS VARCHAR), CAST(src_redeem.n AS VARCHAR) FROM hh, src_redeem
