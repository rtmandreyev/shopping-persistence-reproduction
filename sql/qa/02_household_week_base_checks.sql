-- Phase 2A QA: household_week_base
-- Uniform check output: table_name, check_name, check_class, status, observed, expected
-- HARD_INVARIANT -> violation means the pipeline/code is wrong
-- EMPIRICAL_FINDING -> reported, never asserted

WITH
hwb AS (
    SELECT
        count(*)                                                        AS n_rows,
        count(DISTINCT household_key)                                   AS n_households,
        count(DISTINCT week_no)                                         AS n_weeks,
        count(DISTINCT (household_key, week_no))                        AS n_keys,
        min(week_no)                                                    AS min_week,
        max(week_no)                                                    AS max_week,
        sum(CASE WHEN retailer_purchase = 1 THEN 1 ELSE 0 END)          AS active_rows,
        sum(weekly_sales_value)                                         AS sales_sum,
        sum(trips)                                                      AS trips_sum,
        sum(units)                                                      AS units_sum,
        sum(CASE WHEN retailer_purchase = 0
                   AND (trips <> 0 OR units <> 0 OR weekly_sales_value <> 0
                        OR distinct_products <> 0 OR distinct_commodities <> 0
                        OR active_store_count <> 0 OR transaction_line_count <> 0)
                 THEN 1 ELSE 0 END)                                     AS nonpurchase_with_activity,
        sum(CASE WHEN retailer_purchase = 1
                   AND (transaction_line_count < 1 OR trips < 1)
                 THEN 1 ELSE 0 END)                                     AS purchase_without_activity,
        sum(CASE WHEN retailer_purchase NOT IN (0, 1) THEN 1 ELSE 0 END) AS non_binary_flag,
        count(*) FILTER (WHERE household_key IS NULL OR week_no IS NULL) AS null_keys
    FROM household_week_base
),
per_hh AS (
    SELECT min(c) AS min_c, max(c) AS max_c FROM (
        SELECT household_key, count(*) AS c FROM household_week_base GROUP BY household_key
    )
),
src AS (
    SELECT
        count(DISTINCT household_key)                     AS n_households,
        count(DISTINCT WEEK_NO)                           AS n_weeks,
        min(WEEK_NO)                                      AS min_week,
        max(WEEK_NO)                                      AS max_week,
        count(DISTINCT (household_key, WEEK_NO))          AS active_cells,
        sum(SALES_VALUE)                                  AS sales_sum,
        count(DISTINCT (household_key, BASKET_ID))        AS trips_sum,
        sum(QUANTITY)                                     AS units_sum
    FROM transaction_data
)

SELECT 'household_week_base' AS table_name, 'row_count_is_full_grid' AS check_name, 'HARD_INVARIANT' AS check_class,
       CASE WHEN hwb.n_rows = (SELECT n_households FROM src) * (SELECT n_weeks FROM src) THEN 'PASS' ELSE 'FAIL' END AS status,
       CAST(hwb.n_rows AS VARCHAR) AS observed, CAST((SELECT n_households FROM src) * (SELECT n_weeks FROM src) AS VARCHAR) AS expected FROM hwb
UNION ALL SELECT 'household_week_base','distinct_households_match_source','HARD_INVARIANT',
       CASE WHEN hwb.n_households = src.n_households THEN 'PASS' ELSE 'FAIL' END,
       CAST(hwb.n_households AS VARCHAR), CAST(src.n_households AS VARCHAR) FROM hwb, src
UNION ALL SELECT 'household_week_base','distinct_weeks_match_source','HARD_INVARIANT',
       CASE WHEN hwb.n_weeks = src.n_weeks THEN 'PASS' ELSE 'FAIL' END,
       CAST(hwb.n_weeks AS VARCHAR), CAST(src.n_weeks AS VARCHAR) FROM hwb, src
UNION ALL SELECT 'household_week_base','week_domain_matches_source','HARD_INVARIANT',
       CASE WHEN hwb.min_week = src.min_week AND hwb.max_week = src.max_week THEN 'PASS' ELSE 'FAIL' END,
       CAST(hwb.min_week AS VARCHAR) || '..' || CAST(hwb.max_week AS VARCHAR),
       CAST(src.min_week AS VARCHAR) || '..' || CAST(src.max_week AS VARCHAR) FROM hwb, src
UNION ALL SELECT 'household_week_base','weeks_per_household_uniform','HARD_INVARIANT',
       CASE WHEN per_hh.min_c = per_hh.max_c AND per_hh.min_c = (SELECT n_weeks FROM src) THEN 'PASS' ELSE 'FAIL' END,
       CAST(per_hh.min_c AS VARCHAR) || '..' || CAST(per_hh.max_c AS VARCHAR), CAST((SELECT n_weeks FROM src) AS VARCHAR)
       FROM per_hh
UNION ALL SELECT 'household_week_base','declared_key_unique','HARD_INVARIANT',
       CASE WHEN hwb.n_rows = hwb.n_keys THEN 'PASS' ELSE 'FAIL' END,
       CAST(hwb.n_rows AS VARCHAR), CAST(hwb.n_keys AS VARCHAR) FROM hwb
UNION ALL SELECT 'household_week_base','no_null_keys','HARD_INVARIANT',
       CASE WHEN hwb.null_keys = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(hwb.null_keys AS VARCHAR), '0' FROM hwb
UNION ALL SELECT 'household_week_base','active_rows_reconcile_to_source_cells','HARD_INVARIANT',
       CASE WHEN hwb.active_rows = src.active_cells THEN 'PASS' ELSE 'FAIL' END,
       CAST(hwb.active_rows AS VARCHAR), CAST(src.active_cells AS VARCHAR) FROM hwb, src
UNION ALL SELECT 'household_week_base','sales_value_reconciles','HARD_INVARIANT',
       CASE WHEN abs(hwb.sales_sum - src.sales_sum) <= greatest(1e-6, abs(src.sales_sum) * 1e-9) THEN 'PASS' ELSE 'FAIL' END,
       CAST(hwb.sales_sum AS VARCHAR), CAST(src.sales_sum AS VARCHAR) FROM hwb, src
UNION ALL SELECT 'household_week_base','trips_reconcile','HARD_INVARIANT',
       CASE WHEN hwb.trips_sum = src.trips_sum THEN 'PASS' ELSE 'FAIL' END,
       CAST(hwb.trips_sum AS VARCHAR), CAST(src.trips_sum AS VARCHAR) FROM hwb, src
UNION ALL SELECT 'household_week_base','units_reconcile','HARD_INVARIANT',
       CASE WHEN hwb.units_sum = src.units_sum THEN 'PASS' ELSE 'FAIL' END,
       CAST(hwb.units_sum AS VARCHAR), CAST(src.units_sum AS VARCHAR) FROM hwb, src
UNION ALL SELECT 'household_week_base','nonpurchase_weeks_have_no_activity','HARD_INVARIANT',
       CASE WHEN hwb.nonpurchase_with_activity = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(hwb.nonpurchase_with_activity AS VARCHAR), '0' FROM hwb
UNION ALL SELECT 'household_week_base','purchase_weeks_have_activity','HARD_INVARIANT',
       CASE WHEN hwb.purchase_without_activity = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(hwb.purchase_without_activity AS VARCHAR), '0' FROM hwb
UNION ALL SELECT 'household_week_base','retailer_purchase_is_binary','HARD_INVARIANT',
       CASE WHEN hwb.non_binary_flag = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(hwb.non_binary_flag AS VARCHAR), '0' FROM hwb
UNION ALL SELECT 'household_week_base','empirical_observed_coverage_share','EMPIRICAL_FINDING','REPORT',
       CAST(CAST(hwb.active_rows AS DOUBLE) / hwb.n_rows AS VARCHAR), '' FROM hwb
