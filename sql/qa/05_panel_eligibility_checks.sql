-- Phase 2A QA: panel_eligibility (LOCAL-ONLY household-level diagnostic)
-- Uniform check output: table_name, check_name, check_class, status, observed, expected
-- NOTE: this table supplies evidence only. QA asserts there is NO selection flag
-- and that the diagnostics are internally coherent and consistent with the
-- canonical tables -- it never asserts a chosen population.

WITH
pe AS (
    SELECT
        count(*)                                              AS n_rows,
        count(DISTINCT household_key)                         AS n_keys,
        sum(active_weeks)                                     AS active_sum,
        sum(total_trips)                                      AS trips_sum,
        sum(total_sales_value)                                AS sales_sum,
        count(*) FILTER (
            WHERE active_weeks_block_1 + active_weeks_block_2 + active_weeks_block_3 + active_weeks_block_4
                  <> active_weeks
        )                                                     AS block_violations,
        count(*) FILTER (
            WHERE active_weeks_early + active_weeks_middle + active_weeks_late <> active_weeks
        )                                                     AS third_violations,
        count(*) FILTER (
            WHERE abs(active_share_102 - CAST(active_weeks AS DOUBLE) / tw.total_weeks) > 1e-12
        )                                                     AS share_violations
    FROM panel_eligibility, (SELECT count(DISTINCT week_no) AS total_weeks FROM household_week_base) tw
    GROUP BY tw.total_weeks
),
flag_like_columns AS (
    SELECT count(*) AS n
    FROM information_schema.columns
    WHERE table_name = 'panel_eligibility'
      AND regexp_matches(lower(column_name), 'eligib|selected|include|exclude|rank|recommend|score|flag')
),
hist_mismatch AS (
    SELECT count(*) AS n
    FROM panel_eligibility pe JOIN household_history hh ON hh.household_key = pe.household_key
    WHERE pe.active_weeks <> hh.active_weeks
       OR pe.first_purchase_week <> hh.first_purchase_week
       OR pe.last_purchase_week <> hh.last_purchase_week
       OR pe.total_trips <> hh.total_baskets
       OR abs(pe.total_sales_value - hh.total_sales_value) > greatest(1e-6, abs(hh.total_sales_value) * 1e-9)
),
src AS (SELECT count(DISTINCT household_key) AS n FROM transaction_data)

SELECT 'panel_eligibility' AS table_name, 'row_count_is_household_universe' AS check_name, 'HARD_INVARIANT' AS check_class,
       CASE WHEN pe.n_rows = src.n THEN 'PASS' ELSE 'FAIL' END AS status,
       CAST(pe.n_rows AS VARCHAR) AS observed, CAST(src.n AS VARCHAR) AS expected FROM pe, src
UNION ALL SELECT 'panel_eligibility','declared_key_unique','HARD_INVARIANT',
       CASE WHEN pe.n_rows = pe.n_keys THEN 'PASS' ELSE 'FAIL' END,
       CAST(pe.n_rows AS VARCHAR), CAST(pe.n_keys AS VARCHAR) FROM pe
UNION ALL SELECT 'panel_eligibility','no_selection_flag_columns','HARD_INVARIANT',
       CASE WHEN flag_like_columns.n = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(flag_like_columns.n AS VARCHAR), '0' FROM flag_like_columns
UNION ALL SELECT 'panel_eligibility','active_weeks_and_support_match_history','HARD_INVARIANT',
       CASE WHEN hist_mismatch.n = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(hist_mismatch.n AS VARCHAR), '0' FROM hist_mismatch
UNION ALL SELECT 'panel_eligibility','block_partition_complete','HARD_INVARIANT',
       CASE WHEN pe.block_violations = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(pe.block_violations AS VARCHAR), '0' FROM pe
UNION ALL SELECT 'panel_eligibility','third_partition_complete','HARD_INVARIANT',
       CASE WHEN pe.third_violations = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(pe.third_violations AS VARCHAR), '0' FROM pe
UNION ALL SELECT 'panel_eligibility','active_share_consistent','HARD_INVARIANT',
       CASE WHEN pe.share_violations = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(pe.share_violations AS VARCHAR), '0' FROM pe
