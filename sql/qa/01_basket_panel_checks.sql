-- Phase 2A QA: basket_panel
-- Uniform check output: table_name, check_name, check_class, status, observed, expected
-- check_class HARD_INVARIANT -> violation means the pipeline/code is wrong (FAIL blocks the build)
-- check_class EMPIRICAL_FINDING -> reported, never asserted

WITH
bp AS (
    SELECT
        count(*)                          AS n_rows,
        count(DISTINCT (household_key, basket_id)) AS n_keys,
        sum(basket_sales_value)           AS sales_sum,
        sum(basket_units)                 AS units_sum,
        sum(transaction_line_count)       AS line_sum,
        min(transaction_line_count)       AS min_lines,
        sum(qa_quantity_zero_lines)            AS f_zero,
        sum(qa_retail_disc_positive_lines)     AS f_rd_pos,
        sum(qa_implied_paid_negative_lines)    AS f_neg_paid,
        sum(CASE WHEN qa_quantity_zero_lines > 0
                   OR qa_retail_disc_positive_lines > 0
                   OR qa_implied_paid_negative_lines > 0 THEN 1 ELSE 0 END) AS baskets_with_flags
    FROM basket_panel
),
src AS (
    SELECT
        count(*)                       AS n_lines,
        sum(SALES_VALUE)               AS sales_sum,
        sum(QUANTITY)                  AS units_sum,
        sum(CASE WHEN QUANTITY = 0 THEN 1 ELSE 0 END)                 AS f_zero,
        sum(CASE WHEN RETAIL_DISC > 0 THEN 1 ELSE 0 END)              AS f_rd_pos,
        sum(CASE WHEN SALES_VALUE + COUPON_DISC < 0 THEN 1 ELSE 0 END) AS f_neg_paid
    FROM transaction_data
),
coherence AS (
    SELECT
        sum(CASE WHEN n_day > 1 THEN 1 ELSE 0 END)   AS day_split,
        sum(CASE WHEN n_week > 1 THEN 1 ELSE 0 END)  AS week_split,
        sum(CASE WHEN n_store > 1 THEN 1 ELSE 0 END) AS store_split
    FROM (
        SELECT household_key, BASKET_ID,
               count(DISTINCT DAY) AS n_day,
               count(DISTINCT WEEK_NO) AS n_week,
               count(DISTINCT STORE_ID) AS n_store
        FROM transaction_data
        GROUP BY household_key, BASKET_ID
    )
),
basket_households AS (
    SELECT count(*) AS n FROM (
        SELECT BASKET_ID FROM transaction_data GROUP BY BASKET_ID HAVING count(DISTINCT household_key) > 1
    )
),
unmatched AS (
    SELECT count(*) AS n
    FROM transaction_data t LEFT JOIN product p ON p.PRODUCT_ID = t.PRODUCT_ID
    WHERE p.PRODUCT_ID IS NULL
),
line_bound AS (
    SELECT count(*) AS n FROM basket_panel WHERE distinct_products > transaction_line_count
)

SELECT 'basket_panel' AS table_name, 'declared_key_unique' AS check_name, 'HARD_INVARIANT' AS check_class,
       CASE WHEN bp.n_rows = bp.n_keys THEN 'PASS' ELSE 'FAIL' END AS status,
       CAST(bp.n_rows AS VARCHAR) AS observed, CAST(bp.n_keys AS VARCHAR) AS expected FROM bp
UNION ALL SELECT 'basket_panel','source_line_coverage_complete','HARD_INVARIANT',
       CASE WHEN bp.line_sum = src.n_lines THEN 'PASS' ELSE 'FAIL' END,
       CAST(bp.line_sum AS VARCHAR), CAST(src.n_lines AS VARCHAR) FROM bp, src
UNION ALL SELECT 'basket_panel','basket_row_nonempty','HARD_INVARIANT',
       CASE WHEN bp.min_lines >= 1 THEN 'PASS' ELSE 'FAIL' END,
       CAST(bp.min_lines AS VARCHAR), '1' FROM bp
UNION ALL SELECT 'basket_panel','sales_value_reconciles','HARD_INVARIANT',
       CASE WHEN abs(bp.sales_sum - src.sales_sum) <= greatest(1e-6, abs(src.sales_sum) * 1e-9) THEN 'PASS' ELSE 'FAIL' END,
       CAST(bp.sales_sum AS VARCHAR), CAST(src.sales_sum AS VARCHAR) FROM bp, src
UNION ALL SELECT 'basket_panel','units_reconcile','HARD_INVARIANT',
       CASE WHEN bp.units_sum = src.units_sum THEN 'PASS' ELSE 'FAIL' END,
       CAST(bp.units_sum AS VARCHAR), CAST(src.units_sum AS VARCHAR) FROM bp, src
UNION ALL SELECT 'basket_panel','basket_day_coherent','HARD_INVARIANT',
       CASE WHEN coherence.day_split = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(coherence.day_split AS VARCHAR), '0' FROM coherence
UNION ALL SELECT 'basket_panel','basket_week_coherent','HARD_INVARIANT',
       CASE WHEN coherence.week_split = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(coherence.week_split AS VARCHAR), '0' FROM coherence
UNION ALL SELECT 'basket_panel','basket_store_coherent','HARD_INVARIANT',
       CASE WHEN coherence.store_split = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(coherence.store_split AS VARCHAR), '0' FROM coherence
UNION ALL SELECT 'basket_panel','basket_household_unique','HARD_INVARIANT',
       CASE WHEN basket_households.n = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(basket_households.n AS VARCHAR), '0' FROM basket_households
UNION ALL SELECT 'basket_panel','products_fully_matched','HARD_INVARIANT',
       CASE WHEN unmatched.n = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(unmatched.n AS VARCHAR), '0' FROM unmatched
UNION ALL SELECT 'basket_panel','distinct_products_within_lines','HARD_INVARIANT',
       CASE WHEN line_bound.n = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(line_bound.n AS VARCHAR), '0' FROM line_bound
UNION ALL SELECT 'basket_panel','quantity_zero_flag_reconciles','HARD_INVARIANT',
       CASE WHEN bp.f_zero = src.f_zero THEN 'PASS' ELSE 'FAIL' END,
       CAST(bp.f_zero AS VARCHAR), CAST(src.f_zero AS VARCHAR) FROM bp, src
UNION ALL SELECT 'basket_panel','retail_disc_positive_flag_reconciles','HARD_INVARIANT',
       CASE WHEN bp.f_rd_pos = src.f_rd_pos THEN 'PASS' ELSE 'FAIL' END,
       CAST(bp.f_rd_pos AS VARCHAR), CAST(src.f_rd_pos AS VARCHAR) FROM bp, src
UNION ALL SELECT 'basket_panel','implied_paid_negative_flag_reconciles','HARD_INVARIANT',
       CASE WHEN bp.f_neg_paid = src.f_neg_paid THEN 'PASS' ELSE 'FAIL' END,
       CAST(bp.f_neg_paid AS VARCHAR), CAST(src.f_neg_paid AS VARCHAR) FROM bp, src
UNION ALL SELECT 'basket_panel','empirical_baskets_with_anomaly_flags','EMPIRICAL_FINDING','REPORT',
       CAST(bp.baskets_with_flags AS VARCHAR), '' FROM bp
