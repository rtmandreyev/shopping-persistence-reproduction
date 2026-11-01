-- Phase 2B QA: choice_line_price_candidates
WITH
c AS (
    SELECT
        count(*)                                AS n_rows,
        count(DISTINCT (household_key, basket_id, product_id)) AS n_keys,
        sum(quantity)                           AS units_sum,
        sum(sales_value)                        AS sales_sum,
        count(*) FILTER (WHERE quantity <= 0)   AS nonpositive_rows,
        count(*) FILTER (WHERE vendor_loyalty_unit_price_candidate IS NULL
                           OR vendor_nonloyalty_unit_price_candidate IS NULL) AS null_candidates,
        count(*) FILTER (WHERE vendor_loyalty_unit_price_candidate < 0)    AS negative_loyalty,
        count(*) FILTER (WHERE vendor_nonloyalty_unit_price_candidate < 0) AS negative_nonloyalty,
        count(*) FILTER (WHERE vendor_loyalty_unit_price_candidate < vendor_nonloyalty_unit_price_candidate)
            AS loyalty_below_nonloyalty
    FROM choice_line_price_candidates
),
src AS (
    SELECT count(*) AS lines, sum(QUANTITY) AS units, sum(SALES_VALUE) AS sales
    FROM transaction_data WHERE QUANTITY > 0
),
name_check AS (
    SELECT count(*) AS n FROM information_schema.columns
    WHERE table_name = 'choice_line_price_candidates' AND lower(column_name) = 'price'
)

SELECT 'choice_line_price_candidates' AS table_name, 'declared_key_unique' AS check_name, 'HARD_INVARIANT' AS check_class,
       CASE WHEN c.n_rows = c.n_keys THEN 'PASS' ELSE 'FAIL' END AS status,
       CAST(c.n_rows AS VARCHAR) AS observed, CAST(c.n_keys AS VARCHAR) AS expected FROM c
UNION ALL SELECT 'choice_line_price_candidates','row_count_equals_positive_quantity_lines','HARD_INVARIANT',
       CASE WHEN c.n_rows = src.lines THEN 'PASS' ELSE 'FAIL' END, CAST(c.n_rows AS VARCHAR), CAST(src.lines AS VARCHAR) FROM c, src
UNION ALL SELECT 'choice_line_price_candidates','all_rows_have_positive_quantity','HARD_INVARIANT',
       CASE WHEN c.nonpositive_rows = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.nonpositive_rows AS VARCHAR), '0' FROM c
UNION ALL SELECT 'choice_line_price_candidates','quantity_zero_lines_excluded_not_lost','HARD_INVARIANT',
       CASE WHEN (SELECT count(*) FROM transaction_data WHERE QUANTITY = 0) = 14466
            THEN 'PASS' ELSE 'FAIL' END,
       CAST((SELECT count(*) FROM transaction_data WHERE QUANTITY = 0) AS VARCHAR), '14466'
UNION ALL SELECT 'choice_line_price_candidates','units_reconcile_to_positive_quantity_source','HARD_INVARIANT',
       CASE WHEN c.units_sum = src.units THEN 'PASS' ELSE 'FAIL' END, CAST(c.units_sum AS VARCHAR), CAST(src.units AS VARCHAR) FROM c, src
UNION ALL SELECT 'choice_line_price_candidates','sales_reconcile_to_positive_quantity_source','HARD_INVARIANT',
       CASE WHEN abs(c.sales_sum - src.sales) <= greatest(1e-6, abs(src.sales) * 1e-9) THEN 'PASS' ELSE 'FAIL' END,
       CAST(c.sales_sum AS VARCHAR), CAST(src.sales AS VARCHAR) FROM c, src
UNION ALL SELECT 'choice_line_price_candidates','no_null_price_candidates','HARD_INVARIANT',
       CASE WHEN c.null_candidates = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.null_candidates AS VARCHAR), '0' FROM c
UNION ALL SELECT 'choice_line_price_candidates','no_column_named_price','HARD_INVARIANT',
       CASE WHEN name_check.n = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(name_check.n AS VARCHAR), '0' FROM name_check
UNION ALL SELECT 'choice_line_price_candidates','empirical_negative_loyalty_price_candidates','EMPIRICAL_FINDING','REPORT',
       CAST(c.negative_loyalty AS VARCHAR), '' FROM c
UNION ALL SELECT 'choice_line_price_candidates','empirical_negative_nonloyalty_price_candidates','EMPIRICAL_FINDING','REPORT',
       CAST(c.negative_nonloyalty AS VARCHAR), '' FROM c
UNION ALL SELECT 'choice_line_price_candidates','empirical_loyalty_below_nonloyalty_rows','EMPIRICAL_FINDING','REPORT',
       CAST(c.loyalty_below_nonloyalty AS VARCHAR), '' FROM c
