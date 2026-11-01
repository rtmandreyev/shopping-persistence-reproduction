-- Phase 2B QA: choice_household_product_week
WITH
c AS (
    SELECT
        count(*)                                  AS n_rows,
        count(DISTINCT (household_key, product_id, week_no)) AS n_keys,
        sum(purchase_lines)                       AS lines_sum,
        sum(units)                                AS units_sum,
        sum(sales_value)                          AS sales_sum,
        count(*) FILTER (WHERE units <= 0)        AS nonpositive_units,
        count(*) FILTER (WHERE brand IS NULL OR commodity_desc IS NULL) AS null_attrs,
        count(*) FILTER (WHERE distinct_stores < 1) AS bad_stores,
        count(*) FILTER (WHERE prior_purchase_weeks < 0 OR prior_purchase_lines < 0 OR prior_units < 0
                           OR prior_brand_purchase_lines < 0 OR prior_commodity_purchase_lines < 0) AS negative_prior,
        count(*) FILTER (WHERE weeks_since_prior_product_purchase IS NOT NULL AND weeks_since_prior_product_purchase <= 0) AS leak_product,
        count(*) FILTER (WHERE weeks_since_prior_brand_purchase IS NOT NULL AND weeks_since_prior_brand_purchase <= 0) AS leak_brand,
        count(*) FILTER (WHERE weeks_since_prior_commodity_purchase IS NOT NULL AND weeks_since_prior_commodity_purchase <= 0) AS leak_commodity,
        count(*) FILTER (WHERE COALESCE(prior_purchase_weeks, 0) > 0 AND weeks_since_prior_product_purchase IS NULL) AS prior_flag_inconsistent,
        count(*) FILTER (WHERE previously_purchased_product <> CASE WHEN COALESCE(prior_purchase_weeks, 0) > 0 THEN 1 ELSE 0 END) AS prev_flag_mismatch
    FROM choice_household_product_week
),
rank_check AS (
    SELECT count(*) AS n_bad FROM (
        SELECT prior_purchase_weeks,
               row_number() OVER (PARTITION BY household_key, product_id ORDER BY week_no) - 1 AS expected_prior
        FROM choice_household_product_week
    ) WHERE prior_purchase_weeks <> expected_prior
),
src AS (SELECT count(*) AS lines, sum(QUANTITY) AS units, sum(SALES_VALUE) AS sales FROM transaction_data),
cells AS (SELECT count(*) AS n FROM (SELECT DISTINCT household_key, PRODUCT_ID, WEEK_NO FROM transaction_data)),
forbidden_cols AS (
    SELECT count(*) AS n FROM information_schema.columns
    WHERE table_name = 'choice_household_product_week'
      AND regexp_matches(lower(column_name), 'treat|control|expos|availability|chosen|choice_set')
)

SELECT 'choice_household_product_week' AS table_name, 'declared_key_unique' AS check_name, 'HARD_INVARIANT' AS check_class,
       CASE WHEN c.n_rows = c.n_keys THEN 'PASS' ELSE 'FAIL' END AS status,
       CAST(c.n_rows AS VARCHAR) AS observed, CAST(c.n_keys AS VARCHAR) AS expected FROM c
UNION ALL SELECT 'choice_household_product_week','row_count_equals_observed_purchase_cells','HARD_INVARIANT',
       CASE WHEN c.n_rows = cells.n THEN 'PASS' ELSE 'FAIL' END, CAST(c.n_rows AS VARCHAR), CAST(cells.n AS VARCHAR) FROM c, cells
UNION ALL SELECT 'choice_household_product_week','purchase_lines_reconcile','HARD_INVARIANT',
       CASE WHEN c.lines_sum = src.lines THEN 'PASS' ELSE 'FAIL' END, CAST(c.lines_sum AS VARCHAR), CAST(src.lines AS VARCHAR) FROM c, src
UNION ALL SELECT 'choice_household_product_week','units_reconcile','HARD_INVARIANT',
       CASE WHEN c.units_sum = src.units THEN 'PASS' ELSE 'FAIL' END, CAST(c.units_sum AS VARCHAR), CAST(src.units AS VARCHAR) FROM c, src
UNION ALL SELECT 'choice_household_product_week','sales_value_reconciles','HARD_INVARIANT',
       CASE WHEN abs(c.sales_sum - src.sales) <= greatest(1e-6, abs(src.sales) * 1e-9) THEN 'PASS' ELSE 'FAIL' END,
       CAST(c.sales_sum AS VARCHAR), CAST(src.sales AS VARCHAR) FROM c, src
UNION ALL SELECT 'choice_household_product_week','product_attributes_present','HARD_INVARIANT',
       CASE WHEN c.null_attrs = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.null_attrs AS VARCHAR), '0' FROM c
UNION ALL SELECT 'choice_household_product_week','stores_positive','HARD_INVARIANT',
       CASE WHEN c.bad_stores = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.bad_stores AS VARCHAR), '0' FROM c
UNION ALL SELECT 'choice_household_product_week','prior_counters_nonnegative','HARD_INVARIANT',
       CASE WHEN c.negative_prior = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.negative_prior AS VARCHAR), '0' FROM c
UNION ALL SELECT 'choice_household_product_week','prior_history_is_strictly_earlier','HARD_INVARIANT',
       CASE WHEN c.leak_product = 0 AND c.leak_brand = 0 AND c.leak_commodity = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(c.leak_product AS VARCHAR) || '/' || CAST(c.leak_brand AS VARCHAR) || '/' || CAST(c.leak_commodity AS VARCHAR), '0/0/0' FROM c
UNION ALL SELECT 'choice_household_product_week','prior_purchase_weeks_equal_rank_minus_one','HARD_INVARIANT',
       CASE WHEN rank_check.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(rank_check.n_bad AS VARCHAR), '0' FROM rank_check
UNION ALL SELECT 'choice_household_product_week','previously_purchased_flag_consistent','HARD_INVARIANT',
       CASE WHEN c.prev_flag_mismatch = 0 AND c.prior_flag_inconsistent = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(c.prev_flag_mismatch AS VARCHAR) || '/' || CAST(c.prior_flag_inconsistent AS VARCHAR), '0/0' FROM c
UNION ALL SELECT 'choice_household_product_week','no_treatment_or_choice_columns','HARD_INVARIANT',
       CASE WHEN forbidden_cols.n = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(forbidden_cols.n AS VARCHAR), '0' FROM forbidden_cols
UNION ALL SELECT 'choice_household_product_week','empirical_cells_with_nonpositive_units','EMPIRICAL_FINDING','REPORT',
       CAST(c.nonpositive_units AS VARCHAR), '' FROM c
