-- Phase 2A QA: household_category_week
-- Uniform check output: table_name, check_name, check_class, status, observed, expected

WITH
hcw AS (
    SELECT
        count(*)                                       AS n_rows,
        count(DISTINCT (household_key, commodity_desc, week_no)) AS n_keys,
        sum(category_sales_value)                      AS sales_sum,
        sum(category_units)                            AS units_sum,
        count(DISTINCT commodity_desc)                 AS n_commodities,
        count(*) FILTER (WHERE commodity_desc IS NULL) AS null_commodity,
        count(*) FILTER (WHERE category_trips < 1)     AS zero_trip_rows,
        count(*) FILTER (
            WHERE abs(category_sales_value
                      - (category_national_sales_value + category_private_sales_value))
                  > greatest(1e-6, abs(category_sales_value) * 1e-9)
        )                                              AS brand_split_gaps,
        count(*) FILTER (WHERE commodity_desc = 'NO COMMODITY DESCRIPTION') AS sentinel_rows
    FROM household_category_week
),
src AS (
    SELECT
        sum(t.SALES_VALUE)                    AS sales_sum,
        sum(t.QUANTITY)                       AS units_sum,
        count(DISTINCT p.COMMODITY_DESC)      AS n_commodities
    FROM transaction_data t LEFT JOIN product p ON p.PRODUCT_ID = t.PRODUCT_ID
),
dense AS (
    SELECT (SELECT count(DISTINCT household_key) FROM transaction_data)
         * (SELECT count(DISTINCT WEEK_NO) FROM transaction_data) * (SELECT n_commodities FROM src) AS n_dense
)

SELECT 'household_category_week' AS table_name, 'declared_key_unique' AS check_name, 'HARD_INVARIANT' AS check_class,
       CASE WHEN hcw.n_rows = hcw.n_keys THEN 'PASS' ELSE 'FAIL' END AS status,
       CAST(hcw.n_rows AS VARCHAR) AS observed, CAST(hcw.n_keys AS VARCHAR) AS expected FROM hcw
UNION ALL SELECT 'household_category_week','sales_value_reconciles','HARD_INVARIANT',
       CASE WHEN abs(hcw.sales_sum - src.sales_sum) <= greatest(1e-6, abs(src.sales_sum) * 1e-9) THEN 'PASS' ELSE 'FAIL' END,
       CAST(hcw.sales_sum AS VARCHAR), CAST(src.sales_sum AS VARCHAR) FROM hcw, src
UNION ALL SELECT 'household_category_week','units_reconcile','HARD_INVARIANT',
       CASE WHEN hcw.units_sum = src.units_sum THEN 'PASS' ELSE 'FAIL' END,
       CAST(hcw.units_sum AS VARCHAR), CAST(src.units_sum AS VARCHAR) FROM hcw, src
UNION ALL SELECT 'household_category_week','commodity_labels_complete','HARD_INVARIANT',
       CASE WHEN hcw.n_commodities = src.n_commodities THEN 'PASS' ELSE 'FAIL' END,
       CAST(hcw.n_commodities AS VARCHAR), CAST(src.n_commodities AS VARCHAR) FROM hcw, src
UNION ALL SELECT 'household_category_week','no_null_commodity','HARD_INVARIANT',
       CASE WHEN hcw.null_commodity = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(hcw.null_commodity AS VARCHAR), '0' FROM hcw
UNION ALL SELECT 'household_category_week','category_trips_positive','HARD_INVARIANT',
       CASE WHEN hcw.zero_trip_rows = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(hcw.zero_trip_rows AS VARCHAR), '0' FROM hcw
UNION ALL SELECT 'household_category_week','brand_split_complete','HARD_INVARIANT',
       CASE WHEN hcw.brand_split_gaps = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(hcw.brand_split_gaps AS VARCHAR), '0' FROM hcw
UNION ALL SELECT 'household_category_week','empirical_sparse_not_dense','EMPIRICAL_FINDING','REPORT',
       CAST(hcw.n_rows AS VARCHAR), CAST((SELECT n_dense FROM dense) AS VARCHAR) FROM hcw
UNION ALL SELECT 'household_category_week','empirical_sentinel_commodity_rows','EMPIRICAL_FINDING','REPORT',
       CAST(hcw.sentinel_rows AS VARCHAR), '' FROM hcw
