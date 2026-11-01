-- Phase 2B QA: choice_product_store_week
WITH
c AS (
    SELECT
        count(*)                                    AS n_rows,
        count(DISTINCT (product_id, store_id, week_no)) AS n_keys,
        sum(transaction_lines)                      AS lines_sum,
        sum(transaction_units)                      AS units_sum,
        sum(transaction_sales_value)                AS sales_sum,
        sum(causal_record_present)                  AS present,
        count(*) FILTER (
            WHERE causal_record_present = 0
              AND (causal_display_codes IS NOT NULL OR causal_mailer_codes IS NOT NULL
                   OR any_display_descriptive IS NOT NULL OR any_mailer_descriptive IS NOT NULL
                   OR display_conflict IS NOT NULL OR mailer_conflict IS NOT NULL)
        )                                           AS absent_with_metadata,
        count(*) FILTER (
            WHERE causal_record_present = 1
              AND (causal_display_codes IS NULL OR causal_mailer_codes IS NULL)
        )                                           AS present_without_codes,
        count(*) FILTER (WHERE display_conflict = 1) AS display_conflicts,
        count(*) FILTER (WHERE mailer_conflict = 1)  AS mailer_conflicts,
        count(*) FILTER (WHERE transaction_lines < 1) AS empty_rows,
        count(*) FILTER (WHERE distinct_households < 1) AS zero_households
    FROM choice_product_store_week
),
src AS (
    SELECT count(*) AS lines, sum(QUANTITY) AS units, sum(SALES_VALUE) AS sales FROM transaction_data
),
psw AS (SELECT count(*) AS n FROM (SELECT DISTINCT PRODUCT_ID, STORE_ID, WEEK_NO FROM transaction_data)),
src_conf AS (
    SELECT
        count(*)                        AS keys_present,
        count(*) FILTER (WHERE nd > 1)  AS display_conflicts,
        count(*) FILTER (WHERE nm > 1)  AS mailer_conflicts
    FROM (
        SELECT cd.PRODUCT_ID, cd.STORE_ID, cd.WEEK_NO,
               count(DISTINCT cd.display) AS nd, count(DISTINCT cd.mailer) AS nm
        FROM causal_data AS cd
        JOIN (SELECT DISTINCT PRODUCT_ID, STORE_ID, WEEK_NO FROM transaction_data) AS p
          ON p.PRODUCT_ID = cd.PRODUCT_ID AND p.STORE_ID = cd.STORE_ID AND p.WEEK_NO = cd.WEEK_NO
        GROUP BY 1, 2, 3
    )
),
forbidden_cols AS (
    SELECT count(*) AS n FROM information_schema.columns
    WHERE table_name = 'choice_product_store_week'
      AND regexp_matches(lower(column_name), 'treat|control|expos|effect')
)

SELECT 'choice_product_store_week' AS table_name, 'declared_key_unique' AS check_name, 'HARD_INVARIANT' AS check_class,
       CASE WHEN c.n_rows = c.n_keys THEN 'PASS' ELSE 'FAIL' END AS status,
       CAST(c.n_rows AS VARCHAR) AS observed, CAST(c.n_keys AS VARCHAR) AS expected FROM c
UNION ALL SELECT 'choice_product_store_week','row_count_equals_purchased_psw_universe','HARD_INVARIANT',
       CASE WHEN c.n_rows = psw.n THEN 'PASS' ELSE 'FAIL' END, CAST(c.n_rows AS VARCHAR), CAST(psw.n AS VARCHAR) FROM c, psw
UNION ALL SELECT 'choice_product_store_week','transaction_lines_reconcile','HARD_INVARIANT',
       CASE WHEN c.lines_sum = src.lines THEN 'PASS' ELSE 'FAIL' END, CAST(c.lines_sum AS VARCHAR), CAST(src.lines AS VARCHAR) FROM c, src
UNION ALL SELECT 'choice_product_store_week','transaction_units_reconcile','HARD_INVARIANT',
       CASE WHEN c.units_sum = src.units THEN 'PASS' ELSE 'FAIL' END, CAST(c.units_sum AS VARCHAR), CAST(src.units AS VARCHAR) FROM c, src
UNION ALL SELECT 'choice_product_store_week','transaction_sales_reconcile','HARD_INVARIANT',
       CASE WHEN abs(c.sales_sum - src.sales) <= greatest(1e-6, abs(src.sales) * 1e-9) THEN 'PASS' ELSE 'FAIL' END,
       CAST(c.sales_sum AS VARCHAR), CAST(src.sales AS VARCHAR) FROM c, src
UNION ALL SELECT 'choice_product_store_week','every_row_has_purchases','HARD_INVARIANT',
       CASE WHEN c.empty_rows = 0 AND c.zero_households = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(c.empty_rows AS VARCHAR) || '/' || CAST(c.zero_households AS VARCHAR), '0/0' FROM c
UNION ALL SELECT 'choice_product_store_week','absent_record_yields_null_metadata','HARD_INVARIANT',
       CASE WHEN c.absent_with_metadata = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.absent_with_metadata AS VARCHAR), '0' FROM c
UNION ALL SELECT 'choice_product_store_week','recorded_row_has_code_metadata','HARD_INVARIANT',
       CASE WHEN c.present_without_codes = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.present_without_codes AS VARCHAR), '0' FROM c
UNION ALL SELECT 'choice_product_store_week','causal_coverage_reconciles_to_source','HARD_INVARIANT',
       CASE WHEN c.present = src_conf.keys_present THEN 'PASS' ELSE 'FAIL' END,
       CAST(c.present AS VARCHAR), CAST(src_conf.keys_present AS VARCHAR) FROM c, src_conf
UNION ALL SELECT 'choice_product_store_week','display_conflicts_reconcile_to_source','HARD_INVARIANT',
       CASE WHEN c.display_conflicts = src_conf.display_conflicts THEN 'PASS' ELSE 'FAIL' END,
       CAST(c.display_conflicts AS VARCHAR), CAST(src_conf.display_conflicts AS VARCHAR) FROM c, src_conf
UNION ALL SELECT 'choice_product_store_week','mailer_conflicts_reconcile_to_source','HARD_INVARIANT',
       CASE WHEN c.mailer_conflicts = src_conf.mailer_conflicts THEN 'PASS' ELSE 'FAIL' END,
       CAST(c.mailer_conflicts AS VARCHAR), CAST(src_conf.mailer_conflicts AS VARCHAR) FROM c, src_conf
UNION ALL SELECT 'choice_product_store_week','no_causal_treatment_columns','HARD_INVARIANT',
       CASE WHEN forbidden_cols.n = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(forbidden_cols.n AS VARCHAR), '0' FROM forbidden_cols
UNION ALL SELECT 'choice_product_store_week','empirical_causal_record_coverage_share','EMPIRICAL_FINDING','REPORT',
       CAST(CAST(c.present AS DOUBLE) / c.n_rows AS VARCHAR), '' FROM c
