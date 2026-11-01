-- 11_causal_data_audit.sql
-- Purpose: forensic audit of the largest table, and the measured coverage of
--          purchased product-store-weeks within it.
-- Grain: one row per metric.
--
-- CRITICAL INTERPRETATION: display/mailer are PLACEMENT codes, not treatment
-- indicators, and their presence is not evidence of random assignment. This
-- query reports categories and coverage; it does not create treatment variables
-- and does not assert causality.
-- Provenance: A = vendor's documented display/mailer code lists,
--             B = observed, C = our note.

WITH duplicate_psw AS (
    SELECT PRODUCT_ID,
           STORE_ID,
           WEEK_NO,
           count(*) AS physical_rows,
           count(DISTINCT display) AS display_values,
           count(DISTINCT mailer) AS mailer_values,
           count(DISTINCT (display, mailer)) AS placement_variants
    FROM causal_data
    GROUP BY 1, 2, 3
    HAVING count(*) > 1
)
SELECT 'causal_rows' AS metric, count(*)::VARCHAR AS value, 'B' AS provenance FROM causal_data
UNION ALL SELECT 'causal_distinct_products', count(DISTINCT PRODUCT_ID)::VARCHAR, 'B' FROM causal_data
UNION ALL SELECT 'causal_distinct_stores', count(DISTINCT STORE_ID)::VARCHAR, 'B' FROM causal_data
UNION ALL SELECT 'causal_distinct_weeks', count(DISTINCT WEEK_NO)::VARCHAR, 'B' FROM causal_data
UNION ALL SELECT 'causal_week_min', min(WEEK_NO)::VARCHAR, 'B' FROM causal_data
UNION ALL SELECT 'causal_week_max', max(WEEK_NO)::VARCHAR, 'B' FROM causal_data
UNION ALL SELECT 'causal_distinct_psw', count(*)::VARCHAR, 'B'
        FROM (SELECT DISTINCT PRODUCT_ID, STORE_ID, WEEK_NO FROM causal_data)
UNION ALL SELECT 'causal_duplicate_psw_groups', count(*)::VARCHAR, 'B' FROM duplicate_psw
UNION ALL SELECT 'causal_duplicate_psw_physical_rows', coalesce(sum(physical_rows), 0)::VARCHAR, 'B' FROM duplicate_psw
UNION ALL SELECT 'causal_duplicate_psw_excess_rows', coalesce(sum(physical_rows - 1), 0)::VARCHAR, 'B' FROM duplicate_psw
UNION ALL SELECT 'causal_duplicate_psw_display_conflicts_total', count(*) FILTER (WHERE display_values > 1)::VARCHAR, 'B' FROM duplicate_psw
UNION ALL SELECT 'causal_duplicate_psw_mailer_conflicts_total', count(*) FILTER (WHERE mailer_values > 1)::VARCHAR, 'B' FROM duplicate_psw
UNION ALL SELECT 'causal_duplicate_psw_display_only_conflicts', count(*) FILTER (WHERE display_values > 1 AND mailer_values = 1)::VARCHAR, 'B' FROM duplicate_psw
UNION ALL SELECT 'causal_duplicate_psw_mailer_only_conflicts', count(*) FILTER (WHERE display_values = 1 AND mailer_values > 1)::VARCHAR, 'B' FROM duplicate_psw
UNION ALL SELECT 'causal_duplicate_psw_both_conflicts', count(*) FILTER (WHERE display_values > 1 AND mailer_values > 1)::VARCHAR, 'B' FROM duplicate_psw
UNION ALL SELECT 'causal_duplicate_psw_same_placement_groups', count(*) FILTER (WHERE placement_variants = 1)::VARCHAR, 'B' FROM duplicate_psw

-- Display and mailer code distributions. These are CATEGORIES; the vendor lists
-- specific code meanings, preserved here rather than collapsed to binary.
UNION ALL SELECT 'display_values',
       coalesce(string_agg(DISTINCT display, ' | ' ORDER BY display), 'none'), 'A documented codes; B observed'
       FROM causal_data
UNION ALL SELECT 'mailer_values',
       coalesce(string_agg(DISTINCT mailer, ' | ' ORDER BY mailer), 'none'), 'A documented codes; B observed'
       FROM causal_data
UNION ALL SELECT 'display_distinct_codes', count(DISTINCT display)::VARCHAR, 'B' FROM causal_data
UNION ALL SELECT 'mailer_distinct_codes', count(DISTINCT mailer)::VARCHAR, 'B' FROM causal_data
UNION ALL SELECT 'rows_display_ne_0', count(*) FILTER (WHERE display <> '0')::VARCHAR, 'B' FROM causal_data
UNION ALL SELECT 'rows_mailer_ne_0', count(*) FILTER (WHERE mailer <> '0')::VARCHAR, 'B' FROM causal_data
UNION ALL SELECT 'rows_both_ne_0', count(*) FILTER (WHERE display <> '0' AND mailer <> '0')::VARCHAR, 'B' FROM causal_data
UNION ALL SELECT 'rows_display_null', count(*) FILTER (WHERE display IS NULL)::VARCHAR, 'B' FROM causal_data
UNION ALL SELECT 'rows_mailer_null', count(*) FILTER (WHERE mailer IS NULL)::VARCHAR, 'B' FROM causal_data

-- Coverage vs transactions, at the product-store-week level.
UNION ALL SELECT 'causal_products_not_in_transactions', count(*)::VARCHAR, 'B'
        FROM (SELECT DISTINCT c.PRODUCT_ID FROM causal_data c
              LEFT JOIN (SELECT DISTINCT PRODUCT_ID FROM transaction_data) t
                     ON t.PRODUCT_ID = c.PRODUCT_ID
              WHERE t.PRODUCT_ID IS NULL)
UNION ALL SELECT 'causal_stores_not_in_transactions', count(*)::VARCHAR, 'B'
        FROM (SELECT DISTINCT c.STORE_ID FROM causal_data c
              LEFT JOIN (SELECT DISTINCT STORE_ID FROM transaction_data) t
                     ON t.STORE_ID = c.STORE_ID
              WHERE t.STORE_ID IS NULL)
UNION ALL SELECT 'causal_weeks_outside_transaction_range', count(*)::VARCHAR, 'B'
        FROM causal_data WHERE WEEK_NO < (SELECT min(WEEK_NO) FROM transaction_data)
                            OR WEEK_NO > (SELECT max(WEEK_NO) FROM transaction_data)
