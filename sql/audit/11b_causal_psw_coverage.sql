-- 11b_causal_psw_coverage.sql
-- PURPOSE: measure recorded causal_data row existence for PURCHASED
--          product-store-weeks, without assigning meaning to absent rows.
-- Grain: one row per metric.
--
-- The vendor describes causal_data as indicating whether "a given product was
-- featured in the weekly mailer or was part of an in-store display". It does
-- not promise a complete product-store-week universe or define an absent row.
-- The match rates below therefore describe recorded placement-row existence;
-- they are not missing-data rates and do not establish selection into coverage.
--
-- Both forms are reported:
--   * row-weighted match rate - share of transaction LINES whose (p,s,w) exists
--   * unique PSW match rate   - share of DISTINCT purchased (p,s,w) keys that exist
-- Provenance: B counts; C interpretation boundary.

WITH purchased_psw AS (
    SELECT PRODUCT_ID, STORE_ID, WEEK_NO,
           count(*) AS line_count
    FROM transaction_data
    GROUP BY 1, 2, 3
),
causal_psw_flags AS (
    SELECT PRODUCT_ID,
           STORE_ID,
           WEEK_NO,
           bool_or(display <> '0') AS any_display,
           bool_or(mailer <> '0') AS any_mailer
    FROM causal_data
    GROUP BY 1, 2, 3
),
joined AS (
    SELECT p.PRODUCT_ID, p.STORE_ID, p.WEEK_NO, p.line_count,
           (c.PRODUCT_ID IS NOT NULL) AS has_causal_row,
           coalesce(c.any_display, false) AS any_display,
           coalesce(c.any_mailer, false) AS any_mailer
    FROM purchased_psw p
    LEFT JOIN causal_psw_flags c
           ON c.PRODUCT_ID = p.PRODUCT_ID
          AND c.STORE_ID = p.STORE_ID
          AND c.WEEK_NO = p.WEEK_NO
)
SELECT 'purchased_unique_psw' AS metric,
       count(*)::VARCHAR AS value,
       'B observed' AS provenance
FROM joined

UNION ALL SELECT 'purchased_psw_matched_to_causal',
       count(*) FILTER (WHERE has_causal_row)::VARCHAR, 'B observed' FROM joined
UNION ALL SELECT 'purchased_psw_unmatched',
       count(*) FILTER (WHERE NOT has_causal_row)::VARCHAR, 'B observed' FROM joined
UNION ALL SELECT 'purchased_unique_psw_match_rate_pct',
       round(100.0 * count(*) FILTER (WHERE has_causal_row) / count(*), 4)::VARCHAR,
       'B observed' FROM joined

-- Row-level: same question weighted by transaction lines.
UNION ALL SELECT 'transaction_lines_total', sum(line_count)::VARCHAR, 'B observed' FROM joined
UNION ALL SELECT 'transaction_lines_matched_to_causal',
       sum(line_count) FILTER (WHERE has_causal_row)::VARCHAR, 'B observed' FROM joined
UNION ALL SELECT 'transaction_lines_unmatched',
       sum(line_count) FILTER (WHERE NOT has_causal_row)::VARCHAR, 'B observed' FROM joined
UNION ALL SELECT 'transaction_line_match_rate_pct',
       round(100.0 * sum(line_count) FILTER (WHERE has_causal_row) / sum(line_count), 4)::VARCHAR,
       'B observed' FROM joined

-- Descriptive existence indicators at one row per matched PSW. bool_or means a
-- nonzero raw code exists for the key; it is not a treatment definition.
UNION ALL SELECT 'matched_psw_with_any_placement',
       count(*) FILTER (WHERE has_causal_row AND (any_display OR any_mailer))::VARCHAR,
       'B descriptive key-level bool_or existence indicator' FROM joined

UNION ALL SELECT 'matched_psw_with_no_placement',
       count(*) FILTER (WHERE has_causal_row AND NOT any_display AND NOT any_mailer)::VARCHAR,
       'B descriptive key-level bool_or existence indicator' FROM joined

UNION ALL SELECT 'matched_psw_with_any_display',
       count(*) FILTER (WHERE has_causal_row AND any_display)::VARCHAR,
       'B descriptive key-level bool_or existence indicator' FROM joined

UNION ALL SELECT 'matched_psw_with_any_mailer',
       count(*) FILTER (WHERE has_causal_row AND any_mailer)::VARCHAR,
       'B descriptive key-level bool_or existence indicator' FROM joined

UNION ALL SELECT 'matched_psw_display_only',
       count(*) FILTER (WHERE has_causal_row AND any_display AND NOT any_mailer)::VARCHAR,
       'B descriptive key-level bool_or existence indicator' FROM joined

UNION ALL SELECT 'matched_psw_mailer_only',
       count(*) FILTER (WHERE has_causal_row AND NOT any_display AND any_mailer)::VARCHAR,
       'B descriptive key-level bool_or existence indicator' FROM joined

UNION ALL SELECT 'matched_psw_both_display_and_mailer',
       count(*) FILTER (WHERE has_causal_row AND any_display AND any_mailer)::VARCHAR,
       'B descriptive key-level bool_or existence indicator' FROM joined

UNION ALL SELECT 'matched_psw_display_nonzero_share_pct',
       round(100.0 * count(*) FILTER (WHERE has_causal_row AND any_display)
             / nullif(count(*) FILTER (WHERE has_causal_row), 0), 4)::VARCHAR,
       'B descriptive key-level bool_or existence indicator' FROM joined

UNION ALL SELECT 'matched_psw_mailer_nonzero_share_pct',
       round(100.0 * count(*) FILTER (WHERE has_causal_row AND any_mailer)
             / nullif(count(*) FILTER (WHERE has_causal_row), 0), 4)::VARCHAR,
       'B descriptive key-level bool_or existence indicator' FROM joined

-- Partition unmatched join keys by whether the product appears anywhere in
-- causal_data. These are descriptive row-existence diagnostics only; neither
-- branch is labelled missing coverage or no treatment.
UNION ALL SELECT 'unmatched_psw_where_product_absent_from_causal',
       count(*)::VARCHAR,
       'B observed; no causal_data row exists for that product'
FROM joined j
WHERE NOT j.has_causal_row
  AND NOT EXISTS (SELECT 1 FROM causal_data c WHERE c.PRODUCT_ID = j.PRODUCT_ID)

UNION ALL SELECT 'unmatched_psw_where_product_known_but_storeweek_absent',
       count(*)::VARCHAR,
       'B observed; product appears elsewhere in causal_data but not at this (store,week)'
FROM joined j
WHERE NOT j.has_causal_row
  AND EXISTS (SELECT 1 FROM causal_data c WHERE c.PRODUCT_ID = j.PRODUCT_ID)

-- Store-level row presence; absent rows have no treatment or missingness label.
UNION ALL SELECT 'purchased_stores_without_causal_rows',
       count(*)::VARCHAR, 'B observed row existence'
FROM (SELECT DISTINCT STORE_ID FROM transaction_data) t
WHERE NOT EXISTS (SELECT 1 FROM causal_data c WHERE c.STORE_ID = t.STORE_ID)

UNION ALL SELECT 'causal_weeks_present_vs_transaction_weeks',
       (SELECT count(DISTINCT WEEK_NO) FROM causal_data)::VARCHAR || ' of '
       || (SELECT count(DISTINCT WEEK_NO) FROM transaction_data)::VARCHAR,
       'B observed' FROM (SELECT 1)
