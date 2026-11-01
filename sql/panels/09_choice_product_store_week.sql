-- Phase 2B C: choice_product_store_week
-- Grain : product_id x store_id x week_no
-- Key   : (product_id, store_id, week_no) -- unique
-- Population: the PURCHASED product-store-week universe (2,370,784 keys with at
--   least one transaction line). This is the transaction-anchored PSW universe.
--   causal_data keys that were never purchased (the majority of its 36,771,279
--   keys) are deliberately NOT materialised here: doing so would create a
--   ~37M-row mostly-non-purchased placement census. They remain available in the
--   raw `causal_data` table. This scope choice is documented, not hidden.
--
-- causal_data is a PLACEMENT table, not a treatment. Phase 1 established that
-- absent rows have no documented meaning, that 15,245 product-store-weeks carry
-- conflicting placement codes, and that placement is not randomised. Therefore:
--   * an absent record yields causal_display_codes = NULL and
--     causal_mailer_codes = NULL -- NEVER '0', and never "no placement";
--   * display_conflict / mailer_conflict are NULL when no record exists
--     (a conflict is undefined without a record) and 0/1 when a record exists;
--   * any_display_descriptive / any_mailer_descriptive are descriptive bool_or
--     existence summaries over RECORDED rows only, and NULL when absent;
--   * raw code strings are preserved as sorted distinct lists; no canonical
--     display/mailer treatment and no exposure value is chosen.

WITH purchased AS (
    SELECT
        PRODUCT_ID,
        STORE_ID,
        WEEK_NO,
        count(*)                    AS transaction_lines,
        sum(QUANTITY)               AS transaction_units,
        sum(SALES_VALUE)            AS transaction_sales_value,
        count(DISTINCT household_key) AS distinct_households
    FROM transaction_data
    GROUP BY PRODUCT_ID, STORE_ID, WEEK_NO
),
purchased_psw AS (
    SELECT DISTINCT PRODUCT_ID, STORE_ID, WEEK_NO FROM transaction_data
),
-- Restrict causal_data to the PURCHASED PSW universe BEFORE aggregating. This is
-- semantically identical to aggregating all 36.77M causal keys and then keeping
-- the purchased ones, but the full-table group-by materialises ~36.7M hash
-- groups and exhausts memory; only ~483k purchased keys ever carry a record.
causal_purchased AS (
    SELECT c.PRODUCT_ID, c.STORE_ID, c.WEEK_NO, c.display, c.mailer
    FROM causal_data AS c
    SEMI JOIN purchased_psw AS p
      ON p.PRODUCT_ID = c.PRODUCT_ID AND p.STORE_ID = c.STORE_ID AND p.WEEK_NO = c.WEEK_NO
),
causal_agg AS (
    SELECT
        PRODUCT_ID,
        STORE_ID,
        WEEK_NO,
        count(*)                         AS causal_record_rows,
        count(DISTINCT display)          AS distinct_display_codes,
        count(DISTINCT mailer)           AS distinct_mailer_codes,
        array_to_string(list_sort(list(DISTINCT display)), ',') AS display_codes,
        array_to_string(list_sort(list(DISTINCT mailer)), ',')  AS mailer_codes,
        bool_or(display <> '0')          AS display_nonzero_present,
        bool_or(mailer <> '0')           AS mailer_nonzero_present
    FROM causal_purchased
    GROUP BY PRODUCT_ID, STORE_ID, WEEK_NO
)
SELECT
    p.PRODUCT_ID                                          AS product_id,
    p.STORE_ID                                            AS store_id,
    p.WEEK_NO                                             AS week_no,
    p.transaction_lines,
    p.transaction_units,
    p.transaction_sales_value,
    p.distinct_households,
    CASE WHEN c.PRODUCT_ID IS NULL THEN 0 ELSE 1 END      AS causal_record_present,
    COALESCE(c.causal_record_rows, 0)                     AS causal_record_rows,
    c.display_codes                                       AS causal_display_codes,
    c.mailer_codes                                        AS causal_mailer_codes,
    CASE WHEN c.PRODUCT_ID IS NULL THEN NULL
         WHEN c.distinct_display_codes > 1 THEN 1 ELSE 0 END AS display_conflict,
    CASE WHEN c.PRODUCT_ID IS NULL THEN NULL
         WHEN c.distinct_mailer_codes > 1 THEN 1 ELSE 0 END  AS mailer_conflict,
    CASE WHEN c.PRODUCT_ID IS NULL THEN NULL
         WHEN c.display_nonzero_present THEN 1 ELSE 0 END    AS any_display_descriptive,
    CASE WHEN c.PRODUCT_ID IS NULL THEN NULL
         WHEN c.mailer_nonzero_present THEN 1 ELSE 0 END     AS any_mailer_descriptive
FROM purchased AS p
LEFT JOIN causal_agg AS c
    ON c.PRODUCT_ID = p.PRODUCT_ID AND c.STORE_ID = p.STORE_ID AND c.WEEK_NO = p.WEEK_NO
ORDER BY p.PRODUCT_ID, p.STORE_ID, p.WEEK_NO
