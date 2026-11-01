-- Phase 2A canonical table: household_week_base
-- Grain : household_key x week_no
-- Key   : (household_key, week_no) -- unique
-- Population: the COMPLETE retailer-observation grid = every transacting
--   household x every week present in the transaction calendar (1..102),
--   INCLUDING household-weeks with no recorded purchase.
--
-- SEMANTICS (do not misread downstream):
--   retailer_purchase = 0 means "this household recorded NO purchase from THIS
--   retailer during this week". It does NOT mean the household consumed zero
--   groceries. Non-purchase is not observed non-consumption; a real purchase may
--   have occurred at another retailer or been simply unrecorded.
--
-- Zero-filled weeks carry retailer_purchase = 0 and every activity metric = 0.
-- Active weeks carry the factual weekly aggregates. No campaign/treatment
-- variable and no future-looking feature is added here.
--
-- The grid and its week domain are DERIVED from the source (distinct weeks in
-- transaction_data), not hardcoded. Sums keep the source sign conventions, as in
-- basket_panel. product is LEFT-joined so an unmatched product cannot drop a
-- line; QA asserts zero unmatched products.

WITH households AS (
    SELECT DISTINCT household_key FROM transaction_data
),
weeks AS (
    SELECT DISTINCT WEEK_NO AS week_no FROM transaction_data
),
grid AS (
    SELECT h.household_key, w.week_no
    FROM households AS h
    CROSS JOIN weeks AS w
),
week_agg AS (
    SELECT
        t.household_key,
        t.WEEK_NO                                                      AS week_no,
        count(DISTINCT t.BASKET_ID)                                    AS trips,
        sum(t.SALES_VALUE)                                             AS weekly_sales_value,
        sum(t.QUANTITY)                                                AS units,
        count(DISTINCT t.PRODUCT_ID)                                   AS distinct_products,
        count(DISTINCT p.COMMODITY_DESC)                               AS distinct_commodities,
        count(DISTINCT t.STORE_ID)                                     AS active_store_count,
        count(*)                                                       AS transaction_line_count,
        sum(t.RETAIL_DISC)                                             AS weekly_retail_disc,
        sum(t.COUPON_DISC)                                             AS weekly_coupon_disc,
        sum(t.COUPON_MATCH_DISC)                                       AS weekly_coupon_match_disc,
        sum(CASE WHEN t.QUANTITY = 0 THEN 1 ELSE 0 END)                AS qa_quantity_zero_lines,
        sum(CASE WHEN t.RETAIL_DISC > 0 THEN 1 ELSE 0 END)             AS qa_retail_disc_positive_lines,
        sum(CASE WHEN t.SALES_VALUE + t.COUPON_DISC < 0 THEN 1 ELSE 0 END) AS qa_implied_paid_negative_lines
    FROM transaction_data AS t
    LEFT JOIN product AS p ON p.PRODUCT_ID = t.PRODUCT_ID
    GROUP BY t.household_key, t.WEEK_NO
)
SELECT
    g.household_key,
    g.week_no,
    CASE WHEN a.household_key IS NULL THEN 0 ELSE 1 END AS retailer_purchase,
    COALESCE(a.trips, 0)                     AS trips,
    COALESCE(a.weekly_sales_value, 0.0)      AS weekly_sales_value,
    COALESCE(a.units, 0)                     AS units,
    COALESCE(a.distinct_products, 0)         AS distinct_products,
    COALESCE(a.distinct_commodities, 0)      AS distinct_commodities,
    COALESCE(a.active_store_count, 0)        AS active_store_count,
    COALESCE(a.transaction_line_count, 0)    AS transaction_line_count,
    COALESCE(a.weekly_retail_disc, 0.0)      AS weekly_retail_disc,
    COALESCE(a.weekly_coupon_disc, 0.0)      AS weekly_coupon_disc,
    COALESCE(a.weekly_coupon_match_disc, 0.0) AS weekly_coupon_match_disc,
    COALESCE(a.qa_quantity_zero_lines, 0)    AS qa_quantity_zero_lines,
    COALESCE(a.qa_retail_disc_positive_lines, 0) AS qa_retail_disc_positive_lines,
    COALESCE(a.qa_implied_paid_negative_lines, 0) AS qa_implied_paid_negative_lines
FROM grid AS g
LEFT JOIN week_agg AS a
    ON a.household_key = g.household_key AND a.week_no = g.week_no
ORDER BY g.household_key, g.week_no
