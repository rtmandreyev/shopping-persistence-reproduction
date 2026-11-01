-- Phase 2A canonical table: basket_panel
-- Grain : household_key x basket_id  (one row per distinct shopping basket)
-- Key   : (household_key, basket_id) -- unique
--
-- Phase 1 established that BASKET_ID is globally unique and never spans
-- households, days, weeks, or stores, and that transaction_data is unique on
-- (household_key, BASKET_ID, PRODUCT_ID). A basket row therefore aggregates
-- exactly the transaction lines that share it: no deduplication is performed or
-- required, and no transaction line is dropped.
--
-- SALES_VALUE is retailer revenue, not a customer-paid price (Phase 1 price
-- semantics). basket_sales_value is the sum of that source column and keeps its
-- meaning; it is deliberately NOT called spend or price.
--
-- Discount columns are summed with their source sign convention: RETAIL_DISC is
-- recorded negative (a reduction), COUPON_DISC and COUPON_MATCH_DISC are never
-- positive. The sums are factual aggregates of source columns, not prices.
--
-- Empirical anomalies are FLAGGED with counts and never removed:
--   qa_quantity_zero_lines          QUANTITY = 0 on the line
--   qa_retail_disc_positive_lines   RETAIL_DISC > 0 (26 are float epsilon, 10 are
--                                   monetary positives >= $0.01; see DATA_AUDIT.md)
--   qa_implied_paid_negative_lines  SALES_VALUE + COUPON_DISC < 0 (implied negative
--                                   customer-paid line amount; Phase 1 finding)
--
-- product is LEFT-joined so a hypothetical unmatched product cannot silently
-- drop a transaction line; the QA suite asserts that zero unmatched products
-- exist, so distinct_commodities is complete.

SELECT
    t.household_key,
    t.BASKET_ID                                                    AS basket_id,
    min(t.DAY)                                                     AS day,
    min(t.WEEK_NO)                                                 AS week_no,
    min(t.STORE_ID)                                                AS store_id,
    sum(t.SALES_VALUE)                                             AS basket_sales_value,
    sum(t.QUANTITY)                                                AS basket_units,
    count(DISTINCT t.PRODUCT_ID)                                   AS distinct_products,
    count(DISTINCT p.COMMODITY_DESC)                               AS distinct_commodities,
    count(*)                                                       AS transaction_line_count,
    sum(t.RETAIL_DISC)                                             AS basket_retail_disc,
    sum(t.COUPON_DISC)                                             AS basket_coupon_disc,
    sum(t.COUPON_MATCH_DISC)                                       AS basket_coupon_match_disc,
    sum(CASE WHEN t.QUANTITY = 0 THEN 1 ELSE 0 END)                AS qa_quantity_zero_lines,
    sum(CASE WHEN t.RETAIL_DISC > 0 THEN 1 ELSE 0 END)             AS qa_retail_disc_positive_lines,
    sum(CASE WHEN t.SALES_VALUE + t.COUPON_DISC < 0 THEN 1 ELSE 0 END) AS qa_implied_paid_negative_lines
FROM transaction_data AS t
LEFT JOIN product AS p ON p.PRODUCT_ID = t.PRODUCT_ID
GROUP BY t.household_key, t.BASKET_ID
ORDER BY t.household_key, t.BASKET_ID
