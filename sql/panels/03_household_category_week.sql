-- Phase 2A canonical table: household_category_week
-- Grain : household_key x commodity_desc x week_no
-- Key   : (household_key, commodity_desc, week_no) -- unique
-- Population: SPARSE. A row exists only where the household recorded a purchase
--   in that commodity in that week. This is deliberately NOT a dense
--   2,500 x 308 x 102 grid: absent commodity-weeks are unobserved, not zeros,
--   and dense expansion would fabricate millions of non-observations.
--
-- commodity_desc is taken from product.COMMODITY_DESC. Sentinel labels shipped by
-- the vendor (e.g. 'NO COMMODITY DESCRIPTION') are preserved as ordinary
-- categories and are never filtered out.
--
-- BRAND has exactly two observed values, National and Private. The national/private
-- splits are factual conditional sums; QA asserts their completeness against
-- category_sales_value so an unexpected third brand value cannot silently
-- disappear from the split.
--
-- No discrete-choice commodity is selected, no product availability is inferred,
-- and no exposure/treatment variable is created. Discount sums keep source signs.

SELECT
    t.household_key,
    p.COMMODITY_DESC                                                      AS commodity_desc,
    t.WEEK_NO                                                             AS week_no,
    sum(t.SALES_VALUE)                                                    AS category_sales_value,
    sum(t.QUANTITY)                                                       AS category_units,
    count(DISTINCT t.BASKET_ID)                                           AS category_trips,
    count(DISTINCT t.PRODUCT_ID)                                          AS distinct_products,
    sum(CASE WHEN p.BRAND = 'National' THEN t.SALES_VALUE ELSE 0 END)     AS category_national_sales_value,
    sum(CASE WHEN p.BRAND = 'Private' THEN t.SALES_VALUE ELSE 0 END)      AS category_private_sales_value,
    sum(CASE WHEN p.BRAND = 'National' THEN t.QUANTITY ELSE 0 END)        AS category_national_units,
    sum(CASE WHEN p.BRAND = 'Private' THEN t.QUANTITY ELSE 0 END)         AS category_private_units,
    count(DISTINCT CASE WHEN p.BRAND = 'National' THEN t.PRODUCT_ID END)  AS distinct_national_products,
    count(DISTINCT CASE WHEN p.BRAND = 'Private' THEN t.PRODUCT_ID END)   AS distinct_private_products,
    sum(t.RETAIL_DISC)                                                    AS category_retail_disc,
    sum(t.COUPON_DISC)                                                    AS category_coupon_disc,
    sum(t.COUPON_MATCH_DISC)                                              AS category_coupon_match_disc
FROM transaction_data AS t
LEFT JOIN product AS p ON p.PRODUCT_ID = t.PRODUCT_ID
GROUP BY t.household_key, p.COMMODITY_DESC, t.WEEK_NO
ORDER BY t.household_key, p.COMMODITY_DESC, t.WEEK_NO
