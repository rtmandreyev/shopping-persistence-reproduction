-- Phase 2B C: choice_household_product_week
-- Grain : household_key x product_id x week_no
-- Key   : (household_key, product_id, week_no) -- unique
-- Population: OBSERVED purchase cells only (2,519,013 keys). It is SPARSE and
--   event-based; the full household x product x week universe (2,500 x 92,353 x
--   102) is deliberately NOT materialised.
--
-- Reusable factual ingredients for Phase 5 discrete choice: observed purchase
-- facts, household realized discounts, and strictly-prior purchase history. It
-- does NOT choose a commodity, does NOT build choice_long, does NOT define a
-- choice set, and does NOT infer availability from non-purchase.
--
-- PRICE CANDIDATES ARE NOT HERE. The Phase 1 vendor formulas are exact at
-- TRANSACTION-LINE grain (docs/DATA_AUDIT.md, price semantics), so they live in
-- choice_line_price_candidates at that grain. Aggregating them to this
-- household-product-week grain would silently change the measure (Phase 2B
-- verified that 864 line-level negative implied amounts become 0 at week grain),
-- so no aggregated proxy is created here. Any weekly price must be constructed by
-- the analysis with an explicitly chosen aggregation.
--
-- PRIOR HISTORY is strictly earlier information: for a row at week t every
-- `prior_*` / `weeks_since_*` column uses weeks t' < t for the same household
-- (window ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING). `weeks_since_*` is
-- NULL when there is no prior purchase, and `prior_*` counters are 0 there.
-- No future information enters any column.

WITH base AS (
    SELECT
        t.household_key,
        t.PRODUCT_ID             AS product_id,
        t.WEEK_NO                AS week_no,
        min(p.BRAND)             AS brand,
        min(p.COMMODITY_DESC)    AS commodity_desc,
        count(*)                 AS purchase_lines,
        sum(t.QUANTITY)          AS units,
        sum(t.SALES_VALUE)       AS sales_value,
        sum(t.RETAIL_DISC)       AS retail_disc,
        sum(t.COUPON_DISC)       AS coupon_disc,
        sum(t.COUPON_MATCH_DISC) AS coupon_match_disc,
        count(DISTINCT t.STORE_ID) AS distinct_stores
    FROM transaction_data AS t
    LEFT JOIN product AS p ON p.PRODUCT_ID = t.PRODUCT_ID
    GROUP BY t.household_key, t.PRODUCT_ID, t.WEEK_NO
),
product_prior AS (
    SELECT
        household_key, product_id, week_no,
        sum(purchase_lines) OVER (PARTITION BY household_key, product_id ORDER BY week_no
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_purchase_lines,
        count(*) OVER (PARTITION BY household_key, product_id ORDER BY week_no
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_purchase_weeks,
        sum(units) OVER (PARTITION BY household_key, product_id ORDER BY week_no
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_units,
        max(week_no) OVER (PARTITION BY household_key, product_id ORDER BY week_no
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_product_last_week
    FROM base
),
brand_week AS (
    SELECT household_key, brand, week_no, sum(purchase_lines) AS lines
    FROM base GROUP BY household_key, brand, week_no
),
brand_prior AS (
    SELECT
        household_key, brand, week_no,
        sum(lines) OVER (PARTITION BY household_key, brand ORDER BY week_no
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_brand_purchase_lines,
        max(week_no) OVER (PARTITION BY household_key, brand ORDER BY week_no
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_brand_last_week
    FROM brand_week
),
commodity_week AS (
    SELECT household_key, commodity_desc, week_no, sum(purchase_lines) AS lines
    FROM base GROUP BY household_key, commodity_desc, week_no
),
commodity_prior AS (
    SELECT
        household_key, commodity_desc, week_no,
        sum(lines) OVER (PARTITION BY household_key, commodity_desc ORDER BY week_no
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_commodity_purchase_lines,
        max(week_no) OVER (PARTITION BY household_key, commodity_desc ORDER BY week_no
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_commodity_last_week
    FROM commodity_week
)
SELECT
    b.household_key,
    b.product_id,
    b.week_no,
    b.brand,
    b.commodity_desc,
    b.purchase_lines,
    b.units,
    b.sales_value,
    b.retail_disc,
    b.coupon_disc,
    b.coupon_match_disc,
    b.distinct_stores,
    COALESCE(pp.prior_purchase_weeks, 0)   AS prior_purchase_weeks,
    COALESCE(pp.prior_purchase_lines, 0)   AS prior_purchase_lines,
    COALESCE(pp.prior_units, 0)            AS prior_units,
    b.week_no - pp.prior_product_last_week AS weeks_since_prior_product_purchase,
    COALESCE(bp.prior_brand_purchase_lines, 0) AS prior_brand_purchase_lines,
    b.week_no - bp.prior_brand_last_week       AS weeks_since_prior_brand_purchase,
    COALESCE(cp.prior_commodity_purchase_lines, 0) AS prior_commodity_purchase_lines,
    b.week_no - cp.prior_commodity_last_week       AS weeks_since_prior_commodity_purchase,
    CASE WHEN COALESCE(pp.prior_purchase_weeks, 0) > 0 THEN 1 ELSE 0 END AS previously_purchased_product
FROM base AS b
LEFT JOIN product_prior AS pp
    ON pp.household_key = b.household_key AND pp.product_id = b.product_id AND pp.week_no = b.week_no
LEFT JOIN brand_prior AS bp
    ON bp.household_key = b.household_key AND bp.brand = b.brand AND bp.week_no = b.week_no
LEFT JOIN commodity_prior AS cp
    ON cp.household_key = b.household_key AND cp.commodity_desc = b.commodity_desc AND cp.week_no = b.week_no
ORDER BY b.household_key, b.product_id, b.week_no
