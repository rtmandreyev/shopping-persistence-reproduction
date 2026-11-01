-- 06_price_reconciliation.sql
-- Purpose: test the vendor's price-accounting identities empirically WITHOUT
--          constructing a final price measure.
--
-- Vendor guide (A) states for transaction_data:
--   sales_value = amount of dollars the RETAILER receives for the line, after
--                 coupon_match and loyalty discount are taken into account.
--   loyalty card price      = (sales_value - (retail_disc + coupon_match_disc)) / quantity
--   non-loyalty card price  = (sales_value - coupon_match_disc) / quantity
--   if a customer uses a manufacturer coupon, customer price < sales_value,
--   because the manufacturer reimburses the retailer.
--
-- The guide does NOT publish a single algebraic identity tying sales_value and
-- the three discount columns together, so these are RECONSTRUCTION checks:
-- do the guide's own price formulas produce sensible, positive prices, and is
-- the sign of each discount the one the guide implies? We do not assert a
-- reconciliation equation the vendor never stated.
--
-- Provenance: A = the guide's stated formulas, B = empirical test of them.

WITH all_rows AS (
    SELECT
        QUANTITY,
        SALES_VALUE,
        RETAIL_DISC,
        COUPON_DISC,
        COUPON_MATCH_DISC,
        (RETAIL_DISC IS NULL)::INT + (COUPON_DISC IS NULL)::INT
            + (COUPON_MATCH_DISC IS NULL)::INT AS null_discount_cols
    FROM transaction_data
),
base AS (
    SELECT *
    FROM all_rows
    WHERE QUANTITY > 0
)

-- NOTE on rounding: avg() over DOUBLE is summed in whatever order DuckDB's
-- parallel scan produces, so the last decimal places can vary between runs on an
-- identical file. Every mean is rounded to 6 decimals so the artifact is
-- byte-stable across runs. This was observed, not assumed: two consecutive runs
-- differed in the final digit of these means.
SELECT 'guide_formula_loyalty_unit_price' AS check_name,
       'A guide formula, B empirical' AS provenance,
       count(*) AS rows_evaluated,
       round(min((SALES_VALUE - (RETAIL_DISC + COUPON_MATCH_DISC)) / QUANTITY), 6) AS min_value,
       round(avg((SALES_VALUE - (RETAIL_DISC + COUPON_MATCH_DISC)) / QUANTITY), 6) AS mean_value,
       round(max((SALES_VALUE - (RETAIL_DISC + COUPON_MATCH_DISC)) / QUANTITY), 6) AS max_value,
       count(*) FILTER (
           WHERE (SALES_VALUE - (RETAIL_DISC + COUPON_MATCH_DISC)) / QUANTITY < 0
       ) AS negative_count,
       count(*) FILTER (
           WHERE (SALES_VALUE - (RETAIL_DISC + COUPON_MATCH_DISC)) / QUANTITY = 0
       ) AS zero_count
FROM base

UNION ALL
SELECT 'guide_formula_nonloyalty_unit_price',
       'A guide formula, B empirical',
       count(*),
       round(min((SALES_VALUE - COUPON_MATCH_DISC) / QUANTITY), 6),
       round(avg((SALES_VALUE - COUPON_MATCH_DISC) / QUANTITY), 6),
       round(max((SALES_VALUE - COUPON_MATCH_DISC) / QUANTITY), 6),
       count(*) FILTER (WHERE (SALES_VALUE - COUPON_MATCH_DISC) / QUANTITY < 0),
       count(*) FILTER (WHERE (SALES_VALUE - COUPON_MATCH_DISC) / QUANTITY = 0)
FROM base

UNION ALL
-- Guide-derived loyalty line amount before division by positive quantity.
SELECT 'guide_formula_loyalty_line_amount',
       'A guide formula before unit-price division; B empirical',
       count(*),
       round(min(SALES_VALUE - (RETAIL_DISC + COUPON_MATCH_DISC)), 6),
       round(avg(SALES_VALUE - (RETAIL_DISC + COUPON_MATCH_DISC)), 6),
       round(max(SALES_VALUE - (RETAIL_DISC + COUPON_MATCH_DISC)), 6),
       count(*) FILTER (WHERE SALES_VALUE - (RETAIL_DISC + COUPON_MATCH_DISC) < 0),
       count(*) FILTER (WHERE SALES_VALUE - (RETAIL_DISC + COUPON_MATCH_DISC) = 0)
FROM base

UNION ALL
SELECT 'implied_customer_paid_line_amount_quantity_positive',
       'C inferred by us: sales_value is stated to exceed customer-paid line amount when a '
       || 'manufacturer coupon is used',
       count(*),
       round(min(SALES_VALUE + COUPON_DISC), 6),
       round(avg(SALES_VALUE + COUPON_DISC), 6),
       round(max(SALES_VALUE + COUPON_DISC), 6),
       count(*) FILTER (WHERE SALES_VALUE + COUPON_DISC < 0),
       count(*) FILTER (WHERE SALES_VALUE + COUPON_DISC = 0)
FROM base

UNION ALL
SELECT 'implied_customer_paid_line_amount_all_rows',
       'C inferred by us; includes quantity-zero rows and is not a per-unit price',
       count(*),
       round(min(SALES_VALUE + COUPON_DISC), 6),
       round(avg(SALES_VALUE + COUPON_DISC), 6),
       round(max(SALES_VALUE + COUPON_DISC), 6),
       count(*) FILTER (WHERE SALES_VALUE + COUPON_DISC < 0),
       count(*) FILTER (WHERE SALES_VALUE + COUPON_DISC = 0)
FROM all_rows

UNION ALL
-- Distribution of raw SALES_VALUE, for comparison with quantity so the ratio can
-- be sanity-checked later without committing to a price definition now.
SELECT 'raw_sales_line_amount',
       'B observed',
       count(*),
       round(min(SALES_VALUE), 6), round(avg(SALES_VALUE), 6), round(max(SALES_VALUE), 6),
       count(*) FILTER (WHERE SALES_VALUE < 0),
       count(*) FILTER (WHERE SALES_VALUE = 0)
FROM base

UNION ALL
SELECT 'raw_quantity',
       'B observed',
       count(*), min(QUANTITY), round(avg(QUANTITY), 6), max(QUANTITY),
       count(*) FILTER (WHERE QUANTITY < 0),
       count(*) FILTER (WHERE QUANTITY = 0)
FROM base

UNION ALL
SELECT 'retail_disc',
       'B observed; guide implies negative = discount applied',
       count(*), min(RETAIL_DISC), round(avg(RETAIL_DISC), 6), max(RETAIL_DISC),
       count(*) FILTER (WHERE RETAIL_DISC < 0),
       count(*) FILTER (WHERE RETAIL_DISC = 0)
FROM base

UNION ALL
SELECT 'coupon_disc',
       'B observed; guide implies negative = manufacturer discount',
       count(*), min(COUPON_DISC), round(avg(COUPON_DISC), 6), max(COUPON_DISC),
       count(*) FILTER (WHERE COUPON_DISC < 0),
       count(*) FILTER (WHERE COUPON_DISC = 0)
FROM base

UNION ALL
SELECT 'coupon_match_disc',
       'B observed; guide implies negative = retailer match of manufacturer coupon',
       count(*), min(COUPON_MATCH_DISC), round(avg(COUPON_MATCH_DISC), 6), max(COUPON_MATCH_DISC),
       count(*) FILTER (WHERE COUPON_MATCH_DISC < 0),
       count(*) FILTER (WHERE COUPON_MATCH_DISC = 0)
FROM base

UNION ALL
-- Sign-coherence: is coupon_match_disc ever nonzero when coupon_disc is zero?
-- If the match is a top-up to a manufacturer coupon, the two should co-occur.
SELECT 'rows_match_disc_nonzero_but_coupon_disc_zero',
       'C inferred by us: tests internal coherence of the discount pair',
       count(*),
       NULL, NULL, NULL,
       count(*) FILTER (WHERE COUPON_MATCH_DISC <> 0 AND COUPON_DISC = 0),
       count(*) FILTER (WHERE COUPON_MATCH_DISC = 0 AND COUPON_DISC = 0)
FROM base

UNION ALL
SELECT 'rows_null_discount_columns',
       'B observed; a null discount is ambiguous (missing vs genuinely absent)',
       count(*), NULL, NULL, NULL,
       count(*) FILTER (WHERE null_discount_cols > 0), NULL
FROM base
