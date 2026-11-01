-- 07b_product_categories.sql
-- Purpose: category sizes and purchase coverage, to inform the LATER choice of a
--          discrete-choice commodity. This query reports; it does not recommend.
-- Grain: one row per commodity.
--
-- coverage = households buying the commodity / all transacting households.
-- A later design wants high coverage, many products/brands, and repeat choice
-- occasions; all three are reported so the decision can be made on evidence.
-- Provenance: B = observed.

WITH tx_hh AS (SELECT count(DISTINCT household_key) AS n FROM transaction_data),
     purchased AS (
         SELECT p.COMMODITY_DESC,
                count(*) AS line_count,
                count(DISTINCT t.household_key) AS households,
                count(DISTINCT t.PRODUCT_ID) AS products_bought
         FROM transaction_data t
         JOIN product p ON p.PRODUCT_ID = t.PRODUCT_ID
         GROUP BY 1
     )
SELECT
    p.COMMODITY_DESC AS commodity,
    count(*) AS products_in_assortment,
    count(DISTINCT p.PRODUCT_ID) AS distinct_products,
    count(DISTINCT p.SUB_COMMODITY_DESC) AS sub_commodities,
    count(DISTINCT p.MANUFACTURER) AS manufacturers,
    count(DISTINCT p.BRAND) AS brands,
    count(DISTINCT CASE WHEN p.BRAND = 'National' THEN p.PRODUCT_ID END) AS national_products,
    count(DISTINCT CASE WHEN p.BRAND = 'Private' THEN p.PRODUCT_ID END) AS private_products,
    coalesce(pr.line_count, 0) AS transaction_line_count,
    coalesce(pr.households, 0) AS purchasing_households,
    round(100.0 * coalesce(pr.households, 0) / (SELECT n FROM tx_hh), 4) AS household_coverage_pct,
    coalesce(pr.products_bought, 0) AS products_actually_bought
FROM product p
LEFT JOIN purchased pr ON pr.COMMODITY_DESC = p.COMMODITY_DESC
GROUP BY p.COMMODITY_DESC, pr.line_count, pr.households, pr.products_bought
-- Commodity is the declared and unique output grain. Department/commodity pair
-- counts are reported separately in the hierarchy summary, never mixed here.
ORDER BY transaction_line_count DESC, products_in_assortment DESC,
         commodity ASC;
