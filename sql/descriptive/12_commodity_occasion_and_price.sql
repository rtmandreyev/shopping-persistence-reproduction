-- Phase 3 Question C: commodity purchase-occasion diagnostics and candidate-price usability
-- Grain  : commodity_desc
-- Sources: choice_line_price_candidates (accepted), product (accepted raw line-level metadata)
--
-- Semantics and caveats that travel with every column below:
--   * choice_line_price_candidates covers POSITIVE-QUANTITY transaction lines only
--     (2,581,266 lines; the 14,466 QUANTITY = 0 lines are excluded by construction because
--     the price formula is undefined there). Line, basket and household counts here
--     therefore omit zero-quantity lines, and the caller states that wherever a count is
--     reported.
--   * "usable candidate" means the already-accepted Phase 2 formula returns a non-null,
--     strictly positive value for that line. It does NOT mean the candidate has been
--     selected as the economic price for Phase 5. The two candidates are never averaged
--     together and never ranked against each other here.
--   * multiple products from the same commodity in one basket are MULTIPLE COMMODITY
--     PRODUCTS PURCHASED IN THAT BASKET. That is not evidence that the products were
--     mutually exclusive alternatives, that all products were available, or that the basket
--     is the correct discrete-choice occasion. These are purchase-occasion diagnostics.
--   * purchased product-store-week presence is NOT shelf availability, and absence is NOT
--     "no placement"; causal_data is never used as an availability measure in this run.
--   * package size comes from the accepted CURR_SIZE_OF_PRODUCT field and is reported as
--     completeness only, never as a statement that prices are comparable across sizes.

WITH lines AS (
    SELECT
        p.COMMODITY_DESC                        AS commodity_desc,
        l.basket_id,
        l.household_key,
        l.product_id,
        l.quantity,
        l.sales_value,
        l.vendor_loyalty_unit_price_candidate   AS loyal_candidate,
        l.vendor_nonloyalty_unit_price_candidate AS nonloyal_candidate,
        p.CURR_SIZE_OF_PRODUCT                   AS package_size
    FROM choice_line_price_candidates AS l
    JOIN product AS p ON l.product_id = p.PRODUCT_ID
),

line_metrics AS (
    SELECT
        commodity_desc,
        count(*)                                                        AS transaction_lines,
        count(*) FILTER (WHERE quantity > 0)                            AS positive_quantity_lines,
        count(DISTINCT product_id)                                      AS products_with_lines,
        count(DISTINCT household_key)                                   AS households_with_lines,
        count(DISTINCT basket_id)                                       AS commodity_baskets,
        count(*) FILTER (WHERE loyal_candidate IS NOT NULL AND loyal_candidate > 0) AS loyal_candidate_usable_lines,
        count(*) FILTER (WHERE nonloyal_candidate IS NOT NULL AND nonloyal_candidate > 0) AS nonloyal_candidate_usable_lines,
        count(*) FILTER (WHERE loyal_candidate IS NOT NULL AND loyal_candidate > 0)::DOUBLE
            / nullif(count(*), 0)                                       AS loyal_candidate_usable_share,
        count(*) FILTER (WHERE nonloyal_candidate IS NOT NULL AND nonloyal_candidate > 0)::DOUBLE
            / nullif(count(*), 0)                                       AS nonloyal_candidate_usable_share,
        count(DISTINCT product_id) FILTER (WHERE package_size IS NOT NULL AND package_size <> '') AS products_with_package_size,
        count(DISTINCT product_id) FILTER (WHERE package_size IS NOT NULL AND package_size <> '')::DOUBLE
            / nullif(count(DISTINCT product_id), 0)                     AS package_size_completeness_share,
        count(*) FILTER (WHERE quantity >= 1000)                          AS bulk_lines,
        max(quantity)                                                   AS max_line_quantity,
        sum(quantity)                                                   AS recorded_units,
        -- coalesce: a commodity with no QUANTITY >= 1000 line has zero bulk units, not an
        -- undefined share. Reporting NULL there would have hidden the fact that the bulk
        -- phenomenon is concentrated in a couple of labels rather than spread across the
        -- 308 commodities.
        coalesce(sum(quantity) FILTER (WHERE quantity >= 1000), 0)        AS bulk_units,
        coalesce(sum(quantity) FILTER (WHERE quantity >= 1000), 0)::DOUBLE
            / nullif(sum(quantity), 0)                                   AS bulk_unit_share,
        sum(sales_value)                                                AS line_sales_value,
        coalesce(sum(sales_value) FILTER (WHERE quantity >= 1000), 0)     AS bulk_line_sales_value,
        coalesce(sum(sales_value) FILTER (WHERE quantity >= 1000), 0)::DOUBLE
            / nullif(sum(sales_value), 0)                                AS bulk_line_sales_value_share
    FROM lines
    GROUP BY commodity_desc
),

basket_metrics AS (
    SELECT
        commodity_desc,
        count(*)                                                        AS baskets_with_commodity,
        max(products_in_basket)                                         AS max_products_from_commodity_in_a_basket,
        count(*) FILTER (WHERE products_in_basket = 1)                   AS baskets_single_commodity_product,
        count(*) FILTER (WHERE products_in_basket >= 2)                  AS baskets_multi_commodity_product,
        count(*) FILTER (WHERE products_in_basket >= 2)::DOUBLE / nullif(count(*), 0) AS share_baskets_multi_commodity_product,
        median(products_in_basket)                                      AS median_products_from_commodity_per_basket,
        quantile_cont(products_in_basket, 0.90)                         AS p90_products_from_commodity_per_basket,
        sum(commodity_basket_sales_value)                               AS basket_side_line_sales_value,
        max(commodity_basket_sales_value)                                AS top_basket_line_sales_value
    FROM (
        SELECT
            commodity_desc,
            basket_id,
            count(DISTINCT product_id)              AS products_in_basket,
            sum(sales_value)                        AS commodity_basket_sales_value
        FROM lines
        GROUP BY commodity_desc, basket_id
    )
    GROUP BY commodity_desc
),

household_metrics AS (
    SELECT
        commodity_desc,
        count(*)                                                        AS households_with_baskets,
        count(*) FILTER (WHERE commodity_baskets >= 2)                   AS households_with_repeated_commodity_baskets,
        count(*) FILTER (WHERE commodity_baskets >= 2)::DOUBLE / nullif(count(*), 0) AS share_households_with_repeated_commodity_baskets,
        median(commodity_baskets)                                       AS median_baskets_per_buying_household,
        quantile_cont(commodity_baskets, 0.90)                          AS p90_baskets_per_buying_household,
        max(commodity_baskets)                                          AS max_baskets_per_household
    FROM (
        SELECT commodity_desc, household_key, count(DISTINCT basket_id) AS commodity_baskets
        FROM lines
        GROUP BY commodity_desc, household_key
    )
    GROUP BY commodity_desc
)

SELECT
    l.commodity_desc,
    l.transaction_lines,
    l.positive_quantity_lines,
    l.products_with_lines,
    l.households_with_lines,
    l.commodity_baskets,
    l.loyal_candidate_usable_lines,
    l.nonloyal_candidate_usable_lines,
    l.loyal_candidate_usable_share,
    l.nonloyal_candidate_usable_share,
    l.products_with_package_size,
    l.package_size_completeness_share,
    l.bulk_lines,
    l.max_line_quantity,
    l.recorded_units,
    l.bulk_units,
    l.bulk_unit_share,
    l.line_sales_value,
    l.bulk_line_sales_value,
    l.bulk_line_sales_value_share,
    b.baskets_with_commodity,
    b.max_products_from_commodity_in_a_basket,
    b.baskets_single_commodity_product,
    b.baskets_multi_commodity_product,
    b.share_baskets_multi_commodity_product,
    b.median_products_from_commodity_per_basket,
    b.p90_products_from_commodity_per_basket,
    b.top_basket_line_sales_value / nullif(b.basket_side_line_sales_value, 0) AS top_basket_share_of_commodity_line_sales_value,
    h.households_with_baskets,
    h.households_with_repeated_commodity_baskets,
    h.share_households_with_repeated_commodity_baskets,
    h.median_baskets_per_buying_household,
    h.p90_baskets_per_buying_household,
    h.max_baskets_per_household
FROM line_metrics AS l
LEFT JOIN basket_metrics AS b USING (commodity_desc)
LEFT JOIN household_metrics AS h USING (commodity_desc)
ORDER BY l.commodity_desc
