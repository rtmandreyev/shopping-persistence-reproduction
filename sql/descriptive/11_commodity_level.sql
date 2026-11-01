-- Phase 3 Question C: commodity-level descriptive object (all 308 commodity labels)
-- Grain  : commodity_desc (one row per commodity, sentinel retained)
-- Sources: household_category_week (accepted), choice_household_product_week (accepted
--          choice-layer input, used descriptively only)
--
-- Semantics carried through unchanged:
--   * every sales-value column is retailer SALES_VALUE / retailer revenue, never customer
--     spend, grocery expenditure or a price;
--   * household_category_week is sparse and purchase-conditional: absence of a row does
--     NOT imply zero household consumption of that commodity;
--   * observed-household penetration = distinct households with at least one observed
--     purchase in the commodity / 2,500 observed transaction-universe households. It is
--     NEVER called market penetration;
--   * recorded units are carried with the Question A finding QA-03 in mind (unit totals
--     are dominated by very large-quantity lines) and are never the headline ranking;
--   * national/private composition is reported in BOTH retailer SALES_VALUE and recorded
--     units, each labelled, and never as one generic brand share;
--   * `trips` at commodity grain sums the commodity's household-commodity-week trip counts,
--     so it double-counts baskets ACROSS commodities and is not a basket total.
--
-- Thirds are the accepted 34-week thirds: early 1-34, middle 35-68, late 69-102.
-- Question A finding QA-01 applies: early-third levels are affected by observed support
-- formation, so an early-versus-late difference is not automatically product growth.

WITH household_commodity AS (
    SELECT
        commodity_desc,
        household_key,
        count(*)                                    AS purchase_weeks,
        sum(category_sales_value)                   AS household_sales_value,
        sum(category_units)                         AS household_units,
        sum(category_trips)                         AS household_trips,
        max(category_sales_value)                   AS household_max_week_sales_value
    FROM household_category_week
    GROUP BY commodity_desc, household_key
),

breadth AS (
    SELECT
        commodity_desc,
        count(*)                                                        AS buying_households,
        count(*)::DOUBLE / 2500.0                                       AS observed_household_penetration,
        sum(purchase_weeks)                                             AS household_weeks,
        sum(household_trips)                                            AS commodity_trips,
        sum(household_sales_value)                                      AS retailer_sales_value,
        sum(household_units)                                            AS recorded_units,
        median(purchase_weeks)                                          AS median_purchase_weeks,
        quantile_cont(purchase_weeks, 0.75)                             AS p75_purchase_weeks,
        quantile_cont(purchase_weeks, 0.90)                             AS p90_purchase_weeks,
        max(purchase_weeks)                                             AS max_purchase_weeks,
        count(*) FILTER (WHERE purchase_weeks >= 2)                      AS households_2plus_weeks,
        count(*) FILTER (WHERE purchase_weeks >= 3)                      AS households_3plus_weeks,
        count(*) FILTER (WHERE purchase_weeks >= 5)                      AS households_5plus_weeks,
        median(household_sales_value)                                   AS median_household_sales_value,
        quantile_cont(household_sales_value, 0.90)                      AS p90_household_sales_value,
        max(household_sales_value) / nullif(sum(household_sales_value), 0)
                                                                        AS top_household_sales_value_share
    FROM household_commodity
    GROUP BY commodity_desc
),

labels AS (
    SELECT
        commodity_desc,
        sum(category_national_sales_value)                              AS national_sales_value,
        sum(category_private_sales_value)                               AS private_sales_value,
        sum(category_national_units)                                    AS national_units,
        sum(category_private_units)                                     AS private_units,
        max(distinct_national_products)                                 AS max_household_week_national_products,
        max(distinct_private_products)                                  AS max_household_week_private_products
    FROM household_category_week
    GROUP BY commodity_desc
),

thirds AS (
    SELECT
        commodity_desc,
        count(DISTINCT household_key) FILTER (WHERE week_no <= 34)       AS buying_households_early,
        count(DISTINCT household_key) FILTER (WHERE week_no BETWEEN 35 AND 68)  AS buying_households_middle,
        count(DISTINCT household_key) FILTER (WHERE week_no >= 69)       AS buying_households_late,
        count(*) FILTER (WHERE week_no <= 34)                            AS household_weeks_early,
        count(*) FILTER (WHERE week_no BETWEEN 35 AND 68)                AS household_weeks_middle,
        count(*) FILTER (WHERE week_no >= 69)                            AS household_weeks_late,
        sum(category_sales_value) FILTER (WHERE week_no <= 34)           AS sales_value_early,
        sum(category_sales_value) FILTER (WHERE week_no BETWEEN 35 AND 68) AS sales_value_middle,
        sum(category_sales_value) FILTER (WHERE week_no >= 69)           AS sales_value_late,
        sum(category_trips) FILTER (WHERE week_no <= 34)                 AS trips_early,
        sum(category_trips) FILTER (WHERE week_no BETWEEN 35 AND 68)     AS trips_middle,
        sum(category_trips) FILTER (WHERE week_no >= 69)                 AS trips_late,
        count(*) FILTER (WHERE category_sales_value > 0)                 AS weeks_with_activity
    FROM household_category_week
    GROUP BY commodity_desc
),

product_side AS (
    SELECT
        commodity_desc,
        count(DISTINCT product_id)                                      AS distinct_products,
        count(DISTINCT household_key)                                   AS households_with_product_rows,
        count(*)                                                        AS household_product_weeks,
        count(*) FILTER (WHERE prior_commodity_purchase_lines > 0)       AS household_product_weeks_with_prior_commodity_purchase,
        count(*) FILTER (WHERE previously_purchased_product = 1)         AS household_product_weeks_repeat_product,
        sum(purchase_lines)                                             AS purchase_lines,
        sum(units)                                                      AS product_rows_units,
        sum(sales_value)                                                AS product_rows_sales_value
    FROM choice_household_product_week
    GROUP BY commodity_desc
),

assortment AS (
    SELECT
        commodity_desc,
        median(products_per_household)                                  AS median_products_per_household,
        quantile_cont(products_per_household, 0.90)                     AS p90_products_per_household,
        max(products_per_household)                                     AS max_products_per_household
    FROM (
        SELECT commodity_desc, household_key, count(DISTINCT product_id) AS products_per_household
        FROM choice_household_product_week
        GROUP BY commodity_desc, household_key
    )
    GROUP BY commodity_desc
)

SELECT
    w.commodity_desc,
    w.buying_households,
    w.observed_household_penetration,
    w.household_weeks,
    w.commodity_trips,
    w.retailer_sales_value,
    w.recorded_units,
    p.distinct_products,
    w.retailer_sales_value / nullif(w.household_weeks, 0)               AS sales_value_per_buying_household_week,
    w.commodity_trips::DOUBLE / nullif(w.household_weeks, 0)            AS trips_per_buying_household_week,
    w.retailer_sales_value / nullif(w.buying_households, 0)             AS sales_value_per_buying_household,
    w.median_household_sales_value,
    w.p90_household_sales_value,
    w.top_household_sales_value_share,
    w.median_purchase_weeks,
    w.p75_purchase_weeks,
    w.p90_purchase_weeks,
    w.max_purchase_weeks,
    w.households_2plus_weeks,
    w.households_3plus_weeks,
    w.households_5plus_weeks,
    w.households_2plus_weeks::DOUBLE / nullif(w.buying_households, 0)   AS share_households_2plus_weeks,
    w.households_3plus_weeks::DOUBLE / nullif(w.buying_households, 0)   AS share_households_3plus_weeks,
    w.households_5plus_weeks::DOUBLE / nullif(w.buying_households, 0)   AS share_households_5plus_weeks,
    s.median_products_per_household,
    s.p90_products_per_household,
    s.max_products_per_household,
    p.household_product_weeks,
    p.household_product_weeks_with_prior_commodity_purchase,
    p.household_product_weeks_with_prior_commodity_purchase::DOUBLE
        / nullif(p.household_product_weeks, 0)                          AS share_household_product_weeks_with_prior_commodity_purchase,
    p.household_product_weeks_repeat_product::DOUBLE
        / nullif(p.household_product_weeks, 0)                          AS share_household_product_weeks_repeat_product,
    l.national_sales_value,
    l.private_sales_value,
    l.national_sales_value / nullif(l.national_sales_value + l.private_sales_value, 0)
                                                                        AS national_share_of_sales_value,
    l.private_sales_value / nullif(l.national_sales_value + l.private_sales_value, 0)
                                                                        AS private_share_of_sales_value,
    l.national_units,
    l.private_units,
    l.national_units / nullif(l.national_units + l.private_units, 0)     AS national_share_of_units,
    l.private_units / nullif(l.national_units + l.private_units, 0)      AS private_share_of_units,
    t.buying_households_early,
    t.buying_households_middle,
    t.buying_households_late,
    t.household_weeks_early,
    t.household_weeks_middle,
    t.household_weeks_late,
    t.sales_value_early,
    t.sales_value_middle,
    t.sales_value_late,
    t.trips_early,
    t.trips_middle,
    t.trips_late,
    t.weeks_with_activity,
    CASE
        WHEN t.buying_households_early > 0 AND t.buying_households_middle > 0
             AND t.buying_households_late > 0
        THEN 1 ELSE 0
    END                                                                 AS present_in_all_three_thirds
FROM breadth AS w
LEFT JOIN labels AS l USING (commodity_desc)
LEFT JOIN thirds AS t USING (commodity_desc)
LEFT JOIN product_side AS p USING (commodity_desc)
LEFT JOIN assortment AS s USING (commodity_desc)
ORDER BY commodity_desc
