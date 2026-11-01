-- Phase 3 Question A: weekly composition diagnostics (commodity grain)
-- Grain : week_no
-- Sources: household_category_week
--
-- Mechanical-explanation check only (Step 6): when a weekly movement looks like a
-- pattern, ask whether the SAME weeks also show a material change in how broad the
-- purchased mix was or how concentrated it was in a single commodity. Nothing here
-- ranks commodities for Phase 5 and nothing here is a Question C output.
--
-- Denominators are explicit: shares divide by that week's own recorded
-- category-level retailer SALES_VALUE, which is the same universe as the weekly
-- total (asserted by the caller's reconciliation).

WITH weekly_commodity AS (
    SELECT
        week_no,
        commodity_desc,
        sum(category_sales_value)        AS commodity_sales_value,
        sum(category_units)              AS commodity_units,
        count(DISTINCT household_key)    AS buying_households
    FROM household_category_week
    GROUP BY week_no, commodity_desc
),

ranked AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY week_no ORDER BY commodity_sales_value DESC, commodity_desc
        ) AS rank_by_sales_value
    FROM weekly_commodity
)

SELECT
    week_no,
    count(*)                                                             AS active_commodities,
    sum(commodity_sales_value)                                           AS weekly_category_sales_value,
    max(CASE WHEN rank_by_sales_value = 1 THEN commodity_desc END)        AS largest_commodity_desc,
    max(CASE WHEN rank_by_sales_value = 1 THEN commodity_sales_value END) AS largest_commodity_sales_value,
    max(CASE WHEN rank_by_sales_value = 1 THEN commodity_sales_value END)
        / nullif(sum(commodity_sales_value), 0)                          AS largest_commodity_sales_value_share,
    sum(CASE WHEN rank_by_sales_value <= 5 THEN commodity_sales_value ELSE 0 END)
        / nullif(sum(commodity_sales_value), 0)                          AS top5_commodity_sales_value_share,
    sum(CASE WHEN rank_by_sales_value <= 5 THEN 1 ELSE 0 END)             AS commodities_in_top5
FROM ranked
GROUP BY week_no
ORDER BY week_no
