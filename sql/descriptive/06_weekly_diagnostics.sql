-- Phase 3 Question A: per-week mechanical diagnostics for extreme-week challenges
-- Grain : week_no
-- Sources: household_week_base, basket_panel, choice_line_price_candidates
--
-- Step 5 of the Question A brief: for every apparently extreme week, ask whether it
-- is driven by a few households, a few baskets, extreme quantity lines, flagged
-- transaction anomalies, or unusual mix. The headline series stay AS IS; the
-- alternative totals here exist only so a labelled sensitivity can be reported
-- beside the baseline (never instead of it).
--
-- choice_line_price_candidates is the accepted line-grain table. It holds
-- positive-quantity lines only: the 14,466 QUANTITY = 0 lines are absent from it by
-- construction (they remain in the canonical tables), so its window total is
-- 19.28 below the accepted retailer SALES_VALUE total. The caller states that
-- caveat wherever a line-grain figure is reported.

WITH household_week AS (
    SELECT week_no, household_key, weekly_sales_value
    FROM household_week_base
),

ranked_household AS (
    SELECT
        week_no,
        household_key,
        weekly_sales_value,
        row_number() OVER (
            PARTITION BY week_no ORDER BY weekly_sales_value DESC, household_key
        ) AS rank_by_sales_value
    FROM household_week
),

household_concentration AS (
    SELECT
        week_no,
        sum(weekly_sales_value)                                            AS week_sales_value,
        max(weekly_sales_value)                                            AS top_household_sales_value,
        max(CASE WHEN rank_by_sales_value = 1 THEN weekly_sales_value END)
            / nullif(sum(weekly_sales_value), 0)                           AS top1_household_sales_value_share,
        sum(CASE WHEN rank_by_sales_value <= 10 THEN weekly_sales_value ELSE 0 END)
            / nullif(sum(weekly_sales_value), 0)                           AS top10_household_sales_value_share,
        sum(weekly_sales_value) - max(weekly_sales_value)                   AS sales_value_excluding_top_household
    FROM ranked_household
    GROUP BY week_no
),

basket_concentration AS (
    SELECT
        week_no,
        count(*)                                       AS baskets,
        max(basket_sales_value)                        AS max_basket_sales_value,
        max(basket_sales_value) / nullif(sum(basket_sales_value), 0) AS max_basket_sales_value_share
    FROM basket_panel
    GROUP BY week_no
),

line_diagnostics AS (
    SELECT
        week_no,
        count(*)                                                    AS line_rows,
        max(quantity)                                               AS max_line_quantity,
        sum(sales_value)                                            AS line_sales_value,
        sum(quantity)                                               AS line_units,
        count(*) FILTER (WHERE quantity >= 1000)                     AS extreme_quantity_lines,
        sum(quantity) FILTER (WHERE quantity >= 1000)                AS extreme_quantity_units,
        sum(sales_value) FILTER (WHERE quantity >= 1000)             AS extreme_quantity_sales_value,
        sum(sales_value) FILTER (WHERE quantity < 1000)              AS line_sales_value_excluding_extreme_quantity,
        sum(quantity) FILTER (WHERE quantity < 1000)                 AS line_units_excluding_extreme_quantity,
        count(*) FILTER (
            WHERE abs(retail_disc) + abs(coupon_disc) + abs(coupon_match_disc) > sales_value
        )                                                           AS discount_exceedance_lines,
        sum(quantity) FILTER (
            WHERE abs(retail_disc) + abs(coupon_disc) + abs(coupon_match_disc) > sales_value
        )                                                           AS discount_exceedance_units
    FROM choice_line_price_candidates
    GROUP BY week_no
)

SELECT
    h.week_no,
    h.week_sales_value,
    h.top_household_sales_value,
    h.top1_household_sales_value_share,
    h.top10_household_sales_value_share,
    h.sales_value_excluding_top_household,
    b.baskets,
    b.max_basket_sales_value,
    b.max_basket_sales_value_share,
    l.line_rows,
    l.max_line_quantity,
    l.line_sales_value,
    l.line_units,
    l.extreme_quantity_lines,
    l.extreme_quantity_units,
    l.extreme_quantity_units / nullif(l.line_units, 0)              AS extreme_quantity_unit_share,
    l.extreme_quantity_sales_value,
    l.line_sales_value_excluding_extreme_quantity,
    l.line_units_excluding_extreme_quantity,
    l.discount_exceedance_lines,
    l.discount_exceedance_units,
    l.discount_exceedance_units / nullif(l.line_units, 0)           AS discount_exceedance_unit_share
FROM household_concentration AS h
LEFT JOIN basket_concentration AS b ON b.week_no = h.week_no
LEFT JOIN line_diagnostics AS l ON l.week_no = h.week_no
ORDER BY h.week_no
