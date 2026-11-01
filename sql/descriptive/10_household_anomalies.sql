-- Phase 3 Question B: household outlier / anomaly trace
-- Grain  : household_key
-- Sources: household_week_base, basket_panel, choice_line_price_candidates (accepted)
--
-- Purpose: make it possible to ask whether an apparent concentration or heterogeneity
-- result is produced by a handful of households, by extreme baskets, by extreme
-- quantity lines (Question A finding QA-03), by flagged transaction lines, or by
-- unusually short observed purchase-support windows -- and to answer it by tracing the
-- contributing households rather than by trimming them. Nothing is trimmed anywhere.
--
-- Declared absolute screens (mirrored as constants in question_b.py; a test asserts the
-- two agree, and the screens are labelled wherever a count based on them is reported):
--   * EXTREME_QUANTITY_THRESHOLD = 1000  -- same screen as Question A's bulk-line check
--   * EXTREME_BASKET_SALES_VALUE = 500.0 -- a declared absolute basket-value screen
--
-- Units are carried only for the quantity-line diagnostics; they are never a headline
-- household-heterogeneity measure.

WITH week_side AS (
    SELECT
        household_key,
        max(weekly_sales_value)                                             AS max_week_sales_value,
        sum(weekly_sales_value)                                             AS total_sales_value,
        arg_max(week_no, weekly_sales_value)                                 AS max_week_no,
        count(*) FILTER (WHERE qa_quantity_zero_lines > 0)                   AS weeks_with_quantity_zero_lines,
        count(*) FILTER (WHERE qa_retail_disc_positive_lines > 0)            AS weeks_with_positive_retail_disc,
        count(*) FILTER (WHERE qa_implied_paid_negative_lines > 0)           AS weeks_with_negative_implied_paid
    FROM household_week_base
    GROUP BY household_key
),

basket_side AS (
    SELECT
        household_key,
        max(basket_sales_value)                                             AS max_basket_sales_value,
        arg_max(basket_id, basket_sales_value)                               AS max_basket_id,
        arg_max(week_no, basket_sales_value)                                 AS max_basket_week_no,
        sum(basket_sales_value)                                             AS total_basket_sales_value,
        count(*) FILTER (WHERE basket_sales_value >= 500.0)                  AS baskets_over_declared_screen,
        sum(CASE WHEN basket_sales_value >= 500.0 THEN basket_sales_value ELSE 0 END)
                                                                            AS sales_value_in_baskets_over_declared_screen,
        count(*) FILTER (WHERE basket_units >= 1000)                         AS baskets_with_units_over_1000,
        max(basket_units)                                                    AS max_basket_units
    FROM basket_panel
    GROUP BY household_key
),

line_side AS (
    SELECT
        household_key,
        sum(CASE WHEN quantity >= 1000 THEN 1 ELSE 0 END)                    AS bulk_quantity_lines,
        sum(CASE WHEN quantity >= 1000 THEN sales_value ELSE 0 END)          AS bulk_line_sales_value,
        max(quantity)                                                        AS max_line_quantity,
        sum(sales_value)                                                     AS line_sales_value
    FROM choice_line_price_candidates
    GROUP BY household_key
)

SELECT
    hh.household_key,
    hh.active_weeks,
    (hh.last_purchase_week - hh.first_purchase_week + 1)                    AS support_window_weeks,
    hh.leading_no_purchase_weeks,
    hh.trailing_no_purchase_weeks,
    hh.longest_internal_no_purchase_week_gap,
    w.total_sales_value,
    w.max_week_sales_value,
    w.max_week_no,
    w.max_week_sales_value / nullif(w.total_sales_value, 0)                 AS top_week_share_of_household_sales_value,
    b.max_basket_sales_value,
    b.max_basket_id,
    b.max_basket_week_no,
    b.max_basket_sales_value / nullif(w.total_sales_value, 0)               AS top_basket_share_of_household_sales_value,
    b.baskets_over_declared_screen,
    b.sales_value_in_baskets_over_declared_screen,
    b.sales_value_in_baskets_over_declared_screen / nullif(w.total_sales_value, 0)
                                                                            AS share_household_sales_value_in_extreme_baskets,
    b.baskets_with_units_over_1000,
    b.max_basket_units,
    l.bulk_quantity_lines,
    l.bulk_line_sales_value,
    l.bulk_line_sales_value / nullif(w.total_sales_value, 0)                AS share_household_sales_value_in_bulk_lines,
    l.max_line_quantity,
    w.weeks_with_quantity_zero_lines,
    w.weeks_with_positive_retail_disc,
    w.weeks_with_negative_implied_paid
FROM household_history AS hh
LEFT JOIN week_side AS w USING (household_key)
LEFT JOIN basket_side AS b USING (household_key)
LEFT JOIN line_side AS l USING (household_key)
ORDER BY hh.household_key
