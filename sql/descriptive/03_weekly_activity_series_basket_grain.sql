-- Phase 3 Question A: independent basket-grain route for the weekly series
-- Grain : week_no (102 rows)
-- Sources: basket_panel
--
-- This is the INDEPENDENT RECOMPUTATION route for every weekly headline total.
-- It aggregates the atomic trip table instead of the household-week grid, so a
-- week-level number reported from the row grain must agree with this route to be
-- reportable. It also supplies observed_days_in_week (distinct transaction days),
-- which is how partial weeks are identified empirically rather than assumed:
-- week 1 covers days 1-5 and week 102 covers days 706-711.
--
-- basket_panel holds every purchased basket; it is not a dense grid, so
-- active_households here = distinct households with at least one basket.

SELECT
    week_no,
    count(DISTINCT day)                       AS observed_days_in_week,
    min(day)                                  AS first_observed_day,
    max(day)                                  AS last_observed_day,
    sum(basket_sales_value)                   AS total_retailer_sales_value,
    sum(basket_units)                         AS units,
    count(*)                                  AS trips,
    count(DISTINCT household_key)             AS active_households,
    count(DISTINCT basket_id)                 AS distinct_baskets,
    sum(basket_retail_disc)                   AS basket_retail_disc,
    sum(basket_coupon_disc)                   AS basket_coupon_disc,
    sum(basket_coupon_match_disc)             AS basket_coupon_match_disc,
    max(basket_sales_value)                   AS max_basket_sales_value,
    sum(qa_quantity_zero_lines)               AS qa_quantity_zero_lines,
    sum(qa_implied_paid_negative_lines)        AS qa_implied_paid_negative_lines
FROM basket_panel
GROUP BY week_no
ORDER BY week_no
