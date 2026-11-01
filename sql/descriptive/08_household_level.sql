-- Phase 3 Question B: household-level descriptive analytical object
-- Grain  : household_key (one row per household)
-- Sources: household_history, household_week_base, basket_panel (accepted Phase 2),
--          choice_line_price_candidates (accepted line grain; positive-quantity lines
--          only, so its window total sits 19.28 below the accepted retailer SALES_VALUE
--          total -- see LINE_ROUTE_CAVEAT in the caller)
--
-- This is an exploratory Phase 3 analytical object, not a canonical table, and it must
-- never become one. The caller writes it to an ignored exploratory CSV.
--
-- Semantics carried through unchanged:
--   * sales-value columns are retailer SALES_VALUE / retailer revenue, never household
--     grocery expenditure and never a customer-paid price;
--   * an inactive household-week is "no recorded purchase from this retailer", never
--     zero grocery consumption;
--   * trips are baskets;
--   * units are carried for diagnostics only and are NOT a headline heterogeneity
--     measure (Question A finding QA-03: recorded unit totals are dominated by very
--     large-quantity transaction lines);
--   * households are the full observed S0 population of 2,500. S1/S2 are labelled
--     sensitivity flags applied by the caller, never filters here.
--
-- Every ratio names its denominator:
--   *_per_active_week : denominator = that household's own active weeks
--   *_per_trip        : denominator = that household's own recorded baskets
--   active_share_102  : denominator = the fixed 102-week observation grid
--
-- Halves are weeks 1-51 and 52-102, declared in the caller.

WITH week_side AS (
    SELECT
        household_key,
        sum(retailer_purchase)                                          AS active_weeks_recount,
        sum(trips)                                                      AS trips_recount,
        sum(weekly_sales_value)                                         AS sales_value_recount,
        sum(units)                                                      AS units_recount,
        sum(CASE WHEN week_no <= 51 THEN trips ELSE 0 END)               AS trips_first_half,
        sum(CASE WHEN week_no >= 52 THEN trips ELSE 0 END)               AS trips_second_half,
        sum(CASE WHEN week_no <= 51 THEN weekly_sales_value ELSE 0 END)  AS sales_value_first_half,
        sum(CASE WHEN week_no >= 52 THEN weekly_sales_value ELSE 0 END)  AS sales_value_second_half,
        sum(qa_quantity_zero_lines)                                     AS qa_quantity_zero_lines,
        sum(qa_retail_disc_positive_lines)                              AS qa_retail_disc_positive_lines,
        sum(qa_implied_paid_negative_lines)                             AS qa_implied_paid_negative_lines,
        sum(
            CASE
                WHEN qa_quantity_zero_lines + qa_retail_disc_positive_lines
                     + qa_implied_paid_negative_lines > 0
                THEN 1
                ELSE 0
            END
        )                                                               AS weeks_with_any_qa_flag
    FROM household_week_base
    GROUP BY household_key
),

basket_side AS (
    SELECT
        household_key,
        count(*)                                                        AS baskets_recount,
        avg(basket_sales_value)                                         AS mean_basket_sales_value,
        median(basket_sales_value)                                      AS median_basket_sales_value,
        stddev_samp(basket_sales_value)                                 AS sd_basket_sales_value,
        max(basket_sales_value)                                         AS max_basket_sales_value,
        quantile_cont(basket_sales_value, 0.90)                         AS p90_basket_sales_value,
        max(distinct_products)                                          AS max_basket_distinct_products,
        sum(basket_units)                                               AS units_from_baskets
    FROM basket_panel
    GROUP BY household_key
),

line_side AS (
    SELECT
        household_key,
        count(*)                                                        AS lines_recorded,
        sum(sales_value)                                                AS line_sales_value,
        max(quantity)                                                   AS max_line_quantity,
        sum(CASE WHEN quantity >= 1000 THEN 1 ELSE 0 END)                AS bulk_quantity_lines,
        sum(CASE WHEN quantity >= 1000 THEN sales_value ELSE 0 END)      AS bulk_line_sales_value
    FROM choice_line_price_candidates
    GROUP BY household_key
)

SELECT
    hh.household_key,
    hh.first_purchase_week,
    hh.last_purchase_week,
    hh.first_purchase_day,
    hh.last_purchase_day,
    (hh.last_purchase_week - hh.first_purchase_week + 1)                AS support_window_weeks,
    hh.leading_no_purchase_weeks,
    hh.trailing_no_purchase_weeks,
    hh.longest_internal_no_purchase_week_gap,
    hh.active_weeks,
    hh.active_weeks_first_half,
    hh.active_weeks_second_half,
    hh.inactive_grid_weeks,
    hh.active_weeks::DOUBLE / 102.0                                     AS active_share_102,
    hh.total_sales_value,
    hh.total_baskets,
    hh.total_units,
    hh.distinct_products,
    hh.distinct_commodities,
    hh.distinct_stores,
    hh.has_demographics,
    hh.campaign_recipient_ever,
    hh.campaign_count,
    hh.coupon_redeemer_ever,
    hh.coupon_redemption_count,
    hh.total_sales_value / nullif(hh.active_weeks, 0)                   AS sales_value_per_active_week,
    hh.total_baskets::DOUBLE / nullif(hh.active_weeks, 0)               AS trips_per_active_week,
    hh.total_sales_value / nullif(hh.total_baskets, 0)                  AS sales_value_per_trip,
    hh.total_sales_value / nullif((hh.last_purchase_week - hh.first_purchase_week + 1), 0)
                                                                        AS sales_value_per_support_week,
    b.mean_basket_sales_value,
    b.median_basket_sales_value,
    b.sd_basket_sales_value,
    b.max_basket_sales_value,
    b.p90_basket_sales_value,
    b.max_basket_distinct_products,
    w.sales_value_first_half,
    w.sales_value_second_half,
    w.trips_first_half,
    w.trips_second_half,
    hh.active_weeks_first_half::DOUBLE / 51.0                           AS active_share_first_half,
    hh.active_weeks_second_half::DOUBLE / 51.0                          AS active_share_second_half,
    w.sales_value_first_half / nullif(hh.active_weeks_first_half, 0)    AS sales_value_per_active_week_first_half,
    w.sales_value_second_half / nullif(hh.active_weeks_second_half, 0)  AS sales_value_per_active_week_second_half,
    w.sales_value_first_half / nullif(w.trips_first_half, 0)            AS sales_value_per_trip_first_half,
    w.sales_value_second_half / nullif(w.trips_second_half, 0)          AS sales_value_per_trip_second_half,
    w.active_weeks_recount,
    w.trips_recount,
    w.sales_value_recount,
    w.units_recount,
    b.baskets_recount,
    b.units_from_baskets,
    l.lines_recorded,
    l.line_sales_value,
    l.max_line_quantity,
    l.bulk_quantity_lines,
    l.bulk_line_sales_value,
    w.qa_quantity_zero_lines,
    w.qa_retail_disc_positive_lines,
    w.qa_implied_paid_negative_lines,
    w.weeks_with_any_qa_flag
FROM household_history AS hh
LEFT JOIN week_side AS w USING (household_key)
LEFT JOIN basket_side AS b USING (household_key)
LEFT JOIN line_side AS l USING (household_key)
ORDER BY hh.household_key
