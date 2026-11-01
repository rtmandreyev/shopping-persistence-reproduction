-- Phase 3 Question A: weekly retailer activity series (row / household-week grain)
-- Grain : week_no (102 rows)
-- Key   : week_no
-- Sources: household_week_base (levels, flags, discount magnitudes),
--          campaign_week_calendar + campaign_household_week (campaign TIMING CONTEXT ONLY)
--
-- Semantics (frozen plan, sections 4-5):
--   * total_retailer_sales_value = retailer SALES_VALUE (retailer revenue), NOT grocery
--     expenditure, customer spend, or price.
--   * active_households counts household-weeks with retailer_purchase = 1; a
--     non-active household-week means no recorded purchase from THIS retailer and
--     does NOT mean zero household grocery consumption.
--   * trips = baskets. units = SUM(QUANTITY).
--   * weeks 1 and 102 are partial (5 and 6 observed days); they are KEPT, never
--     normalized per day. observed_days_in_week comes from the independent
--     basket-grain query (03).
--
-- Every rate column carries its denominator as a separate column, so no ratio can
-- be read without its denominator. No normalized discount share, no price measure,
-- no per-day normalization and no counterfactual/reweighted series is created here.
--
-- Campaign columns are CONTEXT: "listed" is never a treatment or an exposure
-- overlap is not an effect. They exist so a peak can be checked for calendar
-- coincidence, not so it can be attributed.

WITH levels AS (
    SELECT
        week_no,
        sum(weekly_sales_value)              AS total_retailer_sales_value,
        sum(retailer_purchase)               AS active_households,
        count(*)                             AS household_week_cells,
        sum(trips)                           AS trips,
        sum(units)                           AS units,
        sum(qa_quantity_zero_lines)          AS qa_quantity_zero_lines,
        sum(qa_retail_disc_positive_lines)   AS qa_retail_disc_positive_lines,
        sum(qa_implied_paid_negative_lines)  AS qa_implied_paid_negative_lines,
        -- Discount magnitudes are the accepted definitions (abs of the signed weekly
        -- sums) recomputed from household_week_base rather than joined in from the
        -- mart, so the series has a single source. RETAIL_DISC is recorded negative
        -- and the coupon columns are never positive, so each weekly sum is <= 0 and
        -- abs() is the magnitude; no normalized share is created.
        abs(sum(weekly_retail_disc))         AS retail_disc_magnitude,
        abs(sum(weekly_coupon_disc))         AS coupon_disc_magnitude,
        abs(sum(weekly_coupon_match_disc))   AS coupon_match_disc_magnitude,
        abs(sum(weekly_retail_disc))
            + abs(sum(weekly_coupon_disc))
            + abs(sum(weekly_coupon_match_disc)) AS total_discount_magnitude
    FROM household_week_base
    GROUP BY week_no
),

campaign_context AS (
    SELECT
        week_no,
        count(DISTINCT CASE WHEN is_campaign_week = 1 THEN campaign END) AS active_listed_campaigns,
        sum(CASE WHEN is_campaign_week = 1 THEN 1 ELSE 0 END)            AS listed_pair_weeks_in_span
    FROM campaign_household_week
    GROUP BY week_no
)

SELECT
    l.week_no,
    l.household_week_cells,
    l.total_retailer_sales_value,
    l.active_households,
    l.trips,
    l.units,
    l.total_discount_magnitude,
    l.retail_disc_magnitude,
    l.coupon_disc_magnitude,
    l.coupon_match_disc_magnitude,
    l.qa_quantity_zero_lines,
    l.qa_retail_disc_positive_lines,
    l.qa_implied_paid_negative_lines,
    -- rates: each denominator is printed alongside its numerator
    l.total_retailer_sales_value / nullif(l.active_households, 0) AS retailer_sales_value_per_active_household_week,
    l.trips / nullif(l.active_households, 0)                     AS trips_per_active_household,
    l.units / nullif(l.active_households, 0)                     AS units_per_active_household,
    l.total_retailer_sales_value / nullif(l.trips, 0)            AS retailer_sales_value_per_trip,
    l.units / nullif(l.trips, 0)                                 AS units_per_trip,
    l.active_households / 2500.0                                 AS active_household_share_of_2500,
    coalesce(c.active_listed_campaigns, 0)                       AS active_listed_campaigns,
    coalesce(c.listed_pair_weeks_in_span, 0)                     AS listed_pair_weeks_in_span
FROM levels AS l
LEFT JOIN campaign_context AS c ON c.week_no = l.week_no
ORDER BY l.week_no
