-- Question E: weekly journey object for the four selected households
-- (grain: journey label x week_no = 4 x 102 = 408 rows)
--
-- Requires a view named `journey_selection`, which the caller builds by executing
-- 20_journey_selection.sql and registering its result in memory. The view therefore exists only
-- inside the run; the household keys it carries never reach a tracked or published artifact.
--
-- All 102 observed weeks are kept for every household, including the partial weeks 1 and 102
-- (week 1 has 5 observed days, week 102 has 6). Nothing is trimmed, smoothed or per-day normalised.
--
-- Weekly facts come from the accepted household_week_base. retailer_purchase = 0 means no recorded
-- purchase from this retailer that week; it is NOT zero household grocery consumption. weekly
-- SALES_VALUE is retailer revenue (retailer SALES_VALUE semantics), never customer spend.
--
-- Campaign listing and recorded redemption enter as CONTEXTUAL factual markers only: the weekly
-- listing count is the number of campaigns the household is listed for whose clipped campaign-week
-- span covers that week, and the redemption count is the recorded redemption rows in that week for
-- campaigns the household is listed for. Neither is exposure, assignment or receipt, and no
-- no event-time alignment and no comparison of a period before with a period after is created here.

WITH weeks AS (
    SELECT DISTINCT week_no FROM household_week_base
),
listed_week AS (
    SELECT
        c.household_key,
        c.week_no,
        count(DISTINCT c.campaign) FILTER (WHERE c.is_campaign_week = 1)
            AS listed_campaign_active_count,
        sum(c.coupon_redemption_rows) AS recorded_redemption_rows
    FROM campaign_household_week AS c
    WHERE c.household_key IN (SELECT household_key FROM journey_selection)
    GROUP BY c.household_key, c.week_no
)
SELECT
    s.journey_label AS household_label,
    s.stratum,
    w.week_no,
    coalesce(b.retailer_purchase, 0) AS retailer_purchase,
    coalesce(b.weekly_sales_value, 0) AS weekly_sales_value,
    coalesce(b.trips, 0) AS trips,
    coalesce(b.distinct_commodities, 0) AS distinct_commodities,
    coalesce(b.distinct_products, 0) AS distinct_products,
    coalesce(b.units, 0) AS units,
    coalesce(l.listed_campaign_active_count, 0) AS listed_campaign_active_count,
    coalesce(l.recorded_redemption_rows, 0) AS recorded_redemption_rows,
    CAST(w.week_no = 1 OR w.week_no = 102 AS INTEGER) AS partial_week,
    s.eligible_stratum_size,
    s.stratum_median_sales_value,
    s.selected_sales_value,
    s.distance_to_median
FROM journey_selection AS s
CROSS JOIN weeks AS w
LEFT JOIN household_week_base AS b
    ON b.household_key = s.household_key AND b.week_no = w.week_no
LEFT JOIN listed_week AS l
    ON l.household_key = s.household_key AND l.week_no = w.week_no
ORDER BY s.journey_label, w.week_no
