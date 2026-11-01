-- Question D: PRE-CAMPAIGN descriptive trajectories, listed vs not listed
-- (grain: campaign x group x pre-campaign week)
--
-- Purpose: a descriptive companion to 18_campaign_selection_diagnostics.sql. It reports how
-- mean weekly activity per household moves ACROSS the observed pre-campaign weeks for the two
-- groups.
--
-- This is NOT a parallel-trends test, NOT a pre-trend test, NOT an event study, and it contains
-- NO post-campaign estimate of anything. Only weeks strictly before the campaign start week are
-- used, so no campaign-window outcome appears here at all. Calendar time is preserved (week_no is
-- the x variable); no event-time re-indexing is created.
--
-- "Not listed for this campaign" is NOT a control group.

WITH campaign_window AS (
    SELECT
        campaign,
        campaign_type,
        CAST(min(week_no) FILTER (WHERE is_campaign_week = 1) AS INTEGER) AS start_week
    FROM campaign_week_calendar
    GROUP BY campaign, campaign_type
),
listed AS (
    SELECT DISTINCT household_key, CAMPAIGN AS campaign
    FROM campaign_table
),
households AS (
    SELECT DISTINCT household_key FROM household_week_base
),
groups AS (
    SELECT
        w.campaign,
        w.campaign_type,
        w.start_week,
        h.household_key,
        CASE WHEN l.household_key IS NULL THEN 0 ELSE 1 END AS listed
    FROM campaign_window AS w
    CROSS JOIN households AS h
    LEFT JOIN listed AS l ON l.campaign = w.campaign AND l.household_key = h.household_key
)
SELECT
    g.campaign,
    g.campaign_type,
    g.listed,
    g.start_week,
    b.week_no,
    g.start_week - b.week_no AS weeks_before_campaign_start,
    count(*) AS group_households,
    count(*) FILTER (WHERE b.retailer_purchase = 1) AS active_households,
    round(100.0 * count(*) FILTER (WHERE b.retailer_purchase = 1) / nullif(count(*), 0), 4)
        AS active_share_pct,
    round(avg(b.weekly_sales_value), 4) AS mean_household_retailer_sales_value,
    round(
        sum(b.weekly_sales_value) / nullif(count(*) FILTER (WHERE b.retailer_purchase = 1), 0), 4
    ) AS mean_retailer_sales_value_per_active_household,
    round(avg(b.trips), 4) AS mean_household_trips
FROM groups AS g
JOIN household_week_base AS b
    ON b.household_key = g.household_key AND b.week_no < g.start_week
GROUP BY g.campaign, g.campaign_type, g.listed, g.start_week, b.week_no
ORDER BY g.campaign, g.listed DESC, b.week_no
