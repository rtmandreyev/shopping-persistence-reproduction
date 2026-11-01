-- Question D: strictly PRE-CAMPAIGN observable household characteristics, listed vs not listed
-- (grain: campaign x group)
--
-- Purpose: a CONFOUNDING DIAGNOSTIC only. It asks whether the households listed for a campaign
-- look observationally different from households not listed for it, using only information
-- strictly before the campaign start week.
--
-- It does NOT define a control group, does NOT estimate a campaign effect, does NOT adjust,
-- match or weight anything, and balance is NOT evidence of exchangeability. Observable
-- similarity would not establish random assignment either: campaign assignment is not
-- documented as randomised in this dataset, and Type A household coupon selection is
-- unobserved and based on prior purchase behaviour.
--
-- "Prior" means every observed week strictly before the campaign's clipped start_week. Facts
-- come from household_week_base and household_category_week (accepted Phase 2 tables).

WITH campaign_window AS (
    SELECT
        campaign,
        campaign_type,
        CAST(min(week_no) FILTER (WHERE is_campaign_week = 1) AS INTEGER) AS start_week,
        CAST(max(week_no) FILTER (WHERE is_campaign_week = 1) AS INTEGER) AS end_week
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
),
prior_week AS (
    SELECT
        g.campaign,
        g.campaign_type,
        g.listed,
        g.household_key,
        count(*) FILTER (WHERE b.week_no < g.start_week) AS prior_observed_weeks,
        count(*) FILTER (WHERE b.week_no < g.start_week AND b.retailer_purchase = 1)
            AS prior_active_weeks,
        coalesce(sum(b.weekly_sales_value) FILTER (WHERE b.week_no < g.start_week), 0)
            AS prior_retailer_sales_value,
        coalesce(sum(b.trips) FILTER (WHERE b.week_no < g.start_week), 0) AS prior_trips,
        cast(nullif(count(*) FILTER (WHERE b.week_no < g.start_week AND b.retailer_purchase = 1), 0)
            as double) AS prior_active_weeks_denominator
    FROM groups AS g
    LEFT JOIN household_week_base AS b
        ON b.household_key = g.household_key AND b.week_no < g.start_week
    GROUP BY g.campaign, g.campaign_type, g.listed, g.household_key
),
prior_breadth AS (
    SELECT
        g.campaign,
        g.listed,
        g.household_key,
        count(DISTINCT c.commodity_desc) AS prior_distinct_commodities
    FROM groups AS g
    LEFT JOIN household_category_week AS c
        ON c.household_key = g.household_key AND c.week_no < g.start_week
    GROUP BY g.campaign, g.listed, g.household_key
)
SELECT
    p.campaign,
    p.campaign_type,
    p.listed,
    count(*) AS households,
    min(p.prior_observed_weeks) AS min_prior_observed_weeks,
    round(avg(p.prior_observed_weeks), 4) AS mean_prior_observed_weeks,
    round(avg(p.prior_active_weeks), 4) AS mean_prior_active_weeks,
    round(median(p.prior_active_weeks), 4) AS median_prior_active_weeks,
    round(avg(p.prior_retailer_sales_value), 4) AS mean_prior_retailer_sales_value,
    round(median(p.prior_retailer_sales_value), 4) AS median_prior_retailer_sales_value,
    round(avg(p.prior_trips), 4) AS mean_prior_trips,
    round(median(p.prior_trips), 4) AS median_prior_trips,
    round(avg(b.prior_distinct_commodities), 4) AS mean_prior_distinct_commodities,
    round(avg(p.prior_retailer_sales_value) / nullif(avg(p.prior_observed_weeks), 0), 4)
        AS mean_prior_sales_value_per_observed_week,
    round(avg(p.prior_active_weeks) / nullif(avg(p.prior_observed_weeks), 0), 4)
        AS mean_prior_active_share
FROM prior_week AS p
JOIN prior_breadth AS b
    ON b.campaign = p.campaign AND b.listed = p.listed AND b.household_key = p.household_key
GROUP BY p.campaign, p.campaign_type, p.listed
ORDER BY p.campaign, p.listed DESC
