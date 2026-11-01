-- Question D: listed vs not-listed support in RECONNAISSANCE windows
-- (grain: campaign, one row per campaign, both groups side by side)
--
-- The windows are fixed reconnaissance windows, not the Phase 6 design:
--   pre      = up to 4 observed weeks strictly before start_week, truncated at week 1
--   campaign = the clipped campaign span (weeks start_week..end_week inclusive)
--   post     = up to 4 observed weeks strictly after end_week, truncated at week 102
-- They are labelled reconnaissance everywhere they appear and are NOT an event-study window,
-- an anticipation window, a parallel-trends window or a causal estimand. The Phase 6 window
-- and horizon remain undecided.
--
-- The two groups are "listed for this campaign" (factual membership in campaign_table) and
-- "not listed for this campaign" (the remaining households of the 2,500). The second group is
-- NOT a control group: a household not listed for this campaign may be listed for another one,
-- and campaign assignment is not documented as randomised.
--
-- Activity facts come from household_week_base (accepted Phase 2 table). retailer_purchase = 0
-- means no recorded purchase from this retailer that week; it is not zero grocery consumption.

WITH bounds AS (
    SELECT min(WEEK_NO) AS min_week, max(WEEK_NO) AS max_week
    FROM household_week_base
),
campaign_window AS (
    SELECT
        campaign,
        campaign_type,
        start_week,
        end_week,
        greatest(b.min_week, start_week - 4) AS pre_window_start,
        start_week - 1 AS pre_window_end,
        end_week + 1 AS post_window_start,
        least(b.max_week, end_week + 4) AS post_window_end,
        b.min_week,
        b.max_week
    FROM (
        SELECT
            campaign,
            campaign_type,
            CAST(min(week_no) FILTER (WHERE is_campaign_week = 1) AS INTEGER) AS start_week,
            CAST(max(week_no) FILTER (WHERE is_campaign_week = 1) AS INTEGER) AS end_week
        FROM campaign_week_calendar
        GROUP BY campaign, campaign_type
    ) AS spans
    CROSS JOIN bounds AS b
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
        h.household_key,
        CASE WHEN l.household_key IS NULL THEN 0 ELSE 1 END AS is_listed_for_this_campaign
    FROM campaign_window AS w
    CROSS JOIN households AS h
    LEFT JOIN listed AS l ON l.campaign = w.campaign AND l.household_key = h.household_key
),
activity AS (
    SELECT
        g.campaign,
        g.campaign_type,
        g.is_listed_for_this_campaign AS listed,
        g.household_key,
        b.week_no,
        b.retailer_purchase,
        b.trips,
        b.weekly_sales_value,
        CASE
            WHEN b.week_no BETWEEN w.pre_window_start AND w.pre_window_end THEN 'reconnaissance_pre'
            WHEN b.week_no BETWEEN w.start_week AND w.end_week THEN 'campaign_span'
            WHEN b.week_no BETWEEN w.post_window_start AND w.post_window_end THEN 'reconnaissance_post'
            ELSE 'outside_reconnaissance_windows'
        END AS reconnaissance_window
    FROM groups AS g
    JOIN campaign_window AS w ON w.campaign = g.campaign
    LEFT JOIN household_week_base AS b ON b.household_key = g.household_key
)
SELECT
    campaign,
    campaign_type,
    count(DISTINCT week_no) FILTER (
        WHERE reconnaissance_window = 'reconnaissance_pre'
    ) AS observed_weeks_in_pre_window,
    count(DISTINCT week_no) FILTER (
        WHERE reconnaissance_window = 'campaign_span'
    ) AS observed_weeks_in_campaign_span,
    count(DISTINCT week_no) FILTER (
        WHERE reconnaissance_window = 'reconnaissance_post'
    ) AS observed_weeks_in_post_window,
    count(DISTINCT household_key) FILTER (WHERE listed = 1) AS listed_households,
    count(DISTINCT household_key) FILTER (WHERE listed = 0) AS not_listed_households,
    count(DISTINCT household_key) FILTER (
        WHERE listed = 1 AND retailer_purchase = 1 AND reconnaissance_window = 'reconnaissance_pre'
    ) AS listed_active_pre,
    count(DISTINCT household_key) FILTER (
        WHERE listed = 1 AND retailer_purchase = 1 AND reconnaissance_window = 'campaign_span'
    ) AS listed_active_campaign,
    count(DISTINCT household_key) FILTER (
        WHERE listed = 1 AND retailer_purchase = 1 AND reconnaissance_window = 'reconnaissance_post'
    ) AS listed_active_post,
    count(DISTINCT household_key) FILTER (
        WHERE listed = 0 AND retailer_purchase = 1 AND reconnaissance_window = 'reconnaissance_pre'
    ) AS not_listed_active_pre,
    count(DISTINCT household_key) FILTER (
        WHERE listed = 0 AND retailer_purchase = 1 AND reconnaissance_window = 'campaign_span'
    ) AS not_listed_active_campaign,
    count(DISTINCT household_key) FILTER (
        WHERE listed = 0 AND retailer_purchase = 1 AND reconnaissance_window = 'reconnaissance_post'
    ) AS not_listed_active_post,
    round(
        100.0 * count(DISTINCT household_key) FILTER (
            WHERE listed = 1 AND retailer_purchase = 1 AND reconnaissance_window = 'reconnaissance_pre'
        ) / nullif(count(DISTINCT household_key) FILTER (WHERE listed = 1), 0), 4
    ) AS listed_active_share_pre_pct,
    round(
        100.0 * count(DISTINCT household_key) FILTER (
            WHERE listed = 0 AND retailer_purchase = 1 AND reconnaissance_window = 'reconnaissance_pre'
        ) / nullif(count(DISTINCT household_key) FILTER (WHERE listed = 0), 0), 4
    ) AS not_listed_active_share_pre_pct,
    round(
        100.0 * count(DISTINCT household_key) FILTER (
            WHERE listed = 1 AND retailer_purchase = 1 AND reconnaissance_window = 'campaign_span'
        ) / nullif(count(DISTINCT household_key) FILTER (WHERE listed = 1), 0), 4
    ) AS listed_active_share_campaign_pct,
    round(
        100.0 * count(DISTINCT household_key) FILTER (
            WHERE listed = 0 AND retailer_purchase = 1 AND reconnaissance_window = 'campaign_span'
        ) / nullif(count(DISTINCT household_key) FILTER (WHERE listed = 0), 0), 4
    ) AS not_listed_active_share_campaign_pct,
    round(
        100.0 * count(DISTINCT household_key) FILTER (
            WHERE listed = 1 AND retailer_purchase = 1 AND reconnaissance_window = 'reconnaissance_post'
        ) / nullif(count(DISTINCT household_key) FILTER (WHERE listed = 1), 0), 4
    ) AS listed_active_share_post_pct,
    round(
        100.0 * count(DISTINCT household_key) FILTER (
            WHERE listed = 0 AND retailer_purchase = 1 AND reconnaissance_window = 'reconnaissance_post'
        ) / nullif(count(DISTINCT household_key) FILTER (WHERE listed = 0), 0), 4
    ) AS not_listed_active_share_post_pct,
    count(*) FILTER (WHERE listed = 1 AND reconnaissance_window = 'reconnaissance_pre')
        AS listed_household_weeks_pre,
    count(*) FILTER (WHERE listed = 1 AND reconnaissance_window = 'campaign_span')
        AS listed_household_weeks_campaign,
    count(*) FILTER (WHERE listed = 1 AND reconnaissance_window = 'reconnaissance_post')
        AS listed_household_weeks_post,
    count(*) FILTER (WHERE listed = 0 AND reconnaissance_window = 'reconnaissance_pre')
        AS not_listed_household_weeks_pre,
    count(*) FILTER (WHERE listed = 0 AND reconnaissance_window = 'campaign_span')
        AS not_listed_household_weeks_campaign,
    count(*) FILTER (WHERE listed = 0 AND reconnaissance_window = 'reconnaissance_post')
        AS not_listed_household_weeks_post,
    sum(CASE WHEN listed = 1 AND reconnaissance_window IN
        ('reconnaissance_pre', 'campaign_span', 'reconnaissance_post') THEN weekly_sales_value END)
        AS listed_retailer_sales_value_in_windows,
    sum(CASE WHEN listed = 0 AND reconnaissance_window IN
        ('reconnaissance_pre', 'campaign_span', 'reconnaissance_post') THEN weekly_sales_value END)
        AS not_listed_retailer_sales_value_in_windows
FROM activity
GROUP BY campaign, campaign_type
ORDER BY campaign
