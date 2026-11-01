-- 09_campaign_feasibility.sql
-- Purpose: one row per campaign with descriptive feasibility measurements.
--
-- This artifact does NOT establish a valid control group, realized exposure, a
-- final event window, or DiD validity. "Nonrecipient" means only not listed for
-- this campaign in campaign_table.
--
-- Empirical day/week mapping for this dataset release:
--   WEEK_NO = floor((DAY + 1) / 7) + 1
-- The vendor guide does not formally specify the calendar origin.
--
-- Period definitions, clipped to observed transaction support:
--   pre      = weeks strictly before start_week
--   campaign = start_week through end_week, inclusive
--   post     = weeks strictly after end_week
--
-- Recipient contamination definition:
-- among a campaign's listed recipient households, the share also listed for a
-- different campaign whose nominal day interval overlaps inclusively. This is
-- listed-campaign overlap, not realized exposure.
-- Historical bugs preserved: an earlier activity query transposed pre/post
-- periods. The first overlap version counted all households rather than only
-- listed recipients; a later join version counted join fan-out instead of
-- distinct recipient households. The separated EXISTS-based aggregates below
-- prevent those failures from being reintroduced.

WITH
bounds AS (
    SELECT min(DAY) AS min_day,
           max(DAY) AS max_day,
           min(WEEK_NO) AS min_week,
           max(WEEK_NO) AS max_week
    FROM transaction_data
),
windows AS (
    SELECT d.CAMPAIGN,
           d.DESCRIPTION AS campaign_type,
           d.START_DAY,
           d.END_DAY,
           d.END_DAY - d.START_DAY + 1 AS duration_days,
           CAST(floor((d.START_DAY + 1) / 7.0) + 1 AS BIGINT) AS nominal_start_week,
           CAST(floor((d.END_DAY + 1) / 7.0) + 1 AS BIGINT) AS nominal_end_week,
           greatest(
               b.min_week,
               CAST(floor((d.START_DAY + 1) / 7.0) + 1 AS BIGINT)
           ) AS start_week,
           least(
               b.max_week,
               CAST(floor((d.END_DAY + 1) / 7.0) + 1 AS BIGINT)
           ) AS end_week,
           b.min_day,
           b.max_day,
           b.min_week,
           b.max_week
    FROM campaign_desc d
    CROSS JOIN bounds b
),
recipients AS (
    SELECT DISTINCT CAMPAIGN, household_key
    FROM campaign_table
),
recipient_counts AS (
    SELECT CAMPAIGN, count(*) AS recipient_households
    FROM recipients
    GROUP BY CAMPAIGN
),
all_hh AS (
    SELECT count(DISTINCT household_key) AS n
    FROM transaction_data
),
hh_week AS (
    SELECT DISTINCT household_key, WEEK_NO
    FROM transaction_data
),
recipient_activity AS (
    SELECT r.CAMPAIGN,
           count(DISTINCT (r.household_key, hw.WEEK_NO))
               FILTER (WHERE hw.WEEK_NO < w.start_week)
               AS recipient_active_household_weeks_before,
           count(DISTINCT (r.household_key, hw.WEEK_NO))
               FILTER (WHERE hw.WEEK_NO BETWEEN w.start_week AND w.end_week)
               AS recipient_active_household_weeks_during,
           count(DISTINCT (r.household_key, hw.WEEK_NO))
               FILTER (WHERE hw.WEEK_NO > w.end_week)
               AS recipient_active_household_weeks_after
    FROM recipients r
    JOIN windows w ON w.CAMPAIGN = r.CAMPAIGN
    LEFT JOIN hh_week hw ON hw.household_key = r.household_key
    GROUP BY r.CAMPAIGN
),
calendar_overlap AS (
    SELECT w.CAMPAIGN,
           count(DISTINCT other.CAMPAIGN) AS overlapping_campaigns
    FROM windows w
    LEFT JOIN windows other
      ON other.CAMPAIGN <> w.CAMPAIGN
     AND other.START_DAY <= w.END_DAY
     AND other.END_DAY >= w.START_DAY
    GROUP BY w.CAMPAIGN
),
recipient_overlap AS (
    SELECT w.CAMPAIGN,
           count(DISTINCT r.household_key) FILTER (
               WHERE EXISTS (
                   SELECT 1
                   FROM recipients other_r
                   JOIN windows other_w ON other_w.CAMPAIGN = other_r.CAMPAIGN
                   WHERE other_r.household_key = r.household_key
                     AND other_r.CAMPAIGN <> w.CAMPAIGN
                     AND other_w.START_DAY <= w.END_DAY
                     AND other_w.END_DAY >= w.START_DAY
               )
           ) AS recipient_households_with_overlapping_campaign
    FROM windows w
    JOIN recipients r ON r.CAMPAIGN = w.CAMPAIGN
    GROUP BY w.CAMPAIGN
),
hh_campaign_counts AS (
    SELECT household_key, count(DISTINCT CAMPAIGN) AS campaign_count
    FROM recipients
    GROUP BY household_key
),
campaign_burden AS (
    SELECT round(avg(campaign_count), 4) AS avg_campaigns_per_household,
           max(campaign_count) AS max_campaigns_per_household
    FROM hh_campaign_counts
),
coupon_summary AS (
    SELECT CAMPAIGN,
           count(DISTINCT (CAMPAIGN, COUPON_UPC)) AS coupon_pool_pairs,
           count(DISTINCT PRODUCT_ID) AS coupon_pool_products
    FROM coupon
    GROUP BY CAMPAIGN
),
redemption_summary AS (
    SELECT CAMPAIGN,
           count(*) AS redemption_rows,
           count(DISTINCT household_key) AS redeeming_households
    FROM coupon_redempt
    GROUP BY CAMPAIGN
)
SELECT
    w.CAMPAIGN AS campaign,
    w.campaign_type,
    w.START_DAY AS start_day,
    w.END_DAY AS end_day,
    w.duration_days,
    w.nominal_start_week,
    w.nominal_end_week,
    w.start_week,
    w.end_week,
    coalesce(rc.recipient_households, 0) AS recipient_households,
    (SELECT n FROM all_hh) - coalesce(rc.recipient_households, 0)
        AS nonrecipient_households_descriptive,
    greatest(w.start_week - w.min_week, 0) AS pre_weeks_calendar,
    greatest(w.end_week - w.start_week + 1, 0) AS campaign_weeks_calendar,
    greatest(w.max_week - w.end_week, 0) AS post_weeks_calendar,
    coalesce(ra.recipient_active_household_weeks_before, 0)
        AS recipient_active_household_weeks_before,
    coalesce(ra.recipient_active_household_weeks_during, 0)
        AS recipient_active_household_weeks_during,
    coalesce(ra.recipient_active_household_weeks_after, 0)
        AS recipient_active_household_weeks_after,
    coalesce(co.overlapping_campaigns, 0) AS overlapping_campaigns,
    coalesce(ro.recipient_households_with_overlapping_campaign, 0)
        AS recipient_households_with_overlapping_campaign,
    round(
        100.0 * coalesce(ro.recipient_households_with_overlapping_campaign, 0)
        / nullif(coalesce(rc.recipient_households, 0), 0),
        6
    ) AS recipient_pct_with_overlapping_campaign,
    (SELECT avg_campaigns_per_household FROM campaign_burden)
        AS avg_campaigns_per_household,
    (SELECT max_campaigns_per_household FROM campaign_burden)
        AS max_campaigns_per_household,
    coalesce(cs.coupon_pool_pairs, 0) AS coupon_pool_pairs,
    coalesce(cs.coupon_pool_products, 0) AS coupon_pool_products,
    coalesce(rs.redemption_rows, 0) AS redemption_rows,
    coalesce(rs.redeeming_households, 0) AS redeeming_households,
    w.END_DAY > w.max_day AS ends_after_transaction_window,
    greatest(w.max_week - w.end_week, 0) = 0 AS zero_post_support,
    coalesce(ro.recipient_households_with_overlapping_campaign, 0) > 0
        AS recipient_overlap_warning,
    'nonrecipient is descriptive only; overlapping campaign means listed overlap, not realized exposure'
        AS interpretation_warning
FROM windows w
LEFT JOIN recipient_counts rc ON rc.CAMPAIGN = w.CAMPAIGN
LEFT JOIN recipient_activity ra ON ra.CAMPAIGN = w.CAMPAIGN
LEFT JOIN calendar_overlap co ON co.CAMPAIGN = w.CAMPAIGN
LEFT JOIN recipient_overlap ro ON ro.CAMPAIGN = w.CAMPAIGN
LEFT JOIN coupon_summary cs ON cs.CAMPAIGN = w.CAMPAIGN
LEFT JOIN redemption_summary rs ON rs.CAMPAIGN = w.CAMPAIGN
ORDER BY w.CAMPAIGN;
