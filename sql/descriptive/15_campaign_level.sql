-- Question D: campaign-level descriptive object (one row per campaign, all 30 retained)
--
-- NEUTRAL DESCRIPTIVE INFRASTRUCTURE for an EXPLORATORY/CHALLENGE run. It contains NO
-- treatment/control assignment, NO exposure measure, NO estimand, NO event-study window,
-- NO effect estimate and NO campaign selection. `listed_recipient` is factual database
-- membership; a household not listed for a campaign is NOT a control group.
--
-- Timing and week membership reuse the accepted Phase 1/2 convention
-- (P1-009 / sql/audit/09_campaign_feasibility.sql) as reproduced by campaign_week_calendar:
--   nominal week = floor((DAY + 1) / 7) + 1, clipped to the observed weeks 1..102.
-- Campaign day intervals are preserved exactly (start_day/end_day are the raw vendor day
-- indices); a campaign touching a week does not mean the whole week was under the campaign.
--
-- Two accepted overlap definitions are reported side by side and neither replaces the other:
--   * Phase 1 (campaign-level, sql/audit/09_campaign_feasibility.sql): among a campaign's
--     listed recipients, the share also listed for a different campaign whose nominal day
--     interval overlaps inclusively. Listed-campaign overlap, never realized exposure.
--   * Phase 2 (household-week level, campaign_household_week.overlapping_listed_campaign_count):
--     the number of OTHER campaigns the same household is listed for whose clipped
--     campaign-week span contains that week.
--
-- Redemption columns are recorded redemption FACTS. They are never a take-up, assignment,
-- receipt or compliance rate: for campaign TypeA the household-specific coupons actually
-- received are outside the database (vendor limitation A), so assignment is unobserved.

WITH windows AS (
    SELECT
        CAMPAIGN AS campaign,
        DESCRIPTION AS campaign_type,
        START_DAY,
        END_DAY,
        END_DAY - START_DAY + 1 AS duration_days
    FROM campaign_desc
),
clip AS (
    SELECT
        c.campaign,
        c.nominal_start_week,
        c.nominal_end_week,
        greatest(b.min_week, c.nominal_start_week) AS start_week,
        least(b.max_week, c.nominal_end_week) AS end_week,
        b.min_week,
        b.max_week
    FROM (
        SELECT
            campaign,
            CAST(floor((START_DAY + 1) / 7.0) + 1 AS BIGINT) AS nominal_start_week,
            CAST(floor((END_DAY + 1) / 7.0) + 1 AS BIGINT) AS nominal_end_week
        FROM windows
    ) AS c
    CROSS JOIN (
        SELECT
            min(WEEK_NO) AS min_week,
            max(WEEK_NO) AS max_week
        FROM household_week_base
    ) AS b
),
listed AS (
    SELECT DISTINCT household_key, CAMPAIGN AS campaign
    FROM campaign_table
),
listed_counts AS (
    SELECT campaign, count(*) AS listed_households
    FROM listed
    GROUP BY campaign
),
campaign_weeks AS (
    SELECT
        campaign,
        count(*) FILTER (WHERE is_campaign_week = 1) AS calendar_weeks_touched,
        count(*) FILTER (WHERE period_label = 'pre') AS weeks_before_campaign,
        count(*) FILTER (WHERE period_label = 'post') AS weeks_after_campaign
    FROM campaign_week_calendar
    GROUP BY campaign
),
-- Phase 2 accepted overlap semantics, aggregated from the listed-pair household-week grid.
household_overlap AS (
    SELECT
        campaign,
        max(overlapping_listed_campaign_count) AS max_overlapping_listed_campaigns,
        count(*) FILTER (
            WHERE is_campaign_week = 1 AND overlapping_listed_campaign_count > 0
        ) AS listed_household_weeks_with_any_overlap,
        count(*) FILTER (WHERE is_campaign_week = 1) AS listed_household_weeks_in_span,
        count(DISTINCT household_key) FILTER (
            WHERE is_campaign_week = 1 AND overlapping_listed_campaign_count > 0
        ) AS listed_households_with_any_overlap
    FROM campaign_household_week
    GROUP BY campaign
),
-- Phase 1 accepted overlap semantics, recomputed on its own route.
span_overlap AS (
    SELECT
        w.campaign,
        count(DISTINCT o.campaign) AS overlapping_campaign_spans
    FROM windows AS w
    JOIN windows AS o
        ON o.campaign <> w.campaign
        AND o.START_DAY <= w.END_DAY
        AND o.END_DAY >= w.START_DAY
    GROUP BY w.campaign
),
day_interval_overlap AS (
    SELECT
        w.campaign,
        count(DISTINCT l.household_key) FILTER (
            WHERE EXISTS (
                SELECT 1
                FROM listed AS other_listed
                JOIN windows AS other_window ON other_window.campaign = other_listed.campaign
                WHERE
                    other_listed.household_key = l.household_key
                    AND other_listed.campaign <> w.campaign
                    AND other_window.START_DAY <= w.END_DAY
                    AND other_window.END_DAY >= w.START_DAY
            )
        ) AS listed_households_with_day_interval_overlap
    FROM windows AS w
    JOIN listed AS l ON l.campaign = w.campaign
    GROUP BY w.campaign
),
-- Concurrency among campaign spans (how crowded the calendar is around each campaign).
concurrent AS (
    SELECT
        w.campaign,
        count(DISTINCT o.campaign) AS concurrent_campaign_spans_including_self
    FROM windows AS w
    JOIN windows AS o
        ON o.START_DAY <= w.END_DAY AND o.END_DAY >= w.START_DAY
    GROUP BY w.campaign
),
redemption AS (
    SELECT
        CAMPAIGN AS campaign,
        count(*) AS recorded_redemption_rows,
        count(DISTINCT household_key) AS distinct_redeeming_households,
        count(DISTINCT COUPON_UPC) AS distinct_redeemed_upcs,
        min(DAY) AS first_redemption_day,
        max(DAY) AS last_redemption_day,
        count(DISTINCT household_key) FILTER (
            WHERE EXISTS (
                SELECT 1
                FROM listed AS l
                WHERE l.household_key = coupon_redempt.household_key AND l.campaign = coupon_redempt.CAMPAIGN
            )
        ) AS distinct_redeeming_households_also_listed
    FROM coupon_redempt
    GROUP BY CAMPAIGN
),
-- Descriptive pre/campaign/post support from the accepted listed-pair grid. RECONNAISSANCE
-- windows only: up to 4 observed weeks before, the clipped campaign span, up to 4 after.
support AS (
    SELECT
        campaign,
        count(DISTINCT household_key) FILTER (
            WHERE retailer_purchase = 1 AND period_label = 'pre'
        ) AS listed_households_active_before,
        count(DISTINCT household_key) FILTER (
            WHERE retailer_purchase = 1 AND period_label = 'campaign'
        ) AS listed_households_active_during,
        count(DISTINCT household_key) FILTER (
            WHERE retailer_purchase = 1 AND period_label = 'post'
        ) AS listed_households_active_after,
        count(DISTINCT household_key) FILTER (
            WHERE retailer_purchase = 1 AND relative_week BETWEEN -4 AND -1
        ) AS listed_households_active_within_4_weeks_before,
        count(DISTINCT household_key) FILTER (
            WHERE retailer_purchase = 1 AND relative_week BETWEEN 1 AND 4
                AND is_campaign_week = 0
        ) AS listed_households_active_within_4_weeks_after,
        count(*) FILTER (WHERE retailer_purchase = 1) AS listed_active_household_weeks,
        sum(weekly_sales_value) AS listed_retailer_sales_value,
        count(*) AS listed_household_weeks_total,
        sum(CASE WHEN period_label = 'pre' THEN 1 ELSE 0 END) AS listed_household_weeks_before,
        sum(CASE WHEN period_label = 'campaign' THEN 1 ELSE 0 END) AS listed_household_weeks_during,
        sum(CASE WHEN period_label = 'post' THEN 1 ELSE 0 END) AS listed_household_weeks_after
    FROM campaign_household_week
    GROUP BY campaign
)
SELECT
    w.campaign,
    w.campaign_type,
    w.START_DAY AS start_day,
    w.END_DAY AS end_day,
    w.duration_days,
    c.start_week,
    c.end_week,
    c.start_week AS mapped_start_week,
    c.end_week AS mapped_end_week,
    CAST(floor((w.START_DAY + 1) / 7.0) + 1 AS BIGINT) AS nominal_start_week,
    CAST(floor((w.END_DAY + 1) / 7.0) + 1 AS BIGINT) AS nominal_end_week,
    c.start_week - 1 AS observed_weeks_before_campaign,
    c.max_week - c.end_week AS observed_weeks_after_campaign,
    cw.calendar_weeks_touched,
    cw.weeks_before_campaign,
    cw.weeks_after_campaign,
    CAST(w.END_DAY > (SELECT max(DAY) FROM transaction_data) AS INTEGER)
        AS campaign_extends_beyond_transaction_support,
    CAST(c.end_week <> c.nominal_end_week AS INTEGER) AS end_week_clipped,
    lc.listed_households,
    round(100.0 * lc.listed_households / 2500.0, 4) AS listed_share_of_2500_pct,
    s.listed_household_weeks_total,
    s.listed_household_weeks_during,
    ho.listed_household_weeks_in_span,
    ho.listed_household_weeks_with_any_overlap,
    round(
        100.0 * ho.listed_household_weeks_with_any_overlap
        / nullif(ho.listed_household_weeks_in_span, 0), 4
    ) AS listed_household_week_overlap_pct,
    ho.listed_households_with_any_overlap,
    round(100.0 * ho.listed_households_with_any_overlap / nullif(lc.listed_households, 0), 4)
        AS listed_household_overlap_pct,
    ho.max_overlapping_listed_campaigns,
    so.overlapping_campaign_spans,
    dio.listed_households_with_day_interval_overlap,
    round(
        100.0 * dio.listed_households_with_day_interval_overlap / nullif(lc.listed_households, 0), 4
    ) AS accepted_phase1_recipient_overlap_pct,
    con.concurrent_campaign_spans_including_self,
    con.concurrent_campaign_spans_including_self - 1 AS concurrent_campaign_spans_other,
    coalesce(r.recorded_redemption_rows, 0) AS recorded_redemption_rows,
    coalesce(r.distinct_redeeming_households, 0) AS distinct_redeeming_households,
    coalesce(r.distinct_redeeming_households_also_listed, 0) AS distinct_redeeming_households_also_listed,
    coalesce(r.distinct_redeemed_upcs, 0) AS distinct_redeemed_upcs,
    r.first_redemption_day,
    r.last_redemption_day,
    CAST(coalesce(r.recorded_redemption_rows, 0) > 0 AS INTEGER) AS has_recorded_redemption,
    round(
        100.0 * coalesce(r.distinct_redeeming_households, 0) / nullif(lc.listed_households, 0), 4
    ) AS recorded_redemption_household_share_of_listed_pct,
    s.listed_active_household_weeks,
    round(s.listed_retailer_sales_value, 2) AS listed_retailer_sales_value,
    s.listed_households_active_before,
    s.listed_households_active_during,
    s.listed_households_active_after,
    round(100.0 * s.listed_households_active_before / nullif(lc.listed_households, 0), 4)
        AS listed_active_before_pct,
    round(100.0 * s.listed_households_active_during / nullif(lc.listed_households, 0), 4)
        AS listed_active_during_pct,
    round(100.0 * s.listed_households_active_after / nullif(lc.listed_households, 0), 4)
        AS listed_active_after_pct,
    s.listed_households_active_within_4_weeks_before,
    s.listed_households_active_within_4_weeks_after
FROM windows AS w
JOIN clip AS c ON c.campaign = w.campaign
JOIN listed_counts AS lc ON lc.campaign = w.campaign
JOIN campaign_weeks AS cw ON cw.campaign = w.campaign
JOIN household_overlap AS ho ON ho.campaign = w.campaign
JOIN span_overlap AS so ON so.campaign = w.campaign
JOIN day_interval_overlap AS dio ON dio.campaign = w.campaign
JOIN concurrent AS con ON con.campaign = w.campaign
JOIN support AS s ON s.campaign = w.campaign
LEFT JOIN redemption AS r ON r.campaign = w.campaign
ORDER BY w.campaign
