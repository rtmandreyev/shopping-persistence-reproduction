-- Phase 2B B: campaign_week_calendar
-- Grain : campaign x week_no  (30 campaigns x 102 weeks = 3,060 rows)
-- Key   : (campaign, week_no) -- unique
--
-- NEUTRAL DATA INFRASTRUCTURE, not a DiD dataset. It contains NO household, NO
-- treatment/control assignment, NO parallel-trend or anticipation window, NO
-- event-study reference period, and NO final pre/post horizon. Those belong to
-- Phase 6.
--
-- Week membership uses the accepted Phase 1 convention (P1-009 /
-- sql/audit/09_campaign_feasibility.sql):
--   nominal_start_week = floor((START_DAY+1)/7)+1
--   nominal_end_week   = floor((END_DAY+1)/7)+1
--   start_week         = max(observed min week, nominal_start_week)   [clipped]
--   end_week           = min(observed max week, nominal_end_week)     [clipped]
--   period: pre      = weeks strictly before start_week
--           campaign = start_week..end_week inclusive
--           post     = weeks strictly after end_week
-- Campaign weeks are never labelled "post". Campaigns that start or end inside a
-- week are represented by the week containing that day; that clipping is the
-- accepted convention, not a new choice.
--
-- All 102 observed weeks are emitted per campaign (no window is chosen), so any
-- consumer can join non-listed households without the pipeline implying a
-- control group.

WITH weeks AS (
    SELECT DISTINCT WEEK_NO AS week_no FROM transaction_data
),
bounds AS (
    SELECT max(DAY) AS max_day, min(WEEK_NO) AS min_week, max(WEEK_NO) AS max_week
    FROM transaction_data
),
camp AS (
    SELECT
        d.CAMPAIGN,
        d.DESCRIPTION AS campaign_type,
        d.START_DAY,
        d.END_DAY,
        d.END_DAY - d.START_DAY + 1 AS duration_days,
        CAST(floor((d.START_DAY + 1) / 7.0) + 1 AS BIGINT) AS nominal_start_week,
        CAST(floor((d.END_DAY + 1) / 7.0) + 1 AS BIGINT)   AS nominal_end_week,
        greatest(b.min_week, CAST(floor((d.START_DAY + 1) / 7.0) + 1 AS BIGINT)) AS start_week,
        least(b.max_week, CAST(floor((d.END_DAY + 1) / 7.0) + 1 AS BIGINT))      AS end_week,
        b.max_day
    FROM campaign_desc AS d
    CROSS JOIN bounds AS b
)
SELECT
    c.CAMPAIGN                     AS campaign,
    c.campaign_type,
    c.START_DAY                    AS start_day,
    c.END_DAY                      AS end_day,
    c.duration_days,
    c.nominal_start_week,
    c.nominal_end_week,
    c.start_week,
    c.end_week,
    CAST(c.END_DAY > c.max_day AS INTEGER) AS ends_after_transaction_window,
    w.week_no,
    w.week_no - c.start_week       AS relative_week,
    CASE WHEN w.week_no < c.start_week THEN 'pre'
         WHEN w.week_no BETWEEN c.start_week AND c.end_week THEN 'campaign'
         ELSE 'post' END           AS period_label,
    CASE WHEN w.week_no BETWEEN c.start_week AND c.end_week THEN 1 ELSE 0 END AS is_campaign_week
FROM camp AS c
CROSS JOIN weeks AS w
ORDER BY c.CAMPAIGN, w.week_no
