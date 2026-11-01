-- Phase 2B B: campaign_household_week
-- Grain : household_key x campaign x week_no   (listed recipient pairs x 102 weeks)
-- Key   : (household_key, campaign, week_no) -- unique
-- Population: every (household, campaign) pair listed in campaign_table
--   (7,208 pairs) x all 102 observed weeks = 735,216 rows.
--
-- NEUTRAL DATA INFRASTRUCTURE, not a DiD dataset and not an assignment table.
-- The table's population is the LISTED recipient set; `listed_recipient` is a
-- constant 1 that states the population explicitly so nobody misreads it as all
-- households. NON-LISTED households are deliberately NOT enumerated here: they
-- are reachable by joining campaign_week_calendar to household_week_base (or the
-- econometric mart). This pipeline defines NO control group.
--
-- `campaign_id` is listed only if the household appears in campaign_table for it;
-- listing is a FACT, never "treatment". Nothing here decides treatment
-- assignment, control eligibility, a parallel-trend or anticipation window, an
-- event-study reference period, a final pre/post horizon, campaign selection,
-- overlap exclusion, matching/weighting, or a causal estimand -- those are
-- Phase 6.
--
-- Week membership and period labels reuse the accepted Phase 1 convention
-- (P1-009 / sql/audit/09_campaign_feasibility.sql) via campaign_week_calendar.
-- Campaigns may start/end inside a week (represented by the containing week) and
-- campaign 24 extends beyond the observed transaction window; both are exposed,
-- not hidden.
--
-- overlapping_listed_campaign_count = the number of OTHER campaigns the same
-- household is listed for whose campaign-week span contains this week. It is a
-- factual listing-overlap count, not realized exposure.

WITH calendar AS (
    SELECT campaign, campaign_type, week_no, relative_week, period_label, is_campaign_week
    FROM campaign_week_calendar
),
listed AS (
    SELECT DISTINCT household_key, CAMPAIGN AS campaign
    FROM campaign_table
),
activity AS (
    SELECT household_key, week_no, retailer_purchase, trips, weekly_sales_value, units
    FROM household_week_base
),
hh_week_active AS (
    SELECT l.household_key, c.week_no, count(DISTINCT c.campaign) AS active_listed_campaigns
    FROM listed AS l
    JOIN calendar AS c ON c.campaign = l.campaign
    WHERE c.is_campaign_week = 1
    GROUP BY 1, 2
),
redemption_week AS (
    SELECT household_key,
           CAMPAIGN AS campaign,
           CAST(floor((DAY + 1) / 7.0) + 1 AS BIGINT) AS week_no,
           count(*) AS coupon_redemption_rows
    FROM coupon_redempt
    GROUP BY 1, 2, 3
)
SELECT
    l.household_key,
    c.campaign,
    c.campaign_type,
    c.week_no,
    c.relative_week,
    c.period_label,
    c.is_campaign_week,
    1 AS listed_recipient,
    a.retailer_purchase,
    a.trips,
    a.weekly_sales_value,
    a.units,
    COALESCE(hw.active_listed_campaigns, 0)
        - CASE WHEN c.is_campaign_week = 1 THEN 1 ELSE 0 END AS overlapping_listed_campaign_count,
    COALESCE(rw.coupon_redemption_rows, 0) AS coupon_redemption_rows
FROM listed AS l
JOIN calendar AS c ON c.campaign = l.campaign
LEFT JOIN activity AS a
    ON a.household_key = l.household_key AND a.week_no = c.week_no
LEFT JOIN hh_week_active AS hw
    ON hw.household_key = l.household_key AND hw.week_no = c.week_no
LEFT JOIN redemption_week AS rw
    ON rw.household_key = l.household_key AND rw.campaign = c.campaign AND rw.week_no = c.week_no
ORDER BY l.household_key, c.campaign, c.week_no
