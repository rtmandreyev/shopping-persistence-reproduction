-- Phase 2B A: econometrics_household_week
-- Grain : household_key x week_no
-- Key   : (household_key, week_no) -- unique
-- Population: all 2,500 transaction-universe households x all 102 weeks = 255,000
--   rows. This is the S0 baseline population. It is a DATA PREPARATION table,
--   not an estimation sample and not a model specification.
--
-- SAMPLE FLAGS (constant within household; S2 subset of S1 subset of S0):
--   sample_full              = 1 for all 2,500 households (S0 baseline)
--   sample_sustained_support = 1 iff active_weeks >= 20 AND active in all three
--                              34-week thirds (S1 sensitivity population)
--   sample_higher_support    = 1 iff active_weeks >= 30 AND active in all three
--                              34-week thirds (S2 sensitivity population)
-- S0 conditions on NOTHING. S1/S2 deliberately condition on full-period realized
-- activity and are therefore SENSITIVITY populations only: full-period activity
-- can be affected by variables later studied, so S1/S2 must NOT be reused
-- automatically for campaign causal identification, treatment/control
-- eligibility, or ML prediction eligibility. They are not "cleaner", "unbiased",
-- "valid", "better" or "causal".
--
-- LAGS: strictly prior-week information only. lag1_* at week t is the value at
-- week t-1 for the SAME household; the household's first week is NULL (no
-- invented zero history). No future information enters any variable.
--
-- CAMPAIGN STATE: listed_campaign_active_count / listed_campaign_recipient_any
-- are FACTUAL listings, not treatment. Week membership uses the accepted Phase 1
-- convention (P1-009 / sql/audit/09_campaign_feasibility.sql): nominal weeks from
-- WEEK_NO = floor((DAY+1)/7)+1, clipped to observed weeks 1..102. "Listed" means
-- only "appears in campaign_table"; it is never called treated.
--
-- Discount aggregates keep recorded sign conventions; the *_magnitude columns are
-- the absolute values of the signed weekly sums. No discount share is created
-- here: more than one defensible denominator exists, so the main share variable
-- is left to a researcher decision.

WITH bounds AS (
    SELECT min(WEEK_NO) AS min_week, max(WEEK_NO) AS max_week
    FROM transaction_data
),
campaign_weeks AS (
    SELECT
        d.CAMPAIGN,
        greatest(b.min_week, CAST(floor((d.START_DAY + 1) / 7.0) + 1 AS BIGINT)) AS start_week,
        least(b.max_week, CAST(floor((d.END_DAY + 1) / 7.0) + 1 AS BIGINT))      AS end_week
    FROM campaign_desc AS d
    CROSS JOIN bounds AS b
),
listed_campaign_week AS (
    SELECT ct.household_key, cw.CAMPAIGN, w.week_no
    FROM campaign_table AS ct
    JOIN campaign_weeks AS cw ON cw.CAMPAIGN = ct.CAMPAIGN
    CROSS JOIN (SELECT DISTINCT WEEK_NO AS week_no FROM transaction_data) AS w
    WHERE w.week_no BETWEEN cw.start_week AND cw.end_week
),
listed_active AS (
    SELECT household_key, week_no, count(DISTINCT CAMPAIGN) AS listed_campaign_active_count
    FROM listed_campaign_week
    GROUP BY household_key, week_no
),
sample_flags AS (
    SELECT
        household_key,
        CASE WHEN active_weeks >= 20
              AND active_weeks_early > 0 AND active_weeks_middle > 0 AND active_weeks_late > 0
             THEN 1 ELSE 0 END AS sample_sustained_support,
        CASE WHEN active_weeks >= 30
              AND active_weeks_early > 0 AND active_weeks_middle > 0 AND active_weeks_late > 0
             THEN 1 ELSE 0 END AS sample_higher_support
    FROM panel_eligibility
),
lagged AS (
    SELECT
        household_key,
        week_no,
        lag(retailer_purchase)  OVER (PARTITION BY household_key ORDER BY week_no) AS lag1_retailer_purchase,
        lag(weekly_sales_value) OVER (PARTITION BY household_key ORDER BY week_no) AS lag1_weekly_sales_value,
        lag(trips)              OVER (PARTITION BY household_key ORDER BY week_no) AS lag1_trips,
        lag(units)              OVER (PARTITION BY household_key ORDER BY week_no) AS lag1_units
    FROM household_week_base
)
SELECT
    h.household_key,
    h.week_no,
    h.retailer_purchase,
    h.weekly_sales_value,
    h.trips,
    h.units,
    h.distinct_products,
    h.distinct_commodities,
    h.active_store_count,
    h.transaction_line_count,
    h.weekly_retail_disc,
    h.weekly_coupon_disc,
    h.weekly_coupon_match_disc,
    abs(h.weekly_retail_disc)                                              AS retail_disc_magnitude,
    abs(h.weekly_coupon_disc)                                              AS coupon_disc_magnitude,
    abs(h.weekly_coupon_match_disc)                                        AS coupon_match_disc_magnitude,
    abs(h.weekly_retail_disc) + abs(h.weekly_coupon_disc)
        + abs(h.weekly_coupon_match_disc)                                  AS total_discount_magnitude,
    h.qa_quantity_zero_lines,
    h.qa_retail_disc_positive_lines,
    h.qa_implied_paid_negative_lines,
    l.lag1_retailer_purchase,
    l.lag1_weekly_sales_value,
    l.lag1_trips,
    l.lag1_units,
    COALESCE(la.listed_campaign_active_count, 0)                           AS listed_campaign_active_count,
    CASE WHEN la.household_key IS NULL THEN 0 ELSE 1 END                   AS listed_campaign_recipient_any,
    1                                                                      AS sample_full,
    sf.sample_sustained_support,
    sf.sample_higher_support
FROM household_week_base AS h
JOIN sample_flags AS sf ON sf.household_key = h.household_key
JOIN lagged AS l ON l.household_key = h.household_key AND l.week_no = h.week_no
LEFT JOIN listed_active AS la ON la.household_key = h.household_key AND la.week_no = h.week_no
ORDER BY h.household_key, h.week_no
