-- Phase 2C: predictive_household_week_features
-- Grain : household_key x as_of_week
-- Key   : (household_key, as_of_week) -- unique, 255,000 rows
-- Population: all 2,500 transaction-universe households x all 102 observed weeks.
--
-- ============================================================================
-- AS-OF CONVENTION (authoritative for this table; do NOT mix with the mart's
-- lag convention, which is a separate analytical layer)
-- ============================================================================
-- A row with as_of_week = t means: "information available through the END of
-- observed retailer week t".
--   FEATURES MAY USE   : weeks <= t
--   FEATURES MUST NOT USE: weeks > t
--   Any eventual prediction target must be STRICTLY AFTER t.
-- Example: at as_of_week 40, weeks 1..40 (including the current week 40) may be
-- features; week 41+ may later define targets and must never enter a feature.
--
-- ============================================================================
-- WHAT THIS TABLE IS NOT
-- ============================================================================
-- No target column, no train/validation/test label, no prediction, no model.
-- No S1/S2 flag and no eligibility restriction of any kind (S0/S1/S2 are
-- full-period econometric sensitivity samples and would leak future realized
-- activity if used as prediction eligibility). Demographics never restrict rows:
-- all 2,500 households are present, including the 1,699 without demographics.
-- causal_data placement is deliberately NOT included (see docs).
-- ============================================================================

WITH
-- Rolling window lengths, expanded once so the three windows are computed in one
-- pass rather than three separate scans.
window_lengths AS (
    SELECT unnest([4, 8, 12]) AS w
),
spine AS (
    SELECT household_key, week_no AS as_of_week
    FROM household_week_base
),
-- Current week t state + the accepted factual discount magnitudes, computed from
-- the canonical grid with the same definitions as the econometric mart.
week_state AS (
    SELECT
        household_key,
        week_no                                              AS as_of_week,
        retailer_purchase,
        weekly_sales_value,
        trips,
        units,
        distinct_products,
        distinct_commodities,
        active_store_count,
        abs(weekly_retail_disc)                              AS retail_disc_magnitude,
        abs(weekly_coupon_disc)                              AS coupon_disc_magnitude,
        abs(weekly_coupon_match_disc)                        AS coupon_match_disc_magnitude,
        abs(weekly_retail_disc) + abs(weekly_coupon_disc)
            + abs(weekly_coupon_match_disc)                  AS total_discount_magnitude
    FROM household_week_base
),
-- Lags (1, 2, 4): NULL when the referenced week precedes week 1. A lagged
-- retailer-nonpurchase week is a genuine 0, not NULL.
lagged AS (
    SELECT
        household_key,
        week_no AS as_of_week,
        lag(retailer_purchase, 1)  OVER (PARTITION BY household_key ORDER BY week_no) AS lag1_retailer_purchase,
        lag(retailer_purchase, 2)  OVER (PARTITION BY household_key ORDER BY week_no) AS lag2_retailer_purchase,
        lag(retailer_purchase, 4)  OVER (PARTITION BY household_key ORDER BY week_no) AS lag4_retailer_purchase,
        lag(weekly_sales_value, 1) OVER (PARTITION BY household_key ORDER BY week_no) AS lag1_weekly_sales_value,
        lag(weekly_sales_value, 2) OVER (PARTITION BY household_key ORDER BY week_no) AS lag2_weekly_sales_value,
        lag(weekly_sales_value, 4) OVER (PARTITION BY household_key ORDER BY week_no) AS lag4_weekly_sales_value,
        lag(trips, 1)              OVER (PARTITION BY household_key ORDER BY week_no) AS lag1_trips,
        lag(trips, 2)              OVER (PARTITION BY household_key ORDER BY week_no) AS lag2_trips,
        lag(trips, 4)              OVER (PARTITION BY household_key ORDER BY week_no) AS lag4_trips,
        lag(units, 1)              OVER (PARTITION BY household_key ORDER BY week_no) AS lag1_units,
        lag(units, 2)              OVER (PARTITION BY household_key ORDER BY week_no) AS lag2_units,
        lag(units, 4)              OVER (PARTITION BY household_key ORDER BY week_no) AS lag4_units
    FROM household_week_base
),
-- Rolling windows ENDING at t over the FULL retailer-observation window,
-- including retailer-nonpurchase zero weeks. Early weeks use the shorter
-- available window (t < W), never a dropped row and never a backfilled zero.
rolling AS (
    SELECT
        household_key, as_of_week,
        sum(retailer_purchase) OVER w4   AS active_weeks_last4,
        sum(retailer_purchase) OVER w8   AS active_weeks_last8,
        sum(retailer_purchase) OVER w12  AS active_weeks_last12,
        avg(retailer_purchase) OVER w4   AS active_share_last4,
        avg(retailer_purchase) OVER w8   AS active_share_last8,
        avg(retailer_purchase) OVER w12  AS active_share_last12,
        sum(weekly_sales_value) OVER w4  AS sales_value_last4,
        sum(weekly_sales_value) OVER w8  AS sales_value_last8,
        sum(weekly_sales_value) OVER w12 AS sales_value_last12,
        avg(weekly_sales_value) OVER w4  AS mean_weekly_sales_value_last4,
        avg(weekly_sales_value) OVER w8  AS mean_weekly_sales_value_last8,
        avg(weekly_sales_value) OVER w12 AS mean_weekly_sales_value_last12,
        sum(trips) OVER w4               AS trips_last4,
        sum(trips) OVER w8               AS trips_last8,
        sum(trips) OVER w12              AS trips_last12,
        avg(trips) OVER w4               AS mean_trips_last4,
        avg(trips) OVER w8               AS mean_trips_last8,
        avg(trips) OVER w12              AS mean_trips_last12,
        sum(units) OVER w4               AS units_last4,
        sum(units) OVER w8               AS units_last8,
        sum(units) OVER w12              AS units_last12,
        avg(units) OVER w4               AS mean_units_last4,
        avg(units) OVER w8               AS mean_units_last8,
        avg(units) OVER w12              AS mean_units_last12,
        sum(retail_disc_magnitude) OVER w4       AS retail_disc_magnitude_last4,
        sum(retail_disc_magnitude) OVER w8       AS retail_disc_magnitude_last8,
        sum(retail_disc_magnitude) OVER w12      AS retail_disc_magnitude_last12,
        sum(coupon_disc_magnitude) OVER w4       AS coupon_disc_magnitude_last4,
        sum(coupon_disc_magnitude) OVER w8       AS coupon_disc_magnitude_last8,
        sum(coupon_disc_magnitude) OVER w12      AS coupon_disc_magnitude_last12,
        sum(coupon_match_disc_magnitude) OVER w4  AS coupon_match_disc_magnitude_last4,
        sum(coupon_match_disc_magnitude) OVER w8  AS coupon_match_disc_magnitude_last8,
        sum(coupon_match_disc_magnitude) OVER w12 AS coupon_match_disc_magnitude_last12,
        sum(total_discount_magnitude) OVER w4    AS total_discount_magnitude_last4,
        sum(total_discount_magnitude) OVER w8    AS total_discount_magnitude_last8,
        sum(total_discount_magnitude) OVER w12   AS total_discount_magnitude_last12
    FROM week_state
    WINDOW
        w4  AS (PARTITION BY household_key ORDER BY as_of_week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW),
        w8  AS (PARTITION BY household_key ORDER BY as_of_week ROWS BETWEEN 7 PRECEDING AND CURRENT ROW),
        w12 AS (PARTITION BY household_key ORDER BY as_of_week ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)
),
-- Recency / frequency / monetary, all through t (weeks <= t only).
rfm AS (
    SELECT
        household_key, as_of_week,
        CASE WHEN max(CASE WHEN retailer_purchase = 1 THEN as_of_week END)
                  OVER (PARTITION BY household_key ORDER BY as_of_week ROWS UNBOUNDED PRECEDING) IS NULL
             THEN 0 ELSE 1 END AS ever_purchased_as_of_week,
        as_of_week - max(CASE WHEN retailer_purchase = 1 THEN as_of_week END)
              OVER (PARTITION BY household_key ORDER BY as_of_week ROWS UNBOUNDED PRECEDING)
            AS weeks_since_last_retailer_purchase,
        sum(retailer_purchase) OVER (PARTITION BY household_key ORDER BY as_of_week ROWS UNBOUNDED PRECEDING)
            AS active_weeks_to_date,
        CAST(sum(retailer_purchase) OVER (PARTITION BY household_key ORDER BY as_of_week ROWS UNBOUNDED PRECEDING) AS DOUBLE)
            / as_of_week AS active_share_to_date,
        sum(trips) OVER (PARTITION BY household_key ORDER BY as_of_week ROWS UNBOUNDED PRECEDING)  AS trips_to_date,
        sum(weekly_sales_value) OVER (PARTITION BY household_key ORDER BY as_of_week ROWS UNBOUNDED PRECEDING) AS sales_value_to_date,
        sum(units) OVER (PARTITION BY household_key ORDER BY as_of_week ROWS UNBOUNDED PRECEDING)  AS units_to_date
    FROM week_state
),
-- ---------------------------------------------------------------------------
-- Category-history: distinct commodities in the windows ending at t, and
-- cumulative distinct commodities to date.
-- An event at week w keeps its commodity "active" for t in
--   [w, min(next_event_week - 1, w + W - 1)]
-- which is exactly the set of t whose W-window contains w and for which w is the
-- OLDEST event in that window. Those intervals are disjoint per commodity, so a
-- running sum of +1/-1 deltas counts each commodity at most once.
-- ---------------------------------------------------------------------------
commodity_span AS (
    SELECT
        household_key, commodity_desc, week_no,
        lead(week_no) OVER (PARTITION BY household_key, commodity_desc ORDER BY week_no) AS next_week
    FROM household_category_week
),
commodity_deltas AS (
    SELECT household_key, w, week_no AS delta_week, 1 AS delta
    FROM commodity_span CROSS JOIN window_lengths
    UNION ALL
    SELECT household_key, w,
           least(coalesce(next_week - 1, (SELECT max(week_no) FROM household_week_base)),
                 week_no + w - 1) + 1 AS delta_week,
           -1 AS delta
    FROM commodity_span CROSS JOIN window_lengths
),
commodity_deltas_pivot AS (
    SELECT
        household_key, delta_week,
        sum(CASE WHEN w = 4  THEN delta ELSE 0 END) AS d4,
        sum(CASE WHEN w = 8  THEN delta ELSE 0 END) AS d8,
        sum(CASE WHEN w = 12 THEN delta ELSE 0 END) AS d12
    FROM commodity_deltas
    WHERE delta_week <= (SELECT max(week_no) FROM household_week_base)
    GROUP BY household_key, delta_week
),
commodity_rolling AS (
    SELECT
        s.household_key, s.as_of_week,
        sum(coalesce(d.d4, 0))  OVER (PARTITION BY s.household_key ORDER BY s.as_of_week ROWS UNBOUNDED PRECEDING)
            AS distinct_commodities_last4,
        sum(coalesce(d.d8, 0))  OVER (PARTITION BY s.household_key ORDER BY s.as_of_week ROWS UNBOUNDED PRECEDING)
            AS distinct_commodities_last8,
        sum(coalesce(d.d12, 0)) OVER (PARTITION BY s.household_key ORDER BY s.as_of_week ROWS UNBOUNDED PRECEDING)
            AS distinct_commodities_last12
    FROM spine AS s
    LEFT JOIN commodity_deltas_pivot AS d
        ON d.household_key = s.household_key AND d.delta_week = s.as_of_week
),
commodity_first AS (
    SELECT household_key, commodity_desc, min(week_no) AS first_week
    FROM household_category_week
    GROUP BY household_key, commodity_desc
),
commodity_first_count AS (
    SELECT household_key, first_week, count(*) AS n
    FROM commodity_first
    GROUP BY household_key, first_week
),
commodity_cumulative AS (
    SELECT
        s.household_key, s.as_of_week,
        sum(coalesce(cf.n, 0)) OVER (PARTITION BY s.household_key ORDER BY s.as_of_week ROWS UNBOUNDED PRECEDING)
            AS distinct_commodities_to_date
    FROM spine AS s
    LEFT JOIN commodity_first_count AS cf
        ON cf.household_key = s.household_key AND cf.first_week = s.as_of_week
),
-- ---------------------------------------------------------------------------
-- Campaign history: FACTUAL listed-campaign information through t only.
-- "Listed" never means treated. campaign_household_week is one row per
-- household x campaign x week, so every aggregate below is a DISTINCT count or a
-- per-(household, week) aggregate -- never a sum of rows (which would double
-- count campaigns).
-- ---------------------------------------------------------------------------
campaign_week_state AS (
    SELECT
        household_key, week_no,
        count(DISTINCT campaign) FILTER (WHERE is_campaign_week = 1) AS listed_campaign_active_count_current
    FROM campaign_household_week
    GROUP BY household_key, week_no
),
campaign_listing_start AS (
    SELECT household_key, campaign, min(week_no) FILTER (WHERE is_campaign_week = 1) AS start_week
    FROM campaign_household_week
    GROUP BY household_key, campaign
),
campaign_start_count AS (
    SELECT household_key, start_week, count(*) AS n
    FROM campaign_listing_start
    GROUP BY household_key, start_week
),
campaign_features AS (
    SELECT
        s.household_key, s.as_of_week,
        coalesce(cws.listed_campaign_active_count_current, 0) AS listed_campaign_active_count_current,
        sum(CASE WHEN coalesce(cws.listed_campaign_active_count_current, 0) > 0 THEN 1 ELSE 0 END)
            OVER (PARTITION BY s.household_key ORDER BY s.as_of_week ROWS UNBOUNDED PRECEDING)
            AS listed_campaign_weeks_to_date,
        sum(coalesce(csc.n, 0))
            OVER (PARTITION BY s.household_key ORDER BY s.as_of_week ROWS UNBOUNDED PRECEDING)
            AS distinct_listed_campaigns_to_date
    FROM spine AS s
    LEFT JOIN campaign_week_state AS cws
        ON cws.household_key = s.household_key AND cws.week_no = s.as_of_week
    LEFT JOIN campaign_start_count AS csc
        ON csc.household_key = s.household_key AND csc.start_week = s.as_of_week
),
-- ---------------------------------------------------------------------------
-- Coupon redemption history: ACTUAL redemption records only. No assignment is
-- inferred from the global Type A pool and no take-up rate is constructed.
-- `prior_k` = weeks [t-k, t-1] (strictly prior, the current week is reported
-- separately as coupon_redemption_rows_current_week).
-- ---------------------------------------------------------------------------
redemption_week AS (
    SELECT household_key, CAST(floor((DAY + 1) / 7.0) + 1 AS BIGINT) AS week_no, count(*) AS rows_n
    FROM coupon_redempt
    GROUP BY household_key, 2
),
coupon_features AS (
    SELECT
        s.household_key, s.as_of_week,
        coalesce(rw.rows_n, 0) AS coupon_redemption_rows_current_week,
        sum(coalesce(rw.rows_n, 0)) OVER (PARTITION BY s.household_key ORDER BY s.as_of_week
            ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING)  AS coupon_redemption_rows_prior_4w,
        sum(coalesce(rw.rows_n, 0)) OVER (PARTITION BY s.household_key ORDER BY s.as_of_week
            ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING)  AS coupon_redemption_rows_prior_8w,
        sum(coalesce(rw.rows_n, 0)) OVER (PARTITION BY s.household_key ORDER BY s.as_of_week
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) AS coupon_redemption_rows_prior_12w,
        sum(coalesce(rw.rows_n, 0)) OVER (PARTITION BY s.household_key ORDER BY s.as_of_week
            ROWS UNBOUNDED PRECEDING)                  AS coupon_redemption_rows_to_date
    FROM spine AS s
    LEFT JOIN redemption_week AS rw
        ON rw.household_key = s.household_key AND rw.week_no = s.as_of_week
),
-- Demographics: LEFT joined so rows are never restricted. Original categorical
-- values are preserved as text; unavailable demographics stay NULL. No ordinal
-- encoding and no one-hot encoding is imposed here.
demo AS (
    SELECT
        household_key,
        1 AS has_demographics,
        classification_1     AS demographic_classification_1,
        classification_2     AS demographic_classification_2,
        classification_3     AS demographic_classification_3,
        classification_4     AS demographic_classification_4,
        classification_5     AS demographic_classification_5,
        HOMEOWNER_DESC       AS demographic_homeowner_desc,
        KID_CATEGORY_DESC    AS demographic_kid_category_desc
    FROM hh_demographic
)
SELECT
    s.household_key,
    s.as_of_week,
    s.as_of_week                                  AS history_weeks_available,
    -- current week (through the as-of cutoff)
    ws.retailer_purchase                          AS current_retailer_purchase,
    ws.weekly_sales_value                         AS current_weekly_sales_value,
    ws.trips                                      AS current_trips,
    ws.units                                      AS current_units,
    ws.distinct_products                          AS current_distinct_products,
    ws.distinct_commodities                       AS current_distinct_commodities,
    ws.active_store_count                         AS current_active_store_count,
    -- lags (NULL before week 1)
    lg.lag1_retailer_purchase,
    lg.lag2_retailer_purchase,
    lg.lag4_retailer_purchase,
    lg.lag1_weekly_sales_value,
    lg.lag2_weekly_sales_value,
    lg.lag4_weekly_sales_value,
    lg.lag1_trips,
    lg.lag2_trips,
    lg.lag4_trips,
    lg.lag1_units,
    lg.lag2_units,
    lg.lag4_units,
    -- rolling windows ending at t (full window, including non-purchase weeks)
    rl.active_weeks_last4, rl.active_weeks_last8, rl.active_weeks_last12,
    rl.active_share_last4, rl.active_share_last8, rl.active_share_last12,
    rl.sales_value_last4, rl.sales_value_last8, rl.sales_value_last12,
    rl.mean_weekly_sales_value_last4, rl.mean_weekly_sales_value_last8, rl.mean_weekly_sales_value_last12,
    rl.trips_last4, rl.trips_last8, rl.trips_last12,
    rl.mean_trips_last4, rl.mean_trips_last8, rl.mean_trips_last12,
    rl.units_last4, rl.units_last8, rl.units_last12,
    rl.mean_units_last4, rl.mean_units_last8, rl.mean_units_last12,
    -- category history
    cr.distinct_commodities_last4, cr.distinct_commodities_last8, cr.distinct_commodities_last12,
    cc.distinct_commodities_to_date,
    -- recency / frequency / monetary
    rf.ever_purchased_as_of_week,
    rf.weeks_since_last_retailer_purchase,
    rf.active_weeks_to_date,
    rf.active_share_to_date,
    rf.trips_to_date,
    rf.sales_value_to_date,
    rf.units_to_date,
    -- discount magnitudes (factual; no normalized share -- see P2-003)
    rl.retail_disc_magnitude_last4, rl.retail_disc_magnitude_last8, rl.retail_disc_magnitude_last12,
    rl.coupon_disc_magnitude_last4, rl.coupon_disc_magnitude_last8, rl.coupon_disc_magnitude_last12,
    rl.coupon_match_disc_magnitude_last4, rl.coupon_match_disc_magnitude_last8, rl.coupon_match_disc_magnitude_last12,
    rl.total_discount_magnitude_last4, rl.total_discount_magnitude_last8, rl.total_discount_magnitude_last12,
    -- campaign listing history (factual; never treatment)
    cf.listed_campaign_active_count_current,
    cf.listed_campaign_weeks_to_date,
    cf.distinct_listed_campaigns_to_date,
    -- coupon redemption history (actual records only)
    cp.coupon_redemption_rows_current_week,
    cp.coupon_redemption_rows_prior_4w,
    cp.coupon_redemption_rows_prior_8w,
    cp.coupon_redemption_rows_prior_12w,
    cp.coupon_redemption_rows_to_date,
    -- demographics (never restrict rows)
    coalesce(dm.has_demographics, 0)              AS has_demographics,
    dm.demographic_classification_1,
    dm.demographic_classification_2,
    dm.demographic_classification_3,
    dm.demographic_classification_4,
    dm.demographic_classification_5,
    dm.demographic_homeowner_desc,
    dm.demographic_kid_category_desc
FROM spine AS s
JOIN week_state   AS ws ON ws.household_key = s.household_key AND ws.as_of_week = s.as_of_week
JOIN lagged       AS lg ON lg.household_key = s.household_key AND lg.as_of_week = s.as_of_week
JOIN rolling      AS rl ON rl.household_key = s.household_key AND rl.as_of_week = s.as_of_week
JOIN rfm          AS rf ON rf.household_key = s.household_key AND rf.as_of_week = s.as_of_week
JOIN commodity_rolling    AS cr ON cr.household_key = s.household_key AND cr.as_of_week = s.as_of_week
JOIN commodity_cumulative AS cc ON cc.household_key = s.household_key AND cc.as_of_week = s.as_of_week
JOIN campaign_features    AS cf ON cf.household_key = s.household_key AND cf.as_of_week = s.as_of_week
JOIN coupon_features      AS cp ON cp.household_key = s.household_key AND cp.as_of_week = s.as_of_week
LEFT JOIN demo            AS dm ON dm.household_key = s.household_key
ORDER BY s.household_key, s.as_of_week
