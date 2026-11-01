-- Phase 2C QA: predictive_household_week_features
-- Point-in-time / leakage verification. Every historical family is re-derived by
-- an INDEPENDENT formulation (joins and conditional aggregation) and compared to
-- the window-function production values, rather than re-checking the production
-- expression against itself.
--
-- Independent recomputation is full-population for cheap families (lags, rolling,
-- cumulative, recency) and restricted to a deterministic 40-household sample for
-- the expensive families (category, campaign, coupon), which is the documented
-- "spot check" approach.

WITH
sample_hh AS (
    SELECT household_key FROM (
        SELECT DISTINCT household_key FROM household_week_base ORDER BY household_key LIMIT 40
    )
),
base AS (
    SELECT
        count(*)                                    AS n_rows,
        count(DISTINCT household_key)               AS n_households,
        count(DISTINCT as_of_week)                  AS n_weeks,
        count(DISTINCT (household_key, as_of_week)) AS n_keys,
        count(*) FILTER (WHERE history_weeks_available <> as_of_week) AS bad_history_len,
        count(DISTINCT household_key) FILTER (WHERE has_demographics = 1) AS n_demo_households,
        count(*) FILTER (WHERE has_demographics = 1) AS n_demo_rows
    FROM predictive_household_week_features
),
per_hh AS (
    SELECT min(c) AS min_c, max(c) AS max_c FROM (
        SELECT household_key, count(*) AS c FROM predictive_household_week_features GROUP BY household_key
    )
),
forbidden_cols AS (
    SELECT count(*) AS n FROM information_schema.columns
    WHERE table_name = 'predictive_household_week_features'
      AND (regexp_matches(lower(column_name), 'target|label|split|train|test|validation|sample_full|sustained|higher_support')
           OR lower(column_name) IN ('y', 'outcome'))
),
lag_check AS (
    SELECT count(*) AS n_bad
    FROM predictive_household_week_features f
    JOIN household_week_base h
      ON h.household_key = f.household_key
     AND h.week_no = f.as_of_week - 1
    WHERE f.lag1_retailer_purchase <> h.retailer_purchase
       OR f.lag1_trips <> h.trips OR f.lag1_units <> h.units
       OR abs(f.lag1_weekly_sales_value - h.weekly_sales_value) > 1e-9
),
lag2_check AS (
    SELECT count(*) AS n_bad
    FROM predictive_household_week_features f
    JOIN household_week_base h
      ON h.household_key = f.household_key AND h.week_no = f.as_of_week - 2
    WHERE f.lag2_retailer_purchase <> h.retailer_purchase
       OR f.lag2_trips <> h.trips OR f.lag2_units <> h.units
       OR abs(f.lag2_weekly_sales_value - h.weekly_sales_value) > 1e-9
),
lag4_check AS (
    SELECT count(*) AS n_bad
    FROM predictive_household_week_features f
    JOIN household_week_base h
      ON h.household_key = f.household_key AND h.week_no = f.as_of_week - 4
    WHERE f.lag4_retailer_purchase <> h.retailer_purchase
       OR f.lag4_trips <> h.trips OR f.lag4_units <> h.units
       OR abs(f.lag4_weekly_sales_value - h.weekly_sales_value) > 1e-9
),
lag_null_check AS (
    SELECT count(*) AS n_bad FROM predictive_household_week_features
    WHERE (as_of_week = 1 AND (lag1_retailer_purchase IS NOT NULL OR lag2_retailer_purchase IS NOT NULL
                              OR lag4_retailer_purchase IS NOT NULL OR lag1_weekly_sales_value IS NOT NULL))
       OR (as_of_week = 2 AND (lag2_retailer_purchase IS NOT NULL OR lag4_retailer_purchase IS NOT NULL))
       OR (as_of_week BETWEEN 1 AND 3 AND lag4_retailer_purchase IS NOT NULL)
),
rolling4 AS (
    SELECT count(*) AS n_bad FROM (
        SELECT f.active_weeks_last4, f.sales_value_last4, f.trips_last4, f.units_last4,
               sum(h.retailer_purchase) AS sp, sum(h.weekly_sales_value) AS ss,
               sum(h.trips) AS st, sum(h.units) AS su, count(*) AS n_weeks
        FROM predictive_household_week_features f
        JOIN household_week_base h
          ON h.household_key = f.household_key
         AND h.week_no BETWEEN greatest(1, f.as_of_week - 3) AND f.as_of_week
        GROUP BY f.household_key, f.as_of_week, f.active_weeks_last4, f.sales_value_last4, f.trips_last4, f.units_last4
    ) WHERE active_weeks_last4 <> sp OR abs(sales_value_last4 - ss) > 1e-9
         OR trips_last4 <> st OR units_last4 <> su
),
rolling8 AS (
    SELECT count(*) AS n_bad FROM (
        SELECT f.active_weeks_last8, f.sales_value_last8, f.trips_last8, f.units_last8,
               sum(h.retailer_purchase) AS sp, sum(h.weekly_sales_value) AS ss,
               sum(h.trips) AS st, sum(h.units) AS su
        FROM predictive_household_week_features f
        JOIN household_week_base h
          ON h.household_key = f.household_key
         AND h.week_no BETWEEN greatest(1, f.as_of_week - 7) AND f.as_of_week
        GROUP BY f.household_key, f.as_of_week, f.active_weeks_last8, f.sales_value_last8, f.trips_last8, f.units_last8
    ) WHERE active_weeks_last8 <> sp OR abs(sales_value_last8 - ss) > 1e-9
         OR trips_last8 <> st OR units_last8 <> su
),
rolling12 AS (
    SELECT count(*) AS n_bad FROM (
        SELECT f.active_weeks_last12, f.sales_value_last12, f.trips_last12, f.units_last12,
               f.active_share_last12, f.mean_weekly_sales_value_last12, f.mean_units_last12,
               sum(h.retailer_purchase) AS sp, sum(h.weekly_sales_value) AS ss,
               sum(h.trips) AS st, sum(h.units) AS su, count(*) AS n_weeks
        FROM predictive_household_week_features f
        JOIN household_week_base h
          ON h.household_key = f.household_key
         AND h.week_no BETWEEN greatest(1, f.as_of_week - 11) AND f.as_of_week
        GROUP BY f.household_key, f.as_of_week, f.active_weeks_last12, f.sales_value_last12,
                 f.trips_last12, f.units_last12, f.active_share_last12,
                 f.mean_weekly_sales_value_last12, f.mean_units_last12
    ) WHERE active_weeks_last12 <> sp OR abs(sales_value_last12 - ss) > 1e-9
         OR trips_last12 <> st OR units_last12 <> su
         OR abs(active_share_last12 - CAST(sp AS DOUBLE) / n_weeks) > 1e-12
         OR abs(mean_weekly_sales_value_last12 - ss / n_weeks) > 1e-9
         OR abs(mean_units_last12 - CAST(su AS DOUBLE) / n_weeks) > 1e-9
),
cumulative_check AS (
    SELECT count(*) AS n_bad FROM (
        SELECT f.active_weeks_to_date, f.trips_to_date, f.sales_value_to_date, f.units_to_date,
               f.active_share_to_date,
               sum(h.retailer_purchase) AS sp, sum(h.trips) AS st,
               sum(h.weekly_sales_value) AS ss, sum(h.units) AS su, count(*) AS n_weeks
        FROM predictive_household_week_features f
        JOIN household_week_base h
          ON h.household_key = f.household_key AND h.week_no <= f.as_of_week
        GROUP BY f.household_key, f.as_of_week, f.active_weeks_to_date, f.trips_to_date,
                 f.sales_value_to_date, f.units_to_date, f.active_share_to_date
    ) WHERE active_weeks_to_date <> sp OR trips_to_date <> st OR units_to_date <> su
         OR abs(sales_value_to_date - ss) > 1e-9
         OR abs(active_share_to_date - CAST(sp AS DOUBLE) / n_weeks) > 1e-12
),
recency_check AS (
    SELECT count(*) AS n_bad FROM (
        SELECT f.as_of_week,
               f.ever_purchased_as_of_week, f.weeks_since_last_retailer_purchase,
               max(CASE WHEN h.retailer_purchase = 1 THEN h.week_no END) AS last_active
        FROM predictive_household_week_features f
        JOIN household_week_base h
          ON h.household_key = f.household_key AND h.week_no <= f.as_of_week
        GROUP BY f.household_key, f.as_of_week, f.ever_purchased_as_of_week, f.weeks_since_last_retailer_purchase
    ) WHERE (last_active IS NULL AND (ever_purchased_as_of_week <> 0 OR weeks_since_last_retailer_purchase IS NOT NULL))
         OR (last_active IS NOT NULL AND (ever_purchased_as_of_week <> 1
              OR weeks_since_last_retailer_purchase <> as_of_week - last_active))
),
commodity_check AS (
    SELECT count(*) AS n_bad FROM (
        SELECT f.distinct_commodities_last4, f.distinct_commodities_last8, f.distinct_commodities_last12,
               f.distinct_commodities_to_date,
               count(DISTINCT CASE WHEN c.week_no >= f.as_of_week - 3 THEN c.commodity_desc END) AS w4,
               count(DISTINCT CASE WHEN c.week_no >= f.as_of_week - 7 THEN c.commodity_desc END) AS w8,
               count(DISTINCT CASE WHEN c.week_no >= f.as_of_week - 11 THEN c.commodity_desc END) AS w12,
               count(DISTINCT c.commodity_desc) AS td
        FROM predictive_household_week_features f
        LEFT JOIN household_category_week c
          ON c.household_key = f.household_key AND c.week_no <= f.as_of_week
        WHERE f.household_key IN (SELECT household_key FROM sample_hh)
        GROUP BY f.household_key, f.as_of_week, f.distinct_commodities_last4,
                 f.distinct_commodities_last8, f.distinct_commodities_last12, f.distinct_commodities_to_date
    ) WHERE distinct_commodities_last4 <> w4 OR distinct_commodities_last8 <> w8
         OR distinct_commodities_last12 <> w12 OR distinct_commodities_to_date <> td
),
coupon_check AS (
    SELECT count(*) AS n_bad FROM (
        SELECT f.coupon_redemption_rows_current_week, f.coupon_redemption_rows_prior_4w,
               f.coupon_redemption_rows_prior_8w, f.coupon_redemption_rows_prior_12w,
               f.coupon_redemption_rows_to_date,
               count(CASE WHEN r.week_no = f.as_of_week THEN 1 END) AS cur,
               count(CASE WHEN r.week_no BETWEEN f.as_of_week - 4 AND f.as_of_week - 1 THEN 1 END) AS p4,
               count(CASE WHEN r.week_no BETWEEN f.as_of_week - 8 AND f.as_of_week - 1 THEN 1 END) AS p8,
               count(CASE WHEN r.week_no BETWEEN f.as_of_week - 12 AND f.as_of_week - 1 THEN 1 END) AS p12,
               count(r.week_no) AS td
        FROM predictive_household_week_features f
        LEFT JOIN (
            SELECT household_key, CAST(floor((DAY + 1) / 7.0) + 1 AS BIGINT) AS week_no FROM coupon_redempt
        ) r ON r.household_key = f.household_key AND r.week_no <= f.as_of_week
        WHERE f.household_key IN (SELECT household_key FROM sample_hh)
        GROUP BY f.household_key, f.as_of_week, f.coupon_redemption_rows_current_week,
                 f.coupon_redemption_rows_prior_4w, f.coupon_redemption_rows_prior_8w,
                 f.coupon_redemption_rows_prior_12w, f.coupon_redemption_rows_to_date
    ) WHERE coupon_redemption_rows_current_week <> cur
         OR coupon_redemption_rows_prior_4w <> p4 OR coupon_redemption_rows_prior_8w <> p8
         OR coupon_redemption_rows_prior_12w <> p12 OR coupon_redemption_rows_to_date <> td
),
campaign_check AS (
    SELECT count(*) AS n_bad FROM (
        SELECT f.listed_campaign_active_count_current, f.listed_campaign_weeks_to_date,
               f.distinct_listed_campaigns_to_date,
               count(DISTINCT CASE WHEN cw.week_no = f.as_of_week AND cw.is_campaign_week = 1
                                   THEN cw.campaign END) AS cur,
               count(DISTINCT CASE WHEN cw.is_campaign_week = 1 AND cw.week_no <= f.as_of_week
                                   THEN cw.week_no END) AS weeks_td,
               count(DISTINCT CASE WHEN cw.start_week <= f.as_of_week THEN cw.campaign END) AS camps_td
        FROM predictive_household_week_features f
        LEFT JOIN (
            SELECT household_key, campaign, week_no, is_campaign_week,
                   min(week_no) FILTER (WHERE is_campaign_week = 1)
                       OVER (PARTITION BY household_key, campaign) AS start_week
            FROM campaign_household_week
        ) cw ON cw.household_key = f.household_key AND cw.week_no <= f.as_of_week
        WHERE f.household_key IN (SELECT household_key FROM sample_hh)
        GROUP BY f.household_key, f.as_of_week, f.listed_campaign_active_count_current,
                 f.listed_campaign_weeks_to_date, f.distinct_listed_campaigns_to_date
    ) WHERE listed_campaign_active_count_current <> cur
         OR listed_campaign_weeks_to_date <> weeks_td
         OR distinct_listed_campaigns_to_date <> camps_td
),
early_week_check AS (
    SELECT count(*) AS n_bad FROM predictive_household_week_features
    WHERE as_of_week = 1 AND (active_weeks_last12 <> current_retailer_purchase
                              OR sales_value_last12 <> current_weekly_sales_value
                              OR distinct_commodities_last12 <> current_distinct_commodities)
),
demographics_check AS (
    SELECT count(*) AS n_bad FROM predictive_household_week_features
    WHERE (household_key IN (SELECT household_key FROM hh_demographic) AND has_demographics <> 1)
       OR (household_key NOT IN (SELECT household_key FROM hh_demographic) AND has_demographics <> 0)
)

SELECT 'predictive_household_week_features' AS table_name, 'row_count_255000' AS check_name, 'HARD_INVARIANT' AS check_class,
       CASE WHEN b.n_rows = 255000 THEN 'PASS' ELSE 'FAIL' END AS status,
       CAST(b.n_rows AS VARCHAR) AS observed, '255000' AS expected FROM base b
UNION ALL SELECT 'predictive_household_week_features','distinct_households_2500','HARD_INVARIANT',
       CASE WHEN b.n_households = 2500 THEN 'PASS' ELSE 'FAIL' END, CAST(b.n_households AS VARCHAR), '2500' FROM base b
UNION ALL SELECT 'predictive_household_week_features','weeks_per_household_102','HARD_INVARIANT',
       CASE WHEN per_hh.min_c = 102 AND per_hh.max_c = 102 THEN 'PASS' ELSE 'FAIL' END,
       CAST(per_hh.min_c AS VARCHAR) || '..' || CAST(per_hh.max_c AS VARCHAR), '102' FROM per_hh
UNION ALL SELECT 'predictive_household_week_features','declared_key_unique','HARD_INVARIANT',
       CASE WHEN b.n_rows = b.n_keys THEN 'PASS' ELSE 'FAIL' END, CAST(b.n_rows AS VARCHAR), CAST(b.n_keys AS VARCHAR) FROM base b
UNION ALL SELECT 'predictive_household_week_features','history_weeks_available_equals_as_of_week','HARD_INVARIANT',
       CASE WHEN b.bad_history_len = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(b.bad_history_len AS VARCHAR), '0' FROM base b
UNION ALL SELECT 'predictive_household_week_features','demographics_do_not_restrict_population','HARD_INVARIANT',
       CASE WHEN b.n_households = 2500 AND b.n_demo_households = 801 THEN 'PASS' ELSE 'FAIL' END,
       CAST(b.n_households AS VARCHAR) || '/' || CAST(b.n_demo_households AS VARCHAR), '2500/801' FROM base b
UNION ALL SELECT 'predictive_household_week_features','demographic_rows_match_coverage','HARD_INVARIANT',
       CASE WHEN b.n_demo_rows = 801 * 102 THEN 'PASS' ELSE 'FAIL' END, CAST(b.n_demo_rows AS VARCHAR), '81702' FROM base b
UNION ALL SELECT 'predictive_household_week_features','no_target_split_or_sample_flag_columns','HARD_INVARIANT',
       CASE WHEN forbidden_cols.n = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(forbidden_cols.n AS VARCHAR), '0' FROM forbidden_cols
UNION ALL SELECT 'predictive_household_week_features','no_causal_data_columns','HARD_INVARIANT',
       CASE WHEN (SELECT count(*) FROM information_schema.columns
                  WHERE table_name = 'predictive_household_week_features'
                    AND regexp_matches(lower(column_name), 'display|mailer|causal|placement')) = 0
            THEN 'PASS' ELSE 'FAIL' END,
       (SELECT CAST(count(*) AS VARCHAR) FROM information_schema.columns
         WHERE table_name = 'predictive_household_week_features'
           AND regexp_matches(lower(column_name), 'display|mailer|causal|placement')), '0'
UNION ALL SELECT 'predictive_household_week_features','lag1_reconciles_to_source_t_minus_1','HARD_INVARIANT',
       CASE WHEN lag_check.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(lag_check.n_bad AS VARCHAR), '0' FROM lag_check
UNION ALL SELECT 'predictive_household_week_features','lag2_reconciles_to_source_t_minus_2','HARD_INVARIANT',
       CASE WHEN lag2_check.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(lag2_check.n_bad AS VARCHAR), '0' FROM lag2_check
UNION ALL SELECT 'predictive_household_week_features','lag4_reconciles_to_source_t_minus_4','HARD_INVARIANT',
       CASE WHEN lag4_check.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(lag4_check.n_bad AS VARCHAR), '0' FROM lag4_check
UNION ALL SELECT 'predictive_household_week_features','lags_are_null_before_sample_start','HARD_INVARIANT',
       CASE WHEN lag_null_check.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(lag_null_check.n_bad AS VARCHAR), '0' FROM lag_null_check
UNION ALL SELECT 'predictive_household_week_features','rolling_4_uses_max_1_t_minus_3_through_t','HARD_INVARIANT',
       CASE WHEN rolling4.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(rolling4.n_bad AS VARCHAR), '0' FROM rolling4
UNION ALL SELECT 'predictive_household_week_features','rolling_8_uses_max_1_t_minus_7_through_t','HARD_INVARIANT',
       CASE WHEN rolling8.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(rolling8.n_bad AS VARCHAR), '0' FROM rolling8
UNION ALL SELECT 'predictive_household_week_features','rolling_12_uses_max_1_t_minus_11_through_t','HARD_INVARIANT',
       CASE WHEN rolling12.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(rolling12.n_bad AS VARCHAR), '0' FROM rolling12
UNION ALL SELECT 'predictive_household_week_features','cumulative_uses_weeks_le_t_only','HARD_INVARIANT',
       CASE WHEN cumulative_check.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(cumulative_check.n_bad AS VARCHAR), '0' FROM cumulative_check
UNION ALL SELECT 'predictive_household_week_features','recency_never_uses_future_purchases','HARD_INVARIANT',
       CASE WHEN recency_check.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(recency_check.n_bad AS VARCHAR), '0' FROM recency_check
UNION ALL SELECT 'predictive_household_week_features','category_history_reconciles_on_sample','HARD_INVARIANT',
       CASE WHEN commodity_check.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(commodity_check.n_bad AS VARCHAR), '0' FROM commodity_check
UNION ALL SELECT 'predictive_household_week_features','coupon_history_reconciles_on_sample','HARD_INVARIANT',
       CASE WHEN coupon_check.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(coupon_check.n_bad AS VARCHAR), '0' FROM coupon_check
UNION ALL SELECT 'predictive_household_week_features','campaign_history_reconciles_on_sample','HARD_INVARIANT',
       CASE WHEN campaign_check.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(campaign_check.n_bad AS VARCHAR), '0' FROM campaign_check
UNION ALL SELECT 'predictive_household_week_features','early_week_windows_use_available_history','HARD_INVARIANT',
       CASE WHEN early_week_check.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(early_week_check.n_bad AS VARCHAR), '0' FROM early_week_check
UNION ALL SELECT 'predictive_household_week_features','demographics_indicator_matches_coverage','HARD_INVARIANT',
       CASE WHEN demographics_check.n_bad = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(demographics_check.n_bad AS VARCHAR), '0' FROM demographics_check
UNION ALL SELECT 'predictive_household_week_features','empirical_never_purchased_as_of_week1_share','EMPIRICAL_FINDING','REPORT',
       CAST(CAST(count(*) FILTER (WHERE as_of_week = 1 AND ever_purchased_as_of_week = 0) AS DOUBLE) / 2500 AS VARCHAR), ''
       FROM predictive_household_week_features
