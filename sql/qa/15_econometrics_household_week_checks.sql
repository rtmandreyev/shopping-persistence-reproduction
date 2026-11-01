-- Phase 2B QA: econometrics_household_week
-- Uniform check output: table_name, check_name, check_class, status, observed, expected
-- HARD_INVARIANT -> violation means the pipeline/code is wrong

WITH
e AS (
    SELECT
        count(*)                                                        AS n_rows,
        count(DISTINCT household_key)                                   AS n_households,
        count(DISTINCT week_no)                                         AS n_weeks,
        count(DISTINCT (household_key, week_no))                        AS n_keys,
        count(*) FILTER (WHERE week_no IS NULL OR household_key IS NULL) AS null_keys,
        sum(sample_full)                                                AS s0_rows,
        sum(CASE WHEN sample_full = 1 THEN 1 ELSE 0 END)                AS s0_cells,
        sum(CASE WHEN sample_sustained_support = 1 THEN 1 ELSE 0 END)   AS s1_cells,
        sum(CASE WHEN sample_higher_support = 1 THEN 1 ELSE 0 END)      AS s2_cells,
        count(*) FILTER (WHERE sample_higher_support = 1 AND sample_sustained_support = 0) AS s2_not_in_s1,
        count(*) FILTER (WHERE sample_sustained_support = 1 AND sample_full = 0)           AS s1_not_in_s0,
        count(*) FILTER (WHERE retail_disc_magnitude < 0 OR coupon_disc_magnitude < 0
                           OR coupon_match_disc_magnitude < 0 OR total_discount_magnitude < 0) AS negative_magnitudes,
        count(*) FILTER (WHERE lag1_retailer_purchase IS NOT NULL AND lag1_retailer_purchase NOT IN (0, 1)) AS bad_lag_flag,
        count(*) FILTER (WHERE listed_campaign_active_count < 0)        AS negative_campaign_count,
        count(*) FILTER (WHERE (listed_campaign_active_count > 0) <> (listed_campaign_recipient_any = 1)) AS campaign_flag_mismatch
    FROM econometrics_household_week
),
per_hh AS (
    SELECT min(c) AS min_c, max(c) AS max_c FROM (
        SELECT household_key, count(*) AS c FROM econometrics_household_week GROUP BY household_key
    )
),
hh_flags AS (
    SELECT count(*) AS n_nonconstant FROM (
        SELECT household_key,
               count(DISTINCT (sample_full, sample_sustained_support, sample_higher_support)) AS variants
        FROM econometrics_household_week GROUP BY household_key
    ) WHERE variants > 1
),
reconcile AS (
    SELECT count(*) AS n_mismatch
    FROM econometrics_household_week e JOIN household_week_base h
      ON h.household_key = e.household_key AND h.week_no = e.week_no
    WHERE e.retailer_purchase <> h.retailer_purchase
       OR e.trips <> h.trips
       OR e.units <> h.units
       OR e.distinct_products <> h.distinct_products
       OR e.distinct_commodities <> h.distinct_commodities
       OR e.active_store_count <> h.active_store_count
       OR e.transaction_line_count <> h.transaction_line_count
       OR abs(e.weekly_sales_value - h.weekly_sales_value) > 1e-9
),
lag_mismatch AS (
    SELECT count(*) AS n_mismatch
    FROM econometrics_household_week e JOIN household_week_base p
      ON p.household_key = e.household_key AND p.week_no = e.week_no - 1
    WHERE e.lag1_retailer_purchase <> p.retailer_purchase
       OR e.lag1_trips <> p.trips
       OR e.lag1_units <> p.units
       OR abs(e.lag1_weekly_sales_value - p.weekly_sales_value) > 1e-9
),
first_week AS (
    SELECT count(*) AS n_nonnull
    FROM econometrics_household_week e
    WHERE e.week_no = (SELECT min(week_no) FROM household_week_base)
      AND (e.lag1_retailer_purchase IS NOT NULL OR e.lag1_trips IS NOT NULL
           OR e.lag1_units IS NOT NULL OR e.lag1_weekly_sales_value IS NOT NULL)
),
forbidden_cols AS (
    SELECT count(*) AS n FROM information_schema.columns
    WHERE table_name = 'econometrics_household_week'
      AND regexp_matches(lower(column_name), 'treat|control|expos|eligible|selected|recommend')
)

SELECT 'econometrics_household_week' AS table_name, 'row_count_255000' AS check_name, 'HARD_INVARIANT' AS check_class,
       CASE WHEN e.n_rows = 255000 THEN 'PASS' ELSE 'FAIL' END AS status,
       CAST(e.n_rows AS VARCHAR) AS observed, '255000' AS expected FROM e
UNION ALL SELECT 'econometrics_household_week','distinct_households_2500','HARD_INVARIANT',
       CASE WHEN e.n_households = 2500 THEN 'PASS' ELSE 'FAIL' END, CAST(e.n_households AS VARCHAR), '2500' FROM e
UNION ALL SELECT 'econometrics_household_week','weeks_per_household_uniform_102','HARD_INVARIANT',
       CASE WHEN per_hh.min_c = 102 AND per_hh.max_c = 102 THEN 'PASS' ELSE 'FAIL' END,
       CAST(per_hh.min_c AS VARCHAR) || '..' || CAST(per_hh.max_c AS VARCHAR), '102' FROM per_hh
UNION ALL SELECT 'econometrics_household_week','declared_key_unique','HARD_INVARIANT',
       CASE WHEN e.n_rows = e.n_keys THEN 'PASS' ELSE 'FAIL' END, CAST(e.n_rows AS VARCHAR), CAST(e.n_keys AS VARCHAR) FROM e
UNION ALL SELECT 'econometrics_household_week','no_null_keys','HARD_INVARIANT',
       CASE WHEN e.null_keys = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(e.null_keys AS VARCHAR), '0' FROM e
UNION ALL SELECT 'econometrics_household_week','sample_full_households_2500','HARD_INVARIANT',
       CASE WHEN e.s0_cells = 255000 THEN 'PASS' ELSE 'FAIL' END, CAST(e.s0_cells AS VARCHAR), '255000' FROM e
UNION ALL SELECT 'econometrics_household_week','sample_sustained_support_households_2115','HARD_INVARIANT',
       CASE WHEN (SELECT count(DISTINCT household_key) FROM econometrics_household_week WHERE sample_sustained_support = 1) = 2115
            THEN 'PASS' ELSE 'FAIL' END,
       CAST((SELECT count(DISTINCT household_key) FROM econometrics_household_week WHERE sample_sustained_support = 1) AS VARCHAR), '2115'
UNION ALL SELECT 'econometrics_household_week','sample_higher_support_households_1807','HARD_INVARIANT',
       CASE WHEN (SELECT count(DISTINCT household_key) FROM econometrics_household_week WHERE sample_higher_support = 1) = 1807
            THEN 'PASS' ELSE 'FAIL' END,
       CAST((SELECT count(DISTINCT household_key) FROM econometrics_household_week WHERE sample_higher_support = 1) AS VARCHAR), '1807'
UNION ALL SELECT 'econometrics_household_week','sample_nesting_s2_subset_s1_subset_s0','HARD_INVARIANT',
       CASE WHEN e.s2_not_in_s1 = 0 AND e.s1_not_in_s0 = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(e.s2_not_in_s1 AS VARCHAR) || '/' || CAST(e.s1_not_in_s0 AS VARCHAR), '0/0' FROM e
UNION ALL SELECT 'econometrics_household_week','sample_flags_constant_within_household','HARD_INVARIANT',
       CASE WHEN hh_flags.n_nonconstant = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(hh_flags.n_nonconstant AS VARCHAR), '0' FROM hh_flags
UNION ALL SELECT 'econometrics_household_week','base_outcomes_reconcile_to_household_week_base','HARD_INVARIANT',
       CASE WHEN reconcile.n_mismatch = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(reconcile.n_mismatch AS VARCHAR), '0' FROM reconcile
UNION ALL SELECT 'econometrics_household_week','lag1_reconciles_to_prior_week','HARD_INVARIANT',
       CASE WHEN lag_mismatch.n_mismatch = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(lag_mismatch.n_mismatch AS VARCHAR), '0' FROM lag_mismatch
UNION ALL SELECT 'econometrics_household_week','first_week_lag_is_null','HARD_INVARIANT',
       CASE WHEN first_week.n_nonnull = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(first_week.n_nonnull AS VARCHAR), '0' FROM first_week
UNION ALL SELECT 'econometrics_household_week','discount_magnitudes_nonnegative','HARD_INVARIANT',
       CASE WHEN e.negative_magnitudes = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(e.negative_magnitudes AS VARCHAR), '0' FROM e
UNION ALL SELECT 'econometrics_household_week','campaign_state_flags_wellformed','HARD_INVARIANT',
       CASE WHEN e.campaign_flag_mismatch = 0 AND e.negative_campaign_count = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(e.campaign_flag_mismatch AS VARCHAR), '0' FROM e
UNION ALL SELECT 'econometrics_household_week','no_treatment_or_control_columns','HARD_INVARIANT',
       CASE WHEN forbidden_cols.n = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(forbidden_cols.n AS VARCHAR), '0' FROM forbidden_cols
