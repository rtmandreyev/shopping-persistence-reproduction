-- Phase 2B QA: campaign_household_week
WITH
c AS (
    SELECT
        count(*)                                          AS n_rows,
        count(DISTINCT (household_key, campaign, week_no)) AS n_keys,
        count(DISTINCT household_key)                     AS n_households,
        count(DISTINCT campaign)                          AS n_campaigns,
        count(*) FILTER (WHERE listed_recipient <> 1)     AS bad_listed,
        count(*) FILTER (WHERE overlapping_listed_campaign_count < 0) AS neg_overlap,
        count(*) FILTER (WHERE period_label NOT IN ('pre', 'campaign', 'post')) AS bad_period,
        sum(coupon_redemption_rows)                       AS redemption_rows
    FROM campaign_household_week
),
pairs AS (
    SELECT count(*) AS n FROM (SELECT DISTINCT household_key, campaign FROM campaign_household_week)
),
src_pairs AS (
    SELECT count(*) AS n FROM (SELECT DISTINCT household_key, CAMPAIGN FROM campaign_table)
),
reconcile AS (
    SELECT count(*) AS n FROM campaign_household_week chw
    JOIN household_week_base h
      ON h.household_key = chw.household_key AND h.week_no = chw.week_no
    WHERE chw.retailer_purchase <> h.retailer_purchase
       OR chw.trips <> h.trips
       OR chw.units <> h.units
       OR abs(chw.weekly_sales_value - h.weekly_sales_value) > 1e-9
),
src_redemption AS (
    SELECT count(*) AS n FROM coupon_redempt
),
forbidden_cols AS (
    SELECT count(*) AS n FROM information_schema.columns
    WHERE table_name = 'campaign_household_week'
      AND regexp_matches(lower(column_name), 'treat|control|expos|eligible|effect|parallel|anticipat')
)

SELECT 'campaign_household_week' AS table_name, 'row_count_listed_pairs_x_102' AS check_name, 'HARD_INVARIANT' AS check_class,
       CASE WHEN c.n_rows = (SELECT n FROM src_pairs) * 102 THEN 'PASS' ELSE 'FAIL' END AS status,
       CAST(c.n_rows AS VARCHAR) AS observed, CAST((SELECT n FROM src_pairs) * 102 AS VARCHAR) AS expected FROM c
UNION ALL SELECT 'campaign_household_week','declared_key_unique','HARD_INVARIANT',
       CASE WHEN c.n_rows = c.n_keys THEN 'PASS' ELSE 'FAIL' END, CAST(c.n_rows AS VARCHAR), CAST(c.n_keys AS VARCHAR) FROM c
UNION ALL SELECT 'campaign_household_week','listed_pairs_match_campaign_table','HARD_INVARIANT',
       CASE WHEN pairs.n = src_pairs.n THEN 'PASS' ELSE 'FAIL' END, CAST(pairs.n AS VARCHAR), CAST(src_pairs.n AS VARCHAR) FROM pairs, src_pairs
UNION ALL SELECT 'campaign_household_week','households_and_campaigns_complete','HARD_INVARIANT',
       CASE WHEN c.n_households = 1584 AND c.n_campaigns = 30 THEN 'PASS' ELSE 'FAIL' END,
       CAST(c.n_households AS VARCHAR) || '/' || CAST(c.n_campaigns AS VARCHAR), '1584/30' FROM c
UNION ALL SELECT 'campaign_household_week','population_is_listed_recipients','HARD_INVARIANT',
       CASE WHEN c.bad_listed = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.bad_listed AS VARCHAR), '0' FROM c
UNION ALL SELECT 'campaign_household_week','period_label_domain','HARD_INVARIANT',
       CASE WHEN c.bad_period = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.bad_period AS VARCHAR), '0' FROM c
UNION ALL SELECT 'campaign_household_week','overlap_count_nonnegative','HARD_INVARIANT',
       CASE WHEN c.neg_overlap = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.neg_overlap AS VARCHAR), '0' FROM c
UNION ALL SELECT 'campaign_household_week','retailer_activity_reconciles_to_household_week_base','HARD_INVARIANT',
       CASE WHEN reconcile.n = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(reconcile.n AS VARCHAR), '0' FROM reconcile
UNION ALL SELECT 'campaign_household_week','coupon_redemption_rows_reconcile','HARD_INVARIANT',
       CASE WHEN c.redemption_rows = src_redemption.n THEN 'PASS' ELSE 'FAIL' END,
       CAST(c.redemption_rows AS VARCHAR), CAST(src_redemption.n AS VARCHAR) FROM c, src_redemption
UNION ALL SELECT 'campaign_household_week','no_treatment_or_control_columns','HARD_INVARIANT',
       CASE WHEN forbidden_cols.n = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(forbidden_cols.n AS VARCHAR), '0' FROM forbidden_cols
