-- 08_household_coverage.sql
-- Purpose: household coverage across transactions, demographics, campaigns,
--          redemptions, and the overlap between them.
-- Grain: one row per metric.
--
-- CRITICAL: demographics are an optional attribute for a SUBSET of households
-- (vendor: "demographic information ... for a portion of households"). Households
-- without them are NOT missing data to be filled, and the main analysis sample
-- must not be restricted to households that have demographics.
-- Provenance: A = vendor "portion of households", B = observed.

WITH tx_hh AS (SELECT count(DISTINCT household_key) AS n FROM transaction_data),
     demo_hh AS (SELECT count(DISTINCT household_key) AS n FROM hh_demographic)

SELECT 'households_in_transactions' AS metric,
       (SELECT n FROM tx_hh)::VARCHAR AS value, 'B' AS provenance
UNION ALL SELECT 'households_in_hh_demographic', (SELECT n FROM demo_hh)::VARCHAR, 'B'
UNION ALL SELECT 'households_in_both',
       (SELECT count(*) FROM (SELECT household_key FROM transaction_data
                              INTERSECT SELECT household_key FROM hh_demographic))::VARCHAR, 'B'
UNION ALL SELECT 'transacting_households_without_demographics',
       (SELECT n FROM tx_hh) -
       (SELECT count(*) FROM (SELECT household_key FROM transaction_data
                              INTERSECT SELECT household_key FROM hh_demographic)), 'B'
UNION ALL SELECT 'demographic_households_without_transactions',
       (SELECT n FROM demo_hh) -
       (SELECT count(*) FROM (SELECT household_key FROM transaction_data
                              INTERSECT SELECT household_key FROM hh_demographic)), 'B'
UNION ALL SELECT 'demographic_coverage_pct_of_transactors',
       round(100.0 * (SELECT count(*) FROM (SELECT household_key FROM transaction_data
                                           INTERSECT SELECT household_key FROM hh_demographic))
             / (SELECT n FROM tx_hh), 4)::VARCHAR, 'B'
UNION ALL SELECT 'households_in_campaign_table',
       (SELECT count(DISTINCT household_key) FROM campaign_table)::VARCHAR, 'B'
UNION ALL SELECT 'households_in_coupon_redempt',
       (SELECT count(DISTINCT household_key) FROM coupon_redempt)::VARCHAR, 'B'
UNION ALL SELECT 'households_in_campaign_table_not_in_transactions',
       (SELECT count(*) FROM (SELECT household_key FROM campaign_table
                              EXCEPT SELECT household_key FROM transaction_data))::VARCHAR, 'B'
UNION ALL SELECT 'households_in_coupon_redempt_not_in_transactions',
       (SELECT count(*) FROM (SELECT household_key FROM coupon_redempt
                              EXCEPT SELECT household_key FROM transaction_data))::VARCHAR, 'B'
UNION ALL SELECT 'households_in_coupon_redempt_not_in_campaign_table',
       (SELECT count(*) FROM (SELECT household_key FROM coupon_redempt
                              EXCEPT SELECT household_key FROM campaign_table))::VARCHAR, 'B'
UNION ALL SELECT 'household_key_min', (SELECT min(household_key) FROM transaction_data)::VARCHAR, 'B'
UNION ALL SELECT 'household_key_max', (SELECT max(household_key) FROM transaction_data)::VARCHAR, 'B'

-- Observed levels of each demographic variable. Vendor says values are ordered
-- and meaningful; listing them preserves that rather than assuming an ordering.
UNION ALL SELECT 'classification_1_values',
       (SELECT coalesce(string_agg(DISTINCT classification_1, ' | ' ORDER BY classification_1), 'none')
        FROM hh_demographic)::VARCHAR, 'A ordered values; B observed levels'
UNION ALL SELECT 'classification_2_values',
       (SELECT coalesce(string_agg(DISTINCT classification_2, ' | ' ORDER BY classification_2), 'none')
        FROM hh_demographic)::VARCHAR, 'A ordered values; B observed levels'
UNION ALL SELECT 'classification_3_values',
       (SELECT coalesce(string_agg(DISTINCT classification_3, ' | ' ORDER BY classification_3), 'none')
        FROM hh_demographic)::VARCHAR, 'A ordered values; B observed levels'
UNION ALL SELECT 'classification_4_values',
       (SELECT coalesce(string_agg(DISTINCT classification_4, ' | ' ORDER BY classification_4), 'none')
        FROM hh_demographic)::VARCHAR, 'A ordered values; B observed levels'
UNION ALL SELECT 'classification_5_values',
       (SELECT coalesce(string_agg(DISTINCT classification_5, ' | ' ORDER BY classification_5), 'none')
        FROM hh_demographic)::VARCHAR, 'A ordered values; B observed levels'
UNION ALL SELECT 'homeowner_values',
       (SELECT coalesce(string_agg(DISTINCT HOMEOWNER_DESC, ' | ' ORDER BY HOMEOWNER_DESC), 'none')
        FROM hh_demographic)::VARCHAR, 'B observed levels'
UNION ALL SELECT 'kid_category_values',
       (SELECT coalesce(string_agg(DISTINCT KID_CATEGORY_DESC, ' | ' ORDER BY KID_CATEGORY_DESC), 'none')
        FROM hh_demographic)::VARCHAR, 'B observed levels'

-- Null counts per demographic column.
UNION ALL SELECT 'hh_demo_null_classification_1', count(*) FILTER (WHERE classification_1 IS NULL)::VARCHAR, 'B' FROM hh_demographic
UNION ALL SELECT 'hh_demo_null_classification_2', count(*) FILTER (WHERE classification_2 IS NULL)::VARCHAR, 'B' FROM hh_demographic
UNION ALL SELECT 'hh_demo_null_classification_3', count(*) FILTER (WHERE classification_3 IS NULL)::VARCHAR, 'B' FROM hh_demographic
UNION ALL SELECT 'hh_demo_null_classification_4', count(*) FILTER (WHERE classification_4 IS NULL)::VARCHAR, 'B' FROM hh_demographic
UNION ALL SELECT 'hh_demo_null_classification_5', count(*) FILTER (WHERE classification_5 IS NULL)::VARCHAR, 'B' FROM hh_demographic
UNION ALL SELECT 'hh_demo_null_homeowner', count(*) FILTER (WHERE HOMEOWNER_DESC IS NULL)::VARCHAR, 'B' FROM hh_demographic
UNION ALL SELECT 'hh_demo_null_kid_category', count(*) FILTER (WHERE KID_CATEGORY_DESC IS NULL)::VARCHAR, 'B' FROM hh_demographic

ORDER BY metric;
