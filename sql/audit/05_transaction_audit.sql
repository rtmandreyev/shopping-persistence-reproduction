-- 05_transaction_audit.sql
-- Purpose: entity/time coverage, value-sign anomalies, and structural integrity
--          of transaction_data.
-- Grain: one row per audit metric.
--
-- Nothing here is cleaned or dropped. Anomalies are counted and reported.
-- Historical bug preserved: the earlier mapping floor((DAY - 1) / 7) + 1
-- produced 694,450 false mismatches. Direct boundary checks recover the exact
-- release-specific mapping floor((DAY + 1) / 7) + 1 used below.
-- Provenance: B = observed empirically. Sign expectations from the vendor guide
-- (A) are stated in the metric label.

SELECT 'row_count' AS metric,
       count(*)::VARCHAR AS value,
       'B' AS provenance
FROM transaction_data

UNION ALL SELECT 'distinct_households', count(DISTINCT household_key)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'distinct_baskets', count(DISTINCT BASKET_ID)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'distinct_products', count(DISTINCT PRODUCT_ID)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'distinct_stores', count(DISTINCT STORE_ID)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'distinct_days', count(DISTINCT DAY)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'distinct_weeks', count(DISTINCT WEEK_NO)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'distinct_trans_time', count(DISTINCT TRANS_TIME)::VARCHAR, 'B' FROM transaction_data

UNION ALL SELECT 'day_min', min(DAY)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'day_max', max(DAY)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'week_no_min', min(WEEK_NO)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'week_no_max', max(WEEK_NO)::VARCHAR, 'B' FROM transaction_data

-- Documented (A): week_no ranges 1-102 per the vendor guide.
UNION ALL SELECT 'weeks_missing_from_1_102',
       (SELECT string_agg(w::VARCHAR, ',' ORDER BY w) FROM generate_series(1, 102) g(w)
        WHERE w NOT IN (SELECT DISTINCT WEEK_NO FROM transaction_data))::VARCHAR, 'B'
UNION ALL SELECT 'weeks_present_beyond_102',
       (SELECT coalesce(string_agg(DISTINCT WEEK_NO::VARCHAR, ','), 'none')
        FROM transaction_data WHERE WEEK_NO < 1 OR WEEK_NO > 102)::VARCHAR, 'B'

-- Quantity / value sign anomalies.
UNION ALL SELECT 'quantity_lt_0', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE QUANTITY < 0
UNION ALL SELECT 'quantity_eq_0', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE QUANTITY = 0
UNION ALL SELECT 'quantity_gt_0', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE QUANTITY > 0
UNION ALL SELECT 'quantity_max', max(QUANTITY)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'quantity_min', min(QUANTITY)::VARCHAR, 'B' FROM transaction_data

UNION ALL SELECT 'sales_value_lt_0', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE SALES_VALUE < 0
UNION ALL SELECT 'sales_value_eq_0', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE SALES_VALUE = 0
UNION ALL SELECT 'sales_value_max', max(SALES_VALUE)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'sales_value_min', min(SALES_VALUE)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'sales_value_sum', round(sum(SALES_VALUE), 2)::VARCHAR, 'B' FROM transaction_data

-- Discount sign conventions. The guide defines these as discounts applied, so a
-- NEGATIVE value is the expected representation; a positive value would be a
-- sign-convention surprise. Both are counted rather than assumed.
UNION ALL SELECT 'retail_disc_lt_0', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE RETAIL_DISC < 0
UNION ALL SELECT 'retail_disc_eq_0', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE RETAIL_DISC = 0
UNION ALL SELECT 'retail_disc_gt_0', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE RETAIL_DISC > 0
UNION ALL SELECT 'retail_disc_min', min(RETAIL_DISC)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'retail_disc_max', max(RETAIL_DISC)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'retail_disc_abs_max', max(abs(RETAIL_DISC))::VARCHAR, 'B' FROM transaction_data

UNION ALL SELECT 'coupon_disc_lt_0', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE COUPON_DISC < 0
UNION ALL SELECT 'coupon_disc_gt_0', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE COUPON_DISC > 0
UNION ALL SELECT 'coupon_disc_abs_max', max(abs(COUPON_DISC))::VARCHAR, 'B' FROM transaction_data

UNION ALL SELECT 'coupon_match_disc_lt_0', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE COUPON_MATCH_DISC < 0
UNION ALL SELECT 'coupon_match_disc_gt_0', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE COUPON_MATCH_DISC > 0
UNION ALL SELECT 'coupon_match_disc_abs_max', max(abs(COUPON_MATCH_DISC))::VARCHAR, 'B'
        FROM transaction_data

-- Extreme magnitudes, using a high quantile rather than an arbitrary cut.
UNION ALL SELECT 'quantity_ge_10', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE QUANTITY >= 10
UNION ALL SELECT 'quantity_ge_50', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE QUANTITY >= 50
UNION ALL SELECT 'sales_value_ge_100', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE SALES_VALUE >= 100
UNION ALL SELECT 'sales_value_ge_500', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE SALES_VALUE >= 500
UNION ALL SELECT 'abs_retail_disc_ge_50', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE abs(RETAIL_DISC) >= 50
UNION ALL SELECT 'abs_coupon_disc_ge_50', count(*)::VARCHAR, 'B'
        FROM transaction_data WHERE abs(COUPON_DISC) >= 50

-- Descriptive magnitude comparison only. The vendor guide does not state an
-- accounting identity equating these fields.
UNION ALL SELECT 'rows_sum_abs_recorded_discount_magnitudes_gt_retailer_revenue', count(*)::VARCHAR, 'B; descriptive magnitude comparison'
        FROM transaction_data
        WHERE abs(RETAIL_DISC) + abs(COUPON_DISC) + abs(COUPON_MATCH_DISC) > SALES_VALUE
           AND SALES_VALUE >= 0

-- Structural integrity: a basket should map to one household, one day, one week.
UNION ALL SELECT 'baskets_spanning_multiple_households', count(*)::VARCHAR, 'B'
        FROM (SELECT BASKET_ID FROM transaction_data GROUP BY BASKET_ID
              HAVING count(DISTINCT household_key) > 1)
UNION ALL SELECT 'baskets_spanning_multiple_days', count(*)::VARCHAR, 'B'
        FROM (SELECT BASKET_ID FROM transaction_data GROUP BY BASKET_ID
              HAVING count(DISTINCT DAY) > 1)
UNION ALL SELECT 'baskets_spanning_multiple_weeks', count(*)::VARCHAR, 'B'
        FROM (SELECT BASKET_ID FROM transaction_data GROUP BY BASKET_ID
              HAVING count(DISTINCT WEEK_NO) > 1)
UNION ALL SELECT 'baskets_spanning_multiple_stores', count(*)::VARCHAR, 'B'
        FROM (SELECT BASKET_ID FROM transaction_data GROUP BY BASKET_ID
              HAVING count(DISTINCT STORE_ID) > 1)

-- DAY/WEEK_NO internal consistency. The mapping is empirically recovered from
-- this dataset release; the vendor guide does not formally specify its origin.
-- Exact observed mapping: floor((DAY+1)/7)+1. Week 1 = days 1-5, weeks 2-101
-- = 7 days each, week 102 = days 706-711, so both endpoints are truncated.
UNION ALL SELECT 'rows_week_not_equal_floor_day_plus_1_div_7_plus_1', count(*)::VARCHAR, 'B'
        FROM transaction_data
        WHERE WEEK_NO <> CAST(floor((DAY + 1) / 7.0) + 1 AS BIGINT)
UNION ALL SELECT 'distinct_week_day_pairs', count(*)::VARCHAR, 'B'
        FROM (SELECT DISTINCT WEEK_NO, DAY FROM transaction_data)
UNION ALL SELECT 'weeks_spanning_multiple_days', count(*)::VARCHAR, 'B'
        FROM (SELECT WEEK_NO FROM transaction_data GROUP BY WEEK_NO
              HAVING count(DISTINCT DAY) > 1)
UNION ALL SELECT 'weeks_with_seven_days', count(*)::VARCHAR, 'B'
        FROM (SELECT WEEK_NO FROM transaction_data GROUP BY WEEK_NO
              HAVING count(DISTINCT DAY) = 7)
UNION ALL SELECT 'week_1_first_day', (SELECT min(DAY) FROM transaction_data WHERE WEEK_NO = 1)::VARCHAR, 'B'
UNION ALL SELECT 'week_102_first_day', (SELECT min(DAY) FROM transaction_data WHERE WEEK_NO = 102)::VARCHAR, 'B'
UNION ALL SELECT 'days_per_week_min', (SELECT min(nd) FROM (SELECT count(DISTINCT DAY) nd FROM transaction_data GROUP BY WEEK_NO))::VARCHAR, 'B'
UNION ALL SELECT 'days_per_week_max', (SELECT max(nd) FROM (SELECT count(DISTINCT DAY) nd FROM transaction_data GROUP BY WEEK_NO))::VARCHAR, 'B'

-- Sign-convention exceptions. A 1e-12 tolerance isolates machine-epsilon noise;
-- exact one-cent values are monetary, not floating residue.
UNION ALL SELECT 'retail_disc_positive_within_float_tolerance',
       count(*) FILTER (WHERE RETAIL_DISC > 0 AND RETAIL_DISC <= 1e-12)::VARCHAR,
       'B; tolerance=1e-12; candidate floating-point residue'
        FROM transaction_data
UNION ALL SELECT 'retail_disc_positive_between_tolerance_and_one_cent',
       count(*) FILTER (WHERE RETAIL_DISC > 1e-12 AND RETAIL_DISC < 0.01)::VARCHAR,
       'B; positive values below one cent but above float tolerance'
        FROM transaction_data
UNION ALL SELECT 'retail_disc_positive_monetary',
       count(*) FILTER (WHERE RETAIL_DISC >= 0.01)::VARCHAR,
       'B; monetary positive values at least one cent'
        FROM transaction_data

-- TRANS_TIME: documented as time of day. Check it is plausibly HHMM.
UNION ALL SELECT 'trans_time_min', min(TRANS_TIME)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'trans_time_max', max(TRANS_TIME)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'trans_time_not_4digit', count(*)::VARCHAR, 'B'
        FROM transaction_data
        WHERE length(trim(TRANS_TIME)) <> 4 OR NOT regexp_matches(trim(TRANS_TIME), '^[0-9]+$')
UNION ALL SELECT 'trans_time_outside_0000_2359', count(*)::VARCHAR, 'B'
        FROM transaction_data
        WHERE regexp_matches(trim(TRANS_TIME), '^[0-9]{4}$')
          AND (CAST(left(trim(TRANS_TIME), 2) AS INTEGER) > 23
               OR CAST(right(trim(TRANS_TIME), 2) AS INTEGER) > 59)

-- Household x week x store recombination: is BASKET_ID globally unique?
UNION ALL SELECT 'basket_id_reused_across_households', count(*)::VARCHAR, 'B'
        FROM (SELECT BASKET_ID FROM transaction_data GROUP BY BASKET_ID
              HAVING count(DISTINCT household_key) > 1)

ORDER BY metric;
