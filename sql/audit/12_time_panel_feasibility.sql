-- 12_time_panel_feasibility.sql
-- Purpose: what information exists to support a household-week panel in Phase 2.
-- Grain: distribution summaries + per-household activity counts.
--
-- CONCEPTUAL POINT, stated by the project plan and preserved here: a household
-- with no purchase in a week is NOT automatically a missing observation. The
-- dataset records purchases, not non-purchases. Whether a zero-fill panel,
-- an observed-activity panel, or a trip-conditioned panel is appropriate is a
-- Phase 2 decision. This query reports the evidence needed to make it; it does
-- NOT construct any panel.
-- Provenance: A = documented 2-year window, B = observed.

SELECT 'week_min' AS metric, min(WEEK_NO)::VARCHAR AS value, 'B' AS provenance FROM transaction_data
UNION ALL SELECT 'week_max', max(WEEK_NO)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'distinct_weeks_observed', count(DISTINCT WEEK_NO)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'weeks_in_1_102_not_observed',
       (SELECT count(*)::VARCHAR FROM generate_series(1, 102) g(w)
        WHERE w NOT IN (SELECT DISTINCT WEEK_NO FROM transaction_data)), 'B'
UNION ALL SELECT 'households', count(DISTINCT household_key)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'household_week_observed_pairs', count(*)::VARCHAR, 'B'
        FROM (SELECT DISTINCT household_key, WEEK_NO FROM transaction_data)
UNION ALL SELECT 'potential_balanced_cells_2500x102',
       (count(DISTINCT household_key) * 102)::VARCHAR, 'C arithmetic reference'
        FROM transaction_data
UNION ALL SELECT 'observed_cell_coverage_pct',
       round(100.0 * (SELECT count(*) FROM (SELECT DISTINCT household_key, WEEK_NO FROM transaction_data))
             / (count(DISTINCT household_key) * 102), 4)::VARCHAR,
       'B observed; C share of a full 2500x102 grid that is observed'
        FROM transaction_data

-- Per-household activity distribution.
UNION ALL SELECT 'active_weeks_per_hh_min', min(cnt)::VARCHAR, 'B'
        FROM (SELECT household_key, count(DISTINCT WEEK_NO) AS cnt FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'active_weeks_per_hh_p25', quantile_cont(cnt, 0.25)::VARCHAR, 'B'
        FROM (SELECT household_key, count(DISTINCT WEEK_NO) AS cnt FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'active_weeks_per_hh_median', median(cnt)::VARCHAR, 'B'
        FROM (SELECT household_key, count(DISTINCT WEEK_NO) AS cnt FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'active_weeks_per_hh_p75', quantile_cont(cnt, 0.75)::VARCHAR, 'B'
        FROM (SELECT household_key, count(DISTINCT WEEK_NO) AS cnt FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'active_weeks_per_hh_max', max(cnt)::VARCHAR, 'B'
        FROM (SELECT household_key, count(DISTINCT WEEK_NO) AS cnt FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'active_weeks_per_hh_mean', round(avg(cnt), 4)::VARCHAR, 'B'
        FROM (SELECT household_key, count(DISTINCT WEEK_NO) AS cnt FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'households_active_all_102_weeks', count(*) FILTER (WHERE cnt = 102)::VARCHAR, 'B'
        FROM (SELECT household_key, count(DISTINCT WEEK_NO) AS cnt FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'households_active_under_52_weeks', count(*) FILTER (WHERE cnt < 52)::VARCHAR, 'B'
        FROM (SELECT household_key, count(DISTINCT WEEK_NO) AS cnt FROM transaction_data GROUP BY 1)

-- First/last observed week, and the resulting panel window per household.
UNION ALL SELECT 'hh_first_week_min', min(fw)::VARCHAR, 'B'
        FROM (SELECT household_key, min(WEEK_NO) AS fw FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'hh_first_week_max', max(fw)::VARCHAR, 'B'
        FROM (SELECT household_key, min(WEEK_NO) AS fw FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'hh_last_week_min', min(lw)::VARCHAR, 'B'
        FROM (SELECT household_key, max(WEEK_NO) AS lw FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'hh_last_week_max', max(lw)::VARCHAR, 'B'
        FROM (SELECT household_key, max(WEEK_NO) AS lw FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'households_starting_after_week_1', count(*) FILTER (WHERE fw > 1)::VARCHAR,
       'B; C indicates left-truncation, not churn'
        FROM (SELECT household_key, min(WEEK_NO) AS fw FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'households_ending_before_week_102', count(*) FILTER (WHERE lw < 102)::VARCHAR,
       'B; C indicates right-truncation, not churn'
        FROM (SELECT household_key, max(WEEK_NO) AS lw FROM transaction_data GROUP BY 1)

-- Trips per household.
UNION ALL SELECT 'trips_per_hh_min', min(cnt)::VARCHAR, 'B'
        FROM (SELECT household_key, count(DISTINCT BASKET_ID) AS cnt FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'trips_per_hh_median', median(cnt)::VARCHAR, 'B'
        FROM (SELECT household_key, count(DISTINCT BASKET_ID) AS cnt FROM transaction_data GROUP BY 1)
UNION ALL SELECT 'trips_per_hh_max', max(cnt)::VARCHAR, 'B'
        FROM (SELECT household_key, count(DISTINCT BASKET_ID) AS cnt FROM transaction_data GROUP BY 1)

-- Internal gaps between consecutive active weeks, for the serial-dependence
-- discussion in the panel stage.
UNION ALL SELECT 'households_with_internal_week_gaps', count(*) FILTER (WHERE has_gap)::VARCHAR, 'B'
        FROM (SELECT household_key,
                     max(gap) > 1 AS has_gap
              FROM (SELECT household_key, WEEK_NO,
                           WEEK_NO - lag(WEEK_NO) OVER (PARTITION BY household_key ORDER BY WEEK_NO) AS gap
                    FROM (SELECT DISTINCT household_key, WEEK_NO FROM transaction_data))
              GROUP BY 1)
UNION ALL SELECT 'median_gap_between_active_weeks', median(gap)::VARCHAR, 'B'
        FROM (SELECT household_key, WEEK_NO,
                     WEEK_NO - lag(WEEK_NO) OVER (PARTITION BY household_key ORDER BY WEEK_NO) AS gap
              FROM (SELECT DISTINCT household_key, WEEK_NO FROM transaction_data))
        WHERE gap IS NOT NULL

-- Store activity over time (relevant to product-store-week construction).
UNION ALL SELECT 'distinct_stores', count(DISTINCT STORE_ID)::VARCHAR, 'B' FROM transaction_data
UNION ALL SELECT 'store_week_observed_pairs', count(*)::VARCHAR, 'B'
        FROM (SELECT DISTINCT STORE_ID, WEEK_NO FROM transaction_data)
UNION ALL SELECT 'product_week_observed_pairs', count(*)::VARCHAR, 'B'
        FROM (SELECT DISTINCT PRODUCT_ID, WEEK_NO FROM transaction_data)
UNION ALL SELECT 'product_store_week_observed_pairs', count(*)::VARCHAR, 'B'
        FROM (SELECT DISTINCT PRODUCT_ID, STORE_ID, WEEK_NO FROM transaction_data)
