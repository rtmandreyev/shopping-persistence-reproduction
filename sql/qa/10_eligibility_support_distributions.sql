-- Phase 2A extension:  panel_eligibility_support_distributions
-- Longitudinal-support distributions for the researcher eligibility decision.
-- Output: outputs/tables/panels/panel_eligibility_support_distributions.csv
--
-- DIAGNOSTIC ONLY. Nothing here is desirable, sufficient, stable, eligible,
-- clean, or recommended. No household is ranked and no identifier is emitted.
-- Deterministic (ORDER BY measure).

WITH measures AS (
    SELECT 'observed_span_weeks' AS measure,
           CAST(last_purchase_week - first_purchase_week + 1 AS DOUBLE) AS value
    FROM panel_eligibility
    UNION ALL SELECT 'active_weeks', CAST(active_weeks AS DOUBLE) FROM panel_eligibility
    UNION ALL SELECT 'leading_no_purchase_weeks', CAST(leading_no_purchase_weeks AS DOUBLE) FROM panel_eligibility
    UNION ALL SELECT 'trailing_no_purchase_weeks', CAST(trailing_no_purchase_weeks AS DOUBLE) FROM panel_eligibility
    UNION ALL SELECT 'longest_internal_no_purchase_gap',
        CAST(longest_internal_no_purchase_week_gap AS DOUBLE) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_first_half', CAST(active_weeks_first_half AS DOUBLE) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_second_half', CAST(active_weeks_second_half AS DOUBLE) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_early_third', CAST(active_weeks_early AS DOUBLE) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_middle_third', CAST(active_weeks_middle AS DOUBLE) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_late_third', CAST(active_weeks_late AS DOUBLE) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_block_1', CAST(active_weeks_block_1 AS DOUBLE) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_block_2', CAST(active_weeks_block_2 AS DOUBLE) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_block_3', CAST(active_weeks_block_3 AS DOUBLE) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_block_4', CAST(active_weeks_block_4 AS DOUBLE) FROM panel_eligibility
)
SELECT
    measure,
    round(min(value), 4)                 AS "min",
    round(quantile_cont(value, 0.05), 4) AS p05,
    round(quantile_cont(value, 0.10), 4) AS p10,
    round(quantile_cont(value, 0.25), 4) AS p25,
    round(quantile_cont(value, 0.50), 4) AS median,
    round(quantile_cont(value, 0.75), 4) AS p75,
    round(quantile_cont(value, 0.90), 4) AS p90,
    round(quantile_cont(value, 0.95), 4) AS p95,
    round(max(value), 4)                 AS "max",
    round(avg(value), 4)                 AS mean
FROM measures
GROUP BY measure
ORDER BY measure
