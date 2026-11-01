-- Phase 2B tracked aggregate: sample_flag_counts
-- Non-identifying counts for the researcher's sample definition (S0/S1/S2).
-- Output: outputs/tables/panels/sample_flag_counts.csv
-- Deterministic (ORDER BY sample). Counts only: no household identifiers.

SELECT * FROM (
    SELECT 'sample_full' AS sample, 1 AS is_baseline,
           count(DISTINCT household_key) AS households, count(*) AS household_weeks
    FROM econometrics_household_week WHERE sample_full = 1
    UNION ALL
    SELECT 'sample_sustained_support', 0,
           count(DISTINCT household_key), count(*)
    FROM econometrics_household_week WHERE sample_sustained_support = 1
    UNION ALL
    SELECT 'sample_higher_support', 0,
           count(DISTINCT household_key), count(*)
    FROM econometrics_household_week WHERE sample_higher_support = 1
)
ORDER BY sample
