-- Phase 2A tracked aggregate: panel_eligibility_active_weeks_thresholds
-- Counts and shares of households at or above diagnostic active-week thresholds.
-- THESE THRESHOLDS ARE DIAGNOSTIC ONLY: they are not recommendations, not
-- eligibility rules, and imply no sample selection.
-- Output: outputs/tables/panels/panel_eligibility_active_weeks_thresholds.csv
-- Deterministic (ORDER BY threshold).

WITH thresholds AS (
    SELECT unnest([1, 10, 20, 30, 40, 50, 60, 70, 80, 90]) AS threshold
),
total AS (
    SELECT count(*) AS n FROM panel_eligibility
)
SELECT
    t.threshold                                                              AS active_weeks_threshold,
    count(*) FILTER (WHERE pe.active_weeks >= t.threshold)                   AS households_at_or_above,
    CAST(count(*) FILTER (WHERE pe.active_weeks >= t.threshold) AS DOUBLE)
        / (SELECT n FROM total)                                              AS share_at_or_above
FROM thresholds AS t
CROSS JOIN panel_eligibility AS pe
GROUP BY t.threshold
ORDER BY t.threshold
