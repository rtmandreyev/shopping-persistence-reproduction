-- Phase 2A extension:  panel_eligibility_gap_thresholds
-- Counts/shares of households whose longest internal no-purchase gap is within a
-- threshold, alone and combined with active-week thresholds.
-- Output: outputs/tables/panels/panel_eligibility_gap_thresholds.csv
--
-- DIAGNOSTIC ONLY: gap thresholds are not desirable, sufficient, stable,
-- eligible, clean, or recommended; they are not rules and imply no sample
-- selection. active_weeks_threshold = 0 means "no activity constraint".
-- share_of_households is out of all 2,500 households.
-- Deterministic (ORDER BY gap_threshold, active_weeks_threshold).

WITH gap_thresholds AS (
    SELECT unnest([4, 8, 12, 20]) AS gap_threshold_weeks
),
activity_levels AS (
    SELECT unnest([0, 20, 30]) AS active_weeks_threshold
),
total AS (SELECT count(*) AS n FROM panel_eligibility)
SELECT
    g.gap_threshold_weeks,
    a.active_weeks_threshold,
    count(pe.household_key)                                        AS households,
    CAST(count(pe.household_key) AS DOUBLE) / (SELECT n FROM total) AS share_of_households
FROM gap_thresholds AS g
CROSS JOIN activity_levels AS a
LEFT JOIN panel_eligibility AS pe
    ON pe.longest_internal_no_purchase_week_gap <= g.gap_threshold_weeks
   AND (a.active_weeks_threshold = 0 OR pe.active_weeks >= a.active_weeks_threshold)
GROUP BY g.gap_threshold_weeks, a.active_weeks_threshold
ORDER BY g.gap_threshold_weeks, a.active_weeks_threshold
