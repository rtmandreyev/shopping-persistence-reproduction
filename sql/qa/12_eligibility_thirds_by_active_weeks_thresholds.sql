-- Phase 2A extension:  panel_eligibility_thirds_by_active_weeks_thresholds
-- Households active in ALL THREE sample thirds (weeks 1-34, 35-68, 69-102),
-- crossed with diagnostic active-week thresholds.
-- Output: outputs/tables/panels/panel_eligibility_thirds_by_active_weeks_thresholds.csv
--
-- DIAGNOSTIC ONLY: not a recommendation, rule, or ranking; no sample selection.
-- share_of_households is out of all 2,500 households. Both flag values are
-- emitted (explicit zero rows); for each threshold the two rows sum to the full
-- household universe.
-- Deterministic (ORDER BY threshold, active_in_all_three_thirds).

WITH thresholds AS (
    SELECT unnest([1, 10, 20, 30, 40, 50]) AS active_weeks_threshold
),
flags AS (SELECT unnest([0, 1]) AS active_in_all_three_thirds),
total AS (SELECT count(*) AS n FROM panel_eligibility)
SELECT
    t.active_weeks_threshold,
    f.active_in_all_three_thirds,
    count(pe.household_key)                                        AS households,
    CAST(count(pe.household_key) AS DOUBLE) / (SELECT n FROM total) AS share_of_households
FROM thresholds AS t
CROSS JOIN flags AS f
LEFT JOIN panel_eligibility AS pe
    ON pe.active_weeks >= t.active_weeks_threshold
   AND (CASE WHEN pe.active_weeks_early > 0 AND pe.active_weeks_middle > 0 AND pe.active_weeks_late > 0
             THEN 1 ELSE 0 END) = f.active_in_all_three_thirds
GROUP BY t.active_weeks_threshold, f.active_in_all_three_thirds
ORDER BY t.active_weeks_threshold, f.active_in_all_three_thirds
