-- Phase 2A extension:  panel_eligibility_halves_by_active_weeks_thresholds
-- Households active in BOTH sample halves (weeks 1-51 and 52-102), crossed with
-- diagnostic active-week thresholds.
-- Output: outputs/tables/panels/panel_eligibility_halves_by_active_weeks_thresholds.csv
--
-- DIAGNOSTIC ONLY: thresholds are not recommendations, rules, or rankings, and
-- imply no sample selection. share_of_households is out of all 2,500 households.
-- Both flag values are emitted (explicit zero rows); for each threshold the two
-- rows sum to the full household universe.
-- Deterministic (ORDER BY threshold, active_in_both_halves).

WITH thresholds AS (
    SELECT unnest([1, 10, 20, 30, 40, 50]) AS active_weeks_threshold
),
flags AS (SELECT unnest([0, 1]) AS active_in_both_halves),
total AS (SELECT count(*) AS n FROM panel_eligibility)
SELECT
    t.active_weeks_threshold,
    f.active_in_both_halves,
    count(pe.household_key)                                        AS households,
    CAST(count(pe.household_key) AS DOUBLE) / (SELECT n FROM total) AS share_of_households
FROM thresholds AS t
CROSS JOIN flags AS f
LEFT JOIN panel_eligibility AS pe
    ON pe.active_weeks >= t.active_weeks_threshold
   AND (CASE WHEN pe.active_weeks_first_half > 0 AND pe.active_weeks_second_half > 0 THEN 1 ELSE 0 END)
        = f.active_in_both_halves
GROUP BY t.active_weeks_threshold, f.active_in_both_halves
ORDER BY t.active_weeks_threshold, f.active_in_both_halves
