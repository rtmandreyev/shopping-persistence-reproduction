-- Phase 2A extension:  panel_eligibility_observed_span_by_active_weeks
-- observed_span_weeks threshold x active_weeks threshold cross-tab.
--   observed_span_weeks = last_purchase_week - first_purchase_week + 1
-- Output: outputs/tables/panels/panel_eligibility_observed_span_by_active_weeks.csv
--
-- DIAGNOSTIC ONLY: thresholds are not recommendations, rules, or rankings, and
-- imply no sample selection. share_of_households is out of all 2,500 households.
-- Every threshold pair is emitted (explicit zero rows).
-- Deterministic (ORDER BY both thresholds).

WITH span_thresholds AS (
    SELECT unnest([52, 70, 80, 90, 100]) AS observed_span_weeks_threshold
),
activity_thresholds AS (
    SELECT unnest([10, 20, 30, 40, 50]) AS active_weeks_threshold
),
total AS (SELECT count(*) AS n FROM panel_eligibility)
SELECT
    s.observed_span_weeks_threshold,
    a.active_weeks_threshold,
    count(pe.household_key)                                        AS households,
    CAST(count(pe.household_key) AS DOUBLE) / (SELECT n FROM total) AS share_of_households
FROM span_thresholds AS s
CROSS JOIN activity_thresholds AS a
LEFT JOIN panel_eligibility AS pe
    ON (pe.last_purchase_week - pe.first_purchase_week + 1) >= s.observed_span_weeks_threshold
   AND pe.active_weeks >= a.active_weeks_threshold
GROUP BY s.observed_span_weeks_threshold, a.active_weeks_threshold
ORDER BY s.observed_span_weeks_threshold, a.active_weeks_threshold
