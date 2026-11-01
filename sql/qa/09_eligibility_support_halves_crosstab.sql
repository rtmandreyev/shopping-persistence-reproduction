-- Phase 2A tracked aggregate: panel_eligibility_support_halves_crosstab
-- Two-dimensional support diagnostic: first-half (weeks 1-51) activity x
-- second-half (weeks 52-102) activity. Aggregate counts only.
-- Diagnostic only; implies no eligibility rule.
-- Output: outputs/tables/panels/panel_eligibility_support_halves_crosstab.csv
-- Deterministic (ORDER BY first_half_active, second_half_active).

SELECT
    CASE WHEN active_weeks_first_half > 0 THEN 1 ELSE 0 END  AS first_half_active,
    CASE WHEN active_weeks_second_half > 0 THEN 1 ELSE 0 END AS second_half_active,
    count(*)                                                 AS households
FROM panel_eligibility
GROUP BY first_half_active, second_half_active
ORDER BY first_half_active, second_half_active
