-- 12b_household_week_activity.sql
-- PURPOSE: per-household activity profile, for reviewing how a household-week
--          panel would have to be constructed in Phase 2.
-- Grain: one row per household.
--
-- A household with no purchase in a week is NOT a missing observation: the
-- dataset records purchases, not non-purchases. The gap between active_weeks
-- and window_length is the size of that ambiguity, reported here rather than
-- resolved.
-- Provenance: B = observed.

-- Per-household week coverage, written out for later panel construction review.
SELECT household_key,
       count(*) AS active_weeks,
       min(WEEK_NO) AS first_week,
       max(WEEK_NO) AS last_week,
       (max(WEEK_NO) - min(WEEK_NO) + 1) AS window_length
FROM (SELECT DISTINCT household_key, WEEK_NO FROM transaction_data)
GROUP BY household_key
ORDER BY household_key