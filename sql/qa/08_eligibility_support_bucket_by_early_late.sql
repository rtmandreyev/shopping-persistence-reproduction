-- Phase 2A tracked aggregate: panel_eligibility_support_bucket_by_early_late
-- Two-dimensional support diagnostic: active-history decile bucket x early
-- (weeks 1-34) and late (weeks 69-102) activity. Aggregate counts only.
-- Diagnostic only; implies no eligibility rule.
-- Output: outputs/tables/panels/panel_eligibility_support_bucket_by_early_late.csv
-- Deterministic (ORDER BY bucket, early, late).

WITH buckets AS (
    SELECT
        household_key,
        CAST(floor(active_weeks / 10.0) * 10 AS INTEGER) AS bucket_start,
        CASE WHEN active_weeks_early > 0 THEN 1 ELSE 0 END AS early_active,
        CASE WHEN active_weeks_late > 0 THEN 1 ELSE 0 END AS late_active
    FROM panel_eligibility
)
SELECT
    bucket_start,
    CAST(bucket_start AS VARCHAR) || '-' || CAST(bucket_start + 9 AS VARCHAR) AS active_weeks_bucket,
    early_active,
    late_active,
    count(*) AS households
FROM buckets
GROUP BY bucket_start, early_active, late_active
ORDER BY bucket_start, early_active, late_active
