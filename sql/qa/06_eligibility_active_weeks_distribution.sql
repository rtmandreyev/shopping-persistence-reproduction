-- Phase 2A tracked aggregate: panel_eligibility_active_weeks_distribution
-- Non-identifying distribution summary of the local household-level diagnostic.
-- Output: outputs/tables/panels/panel_eligibility_active_weeks_distribution.csv
-- Deterministic (ORDER BY statistic).

SELECT * FROM (
    SELECT 'households' AS statistic, CAST(count(*) AS VARCHAR) AS value FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_min', CAST(min(active_weeks) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_p05', CAST(quantile_cont(active_weeks, 0.05) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_p10', CAST(quantile_cont(active_weeks, 0.10) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_p25', CAST(quantile_cont(active_weeks, 0.25) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_median', CAST(quantile_cont(active_weeks, 0.50) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_p75', CAST(quantile_cont(active_weeks, 0.75) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_p90', CAST(quantile_cont(active_weeks, 0.90) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_p95', CAST(quantile_cont(active_weeks, 0.95) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_max', CAST(max(active_weeks) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'active_weeks_mean', CAST(avg(active_weeks) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'active_share_102_mean', CAST(avg(active_share_102) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'leading_no_purchase_weeks_median',
        CAST(quantile_cont(leading_no_purchase_weeks, 0.50) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'trailing_no_purchase_weeks_median',
        CAST(quantile_cont(trailing_no_purchase_weeks, 0.50) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'longest_internal_gap_median',
        CAST(quantile_cont(longest_internal_no_purchase_week_gap, 0.50) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'households_with_demographics',
        CAST(count(*) FILTER (WHERE has_demographics = 1) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'households_campaign_recipient_ever',
        CAST(count(*) FILTER (WHERE campaign_recipient_ever = 1) AS VARCHAR) FROM panel_eligibility
    UNION ALL SELECT 'households_coupon_redeemer_ever',
        CAST(count(*) FILTER (WHERE coupon_redeemer_ever = 1) AS VARCHAR) FROM panel_eligibility
)
ORDER BY statistic
