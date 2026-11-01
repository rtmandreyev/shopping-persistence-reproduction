-- Phase 2B QA: campaign_week_calendar
WITH
c AS (
    SELECT
        count(*)                                   AS n_rows,
        count(DISTINCT (campaign, week_no))        AS n_keys,
        count(DISTINCT campaign)                   AS n_campaigns,
        count(DISTINCT week_no)                    AS n_weeks,
        count(*) FILTER (WHERE period_label NOT IN ('pre', 'campaign', 'post'))   AS bad_period,
        count(*) FILTER (WHERE (period_label = 'campaign') <> (is_campaign_week = 1)) AS period_mismatch,
        count(*) FILTER (WHERE is_campaign_week = 1 AND period_label = 'post')    AS campaign_labelled_post,
        count(*) FILTER (WHERE relative_week <> week_no - start_week)             AS rel_mismatch,
        count(*) FILTER (WHERE is_campaign_week = 1 AND (week_no < start_week OR week_no > end_week)) AS out_of_span,
        count(*) FILTER (WHERE is_campaign_week = 0 AND week_no BETWEEN start_week AND end_week)      AS in_span_not_flagged,
        count(*) FILTER (WHERE campaign_type NOT IN ('TypeA', 'TypeB', 'TypeC'))  AS bad_type,
        count(*) FILTER (WHERE start_week > end_week)                            AS empty_window
    FROM campaign_week_calendar
),
c24 AS (
    SELECT count(*) AS n
    FROM campaign_week_calendar
    WHERE campaign = 24 AND end_week = 102 AND ends_after_transaction_window
),
forbidden_cols AS (
    SELECT count(*) AS n FROM information_schema.columns
    WHERE table_name = 'campaign_week_calendar'
      AND regexp_matches(lower(column_name), 'treat|control|expos|eligible|estimated|effect')
)

SELECT 'campaign_week_calendar' AS table_name, 'row_count_30x102' AS check_name, 'HARD_INVARIANT' AS check_class,
       CASE WHEN c.n_rows = 3060 THEN 'PASS' ELSE 'FAIL' END AS status,
       CAST(c.n_rows AS VARCHAR) AS observed, '3060' AS expected FROM c
UNION ALL SELECT 'campaign_week_calendar','declared_key_unique','HARD_INVARIANT',
       CASE WHEN c.n_rows = c.n_keys THEN 'PASS' ELSE 'FAIL' END, CAST(c.n_rows AS VARCHAR), CAST(c.n_keys AS VARCHAR) FROM c
UNION ALL SELECT 'campaign_week_calendar','campaigns_and_weeks_complete','HARD_INVARIANT',
       CASE WHEN c.n_campaigns = 30 AND c.n_weeks = 102 THEN 'PASS' ELSE 'FAIL' END,
       CAST(c.n_campaigns AS VARCHAR) || '/' || CAST(c.n_weeks AS VARCHAR), '30/102' FROM c
UNION ALL SELECT 'campaign_week_calendar','period_label_domain','HARD_INVARIANT',
       CASE WHEN c.bad_period = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.bad_period AS VARCHAR), '0' FROM c
UNION ALL SELECT 'campaign_week_calendar','period_label_matches_flag','HARD_INVARIANT',
       CASE WHEN c.period_mismatch = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.period_mismatch AS VARCHAR), '0' FROM c
UNION ALL SELECT 'campaign_week_calendar','campaign_weeks_never_labelled_post','HARD_INVARIANT',
       CASE WHEN c.campaign_labelled_post = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.campaign_labelled_post AS VARCHAR), '0' FROM c
UNION ALL SELECT 'campaign_week_calendar','relative_week_consistent','HARD_INVARIANT',
       CASE WHEN c.rel_mismatch = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.rel_mismatch AS VARCHAR), '0' FROM c
UNION ALL SELECT 'campaign_week_calendar','campaign_span_consistent','HARD_INVARIANT',
       CASE WHEN c.out_of_span = 0 AND c.in_span_not_flagged = 0 THEN 'PASS' ELSE 'FAIL' END,
       CAST(c.out_of_span AS VARCHAR) || '/' || CAST(c.in_span_not_flagged AS VARCHAR), '0/0' FROM c
UNION ALL SELECT 'campaign_week_calendar','campaign_type_vocabulary_closed','HARD_INVARIANT',
       CASE WHEN c.bad_type = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.bad_type AS VARCHAR), '0' FROM c
UNION ALL SELECT 'campaign_week_calendar','no_empty_campaign_window','HARD_INVARIANT',
       CASE WHEN c.empty_window = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c.empty_window AS VARCHAR), '0' FROM c
UNION ALL SELECT 'campaign_week_calendar','campaign_24_clipped_to_week_102','HARD_INVARIANT',
       CASE WHEN c24.n > 0 THEN 'PASS' ELSE 'FAIL' END, CAST(c24.n AS VARCHAR), '>0' FROM c24
UNION ALL SELECT 'campaign_week_calendar','no_treatment_or_control_columns','HARD_INVARIANT',
       CASE WHEN forbidden_cols.n = 0 THEN 'PASS' ELSE 'FAIL' END, CAST(forbidden_cols.n AS VARCHAR), '0' FROM forbidden_cols
