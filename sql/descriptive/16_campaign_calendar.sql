-- Question D: campaign calendar density and concurrency (grain: week_no, 102 rows)
--
-- One row per observed week describing how crowded the campaign calendar is. Week membership
-- and period labels come from the accepted campaign_week_calendar (Phase 1 convention P1-009);
-- campaign start/end days are preserved exactly in 15_campaign_level.sql and are only mapped to
-- weeks here. A campaign touching a week does not mean the whole week was inside the campaign
-- window, so this artifact describes calendar density, never exposure, and it is not an event
-- window, an event-study reference period or a final pre/post design.

WITH per_campaign_week AS (
    SELECT
        week_no,
        campaign,
        campaign_type,
        is_campaign_week,
        relative_week
    FROM campaign_week_calendar
)
SELECT
    week_no,
    count(*) FILTER (WHERE is_campaign_week = 1) AS campaign_spans_touching_week,
    count(DISTINCT campaign) AS campaigns_touching_week,
    count(DISTINCT campaign) FILTER (WHERE campaign_type = 'TypeA') AS typea_spans_touching_week,
    count(DISTINCT campaign) FILTER (WHERE campaign_type = 'TypeB') AS typeb_spans_touching_week,
    count(DISTINCT campaign) FILTER (WHERE campaign_type = 'TypeC') AS typec_spans_touching_week,
    count(DISTINCT campaign) FILTER (WHERE is_campaign_week = 1 AND relative_week = 0)
        AS campaigns_starting_this_week,
    count(DISTINCT campaign) FILTER (
        WHERE is_campaign_week = 1
            AND relative_week = (
                SELECT max(inner_calendar.relative_week)
                FROM campaign_week_calendar AS inner_calendar
                WHERE
                    inner_calendar.campaign = per_campaign_week.campaign
                    AND inner_calendar.is_campaign_week = 1
            )
    ) AS campaigns_ending_this_week,
    CAST(week_no = 1 OR week_no = 102 AS INTEGER) AS edge_of_sample_week,
    CAST(week_no = 1 AS INTEGER) AS partial_week_note_week1,
    CAST(week_no = 102 AS INTEGER) AS partial_week_note_week102
FROM per_campaign_week
GROUP BY week_no
ORDER BY week_no
