-- Phase 3 Question A: observed purchase support and calendar diagnostics
-- Grain : week_no
-- Sources: household_history, household_week_base, basket_panel
--
-- Two mechanical questions the Question A series cannot answer on its own:
--
-- 1. OBSERVED PURCHASE SUPPORT. "active_households" out of 2,500 mixes how many
--    households lie inside their own observed purchase-support window with how
--    likely such a household is to purchase. household_history gives each
--    household's first and last observed purchase week (accepted factual fields),
--    so the in-support set for week w is
--    {first_purchase_week <= w AND last_purchase_week >= w}.
--
--    This is an OBSERVED PURCHASE-SUPPORT ENVELOPE. It is NOT household entry or
--    exit, NOT customer acquisition or churn, and NOT verified retailer
--    membership. It is a property of the observation window: the in-support set
--    forms over the first weeks and declines over the last ones, and the share of
--    in-support households that is active is the participation measure that is
--    comparable across weeks. Leading/trailing truncation is calendar coverage.
--
-- 2. NEAR-EMPTY DAYS. The transaction file contains isolated days with almost no
--    recorded activity (day 278 in week 40 and day 643 in week 93, each carrying a
--    single basket). A week containing such a day has a structurally lower total,
--    so the weekly total excluding its own lowest day is reported alongside it as a
--    labelled sensitivity. The baseline stays as-is.
--
-- weekly_sales_value is retailer SALES_VALUE (retailer revenue), not customer spend.

WITH daily AS (
    SELECT
        day,
        week_no,
        sum(basket_sales_value) AS day_sales_value,
        count(*)                AS day_baskets
    FROM basket_panel
    GROUP BY day, week_no
),

day_baseline AS (
    SELECT median(day_sales_value) AS median_day_sales_value
    FROM daily
),

local_day_baseline AS (
    -- Median day value over the week's centred 7-week neighbourhood. A "near-empty
    -- day" is identified against its own local calendar rather than against the
    -- whole window, because the early weeks are low for a different reason (the
    -- support structure is still forming) and would otherwise be flagged.
    SELECT
        weeks.week_no,
        median(d.day_sales_value) AS local_median_day_sales_value
    FROM (SELECT DISTINCT week_no FROM daily) AS weeks
    JOIN daily AS d ON d.week_no BETWEEN weeks.week_no - 3 AND weeks.week_no + 3
    GROUP BY weeks.week_no
),

weekly_daily AS (
    SELECT
        week_no,
        count(*)                                    AS observed_days_in_week,
        sum(day_sales_value)                        AS week_sales_value_from_daily,
        min(day_sales_value)                        AS min_day_sales_value,
        max(day_sales_value)                        AS max_day_sales_value,
        arg_min(day, day_sales_value)               AS lowest_day_number,
        arg_max(day, day_sales_value)               AS highest_day_number
    FROM daily
    GROUP BY week_no
),

in_support AS (
    SELECT weeks.week_no, count(*) AS in_support_households
    FROM (SELECT DISTINCT week_no FROM household_week_base) AS weeks
    JOIN household_history AS h
      ON h.first_purchase_week <= weeks.week_no
     AND h.last_purchase_week >= weeks.week_no
    GROUP BY weeks.week_no
),

active AS (
    SELECT week_no, sum(retailer_purchase) AS active_households
    FROM household_week_base
    GROUP BY week_no
),

windowed AS (
    SELECT
        p.week_no,
        p.in_support_households,
        a.active_households,
        a.active_households::DOUBLE / nullif(p.in_support_households, 0)
            AS active_share_of_in_support,
        w.observed_days_in_week,
        w.week_sales_value_from_daily,
        w.min_day_sales_value,
        w.max_day_sales_value,
        w.lowest_day_number,
        w.highest_day_number,
        w.week_sales_value_from_daily - w.min_day_sales_value AS week_sales_value_excluding_lowest_day,
        CASE
            WHEN w.min_day_sales_value < 0.25 * ld.local_median_day_sales_value THEN 1
            ELSE 0
        END AS has_low_activity_day,
        b.median_day_sales_value,
        ld.local_median_day_sales_value
    FROM in_support AS p
    JOIN active AS a ON a.week_no = p.week_no
    JOIN weekly_daily AS w ON w.week_no = p.week_no
    JOIN local_day_baseline AS ld ON ld.week_no = p.week_no
    CROSS JOIN day_baseline AS b
)

SELECT
    week_no,
    in_support_households,
    active_households,
    active_share_of_in_support,
    observed_days_in_week,
    min_day_sales_value,
    max_day_sales_value,
    lowest_day_number,
    highest_day_number,
    week_sales_value_excluding_lowest_day,
    has_low_activity_day,
    median_day_sales_value,
    local_median_day_sales_value,
    (SELECT median(w2.week_sales_value_from_daily)
       FROM weekly_daily AS w2
      WHERE w2.week_no BETWEEN windowed.week_no - 3 AND windowed.week_no + 3)
        AS local_median_week_sales_value,
    week_sales_value_from_daily
        / nullif((SELECT median(w2.week_sales_value_from_daily)
                    FROM weekly_daily AS w2
                   WHERE w2.week_no BETWEEN windowed.week_no - 3 AND windowed.week_no + 3), 0)
        AS week_share_of_local_median
FROM windowed
ORDER BY week_no
