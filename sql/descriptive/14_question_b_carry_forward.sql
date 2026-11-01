-- Phase 3 Question C: bounded Question-B carry-forward composition diagnostic
-- Grain  : household group x dimension x label
-- Sources: household_history, household_category_week, the validated department derivation
--
-- Purpose: ask whether BROAD product/department composition differs between the top 1% of
-- households by whole-sample retailer SALES_VALUE and the top 1% by whole-sample trips,
-- as a secondary diagnostic for the Question B carry-forward question (why the extreme
-- value and trip households overlap only modestly).
--
-- Boundaries:
--   * groups are defined factually by a declared rank rule on accepted household totals
--     (top ceil(0.01 x 2,500) = 25 households in each ranking);
--   * NO household identifier is emitted by this query: every row is an aggregate over a
--     group;
--   * composition is DESCRIPTIVE. It says nothing about preferences, demographics, motives
--     or causality, and a composition difference is not a behavioural explanation;
--   * department composition comes from the validated department derivation, and commodity
--     composition from household_category_week. Commodity does NOT map one-to-one to
--     department (36 commodities span more than one department), so the two dimensions are
--     reported separately and never re-derived from each other;
--   * retailer SALES_VALUE means retailer revenue, and the shares below are each label's
--     share of that group's own recorded retailer SALES_VALUE.

WITH totals AS (
    SELECT household_key, total_sales_value, total_baskets
    FROM household_history
),

ranked AS (
    SELECT
        household_key,
        row_number() OVER (ORDER BY total_sales_value DESC, household_key) AS value_rank,
        row_number() OVER (ORDER BY total_baskets DESC, household_key)     AS trips_rank,
        count(*) OVER ()                                                   AS households
    FROM totals
),

groups AS (
    SELECT 'top 1% by retailer SALES_VALUE' AS household_group, household_key
    FROM ranked WHERE value_rank <= ceil(households * 0.01)
    UNION ALL
    SELECT 'top 1% by trips' AS household_group, household_key
    FROM ranked WHERE trips_rank <= ceil(households * 0.01)
),

group_totals AS (
    SELECT household_group, sum(total_sales_value) AS group_sales_value
    FROM groups JOIN totals USING (household_key)
    GROUP BY household_group
),

commodity_rows AS (
    SELECT
        g.household_group                                                       AS household_group,
        'commodity'                                                             AS dimension,
        hcw.commodity_desc                                                      AS label,
        sum(hcw.category_sales_value)                                            AS retailer_sales_value,
        count(DISTINCT hcw.household_key)                                        AS buying_households,
        sum(hcw.category_trips)                                                  AS trips,
        sum(hcw.category_units)                                                  AS recorded_units,
        sum(hcw.category_private_sales_value)                                    AS private_sales_value,
        sum(hcw.category_national_sales_value)                                   AS national_sales_value
    FROM groups AS g
    JOIN household_category_week AS hcw USING (household_key)
    GROUP BY g.household_group, hcw.commodity_desc
),

department_rows AS (
    SELECT
        g.household_group                                                       AS household_group,
        'department'                                                            AS dimension,
        dd.department_desc                                                      AS label,
        sum(dd.department_sales_value)                                           AS retailer_sales_value,
        count(DISTINCT dd.household_key)                                         AS buying_households,
        sum(dd.department_trips)                                                 AS trips,
        sum(dd.department_units)                                                 AS recorded_units,
        sum(dd.department_private_sales_value)                                   AS private_sales_value,
        sum(dd.department_national_sales_value)                                  AS national_sales_value
    FROM groups AS g
    JOIN descriptive_department_week AS dd USING (household_key)
    GROUP BY g.household_group, dd.department_desc
),

combined AS (
    SELECT * FROM commodity_rows
    UNION ALL
    SELECT * FROM department_rows
)

SELECT
    c.household_group,
    c.dimension,
    c.label,
    c.buying_households,
    c.retailer_sales_value,
    c.retailer_sales_value / nullif(gt.group_sales_value, 0)                    AS share_of_group_retailer_sales_value,
    c.trips,
    c.recorded_units,
    c.national_sales_value,
    c.private_sales_value,
    c.national_sales_value / nullif(c.national_sales_value + c.private_sales_value, 0)
                                                                                AS national_share_of_sales_value,
    c.private_sales_value / nullif(c.national_sales_value + c.private_sales_value, 0)
                                                                                AS private_share_of_sales_value,
    gt.group_sales_value,
    (SELECT count(*) FROM groups WHERE household_group = c.household_group)      AS households_in_group
FROM combined AS c
JOIN group_totals AS gt USING (household_group)
ORDER BY c.household_group, c.dimension, c.retailer_sales_value DESC
