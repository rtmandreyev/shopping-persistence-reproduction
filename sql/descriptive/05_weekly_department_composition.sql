-- Phase 3 Question A: weekly department composition (department derivation grain)
-- Grain : week_no
-- Sources: descriptive_department_week (the Phase 3 temporary view built by
--          complete_journey.descriptive.department; NOT a canonical table)
--
-- Mechanical-explanation check only (Step 6), and it exists because the department
-- derivation is the only route that assigns each transaction line its OWN
-- department (commodity is not nested in department). Nothing here is a Question C
-- output and no department ranking is published.
--
-- The caller builds the view in the same connection before running this query.

WITH weekly_department AS (
    SELECT
        week_no,
        department_desc,
        sum(department_sales_value) AS department_sales_value,
        sum(department_units)       AS department_units
    FROM descriptive_department_week
    GROUP BY week_no, department_desc
),

ranked AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY week_no ORDER BY department_sales_value DESC, department_desc
        ) AS rank_by_sales_value
    FROM weekly_department
)

SELECT
    week_no,
    count(*)                                                              AS active_departments,
    sum(department_sales_value)                                           AS weekly_department_sales_value,
    max(CASE WHEN rank_by_sales_value = 1 THEN department_desc END)         AS largest_department_desc,
    max(CASE WHEN rank_by_sales_value = 1 THEN department_sales_value END)
        / nullif(sum(department_sales_value), 0)                           AS largest_department_sales_value_share,
    sum(CASE WHEN rank_by_sales_value <= 3 THEN department_sales_value ELSE 0 END)
        / nullif(sum(department_sales_value), 0)                           AS top3_department_sales_value_share
FROM ranked
GROUP BY week_no
ORDER BY week_no
