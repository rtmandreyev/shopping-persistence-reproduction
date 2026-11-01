-- Phase 3 Question C: department-level descriptive structure
-- Grain  : department_desc (all 44 labels, including the unlabelled sentinel)
-- Source : the validated Phase 3 department derivation (household x department x week),
--          which is a Phase 3 descriptive derivation and NOT a canonical Phase 2 table,
--          plus the accepted raw product table for the department product universe.
--
-- The derivation's own validated properties are relied on and are not re-derived here:
-- product key unique, 1:1 transaction/product join, no fan-out, no unmatched transaction
-- lines, blank/sentinel department preserved, totals reconcile to the accepted anchors.
--
-- Semantics:
--   * sales-value columns are retailer SALES_VALUE / retailer revenue;
--   * observed-household penetration = distinct buying households / 2,500, always called
--     observed-household penetration and never market penetration;
--   * recorded units are reported with the Question A finding QA-03 in mind and are never
--     the headline ranking;
--   * national/private composition is reported in BOTH retailer SALES_VALUE and units, each
--     labelled, and never collapsed into one generic brand share;
--   * `department_trips` sums household-department-week trip counts, so it double-counts
--     baskets ACROSS departments and is not a basket total;
--   * the unlabelled sentinel department is RETAINED, never dropped or silently merged;
--   * thirds are the accepted 34-week thirds, and Question A finding QA-01 applies to
--     early-third levels.

WITH household_department AS (
    SELECT
        department_desc,
        household_key,
        count(*)                                    AS purchase_weeks,
        sum(department_sales_value)                 AS household_sales_value,
        sum(department_trips)                       AS household_trips
    FROM descriptive_department_week
    GROUP BY department_desc, household_key
),

breadth AS (
    SELECT
        hd.department_desc,
        count(*)                                                        AS buying_households,
        count(*)::DOUBLE / 2500.0                                       AS observed_household_penetration,
        sum(hd.purchase_weeks)                                          AS household_weeks,
        sum(hd.household_trips)                                         AS department_trips,
        sum(hd.household_sales_value)                                   AS retailer_sales_value,
        median(hd.purchase_weeks)                                       AS median_purchase_weeks,
        quantile_cont(hd.purchase_weeks, 0.90)                          AS p90_purchase_weeks,
        max(hd.household_sales_value) / nullif(sum(hd.household_sales_value), 0)
                                                                        AS top_household_sales_value_share
    FROM household_department AS hd
    GROUP BY hd.department_desc
),

unlabelled AS (
    SELECT
        department_desc,
        bool_or(department_is_unlabelled <> 0)                             AS department_is_unlabelled
    FROM descriptive_department_week
    GROUP BY department_desc
),

totals AS (
    SELECT
        department_desc,
        sum(department_units)                                           AS recorded_units,
        max(distinct_products)                                          AS max_household_week_distinct_products,
        sum(department_national_sales_value)                            AS national_sales_value,
        sum(department_private_sales_value)                             AS private_sales_value,
        sum(department_national_units)                                  AS national_units,
        sum(department_private_units)                                   AS private_units
    FROM descriptive_department_week
    GROUP BY department_desc
),

product_universe AS (
    SELECT
        DEPARTMENT                                                      AS department_desc,
        count(DISTINCT PRODUCT_ID)                                      AS distinct_products
    FROM product
    GROUP BY DEPARTMENT
),

thirds AS (
    SELECT
        department_desc,
        count(DISTINCT household_key) FILTER (WHERE week_no <= 34)       AS buying_households_early,
        count(DISTINCT household_key) FILTER (WHERE week_no BETWEEN 35 AND 68) AS buying_households_middle,
        count(DISTINCT household_key) FILTER (WHERE week_no >= 69)       AS buying_households_late,
        sum(department_sales_value) FILTER (WHERE week_no <= 34)         AS sales_value_early,
        sum(department_sales_value) FILTER (WHERE week_no BETWEEN 35 AND 68) AS sales_value_middle,
        sum(department_sales_value) FILTER (WHERE week_no >= 69)         AS sales_value_late,
        count(*) FILTER (WHERE week_no <= 34)                            AS household_weeks_early,
        count(*) FILTER (WHERE week_no BETWEEN 35 AND 68)                AS household_weeks_middle,
        count(*) FILTER (WHERE week_no >= 69)                            AS household_weeks_late
    FROM descriptive_department_week
    GROUP BY department_desc
)

SELECT
    b.department_desc,
    u.department_is_unlabelled,
    b.buying_households,
    b.observed_household_penetration,
    b.household_weeks,
    b.department_trips,
    b.retailer_sales_value,
    t.recorded_units,
    p.distinct_products,
    b.retailer_sales_value / nullif(b.household_weeks, 0)               AS sales_value_per_buying_household_week,
    b.retailer_sales_value / nullif(b.buying_households, 0)             AS sales_value_per_buying_household,
    b.median_purchase_weeks,
    b.p90_purchase_weeks,
    b.top_household_sales_value_share,
    t.max_household_week_distinct_products,
    t.national_sales_value,
    t.private_sales_value,
    t.national_sales_value / nullif(t.national_sales_value + t.private_sales_value, 0)
                                                                        AS national_share_of_sales_value,
    t.private_sales_value / nullif(t.national_sales_value + t.private_sales_value, 0)
                                                                        AS private_share_of_sales_value,
    t.national_units,
    t.private_units,
    t.national_units / nullif(t.national_units + t.private_units, 0)     AS national_share_of_units,
    t.private_units / nullif(t.national_units + t.private_units, 0)      AS private_share_of_units,
    th.buying_households_early,
    th.buying_households_middle,
    th.buying_households_late,
    th.household_weeks_early,
    th.household_weeks_middle,
    th.household_weeks_late,
    th.sales_value_early,
    th.sales_value_middle,
    th.sales_value_late,
    CASE
        WHEN th.buying_households_early > 0 AND th.buying_households_middle > 0
             AND th.buying_households_late > 0
        THEN 1 ELSE 0
    END                                                                 AS present_in_all_three_thirds
FROM breadth AS b
LEFT JOIN unlabelled AS u USING (department_desc)
LEFT JOIN totals AS t USING (department_desc)
LEFT JOIN thirds AS th USING (department_desc)
LEFT JOIN product_universe AS p ON p.department_desc IS NOT DISTINCT FROM b.department_desc
ORDER BY b.retailer_sales_value DESC
