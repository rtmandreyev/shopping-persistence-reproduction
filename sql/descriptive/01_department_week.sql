-- Phase 3 descriptive derivation: household x department x week retailer activity
-- (docs/phases/PHASE_03_DESCRIPTIVE_ANALYTICS.md, section 2; authorized as a
--  Phase 3 derivation, NOT a canonical Phase 2 table).
--
-- Grain : household_key x department_desc x week_no  (SPARSE)
-- Key   : (household_key, department_desc, week_no) -- asserted unique by caller
-- Sources: transaction_data (raw, read-only) LEFT JOIN product (raw, read-only)
-- Join  : PRODUCT_ID, expected 1:1 (product.PRODUCT_ID is unique: 92,353 = 92,353).
--         The caller asserts the joined row count equals the transaction-line
--         count, so a duplicated product row FAILS instead of silently fanning out.
--
-- Why the raw line grain and not household_category_week:
--   COMMODITY is NOT nested inside DEPARTMENT in this vendor hierarchy -- 36
--   commodity labels span more than one department. Aggregating the accepted
--   commodity-grain table to department would therefore assign a multi-department
--   commodity to whichever department happened to win the grouping, i.e. it would
--   fabricate a department label that the source does not support. Deriving from
--   the product-level fact keeps every line's own DEPARTMENT.
--
-- Sentinel / missing department values are PRESERVED, never filtered: the vendor
-- ships an empty DEPARTMENT string for 15 products (no 'NO DEPARTMENT
-- DESCRIPTION' sentinel exists here). The blank label is kept as an ordinary value
-- and additionally flagged by department_is_unlabelled (1 when the label is blank
-- after trimming). The flag is a presentation aid for later figures -- it is never
-- an inclusion rule, and no row is dropped.
--
-- Measures mirror the accepted commodity-grain table where the same object exists
-- (value, units, trips, distinct products, national/private splits). Department
-- discount aggregates are deliberately NOT created: no frozen Phase 3 output needs
-- them, and discount normalization is deferred (DECISIONS.md P2-003).
--
-- No exposure, treatment, availability or price variable is created; no row is
-- dropped for carrying a flagged anomaly; no interpretation is attached here.

WITH line_facts AS (
    SELECT
        t.household_key,
        t.BASKET_ID,
        t.PRODUCT_ID,
        t.WEEK_NO    AS week_no,
        t.SALES_VALUE,
        t.QUANTITY,
        p.DEPARTMENT AS department_desc,
        p.BRAND      AS brand
    FROM transaction_data AS t
    LEFT JOIN product AS p ON p.PRODUCT_ID = t.PRODUCT_ID
)

SELECT
    household_key,
    department_desc,
    week_no,
    sum(SALES_VALUE)                                                        AS department_sales_value,
    sum(QUANTITY)                                                           AS department_units,
    count(DISTINCT BASKET_ID)                                               AS department_trips,
    count(DISTINCT PRODUCT_ID)                                              AS distinct_products,
    sum(CASE WHEN brand = 'National' THEN SALES_VALUE ELSE 0 END)           AS department_national_sales_value,
    sum(CASE WHEN brand = 'Private'  THEN SALES_VALUE ELSE 0 END)           AS department_private_sales_value,
    sum(CASE WHEN brand = 'National' THEN QUANTITY    ELSE 0 END)           AS department_national_units,
    sum(CASE WHEN brand = 'Private'  THEN QUANTITY    ELSE 0 END)           AS department_private_units,
    count(DISTINCT CASE WHEN brand = 'National' THEN PRODUCT_ID END)        AS distinct_national_products,
    count(DISTINCT CASE WHEN brand = 'Private'  THEN PRODUCT_ID END)        AS distinct_private_products,
    CASE WHEN trim(coalesce(department_desc, '')) = '' THEN 1 ELSE 0 END    AS department_is_unlabelled
FROM line_facts
GROUP BY household_key, department_desc, week_no
ORDER BY household_key, department_desc, week_no
