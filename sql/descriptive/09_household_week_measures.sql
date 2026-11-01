-- Phase 3 Question B: household-week measures for within/between variation
-- Grain  : household_key x week_no (the full 2,500 x 102 = 255,000-cell grid)
-- Sources: household_week_base, household_history, econometrics_household_week
--          (accepted sample flags only), all accepted Phase 2 tables
--
-- Purpose: give the caller one explicit place to state which denominator a
-- within/between statement is computed on, because the two panels answer different
-- questions and must never be silently mixed:
--
--   in_support = 1  -> weeks inside the household's own observed purchase-support
--                      window (first_purchase_week <= week <= last_purchase_week)
--   retailer_purchase = 1 -> weeks with a recorded purchase from this retailer
--
-- An inactive household-week is "no recorded purchase from this retailer"; it is never
-- zero grocery consumption. Sales-value columns are retailer SALES_VALUE / retailer
-- revenue, never household expenditure.
--
-- Sample flags are carried for labelled sensitivity only. They are never a filter here
-- and they are not causal eligibility rules.

SELECT
    hw.household_key,
    hw.week_no,
    hw.retailer_purchase,
    hw.weekly_sales_value,
    hw.trips,
    hw.units,
    hw.distinct_products,
    hw.distinct_commodities,
    hw.transaction_line_count,
    hw.qa_quantity_zero_lines,
    hw.qa_retail_disc_positive_lines,
    hw.qa_implied_paid_negative_lines,
    hh.first_purchase_week,
    hh.last_purchase_week,
    CASE
        WHEN hw.week_no BETWEEN hh.first_purchase_week AND hh.last_purchase_week
        THEN 1
        ELSE 0
    END                                     AS in_support,
    CASE WHEN hw.week_no <= 51 THEN 'first' ELSE 'second' END AS half,
    ehw.sample_full,
    ehw.sample_sustained_support,
    ehw.sample_higher_support
FROM household_week_base AS hw
LEFT JOIN household_history AS hh USING (household_key)
LEFT JOIN econometrics_household_week AS ehw
    ON hw.household_key = ehw.household_key AND hw.week_no = ehw.week_no
ORDER BY hw.household_key, hw.week_no
