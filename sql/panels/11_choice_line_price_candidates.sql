-- Phase 2B C: choice_line_price_candidates
-- Grain : household_key x basket_id x product_id   (one transaction line)
-- Key   : (household_key, basket_id, product_id) -- unique
-- Population: positive-quantity transaction lines only (2,581,266 of 2,595,732).
--   The 14,466 QUANTITY = 0 lines are EXCLUDED because the vendor unit-price
--   formulas are undefined without a positive quantity; those lines remain in the
--   canonical tables (basket_panel, household_week_base) with their QA flags.
--
-- VENDOR-DERIVED PRICE CANDIDATES (candidate measures only -- NOT the chosen
-- economic price; no column is named simply "price"):
--   vendor_loyalty_unit_price_candidate    = (SALES_VALUE - (RETAIL_DISC + COUPON_MATCH_DISC)) / QUANTITY
--   vendor_nonloyalty_unit_price_candidate = (SALES_VALUE - COUPON_MATCH_DISC) / QUANTITY
-- These are the EXACT Phase 1 accepted vendor formulas, evaluated at the same
-- transaction-line grain on which Phase 1 verified them (docs/DATA_AUDIT.md,
-- "Price semantics"). Because QUANTITY > 0 by construction, neither is NULL here
-- and no zero-division occurs.
--
-- NOTE ON INTERPRETATION: Phase 1 established that these are candidate unit-price
-- measures only. SALES_VALUE is retailer revenue, not necessarily the customer's
-- price; Phase 1 verified the formulas produce no negative values at this grain,
-- and the implied customer-paid line amount SALES_VALUE + COUPON_DISC is negative
-- on some positive-quantity lines. Both conditions are preserved, not hidden.

SELECT
    household_key,
    BASKET_ID              AS basket_id,
    PRODUCT_ID             AS product_id,
    STORE_ID               AS store_id,
    DAY                    AS day,
    WEEK_NO                AS week_no,
    QUANTITY               AS quantity,
    SALES_VALUE            AS sales_value,
    RETAIL_DISC            AS retail_disc,
    COUPON_DISC            AS coupon_disc,
    COUPON_MATCH_DISC      AS coupon_match_disc,
    (SALES_VALUE - (RETAIL_DISC + COUPON_MATCH_DISC)) / QUANTITY AS vendor_loyalty_unit_price_candidate,
    (SALES_VALUE - COUPON_MATCH_DISC) / QUANTITY                 AS vendor_nonloyalty_unit_price_candidate
FROM transaction_data
WHERE QUANTITY > 0
ORDER BY household_key, BASKET_ID, PRODUCT_ID
