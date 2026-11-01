-- Question E: frozen representativeness selection (Run 0 plan section 8)
--
-- Grain: one row per public journey label (A, B, C, D) = 4 rows.
--
-- The selection rule is FROZEN and is applied exactly as written in the plan:
--   A HIGH-FREQUENCY      active_weeks >= 90 AND longest_internal_no_purchase_week_gap <= 4
--   B MEDIAN-FREQUENCY    45 <= active_weeks <= 55
--   C INTERMITTENT        active_weeks <= 25
--   D COUPON/CAMPAIGN-ACTIVE  coupon_redeemer_ever = 1 AND campaign_recipient_ever = 1
--                             AND campaign_count >= median campaign_count among listed households
-- Applied in order A then B then C then D, each excluding households already selected. Within a
-- stratum: take the stratum median whole-sample retailer SALES_VALUE, choose the eligible household
-- whose value is closest to that median, tie-break by lowest household_key.
--
-- This query returns the local mapping (label -> household_key) so that the caller can keep it in a
-- Git-ignored artifact. The key is NEVER written to a tracked or published artifact; the public
-- summary is built from the aggregate columns only.
--
-- retailer SALES_VALUE is retailer revenue, never customer spend or grocery expenditure.

WITH base AS (
    SELECT
        household_key,
        active_weeks,
        longest_internal_no_purchase_week_gap,
        total_sales_value,
        total_baskets,
        total_units,
        distinct_products,
        distinct_commodities,
        campaign_recipient_ever,
        campaign_count,
        coupon_redeemer_ever,
        coupon_redemption_count,
        first_purchase_week,
        last_purchase_week
    FROM household_history
),
rule_parameters AS (
    SELECT
        median(campaign_count) AS median_campaign_count_among_listed
    FROM base
    WHERE campaign_recipient_ever = 1
),
pool_a AS (
    SELECT * FROM base WHERE active_weeks >= 90 AND longest_internal_no_purchase_week_gap <= 4
),
selected_a AS (
    SELECT
        'A' AS journey_label,
        'HIGH-FREQUENCY' AS stratum,
        pool_a.*,
        (SELECT median(total_sales_value) FROM pool_a) AS stratum_median_sales_value,
        (SELECT count(*) FROM pool_a) AS eligible_stratum_size,
        abs(total_sales_value - (SELECT median(total_sales_value) FROM pool_a)) AS distance_to_median
    FROM pool_a
    QUALIFY row_number() OVER (
        ORDER BY
            abs(total_sales_value - (SELECT median(total_sales_value) FROM pool_a)) ASC,
            household_key ASC
    ) = 1
),
pool_b AS (
    SELECT * FROM base
    WHERE active_weeks BETWEEN 45 AND 55
      AND household_key NOT IN (SELECT household_key FROM selected_a)
),
selected_b AS (
    SELECT
        'B' AS journey_label,
        'MEDIAN-FREQUENCY' AS stratum,
        pool_b.*,
        (SELECT median(total_sales_value) FROM pool_b) AS stratum_median_sales_value,
        (SELECT count(*) FROM pool_b) AS eligible_stratum_size,
        abs(total_sales_value - (SELECT median(total_sales_value) FROM pool_b)) AS distance_to_median
    FROM pool_b
    QUALIFY row_number() OVER (
        ORDER BY
            abs(total_sales_value - (SELECT median(total_sales_value) FROM pool_b)) ASC,
            household_key ASC
    ) = 1
),
selected_ab AS (
    SELECT * FROM selected_a UNION ALL SELECT * FROM selected_b
),
pool_c AS (
    SELECT * FROM base
    WHERE active_weeks <= 25
      AND household_key NOT IN (SELECT household_key FROM selected_ab)
),
selected_c AS (
    SELECT
        'C' AS journey_label,
        'INTERMITTENT' AS stratum,
        pool_c.*,
        (SELECT median(total_sales_value) FROM pool_c) AS stratum_median_sales_value,
        (SELECT count(*) FROM pool_c) AS eligible_stratum_size,
        abs(total_sales_value - (SELECT median(total_sales_value) FROM pool_c)) AS distance_to_median
    FROM pool_c
    QUALIFY row_number() OVER (
        ORDER BY
            abs(total_sales_value - (SELECT median(total_sales_value) FROM pool_c)) ASC,
            household_key ASC
    ) = 1
),
selected_abc AS (
    SELECT * FROM selected_ab UNION ALL SELECT * FROM selected_c
),
pool_d AS (
    SELECT * FROM base
    WHERE coupon_redeemer_ever = 1
      AND campaign_recipient_ever = 1
      AND campaign_count >= (SELECT median_campaign_count_among_listed FROM rule_parameters)
      AND household_key NOT IN (SELECT household_key FROM selected_abc)
),
selected_d AS (
    SELECT
        'D' AS journey_label,
        'COUPON/CAMPAIGN-ACTIVE' AS stratum,
        pool_d.*,
        (SELECT median(total_sales_value) FROM pool_d) AS stratum_median_sales_value,
        (SELECT count(*) FROM pool_d) AS eligible_stratum_size,
        abs(total_sales_value - (SELECT median(total_sales_value) FROM pool_d)) AS distance_to_median
    FROM pool_d
    QUALIFY row_number() OVER (
        ORDER BY
            abs(total_sales_value - (SELECT median(total_sales_value) FROM pool_d)) ASC,
            household_key ASC
    ) = 1
),
selected AS (
    SELECT * FROM selected_a
    UNION ALL SELECT * FROM selected_b
    UNION ALL SELECT * FROM selected_c
    UNION ALL SELECT * FROM selected_d
)
SELECT
    journey_label,
    stratum,
    household_key,
    eligible_stratum_size,
    round(stratum_median_sales_value, 2) AS stratum_median_sales_value,
    round(total_sales_value, 2) AS selected_sales_value,
    round(distance_to_median, 4) AS distance_to_median,
    active_weeks,
    longest_internal_no_purchase_week_gap,
    total_baskets AS trips,
    total_units,
    distinct_products,
    distinct_commodities,
    campaign_recipient_ever,
    campaign_count,
    coupon_redeemer_ever,
    coupon_redemption_count,
    first_purchase_week,
    last_purchase_week,
    (SELECT median_campaign_count_among_listed FROM rule_parameters) AS median_campaign_count_among_listed
FROM selected
ORDER BY journey_label
