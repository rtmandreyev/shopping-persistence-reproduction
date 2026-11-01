-- 04_referential_integrity.sql
-- Purpose: foreign-key match rates plus directional physical-row multiplicity,
--          so row-multiplying joins are visible before Phase 2.
-- Grain: one row per relationship.
--
-- source_max_rows_per_key and target_max_rows_per_key report physical rows on
-- each side of the declared key. raw_join_max_rows_per_key reports their product
-- for the worst matched key: the number of rows a naive physical-row join emits
-- for that key. These directions are explicit; an earlier field mixed source
-- multiplicity, target multiplicity, and parent uniqueness across relationships.
--
-- Provenance: A = documented expectation, B = observed counts.

WITH
transaction_by_product AS (
    SELECT PRODUCT_ID, count(*) AS rows_per_key
    FROM transaction_data GROUP BY 1
),
product_by_product AS (
    SELECT PRODUCT_ID, count(*) AS rows_per_key
    FROM product GROUP BY 1
),
campaign_table_by_campaign AS (
    SELECT CAMPAIGN, count(*) AS rows_per_key
    FROM campaign_table GROUP BY 1
),
campaign_desc_by_campaign AS (
    SELECT CAMPAIGN, count(*) AS rows_per_key
    FROM campaign_desc GROUP BY 1
),
coupon_by_campaign AS (
    SELECT CAMPAIGN, count(*) AS rows_per_key
    FROM coupon GROUP BY 1
),
coupon_by_product AS (
    SELECT PRODUCT_ID, count(*) AS rows_per_key
    FROM coupon GROUP BY 1
),
coupon_by_pair AS (
    SELECT COUPON_UPC, CAMPAIGN, count(*) AS rows_per_key
    FROM coupon GROUP BY 1, 2
),
redemption_by_pair AS (
    SELECT COUPON_UPC, CAMPAIGN, count(*) AS rows_per_key
    FROM coupon_redempt GROUP BY 1, 2
),
redemption_by_campaign AS (
    SELECT CAMPAIGN, count(*) AS rows_per_key
    FROM coupon_redempt GROUP BY 1
),
hh_demographic_by_household AS (
    SELECT household_key, count(*) AS rows_per_key
    FROM hh_demographic GROUP BY 1
),
transaction_by_household AS (
    SELECT household_key, count(*) AS rows_per_key
    FROM transaction_data GROUP BY 1
),
campaign_table_by_household AS (
    SELECT household_key, count(*) AS rows_per_key
    FROM campaign_table GROUP BY 1
),
causal_by_product AS (
    SELECT PRODUCT_ID, count(*) AS rows_per_key
    FROM causal_data GROUP BY 1
),
causal_by_store AS (
    SELECT STORE_ID, count(*) AS rows_per_key
    FROM causal_data GROUP BY 1
),
transaction_by_store AS (
    SELECT STORE_ID, count(*) AS rows_per_key
    FROM transaction_data GROUP BY 1
),
redemption_by_household_campaign AS (
    SELECT household_key, CAMPAIGN, count(*) AS rows_per_key
    FROM coupon_redempt GROUP BY 1, 2
),
campaign_table_by_household_campaign AS (
    SELECT household_key, CAMPAIGN, count(*) AS rows_per_key
    FROM campaign_table GROUP BY 1, 2
),
-- Each relationship aggregate has one row and the same directional meanings.
r1 AS (
    SELECT count(*) AS src_ids,
           count(t.PRODUCT_ID) AS matched,
           coalesce(max(s.rows_per_key), 0) AS source_max,
           coalesce(max(t.rows_per_key), 0) AS target_max,
           coalesce(max(s.rows_per_key * t.rows_per_key), 0) AS raw_join_max
    FROM transaction_by_product s
    LEFT JOIN product_by_product t USING (PRODUCT_ID)
),
r2 AS (
    SELECT count(*) AS src_ids,
           count(t.CAMPAIGN) AS matched,
           coalesce(max(s.rows_per_key), 0) AS source_max,
           coalesce(max(t.rows_per_key), 0) AS target_max,
           coalesce(max(s.rows_per_key * t.rows_per_key), 0) AS raw_join_max
    FROM campaign_table_by_campaign s
    LEFT JOIN campaign_desc_by_campaign t USING (CAMPAIGN)
),
r3 AS (
    SELECT count(*) AS src_ids,
           count(t.CAMPAIGN) AS matched,
           coalesce(max(s.rows_per_key), 0) AS source_max,
           coalesce(max(t.rows_per_key), 0) AS target_max,
           coalesce(max(s.rows_per_key * t.rows_per_key), 0) AS raw_join_max
    FROM coupon_by_campaign s
    LEFT JOIN campaign_desc_by_campaign t USING (CAMPAIGN)
),
r4 AS (
    SELECT count(*) AS src_ids,
           count(t.PRODUCT_ID) AS matched,
           coalesce(max(s.rows_per_key), 0) AS source_max,
           coalesce(max(t.rows_per_key), 0) AS target_max,
           coalesce(max(s.rows_per_key * t.rows_per_key), 0) AS raw_join_max
    FROM coupon_by_product s
    LEFT JOIN product_by_product t USING (PRODUCT_ID)
),
r5 AS (
    SELECT count(*) AS src_ids,
           count(t.COUPON_UPC) AS matched,
           coalesce(max(s.rows_per_key), 0) AS source_max,
           coalesce(max(t.rows_per_key), 0) AS target_max,
           coalesce(max(s.rows_per_key * t.rows_per_key), 0) AS raw_join_max
    FROM redemption_by_pair s
    LEFT JOIN coupon_by_pair t USING (COUPON_UPC, CAMPAIGN)
),
r6 AS (
    SELECT count(*) AS src_ids,
           count(t.CAMPAIGN) AS matched,
           coalesce(max(s.rows_per_key), 0) AS source_max,
           coalesce(max(t.rows_per_key), 0) AS target_max,
           coalesce(max(s.rows_per_key * t.rows_per_key), 0) AS raw_join_max
    FROM redemption_by_campaign s
    LEFT JOIN campaign_desc_by_campaign t USING (CAMPAIGN)
),
r7 AS (
    SELECT count(*) AS src_ids,
           count(t.household_key) AS matched,
           coalesce(max(s.rows_per_key), 0) AS source_max,
           coalesce(max(t.rows_per_key), 0) AS target_max,
           coalesce(max(s.rows_per_key * t.rows_per_key), 0) AS raw_join_max
    FROM hh_demographic_by_household s
    LEFT JOIN transaction_by_household t USING (household_key)
),
r8 AS (
    SELECT count(*) AS src_ids,
           count(t.household_key) AS matched,
           coalesce(max(s.rows_per_key), 0) AS source_max,
           coalesce(max(t.rows_per_key), 0) AS target_max,
           coalesce(max(s.rows_per_key * t.rows_per_key), 0) AS raw_join_max
    FROM campaign_table_by_household s
    LEFT JOIN transaction_by_household t USING (household_key)
),
r9 AS (
    SELECT count(*) AS src_ids,
           count(t.PRODUCT_ID) AS matched,
           coalesce(max(s.rows_per_key), 0) AS source_max,
           coalesce(max(t.rows_per_key), 0) AS target_max,
           coalesce(max(s.rows_per_key * t.rows_per_key), 0) AS raw_join_max
    FROM causal_by_product s
    LEFT JOIN product_by_product t USING (PRODUCT_ID)
),
r10 AS (
    SELECT count(*) AS src_ids,
           count(t.STORE_ID) AS matched,
           coalesce(max(s.rows_per_key), 0) AS source_max,
           coalesce(max(t.rows_per_key), 0) AS target_max,
           coalesce(max(s.rows_per_key * t.rows_per_key), 0) AS raw_join_max
    FROM causal_by_store s
    LEFT JOIN transaction_by_store t USING (STORE_ID)
),
r11 AS (
    SELECT count(*) AS src_ids,
           count(t.household_key) AS matched,
           coalesce(max(s.rows_per_key), 0) AS source_max,
           coalesce(max(t.rows_per_key), 0) AS target_max,
           coalesce(max(s.rows_per_key * t.rows_per_key), 0) AS raw_join_max
    FROM redemption_by_household_campaign s
    LEFT JOIN campaign_table_by_household_campaign t USING (household_key, CAMPAIGN)
)

SELECT 'transaction_data.PRODUCT_ID -> product.PRODUCT_ID' AS relationship,
       'many-to-one (many receipt lines, one product row)' AS expected_cardinality,
       r1.src_ids AS source_distinct_ids,
       (SELECT count(*) FROM product_by_product) AS parent_distinct_ids,
       r1.src_ids - r1.matched AS unmatched_ids,
       round(100.0 * r1.matched / r1.src_ids, 6) AS match_rate_pct,
       r1.source_max AS source_max_rows_per_key,
       r1.target_max AS target_max_rows_per_key,
       r1.raw_join_max AS raw_join_max_rows_per_key
FROM r1
UNION ALL
SELECT 'campaign_table.CAMPAIGN -> campaign_desc.CAMPAIGN',
       'many-to-one (many households, one campaign row)',
       r2.src_ids, (SELECT count(*) FROM campaign_desc_by_campaign),
       r2.src_ids - r2.matched, round(100.0 * r2.matched / r2.src_ids, 6),
       r2.source_max, r2.target_max, r2.raw_join_max
FROM r2
UNION ALL
SELECT 'coupon.CAMPAIGN -> campaign_desc.CAMPAIGN',
       'many-to-one (many coupon rows, one campaign row)',
       r3.src_ids, (SELECT count(*) FROM campaign_desc_by_campaign),
       r3.src_ids - r3.matched, round(100.0 * r3.matched / r3.src_ids, 6),
       r3.source_max, r3.target_max, r3.raw_join_max
FROM r3
UNION ALL
SELECT 'coupon.PRODUCT_ID -> product.PRODUCT_ID',
       'many-to-one (many coupon rows, one product row)',
       r4.src_ids, (SELECT count(*) FROM product_by_product),
       r4.src_ids - r4.matched, round(100.0 * r4.matched / r4.src_ids, 6),
       r4.source_max, r4.target_max, r4.raw_join_max
FROM r4
UNION ALL
SELECT 'coupon_redempt.(COUPON_UPC,CAMPAIGN) -> coupon.(COUPON_UPC,CAMPAIGN)',
       'source-key existence with physical coupon-product fan-out reported',
       r5.src_ids, (SELECT count(*) FROM coupon_by_pair),
       r5.src_ids - r5.matched, round(100.0 * r5.matched / r5.src_ids, 6),
       r5.source_max, r5.target_max, r5.raw_join_max
FROM r5
UNION ALL
SELECT 'coupon_redempt.CAMPAIGN -> campaign_desc.CAMPAIGN',
       'many-to-one',
       r6.src_ids, (SELECT count(*) FROM campaign_desc_by_campaign),
       r6.src_ids - r6.matched, round(100.0 * r6.matched / r6.src_ids, 6),
       r6.source_max, r6.target_max, r6.raw_join_max
FROM r6
UNION ALL
SELECT 'hh_demographic.household_key -> transaction_data.household_key',
       'one demographic row to many transaction lines per household',
       r7.src_ids, (SELECT count(*) FROM transaction_by_household),
       r7.src_ids - r7.matched, round(100.0 * r7.matched / r7.src_ids, 6),
       r7.source_max, r7.target_max, r7.raw_join_max
FROM r7
UNION ALL
SELECT 'campaign_table.household_key -> transaction_data.household_key',
       'many campaign memberships to many transaction lines per household',
       r8.src_ids, (SELECT count(*) FROM transaction_by_household),
       r8.src_ids - r8.matched, round(100.0 * r8.matched / r8.src_ids, 6),
       r8.source_max, r8.target_max, r8.raw_join_max
FROM r8
UNION ALL
SELECT 'causal_data.PRODUCT_ID -> product.PRODUCT_ID',
       'many placement rows to one product row',
       r9.src_ids, (SELECT count(*) FROM product_by_product),
       r9.src_ids - r9.matched, round(100.0 * r9.matched / r9.src_ids, 6),
       r9.source_max, r9.target_max, r9.raw_join_max
FROM r9
UNION ALL
SELECT 'causal_data.STORE_ID -> transaction_data.STORE_ID',
       'many placement rows to many transaction lines per store',
       r10.src_ids, (SELECT count(*) FROM transaction_by_store),
       r10.src_ids - r10.matched, round(100.0 * r10.matched / r10.src_ids, 6),
       r10.source_max, r10.target_max, r10.raw_join_max
FROM r10
UNION ALL
SELECT 'coupon_redempt.(household_key,CAMPAIGN) -> campaign_table.(household_key,CAMPAIGN)',
       'redemption rows to listed-recipient membership rows',
       r11.src_ids, (SELECT count(*) FROM campaign_table_by_household_campaign),
       r11.src_ids - r11.matched, round(100.0 * r11.matched / r11.src_ids, 6),
       r11.source_max, r11.target_max, r11.raw_join_max
FROM r11
ORDER BY relationship;
