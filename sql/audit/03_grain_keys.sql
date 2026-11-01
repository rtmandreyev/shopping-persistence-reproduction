-- 03_grain_keys.sql
-- Purpose: candidate-key uniqueness and duplicate characterisation for every table.
-- Grain: one row per (table, candidate_key).
--
-- The project plan names an expected grain per table. This query does NOT
-- assume those grains are correct; it measures them. A key with
-- duplicate_groups > 0 is reported, not deduplicated.
--
-- Provenance: A = documented by vendor/project plan (expected_grain_source),
--             B = observed empirically in the files (all counts).

-- product: vendor states PRODUCT_ID uniquely identifies a product.
SELECT 'product' AS table_name, 'PRODUCT_ID' AS candidate_key,
       'documented: uniquely identifies each product' AS expected_grain_source,
       count(*) AS row_count,
       count(DISTINCT PRODUCT_ID) AS distinct_key,
       count(*) - count(DISTINCT PRODUCT_ID) AS duplicate_rows,
       (SELECT count(*) FROM (SELECT PRODUCT_ID FROM product
                              GROUP BY PRODUCT_ID HAVING count(*) > 1)) AS duplicate_groups
FROM product

UNION ALL
-- campaign_desc: vendor states CAMPAIGN uniquely identifies a campaign, 1-30.
SELECT 'campaign_desc', 'CAMPAIGN',
       'documented: uniquely identifies each campaign, ranges 1-30',
       count(*), count(DISTINCT CAMPAIGN),
       count(*) - count(DISTINCT CAMPAIGN),
       (SELECT count(*) FROM (SELECT CAMPAIGN FROM campaign_desc
                              GROUP BY CAMPAIGN HAVING count(*) > 1))
FROM campaign_desc

UNION ALL
-- campaign_table: plan says household x campaign.
SELECT 'campaign_table', 'household_key + CAMPAIGN',
       'documented: campaigns received by each household',
       count(*), count(DISTINCT (household_key, CAMPAIGN)),
       count(*) - count(DISTINCT (household_key, CAMPAIGN)),
       (SELECT count(*) FROM (SELECT household_key, CAMPAIGN FROM campaign_table
                              GROUP BY 1, 2 HAVING count(*) > 1))
FROM campaign_table

UNION ALL
-- hh_demographic: vendor states HOUSEHOLD_KEY uniquely identifies a household.
SELECT 'hh_demographic', 'household_key',
       'documented: uniquely identifies each household (portion of households)',
       count(*), count(DISTINCT household_key),
       count(*) - count(DISTINCT household_key),
       (SELECT count(*) FROM (SELECT household_key FROM hh_demographic
                              GROUP BY household_key HAVING count(*) > 1))
FROM hh_demographic

UNION ALL
-- coupon: vendor: COUPON_UPC uniquely identifies a coupon, unique to household
-- and campaign; a coupon is redeemable for MULTIPLE products -> product_id is
-- deliberately NOT part of the key.
SELECT 'coupon', 'CAMPAIGN + COUPON_UPC',
       'documented: coupon_upc unique to campaign; one coupon spans many products',
       count(*), count(DISTINCT (CAMPAIGN, COUPON_UPC)),
       count(*) - count(DISTINCT (CAMPAIGN, COUPON_UPC)),
       (SELECT count(*) FROM (SELECT CAMPAIGN, COUPON_UPC FROM coupon
                              GROUP BY 1, 2 HAVING count(*) > 1))
FROM coupon

UNION ALL
-- coupon_redempt: plan says household x day x coupon_upc x campaign.
SELECT 'coupon_redempt', 'household_key + DAY + COUPON_UPC + CAMPAIGN',
       'documented: identifies the coupons each household redeemed',
       count(*),
       count(DISTINCT (household_key, DAY, COUPON_UPC, CAMPAIGN)),
       count(*) - count(DISTINCT (household_key, DAY, COUPON_UPC, CAMPAIGN)),
       (SELECT count(*) FROM (SELECT household_key, DAY, COUPON_UPC, CAMPAIGN
                              FROM coupon_redempt GROUP BY 1, 2, 3, 4 HAVING count(*) > 1))
FROM coupon_redempt

UNION ALL
-- transaction_data: the important one. A receipt line is NOT expected to have a
-- unique key. Test progressively wider candidate keys to find where uniqueness
-- actually begins, and count full-row duplicates separately.
SELECT 'transaction_data', 'household_key + BASKET_ID + PRODUCT_ID',
       'documented: each line is as found on a store receipt',
       count(*),
       count(DISTINCT (household_key, BASKET_ID, PRODUCT_ID)),
       count(*) - count(DISTINCT (household_key, BASKET_ID, PRODUCT_ID)),
       (SELECT count(*) FROM (SELECT household_key, BASKET_ID, PRODUCT_ID
                              FROM transaction_data GROUP BY 1, 2, 3 HAVING count(*) > 1))
FROM transaction_data

UNION ALL
SELECT 'transaction_data', 'household_key + BASKET_ID + PRODUCT_ID + DAY + WEEK_NO',
       'wider candidate: adds day/week',
       count(*),
       count(DISTINCT (household_key, BASKET_ID, PRODUCT_ID, DAY, WEEK_NO)),
       count(*) - count(DISTINCT (household_key, BASKET_ID, PRODUCT_ID, DAY, WEEK_NO)),
       (SELECT count(*) FROM (SELECT household_key, BASKET_ID, PRODUCT_ID, DAY, WEEK_NO
                              FROM transaction_data GROUP BY 1, 2, 3, 4, 5 HAVING count(*) > 1))
FROM transaction_data

UNION ALL
-- Full-row duplicate count (all 12 columns) - tests whether repeats are exact
-- duplicates or genuine separate lines that happen to share a key.
SELECT 'transaction_data', 'ALL COLUMNS (full row)',
       'tests whether repeated keys are exact duplicate lines',
       count(*), count(DISTINCT (household_key, BASKET_ID, DAY, PRODUCT_ID, QUANTITY,
                                 SALES_VALUE, STORE_ID, RETAIL_DISC, TRANS_TIME,
                                 WEEK_NO, COUPON_DISC, COUPON_MATCH_DISC)),
       count(*) - count(DISTINCT (household_key, BASKET_ID, DAY, PRODUCT_ID, QUANTITY,
                                  SALES_VALUE, STORE_ID, RETAIL_DISC, TRANS_TIME,
                                  WEEK_NO, COUPON_DISC, COUPON_MATCH_DISC)),
       (SELECT count(*) FROM (SELECT household_key, BASKET_ID, DAY, PRODUCT_ID, QUANTITY,
                                     SALES_VALUE, STORE_ID, RETAIL_DISC, TRANS_TIME,
                                     WEEK_NO, COUPON_DISC, COUPON_MATCH_DISC
                              FROM transaction_data GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12
                              HAVING count(*) > 1))
FROM transaction_data

UNION ALL
-- causal_data: plan says product x store x week.
SELECT 'causal_data', 'PRODUCT_ID + STORE_ID + WEEK_NO',
       'documented: whether a given product was featured in mailer/display',
       count(*), count(DISTINCT (PRODUCT_ID, STORE_ID, WEEK_NO)),
       count(*) - count(DISTINCT (PRODUCT_ID, STORE_ID, WEEK_NO)),
       (SELECT count(*) FROM (SELECT PRODUCT_ID, STORE_ID, WEEK_NO FROM causal_data
                              GROUP BY 1, 2, 3 HAVING count(*) > 1))
FROM causal_data

ORDER BY table_name, candidate_key;
