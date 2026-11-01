-- 01_table_inventory.sql
-- Purpose: row counts, column counts, and on-disk size for every raw table.
-- Grain: one row per raw table.
-- Provenance: B = observed empirically in the files.

SELECT
    'campaign_desc'      AS table_name,
    count(*)             AS row_count,
    count(DISTINCT CAMPAIGN) AS distinct_campaign,
    (SELECT count(*) FROM (DESCRIBE SELECT * FROM campaign_desc)) AS column_count
FROM campaign_desc
UNION ALL
SELECT 'campaign_table', count(*), count(DISTINCT CAMPAIGN),
       (SELECT count(*) FROM (DESCRIBE SELECT * FROM campaign_table))
FROM campaign_table
UNION ALL
SELECT 'coupon', count(*), count(DISTINCT CAMPAIGN),
       (SELECT count(*) FROM (DESCRIBE SELECT * FROM coupon))
FROM coupon
UNION ALL
SELECT 'coupon_redempt', count(*), count(DISTINCT CAMPAIGN),
       (SELECT count(*) FROM (DESCRIBE SELECT * FROM coupon_redempt))
FROM coupon_redempt
UNION ALL
SELECT 'hh_demographic', count(*), count(DISTINCT household_key),
       (SELECT count(*) FROM (DESCRIBE SELECT * FROM hh_demographic))
FROM hh_demographic
UNION ALL
SELECT 'product', count(*), count(DISTINCT PRODUCT_ID),
       (SELECT count(*) FROM (DESCRIBE SELECT * FROM product))
FROM product
UNION ALL
SELECT 'transaction_data', count(*), count(DISTINCT household_key),
       (SELECT count(*) FROM (DESCRIBE SELECT * FROM transaction_data))
FROM transaction_data
UNION ALL
SELECT 'causal_data', count(*), count(DISTINCT PRODUCT_ID),
       (SELECT count(*) FROM (DESCRIBE SELECT * FROM causal_data))
FROM causal_data
ORDER BY table_name;
