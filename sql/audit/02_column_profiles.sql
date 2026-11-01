-- 02_column_profiles.sql
-- Purpose: per-column null counts, null %, and distinct counts for every raw
--          table, unpivoted into one long table.
-- Grain: one row per (table, column).
--
-- Implementation note: DuckDB cannot pivot a dynamically-built column list, so
-- each table contributes a UNION ALL branch of unpivoted counts. This keeps the
-- query readable and explicit at the cost of repetition, which is deliberate:
-- a reviewer should be able to read exactly what is counted.
--
-- Provenance: B = observed empirically in the files.

-- transaction_data -------------------------------------------------------------
SELECT 'transaction_data' AS table_name, 'household_key' AS column_name,
       count(*) AS row_count,
       count(*) - count(household_key) AS null_count,
       round(100.0 * (count(*) - count(household_key)) / count(*), 6) AS null_pct,
       count(DISTINCT household_key) AS distinct_count FROM transaction_data
UNION ALL SELECT 'transaction_data', 'BASKET_ID', count(*),
       count(*) - count(BASKET_ID),
       round(100.0 * (count(*) - count(BASKET_ID)) / count(*), 6),
       count(DISTINCT BASKET_ID) FROM transaction_data
UNION ALL SELECT 'transaction_data', 'DAY', count(*),
       count(*) - count(DAY), round(100.0 * (count(*) - count(DAY)) / count(*), 6),
       count(DISTINCT DAY) FROM transaction_data
UNION ALL SELECT 'transaction_data', 'PRODUCT_ID', count(*),
       count(*) - count(PRODUCT_ID), round(100.0 * (count(*) - count(PRODUCT_ID)) / count(*), 6),
       count(DISTINCT PRODUCT_ID) FROM transaction_data
UNION ALL SELECT 'transaction_data', 'QUANTITY', count(*),
       count(*) - count(QUANTITY), round(100.0 * (count(*) - count(QUANTITY)) / count(*), 6),
       count(DISTINCT QUANTITY) FROM transaction_data
UNION ALL SELECT 'transaction_data', 'SALES_VALUE', count(*),
       count(*) - count(SALES_VALUE), round(100.0 * (count(*) - count(SALES_VALUE)) / count(*), 6),
       count(DISTINCT SALES_VALUE) FROM transaction_data
UNION ALL SELECT 'transaction_data', 'STORE_ID', count(*),
       count(*) - count(STORE_ID), round(100.0 * (count(*) - count(STORE_ID)) / count(*), 6),
       count(DISTINCT STORE_ID) FROM transaction_data
UNION ALL SELECT 'transaction_data', 'RETAIL_DISC', count(*),
       count(*) - count(RETAIL_DISC), round(100.0 * (count(*) - count(RETAIL_DISC)) / count(*), 6),
       count(DISTINCT RETAIL_DISC) FROM transaction_data
UNION ALL SELECT 'transaction_data', 'TRANS_TIME', count(*),
       count(*) - count(TRANS_TIME), round(100.0 * (count(*) - count(TRANS_TIME)) / count(*), 6),
       count(DISTINCT TRANS_TIME) FROM transaction_data
UNION ALL SELECT 'transaction_data', 'WEEK_NO', count(*),
       count(*) - count(WEEK_NO), round(100.0 * (count(*) - count(WEEK_NO)) / count(*), 6),
       count(DISTINCT WEEK_NO) FROM transaction_data
UNION ALL SELECT 'transaction_data', 'COUPON_DISC', count(*),
       count(*) - count(COUPON_DISC), round(100.0 * (count(*) - count(COUPON_DISC)) / count(*), 6),
       count(DISTINCT COUPON_DISC) FROM transaction_data
UNION ALL SELECT 'transaction_data', 'COUPON_MATCH_DISC', count(*),
       count(*) - count(COUPON_MATCH_DISC),
       round(100.0 * (count(*) - count(COUPON_MATCH_DISC)) / count(*), 6),
       count(DISTINCT COUPON_MATCH_DISC) FROM transaction_data

-- product ----------------------------------------------------------------------
UNION ALL SELECT 'product', 'PRODUCT_ID', count(*),
       count(*) - count(PRODUCT_ID), round(100.0 * (count(*) - count(PRODUCT_ID)) / count(*), 6),
       count(DISTINCT PRODUCT_ID) FROM product
UNION ALL SELECT 'product', 'MANUFACTURER', count(*),
       count(*) - count(MANUFACTURER), round(100.0 * (count(*) - count(MANUFACTURER)) / count(*), 6),
       count(DISTINCT MANUFACTURER) FROM product
UNION ALL SELECT 'product', 'DEPARTMENT', count(*),
       count(*) - count(DEPARTMENT), round(100.0 * (count(*) - count(DEPARTMENT)) / count(*), 6),
       count(DISTINCT DEPARTMENT) FROM product
UNION ALL SELECT 'product', 'BRAND', count(*),
       count(*) - count(BRAND), round(100.0 * (count(*) - count(BRAND)) / count(*), 6),
       count(DISTINCT BRAND) FROM product
UNION ALL SELECT 'product', 'COMMODITY_DESC', count(*),
       count(*) - count(COMMODITY_DESC), round(100.0 * (count(*) - count(COMMODITY_DESC)) / count(*), 6),
       count(DISTINCT COMMODITY_DESC) FROM product
UNION ALL SELECT 'product', 'SUB_COMMODITY_DESC', count(*),
       count(*) - count(SUB_COMMODITY_DESC), round(100.0 * (count(*) - count(SUB_COMMODITY_DESC)) / count(*), 6),
       count(DISTINCT SUB_COMMODITY_DESC) FROM product
UNION ALL SELECT 'product', 'CURR_SIZE_OF_PRODUCT', count(*),
       count(*) - count(CURR_SIZE_OF_PRODUCT), round(100.0 * (count(*) - count(CURR_SIZE_OF_PRODUCT)) / count(*), 6),
       count(DISTINCT CURR_SIZE_OF_PRODUCT) FROM product

-- campaign_table ---------------------------------------------------------------
UNION ALL SELECT 'campaign_table', 'household_key', count(*),
       count(*) - count(household_key), round(100.0 * (count(*) - count(household_key)) / count(*), 6),
       count(DISTINCT household_key) FROM campaign_table
UNION ALL SELECT 'campaign_table', 'CAMPAIGN', count(*),
       count(*) - count(CAMPAIGN), round(100.0 * (count(*) - count(CAMPAIGN)) / count(*), 6),
       count(DISTINCT CAMPAIGN) FROM campaign_table
UNION ALL SELECT 'campaign_table', 'DESCRIPTION', count(*),
       count(*) - count(DESCRIPTION), round(100.0 * (count(*) - count(DESCRIPTION)) / count(*), 6),
       count(DISTINCT DESCRIPTION) FROM campaign_table

-- campaign_desc ----------------------------------------------------------------
UNION ALL SELECT 'campaign_desc', 'CAMPAIGN', count(*),
       count(*) - count(CAMPAIGN), round(100.0 * (count(*) - count(CAMPAIGN)) / count(*), 6),
       count(DISTINCT CAMPAIGN) FROM campaign_desc
UNION ALL SELECT 'campaign_desc', 'DESCRIPTION', count(*),
       count(*) - count(DESCRIPTION), round(100.0 * (count(*) - count(DESCRIPTION)) / count(*), 6),
       count(DISTINCT DESCRIPTION) FROM campaign_desc
UNION ALL SELECT 'campaign_desc', 'START_DAY', count(*),
       count(*) - count(START_DAY), round(100.0 * (count(*) - count(START_DAY)) / count(*), 6),
       count(DISTINCT START_DAY) FROM campaign_desc
UNION ALL SELECT 'campaign_desc', 'END_DAY', count(*),
       count(*) - count(END_DAY), round(100.0 * (count(*) - count(END_DAY)) / count(*), 6),
       count(DISTINCT END_DAY) FROM campaign_desc

-- coupon -----------------------------------------------------------------------
UNION ALL SELECT 'coupon', 'CAMPAIGN', count(*),
       count(*) - count(CAMPAIGN), round(100.0 * (count(*) - count(CAMPAIGN)) / count(*), 6),
       count(DISTINCT CAMPAIGN) FROM coupon
UNION ALL SELECT 'coupon', 'COUPON_UPC', count(*),
       count(*) - count(COUPON_UPC), round(100.0 * (count(*) - count(COUPON_UPC)) / count(*), 6),
       count(DISTINCT COUPON_UPC) FROM coupon
UNION ALL SELECT 'coupon', 'PRODUCT_ID', count(*),
       count(*) - count(PRODUCT_ID), round(100.0 * (count(*) - count(PRODUCT_ID)) / count(*), 6),
       count(DISTINCT PRODUCT_ID) FROM coupon

-- coupon_redempt ---------------------------------------------------------------
UNION ALL SELECT 'coupon_redempt', 'household_key', count(*),
       count(*) - count(household_key), round(100.0 * (count(*) - count(household_key)) / count(*), 6),
       count(DISTINCT household_key) FROM coupon_redempt
UNION ALL SELECT 'coupon_redempt', 'DAY', count(*),
       count(*) - count(DAY), round(100.0 * (count(*) - count(DAY)) / count(*), 6),
       count(DISTINCT DAY) FROM coupon_redempt
UNION ALL SELECT 'coupon_redempt', 'COUPON_UPC', count(*),
       count(*) - count(COUPON_UPC), round(100.0 * (count(*) - count(COUPON_UPC)) / count(*), 6),
       count(DISTINCT COUPON_UPC) FROM coupon_redempt
UNION ALL SELECT 'coupon_redempt', 'CAMPAIGN', count(*),
       count(*) - count(CAMPAIGN), round(100.0 * (count(*) - count(CAMPAIGN)) / count(*), 6),
       count(DISTINCT CAMPAIGN) FROM coupon_redempt

-- hh_demographic ---------------------------------------------------------------
UNION ALL SELECT 'hh_demographic', 'household_key', count(*),
       count(*) - count(household_key), round(100.0 * (count(*) - count(household_key)) / count(*), 6),
       count(DISTINCT household_key) FROM hh_demographic
UNION ALL SELECT 'hh_demographic', 'classification_1', count(*),
       count(*) - count(classification_1), round(100.0 * (count(*) - count(classification_1)) / count(*), 6),
       count(DISTINCT classification_1) FROM hh_demographic
UNION ALL SELECT 'hh_demographic', 'classification_2', count(*),
       count(*) - count(classification_2), round(100.0 * (count(*) - count(classification_2)) / count(*), 6),
       count(DISTINCT classification_2) FROM hh_demographic
UNION ALL SELECT 'hh_demographic', 'classification_3', count(*),
       count(*) - count(classification_3), round(100.0 * (count(*) - count(classification_3)) / count(*), 6),
       count(DISTINCT classification_3) FROM hh_demographic
UNION ALL SELECT 'hh_demographic', 'classification_4', count(*),
       count(*) - count(classification_4), round(100.0 * (count(*) - count(classification_4)) / count(*), 6),
       count(DISTINCT classification_4) FROM hh_demographic
UNION ALL SELECT 'hh_demographic', 'classification_5', count(*),
       count(*) - count(classification_5), round(100.0 * (count(*) - count(classification_5)) / count(*), 6),
       count(DISTINCT classification_5) FROM hh_demographic
UNION ALL SELECT 'hh_demographic', 'HOMEOWNER_DESC', count(*),
       count(*) - count(HOMEOWNER_DESC), round(100.0 * (count(*) - count(HOMEOWNER_DESC)) / count(*), 6),
       count(DISTINCT HOMEOWNER_DESC) FROM hh_demographic
UNION ALL SELECT 'hh_demographic', 'KID_CATEGORY_DESC', count(*),
       count(*) - count(KID_CATEGORY_DESC), round(100.0 * (count(*) - count(KID_CATEGORY_DESC)) / count(*), 6),
       count(DISTINCT KID_CATEGORY_DESC) FROM hh_demographic

-- causal_data -----------------------------------------------------------------
UNION ALL SELECT 'causal_data', 'PRODUCT_ID', count(*),
       count(*) - count(PRODUCT_ID), round(100.0 * (count(*) - count(PRODUCT_ID)) / count(*), 6),
       count(DISTINCT PRODUCT_ID) FROM causal_data
UNION ALL SELECT 'causal_data', 'STORE_ID', count(*),
       count(*) - count(STORE_ID), round(100.0 * (count(*) - count(STORE_ID)) / count(*), 6),
       count(DISTINCT STORE_ID) FROM causal_data
UNION ALL SELECT 'causal_data', 'WEEK_NO', count(*),
       count(*) - count(WEEK_NO), round(100.0 * (count(*) - count(WEEK_NO)) / count(*), 6),
       count(DISTINCT WEEK_NO) FROM causal_data
UNION ALL SELECT 'causal_data', 'display', count(*),
       count(*) - count(display), round(100.0 * (count(*) - count(display)) / count(*), 6),
       count(DISTINCT display) FROM causal_data
UNION ALL SELECT 'causal_data', 'mailer', count(*),
       count(*) - count(mailer), round(100.0 * (count(*) - count(mailer)) / count(*), 6),
       count(DISTINCT mailer) FROM causal_data

ORDER BY table_name, column_name;
