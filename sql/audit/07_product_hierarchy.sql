-- 07_product_hierarchy.sql
-- Purpose: product key integrity, category coverage and sizes, and the
--          product<->transaction coverage both ways.
-- Grain: one row per product table.
-- No category is selected or recommended here.
-- Provenance: A = vendor documented levels, B = observed.

SELECT 'product_rows' AS metric, count(*)::VARCHAR AS value, 'B' AS provenance FROM product
UNION ALL SELECT 'product_distinct_ids', count(DISTINCT PRODUCT_ID)::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_distinct_departments', count(DISTINCT DEPARTMENT)::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_distinct_commodities', count(DISTINCT COMMODITY_DESC)::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_distinct_department_commodity_pairs',
       count(DISTINCT (DEPARTMENT, COMMODITY_DESC))::VARCHAR, 'B' FROM product
UNION ALL SELECT 'commodity_labels_spanning_multiple_departments', count(*)::VARCHAR, 'B'
        FROM (
            SELECT COMMODITY_DESC
            FROM product
            GROUP BY COMMODITY_DESC
            HAVING count(DISTINCT DEPARTMENT) > 1
        )
UNION ALL SELECT 'product_distinct_sub_commodities', count(DISTINCT SUB_COMMODITY_DESC)::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_distinct_manufacturers', count(DISTINCT MANUFACTURER)::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_distinct_brands', count(DISTINCT BRAND)::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_distinct_sizes', count(DISTINCT CURR_SIZE_OF_PRODUCT)::VARCHAR, 'B' FROM product

-- Nulls / sentinels in hierarchy labels. The vendor ships literal
-- 'NO COMMODITY DESCRIPTION' style placeholders, which are NOT nulls.
UNION ALL SELECT 'product_null_department', count(*) FILTER (WHERE DEPARTMENT IS NULL)::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_null_commodity', count(*) FILTER (WHERE COMMODITY_DESC IS NULL)::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_null_sub_commodity', count(*) FILTER (WHERE SUB_COMMODITY_DESC IS NULL)::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_null_manufacturer', count(*) FILTER (WHERE MANUFACTURER IS NULL)::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_null_brand', count(*) FILTER (WHERE BRAND IS NULL)::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_null_size', count(*) FILTER (WHERE CURR_SIZE_OF_PRODUCT IS NULL)::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_blank_size', count(*) FILTER (WHERE trim(CURR_SIZE_OF_PRODUCT) = '')::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_no_commodity_placeholder', count(*) FILTER (WHERE upper(trim(COMMODITY_DESC)) = 'NO COMMODITY DESCRIPTION')::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_no_sub_commodity_placeholder', count(*) FILTER (WHERE upper(trim(SUB_COMMODITY_DESC)) = 'NO SUBCOMMODITY DESCRIPTION')::VARCHAR, 'B' FROM product
UNION ALL SELECT 'product_no_brand_placeholder', count(*) FILTER (WHERE upper(trim(BRAND)) = 'NO BRAND')::VARCHAR, 'B' FROM product

-- Brand vocabulary: the guide says BRAND indicates Private or National label.
UNION ALL SELECT 'brand_values', coalesce(string_agg(DISTINCT BRAND, ' | '), 'none'), 'B' FROM product

-- Coverage against transactions, both directions.
UNION ALL SELECT 'products_in_transaction_not_in_product', count(*)::VARCHAR, 'B'
        FROM (SELECT DISTINCT t.PRODUCT_ID FROM transaction_data t
              LEFT JOIN product p ON p.PRODUCT_ID = t.PRODUCT_ID
              WHERE p.PRODUCT_ID IS NULL)
UNION ALL SELECT 'products_never_purchased', count(*)::VARCHAR, 'B'
        FROM (SELECT p.PRODUCT_ID FROM product p
              LEFT JOIN (SELECT DISTINCT PRODUCT_ID FROM transaction_data) t
                     ON t.PRODUCT_ID = p.PRODUCT_ID
              WHERE t.PRODUCT_ID IS NULL)
UNION ALL SELECT 'products_purchased_at_least_once', count(*)::VARCHAR, 'B'
        FROM (SELECT p.PRODUCT_ID FROM product p
              JOIN (SELECT DISTINCT PRODUCT_ID FROM transaction_data) t
                ON t.PRODUCT_ID = p.PRODUCT_ID)

ORDER BY metric;
