-- 10_coupon_audit.sql
-- Purpose: coupon pool structure, redemption counts/rates where definable, and
--          the vendor-documented Type A assignment limitation.
-- Grain: one row per metric, plus a per-campaign-type breakdown.
--
-- VENDOR LIMITATION (A), quoted: for campaign TypeA the coupon table provides
-- the POOL of possible coupons; each customer received 16 coupons out of that
-- pool, "The 16 coupons were selected based on the customer's prior purchase
-- behaviour. Identifying the specific 16 coupons that each customer received is
-- outside the scope of this database." For TypeB and TypeC, all participating
-- customers receive all coupons for the campaign.
--
-- CONSEQUENCE: redemption rates are NOT campaign coupon-uptake rates for Type A,
-- and coupon assignment is endogenous by construction (selection on prior
-- purchase behaviour). This is reported, not resolved.
-- Provenance: A = vendor statement, B = observed, C = our note.

SELECT 'coupon_rows' AS metric, count(*)::VARCHAR AS value, 'B' AS provenance FROM coupon
UNION ALL SELECT 'coupon_distinct_upc', count(DISTINCT COUPON_UPC)::VARCHAR, 'B' FROM coupon
UNION ALL SELECT 'coupon_distinct_campaign', count(DISTINCT CAMPAIGN)::VARCHAR, 'B' FROM coupon
UNION ALL SELECT 'coupon_distinct_products', count(DISTINCT PRODUCT_ID)::VARCHAR, 'B' FROM coupon
UNION ALL SELECT 'coupon_distinct_campaign_upc_pairs',
       count(*) FILTER (WHERE true)::VARCHAR, 'B' FROM (SELECT DISTINCT CAMPAIGN, COUPON_UPC FROM coupon)
UNION ALL SELECT 'coupon_products_not_in_product', count(*)::VARCHAR, 'B'
        FROM (SELECT DISTINCT c.PRODUCT_ID FROM coupon c
              LEFT JOIN product p ON p.PRODUCT_ID = c.PRODUCT_ID WHERE p.PRODUCT_ID IS NULL)
UNION ALL SELECT 'coupon_campaigns_not_in_campaign_desc', count(*)::VARCHAR, 'B'
        FROM (SELECT DISTINCT CAMPAIGN FROM coupon
              WHERE CAMPAIGN NOT IN (SELECT CAMPAIGN FROM campaign_desc))

-- How many products each coupon is redeemable for (vendor: coupons can cover
-- many products, e.g. 38 for one frozen-vegetable coupon).
UNION ALL SELECT 'products_per_coupon_min', min(cnt)::VARCHAR, 'B'
        FROM (SELECT count(DISTINCT PRODUCT_ID) AS cnt FROM coupon GROUP BY CAMPAIGN, COUPON_UPC)
UNION ALL SELECT 'products_per_coupon_median', median(cnt)::VARCHAR, 'B'
        FROM (SELECT count(DISTINCT PRODUCT_ID) AS cnt FROM coupon GROUP BY CAMPAIGN, COUPON_UPC)
UNION ALL SELECT 'products_per_coupon_max', max(cnt)::VARCHAR, 'B'
        FROM (SELECT count(DISTINCT PRODUCT_ID) AS cnt FROM coupon GROUP BY CAMPAIGN, COUPON_UPC)
UNION ALL SELECT 'products_per_coupon_mean', round(avg(cnt), 4)::VARCHAR, 'B'
        FROM (SELECT count(DISTINCT PRODUCT_ID) AS cnt FROM coupon GROUP BY CAMPAIGN, COUPON_UPC)

-- Redemption table structure.
UNION ALL SELECT 'redempt_rows', count(*)::VARCHAR, 'B' FROM coupon_redempt
UNION ALL SELECT 'redempt_distinct_households', count(DISTINCT household_key)::VARCHAR, 'B' FROM coupon_redempt
UNION ALL SELECT 'redempt_distinct_upc', count(DISTINCT COUPON_UPC)::VARCHAR, 'B' FROM coupon_redempt
UNION ALL SELECT 'redempt_distinct_campaign', count(DISTINCT CAMPAIGN)::VARCHAR, 'B' FROM coupon_redempt
UNION ALL SELECT 'redempt_distinct_campaign_upc_pairs',
       count(*)::VARCHAR, 'B source-key grain'
       FROM (SELECT DISTINCT CAMPAIGN, COUPON_UPC FROM coupon_redempt)
UNION ALL SELECT 'redempt_pairs_with_multiple_coupon_products',
       count(*)::VARCHAR, 'B fan-out diagnostic'
       FROM (
           SELECT r.CAMPAIGN, r.COUPON_UPC
           FROM (SELECT DISTINCT CAMPAIGN, COUPON_UPC FROM coupon_redempt) r
           JOIN coupon c USING (CAMPAIGN, COUPON_UPC)
           GROUP BY 1, 2
           HAVING count(DISTINCT c.PRODUCT_ID) > 1
       )
UNION ALL SELECT 'redempt_pair_coupon_raw_join_rows',
       count(*)::VARCHAR, 'B fan-out diagnostic; not a source-key count'
       FROM (SELECT DISTINCT CAMPAIGN, COUPON_UPC FROM coupon_redempt) r
       JOIN coupon c USING (CAMPAIGN, COUPON_UPC)
UNION ALL SELECT 'redempt_pair_coupon_join_excess_rows',
       (count(*) - count(DISTINCT (r.CAMPAIGN, r.COUPON_UPC)))::VARCHAR,
       'B fan-out diagnostic; raw join rows minus source keys'
       FROM (SELECT DISTINCT CAMPAIGN, COUPON_UPC FROM coupon_redempt) r
       JOIN coupon c USING (CAMPAIGN, COUPON_UPC)
UNION ALL SELECT 'redempt_day_min', min(DAY)::VARCHAR, 'B' FROM coupon_redempt
UNION ALL SELECT 'redempt_day_max', max(DAY)::VARCHAR, 'B' FROM coupon_redempt

-- Unmatched redemptions: redemption pairs with no matching coupon pool entry.
UNION ALL SELECT 'redempt_pairs_not_in_coupon_pool', count(*)::VARCHAR, 'B'
        FROM (SELECT DISTINCT COUPON_UPC, CAMPAIGN FROM coupon_redempt) r
        WHERE NOT EXISTS (SELECT 1 FROM coupon c
                          WHERE c.COUPON_UPC = r.COUPON_UPC AND c.CAMPAIGN = r.CAMPAIGN)
UNION ALL SELECT 'redempt_rows_not_in_coupon_pool', count(*)::VARCHAR, 'B'
        FROM coupon_redempt r
        WHERE NOT EXISTS (SELECT 1 FROM coupon c
                          WHERE c.COUPON_UPC = r.COUPON_UPC AND c.CAMPAIGN = r.CAMPAIGN)
UNION ALL SELECT 'redempt_households_not_receiving_that_campaign', count(DISTINCT r.household_key)::VARCHAR, 'B'
        FROM coupon_redempt r
        WHERE NOT EXISTS (SELECT 1 FROM campaign_table t
                          WHERE t.household_key = r.household_key AND t.CAMPAIGN = r.CAMPAIGN)

-- Redemption timing relative to the campaign window. A redemption outside
-- [START_DAY, END_DAY] contradicts the vendor's "valid within the dates"
-- statement and would need explaining.
UNION ALL SELECT 'redemptions_outside_campaign_window', count(*)::VARCHAR, 'B'
        FROM coupon_redempt r JOIN campaign_desc d ON d.CAMPAIGN = r.CAMPAIGN
        WHERE r.DAY < d.START_DAY OR r.DAY > d.END_DAY
UNION ALL SELECT 'redemptions_before_campaign_start', count(*)::VARCHAR, 'B'
        FROM coupon_redempt r JOIN campaign_desc d ON d.CAMPAIGN = r.CAMPAIGN
        WHERE r.DAY < d.START_DAY
UNION ALL SELECT 'redemptions_after_campaign_end', count(*)::VARCHAR, 'B'
        FROM coupon_redempt r JOIN campaign_desc d ON d.CAMPAIGN = r.CAMPAIGN
        WHERE r.DAY > d.END_DAY

-- Repeated redemption: same household redeeming the same coupon on multiple days.
UNION ALL SELECT 'household_coupon_pairs_redeemed_more_than_once', count(*)::VARCHAR, 'B'
        FROM (SELECT household_key, COUPON_UPC, CAMPAIGN FROM coupon_redempt
              GROUP BY 1, 2, 3 HAVING count(DISTINCT DAY) > 1)
UNION ALL SELECT 'max_days_same_household_same_coupon', max(cnt)::VARCHAR, 'B'
        FROM (SELECT count(DISTINCT DAY) AS cnt FROM coupon_redempt GROUP BY household_key, COUPON_UPC, CAMPAIGN)
UNION ALL SELECT 'max_redemptions_by_one_household', max(cnt)::VARCHAR, 'B'
        FROM (SELECT count(*) AS cnt FROM coupon_redempt GROUP BY household_key)

-- Per campaign type, computed in SEPARATE aggregates.
-- NOTE: coupon and redemption counts remain separate. A historical campaign-only
-- join produced 56,757,409 rows by combining every campaign coupon-product row
-- with every campaign redemption. Even the correct (COUPON_UPC, CAMPAIGN)
-- composite-key raw join produces 2,198,362 rows because coupon pairs expand to
-- many products and can have many redemption rows. Neither fan-out is a coupon
-- count, so the summaries below use independent per-table aggregates.
-- Redemption RATE is not reported as a take-up rate: the pool size is not the
-- number of coupons a given household received.
-- Provenance: C, following the vendor's A-documented limitation.
UNION ALL
SELECT 'campaign_type_' || t.TYPE || '_summary',
       'campaigns=' || t.campaigns
       || ' pool_coupons=' || t.coupons
       || ' coupon_rows=' || t.coupon_rows
       || ' pool_products=' || t.pool_products
       || ' redeeming_households=' || t.redeem_hh
       || ' redemption_rows=' || t.redempt_rows,
       'B counts; C note: pool size != household coupon count for TypeA'
FROM (
    SELECT cd.DESCRIPTION AS TYPE,
           count(DISTINCT cd.CAMPAIGN) AS campaigns,
           (SELECT count(DISTINCT (c.CAMPAIGN, c.COUPON_UPC)) FROM coupon c
             WHERE c.CAMPAIGN IN (SELECT CAMPAIGN FROM campaign_desc WHERE DESCRIPTION = cd.DESCRIPTION))
               AS coupons,
           (SELECT count(*) FROM coupon c
             WHERE c.CAMPAIGN IN (SELECT CAMPAIGN FROM campaign_desc WHERE DESCRIPTION = cd.DESCRIPTION))
               AS coupon_rows,
           (SELECT count(DISTINCT c.PRODUCT_ID) FROM coupon c
             WHERE c.CAMPAIGN IN (SELECT CAMPAIGN FROM campaign_desc WHERE DESCRIPTION = cd.DESCRIPTION))
               AS pool_products,
           (SELECT count(DISTINCT r.household_key) FROM coupon_redempt r
             WHERE r.CAMPAIGN IN (SELECT CAMPAIGN FROM campaign_desc WHERE DESCRIPTION = cd.DESCRIPTION))
               AS redeem_hh,
           (SELECT count(*) FROM coupon_redempt r
             WHERE r.CAMPAIGN IN (SELECT CAMPAIGN FROM campaign_desc WHERE DESCRIPTION = cd.DESCRIPTION))
               AS redempt_rows
    FROM campaign_desc cd
    GROUP BY cd.DESCRIPTION
) t

-- Recipients vs redeemers per campaign type, descriptive only.
UNION ALL
SELECT 'campaign_type_' || d.DESCRIPTION || '_recipients',
       'campaigns=' || count(DISTINCT d.CAMPAIGN)
       || ' recipient_rows=' || count(*)
       || ' recipient_households=' || count(DISTINCT t.household_key),
       'B observed'
FROM campaign_desc d
JOIN campaign_table t ON t.CAMPAIGN = d.CAMPAIGN
GROUP BY d.DESCRIPTION

ORDER BY metric;
