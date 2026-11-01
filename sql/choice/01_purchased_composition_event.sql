-- Phase 5 factual events only. positive_group_lines is the guarded line projection:
-- exact SPAGHETTI DRY, quantity > 0, unique household/basket/product keys,
-- many-to-one product and basket joins, zero unmatched, coherent basket context.
-- Basket identity cannot be recovered from a product-week aggregate.
-- Histories are aggregated by household-week BEFORE accumulating earlier weeks;
-- mixed events remain in history, and pure responses are selected only afterwards.
WITH events AS (
    SELECT
        household_key, basket_id, week_no, store_id,
        count(*) FILTER (WHERE brand = 'National')::BIGINT AS national_lines,
        count(*) FILTER (WHERE brand = 'Private')::BIGINT AS private_lines,
        CASE
            WHEN count(*) FILTER (WHERE brand = 'Private') = 0 THEN 'NATIONAL_ONLY'
            WHEN count(*) FILTER (WHERE brand = 'National') = 0 THEN 'PRIVATE_ONLY'
            ELSE 'MIXED'
        END AS composition
    FROM positive_group_lines
    GROUP BY household_key, basket_id, week_no, store_id
), household_weeks AS (
    SELECT
        household_key, week_no,
        count(*) FILTER (WHERE composition = 'NATIONAL_ONLY') AS national_pure,
        count(*) FILTER (WHERE composition = 'PRIVATE_ONLY') AS private_pure,
        count(*) FILTER (WHERE composition = 'MIXED') AS mixed
    FROM events
    GROUP BY household_key, week_no
), prior_weeks AS (
    SELECT
        household_key, week_no,
        coalesce(sum(national_pure) OVER earlier, 0)::BIGINT AS prior_national_pure,
        coalesce(sum(private_pure) OVER earlier, 0)::BIGINT AS prior_private_pure,
        coalesce(sum(mixed) OVER earlier, 0)::BIGINT AS prior_mixed
    FROM household_weeks
    WINDOW earlier AS (
        PARTITION BY household_key ORDER BY week_no
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
), ledger AS (
    SELECT
        e.*, p.prior_national_pure, p.prior_private_pure, p.prior_mixed,
        p.prior_national_pure + p.prior_private_pure + p.prior_mixed
            AS prior_total_group,
        p.prior_national_pure + p.prior_private_pure AS prior_pure
    FROM events e
    JOIN prior_weeks p USING (household_key, week_no)
)
SELECT
    household_key, basket_id, week_no, store_id,
    national_lines, private_lines, composition,
    prior_national_pure, prior_private_pure, prior_mixed,
    prior_total_group, prior_pure,
    CASE
        WHEN prior_pure = 0 THEN 'H0'
        WHEN prior_private_pure = 0 THEN 'HN'
        WHEN prior_national_pure = 0 THEN 'HP'
        ELSE 'HB'
    END AS history_category,
    -- The runner uses numpy.log1p for the final numerical depth transformation.
    ln(1.0 + prior_pure) AS log1p_prior_pure,
    prior_private_pure::DOUBLE / nullif(prior_pure, 0) AS prior_private_share,
    CASE
        WHEN week_no <= 34 THEN 'THIRD_1'
        WHEN week_no <= 68 THEN 'THIRD_2'
        ELSE 'THIRD_3'
    END AS calendar_third,
    (prior_pure = 0 AND prior_mixed > 0) AS prior_mixed_only
FROM ledger
ORDER BY household_key, week_no, basket_id;
