-- Phase 3 synthesis: T1 core descriptive KPIs (long format, paper-facing)
--
-- Grain: one row per KPI. Populated ONLY from already accepted/reconciled Phase 1-3 evidence: the
-- accepted Phase 2 tables, the accepted Phase 3 derivations (11_commodity_level, 13_department_level,
-- 15_campaign_level) and the accepted totals declared in contracts.py. Nothing here is a new
-- substantive analysis: no model, no test, no selection and no new population is introduced.
--
-- The three top-10% concentration rows are NOT re-derived here: the accepted Question B route selects
-- exactly ceil(n * fraction) = 250 households by VALUE (metrics.top_activity_share), which is
-- count-based and therefore tie-free, whereas a key tie-broken rank would be a different selection
-- rule. The caller appends those three rows from the accepted metric route so T1 carries the accepted
-- figures rather than a near-copy of them.
--
-- Every row carries its unit and its population_or_denominator. Retailer SALES_VALUE rows state that
-- the measure is retailer revenue rather than customer spend or grocery expenditure. Units rows carry
-- the QA-03 caveat. No household identifier appears anywhere in this object.

WITH household AS (
    SELECT
        count(*) AS households,
        sum(active_weeks) AS active_household_weeks,
        sum(total_baskets) AS trips,
        median(active_weeks) AS median_active_weeks,
        median(total_sales_value) AS median_household_sales_value,
        median(total_baskets) AS median_household_trips
    FROM household_history
),
weeks AS (
    SELECT
        count(*) AS observed_weeks,
        sum(weekly_sales_value) AS total_sales_value,
        median(weekly_sales_value) FILTER (WHERE week_no BETWEEN 2 AND 101)
            AS median_full_week_sales_value
    FROM (
        SELECT week_no, sum(weekly_sales_value) AS weekly_sales_value
        FROM household_week_base
        GROUP BY week_no
    )
),
lines AS (
    SELECT
        sum(weekly_sales_value) AS line_grain_sales_value,
        sum(units) AS recorded_units
    FROM household_week_base
),
halves AS (
    SELECT
        household_key,
        sum(weekly_sales_value) FILTER (WHERE week_no <= 51) AS first_half,
        sum(weekly_sales_value) FILTER (WHERE week_no > 51) AS second_half
    FROM household_week_base
    GROUP BY household_key
),
persistence AS (
    SELECT round(corr(first_half, second_half), 4) AS half_correlation FROM halves
),
departments AS (
    SELECT count(*) AS department_labels FROM department_level
),
grocery AS (
    SELECT
        round(100.0 * max(retailer_sales_value) FILTER (WHERE department_desc = 'GROCERY')
            / sum(retailer_sales_value), 4) AS grocery_share_pct
    FROM department_level
),
commodity AS (
    SELECT
        count(*) AS commodity_labels,
        round(100.0 * median(observed_household_penetration), 4) AS median_penetration_pct
    FROM commodity_level
),
campaign AS (
    SELECT
        count(*) AS campaigns,
        sum(listed_households) AS listed_pairs,
        round(median(accepted_phase1_recipient_overlap_pct), 2) AS median_listed_recipient_overlap_pct
    FROM campaign_level
),
listed_households AS (
    SELECT count(DISTINCT household_key) AS households_listed_at_least_once,
           median(campaigns_listed) AS median_campaigns_per_listed_household
    FROM (
        SELECT household_key, count(DISTINCT CAMPAIGN) AS campaigns_listed
        FROM campaign_table GROUP BY household_key
    )
)
SELECT * FROM (
    SELECT 'SAMPLE' AS domain, 'households' AS metric,
           (SELECT households FROM household)::VARCHAR AS value, 'households' AS unit,
           'S0 baseline population' AS population_or_denominator,
           'all 2,500 transacting households; S1/S2 are sensitivity populations only' AS notes
    UNION ALL SELECT 'SAMPLE', 'observed days', '711', 'days', 'days 1-711 of the release',
        'every day present in transaction data'
    UNION ALL SELECT 'SAMPLE', 'observed weeks', (SELECT observed_weeks FROM weeks)::VARCHAR, 'weeks',
        'weeks 1-102', 'weeks 1 and 102 are partial observed weeks (5 and 6 days)'
    UNION ALL SELECT 'SAMPLE', 'household-week grid', '255000', 'household-weeks',
        '2,500 households x 102 weeks', 'balanced grid; zeros are no recorded purchase, not zero grocery consumption'
    UNION ALL SELECT 'SAMPLE', 'active household-weeks',
        (SELECT active_household_weeks FROM household)::VARCHAR, 'household-weeks',
        'of the 255,000-cell grid', 'recorded purchase at this retailer (48.62% of the grid)'
    UNION ALL SELECT 'SAMPLE', 'trips (baskets)',
        (SELECT trips FROM household)::VARCHAR, 'baskets', 'all recorded household-weeks',
        'baskets recorded at this retailer'
    UNION ALL SELECT 'RETAILER VALUE', 'total retailer SALES_VALUE',
        round((SELECT total_sales_value FROM weeks), 2)::VARCHAR, 'retailer currency units',
        'all recorded household-weeks',
        'retailer revenue (retailer SALES_VALUE); not a household grocery bill and not a customer-paid amount'
    UNION ALL SELECT 'RETAILER VALUE', 'median weekly retailer SALES_VALUE (full weeks)',
        round((SELECT median_full_week_sales_value FROM weeks), 2)::VARCHAR, 'retailer currency units',
        'weeks 2-101 (full observed weeks)',
        'retailer revenue; median of the weekly series reproduced by the accepted weekly route'
    UNION ALL SELECT 'RETAILER VALUE', 'total recorded units',
        (SELECT recorded_units FROM lines)::VARCHAR, 'recorded units', 'all recorded household-weeks',
        'dominated by very large-quantity lines (QA-03): not an intuitive physical volume measure'
    UNION ALL SELECT 'HOUSEHOLD', 'median active weeks per household',
        (SELECT median_active_weeks FROM household)::VARCHAR, 'weeks', '2,500 households',
        'active week = week with a recorded purchase at this retailer'
    UNION ALL SELECT 'HOUSEHOLD', 'median household retailer SALES_VALUE',
        round((SELECT median_household_sales_value FROM household), 2)::VARCHAR,
        'retailer currency units', '2,500 households', 'retailer revenue over weeks 1-102'
    UNION ALL SELECT 'HOUSEHOLD', 'median household trips',
        (SELECT median_household_trips FROM household)::VARCHAR, 'baskets', '2,500 households',
        'baskets recorded over weeks 1-102'
    UNION ALL SELECT 'HOUSEHOLD', 'first-half vs second-half household retailer SALES_VALUE correlation',
        (SELECT half_correlation FROM persistence)::VARCHAR, 'Pearson correlation',
        '2,500 households (QB-02)',
        'retailer revenue across sample halves; descriptive persistence only, not a causal household effect'
    UNION ALL SELECT 'PRODUCT', 'department labels',
        (SELECT department_labels FROM departments)::VARCHAR, 'labels',
        'Phase 3 department derivation',
        'includes one unlabelled sentinel department carrying 0.00 recorded value'
    UNION ALL SELECT 'PRODUCT', 'commodity labels',
        (SELECT commodity_labels FROM commodity)::VARCHAR, 'labels', 'all commodity labels retained',
        'includes the unlabelled sentinel label; all 308 labels are retained in T2'
    UNION ALL SELECT 'PRODUCT', 'GROCERY share of retailer SALES_VALUE',
        (SELECT grocery_share_pct FROM grocery)::VARCHAR, '%', 'of 8,057,463.08 retailer SALES_VALUE',
        'retailer revenue; departmental partition, not comparable with the commodity-level dispersion'
    UNION ALL SELECT 'PRODUCT', 'median commodity household penetration',
        (SELECT median_penetration_pct FROM commodity)::VARCHAR, '% of 2,500 households',
        '308 commodity labels', 'observed-household penetration; never market penetration'
    UNION ALL SELECT 'CAMPAIGN', 'campaigns',
        (SELECT campaigns FROM campaign)::VARCHAR, 'campaigns', 'campaign_desc (all retained)',
        'campaign inventory recorded by the vendor'
    UNION ALL SELECT 'CAMPAIGN', 'households listed at least once',
        (SELECT households_listed_at_least_once FROM listed_households)::VARCHAR, 'households',
        'of 2,500 households', 'listed = factual database membership, never an assigned unit'
    UNION ALL SELECT 'CAMPAIGN', 'median campaigns per listed household',
        (SELECT median_campaigns_per_listed_household FROM listed_households)::VARCHAR, 'campaigns',
        '1,584 listed households', 'listing is repeated; maximum is 17 campaigns for one household'
    UNION ALL SELECT 'CAMPAIGN', 'median accepted listed-recipient overlap',
        (SELECT median_listed_recipient_overlap_pct FROM campaign)::VARCHAR, '%',
        '30 campaigns (accepted Phase 1 definition)',
        'listed-campaign overlap among a campaign''s own listed recipients; never realized exposure'
) AS kpis
ORDER BY domain, metric
