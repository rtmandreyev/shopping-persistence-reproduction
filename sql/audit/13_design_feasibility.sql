-- 13_design_feasibility.sql
-- PURPOSE: record, per planned design, whether the audit found enough structure to
--          justify CONSTRUCTING it next - and what remains undecided.
--
-- This is NOT a causal-identification verdict. It says only: does the structure
-- needed to attempt this design exist in the data, and what must be resolved
-- before an attempt is credible?
--
-- Classifications are limited to exactly three:
--   SUPPORTED FOR NEXT-STAGE CONSTRUCTION
--   REQUIRES FURTHER IDENTIFICATION/DEFINITION WORK
--   CURRENTLY UNSUPPORTED BY AUDIT EVIDENCE
--
-- Each row cites the audit metric(s) it rests on, so a reviewer can check the
-- classification against the generated artifacts rather than trusting prose.
-- Provenance: C = our classification, with B evidence behind it.

SELECT 'A. descriptive / product analytics' AS design,
       'SUPPORTED FOR NEXT-STAGE CONSTRUCTION' AS classification,
       '02_table_inventory (8 tables, 2.5M+ transaction lines); 07_product_hierarchy; '
       || '07b_product_categories (assortment + coverage per commodity)' AS supporting_evidence,
       'Choice of outcome definitions (e.g. weekly_spend) still open; that is a Phase 2 '
       || 'definition task, not a blocker for construction.' AS unresolved_questions,
       'Left-truncation: households may predate observation, so cohort cuts are not clean '
       || 'entry cohorts. Sales_value is retailer revenue, not customer price.' AS major_threats
UNION ALL
SELECT 'B. household panel models',
       'SUPPORTED FOR NEXT-STAGE CONSTRUCTION',
       '12_time_panel_feasibility (2,500 households, weeks 1-102, observed household-week '
       || 'pairs); 12b_household_week_activity (per-household active weeks)',
       'Whether to zero-fill non-purchasing weeks, condition on active weeks, or use a '
       || 'trip-conditioned panel is UNDECIDED and belongs to Phase 2.',
       'Unobserved time-varying targeting; serial correlation; a household-week with no '
       || 'purchase is not an observed zero by construction. FE will not make these causal.'
UNION ALL
SELECT 'C. discrete choice',
       'REQUIRES FURTHER IDENTIFICATION/DEFINITION WORK',
       '07b_product_categories reports per-commodity assortment, brands, household coverage '
       || 'and products actually bought, which is the evidence needed to screen candidates',
       'The commodity is NOT chosen here. Choice-set construction rule (products plausibly '
       || 'available in the same store-week vs a rolling rule) is undefined and must be '
       || 'decided with sensitivity analysis.',
       'Non-purchases do not reveal shelf availability, so choice sets are constructed, not '
       || 'observed. Implied price is ambiguous (see 06_price_reconciliation). Price/promo '
       || 'coefficients would be behavioural associations absent exogenous price variation.'
UNION ALL
SELECT 'D. campaign DiD / event study',
       'REQUIRES FURTHER IDENTIFICATION/DEFINITION WORK',
       '09_campaign_feasibility gives, per campaign: type, window, listed recipient households, '
       || 'available pre/campaign/post weeks, and listed-campaign overlap',
       'No campaign is selected here, and NO campaign is declared DiD-valid. Pre-trend tests, '
       || 'control-group validity, and treatment-selection balance are Phase 6 decisions.',
       'The vendor never states campaigns were randomised. "Not listed as a recipient" is not '
       || 'evidence of eligibility. Overlapping campaigns and repeated exposure can contaminate '
       || 'both treatment and control. Target selection is plausibly on prior purchase behaviour.'
UNION ALL
SELECT 'E. IV / 2SLS investigation',
       'CURRENTLY UNSUPPORTED BY AUDIT EVIDENCE',
       '10_coupon_audit: for Type A the household-specific coupon subset is unobserved and was '
       || 'selected on prior purchase behaviour (vendor-documented); 11_causal_data_audit '
       || 'shows display/mailer are placement codes, with recorded-key existence in 11b',
       'NO instrument is proposed by this audit. Candidate instruments and their exclusion '
       || 'paths must be adjudicated separately; the audit found no randomisation to build on.',
       'Type A assignment is endogenous by construction. Display and mailer plausibly affect '
       || 'demand directly, so they are treatments rather than valid instruments. Excluding an '
       || 'instrument here would require an assumption the data cannot test.'
UNION ALL
SELECT 'F. predictive purchase benchmark',
       'SUPPORTED FOR NEXT-STAGE CONSTRUCTION',
       '12_time_panel_feasibility (time coverage, 102 weeks) supports a time-aware split; '
       || '08_household_coverage defines the household universe',
       'Target definition, unit of observation, and prediction horizon are Phase 7 decisions.',
       'Leakage risk from using post-outcome or contemporaneously-determined features. '
       || 'Predictive accuracy must not be reported as evidence of treatment effect.'
UNION ALL
SELECT 'G. empirically calibrated ABM',
       'REQUIRES FURTHER IDENTIFICATION/DEFINITION WORK',
       'Panels (B), choice evidence (C) and product hierarchy (07b) can supply calibration '
       || 'targets once those stages are built',
       'Calibration targets and parameter ranges are undecided; they depend on Phase 4-5 results '
       || 'that do not exist yet.',
       'ABM results are conditional scenario analysis, not causal estimates. Simulation cannot '
       || 'recover identification that the econometrics failed to establish.'
ORDER BY design;
