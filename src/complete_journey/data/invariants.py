"""Hard invariants for the Phase 1 audit.

Distinction enforced throughout this project:

**HARD INVARIANT** - a violation means the *pipeline* is wrong (a file is
missing, a count is internally inconsistent, raw data was modified, an expected
artifact is absent). The audit run fails.

**EMPIRICAL FINDING** - a surprising value in the data (negative quantities, an
unexpected duplicate key, a low match rate). The audit *reports* it. It is never
asserted into a pass/fail condition, because "make the test green by deleting
the anomaly" is exactly the failure mode this project must avoid.

A hard invariant is only ever stated where its violation would indicate a bug in
*our* code, never where it would indicate something interesting about the data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

#: Public aggregate artifacts eligible for tracking and publication. This is a
#: closed allowlist: adding SQL does not make its output publishable.
PUBLIC_ARTIFACTS: set[str] = {
    "01_table_inventory",
    "02_column_profiles",
    "03_grain_keys",
    "04_referential_integrity",
    "05_transaction_audit",
    "06_price_reconciliation",
    "07_product_hierarchy",
    "07b_product_categories",
    "08_household_coverage",
    "09_campaign_feasibility",
    "10_coupon_audit",
    "11_causal_data_audit",
    "11b_causal_psw_coverage",
    "12_time_panel_feasibility",
    "13_design_feasibility",
}

#: Local derived artifacts that are generated but never published or tracked.
PRIVATE_ARTIFACTS: set[str] = {"12b_household_week_activity"}

#: Artifacts the audit must produce, and the columns each must expose.
#:
#: Schema stability of generated outputs is a hard invariant: downstream
#: consumers and reviewers rely on these column names.
EXPECTED_ARTIFACTS: dict[str, list[str]] = {
    "01_table_inventory": ["table_name", "row_count", "distinct_campaign", "column_count"],
    "02_column_profiles": [
        "table_name",
        "column_name",
        "row_count",
        "null_count",
        "null_pct",
        "distinct_count",
    ],
    "03_grain_keys": [
        "table_name",
        "candidate_key",
        "expected_grain_source",
        "row_count",
        "distinct_key",
        "duplicate_rows",
        "duplicate_groups",
    ],
    "04_referential_integrity": [
        "relationship",
        "expected_cardinality",
        "source_distinct_ids",
        "parent_distinct_ids",
        "unmatched_ids",
        "match_rate_pct",
        "source_max_rows_per_key",
        "target_max_rows_per_key",
        "raw_join_max_rows_per_key",
    ],
    "05_transaction_audit": ["metric", "value", "provenance"],
    "06_price_reconciliation": [
        "check_name",
        "provenance",
        "rows_evaluated",
        "min_value",
        "mean_value",
        "max_value",
        "negative_count",
        "zero_count",
    ],
    "07_product_hierarchy": ["metric", "value", "provenance"],
    "07b_product_categories": [
        "commodity",
        "products_in_assortment",
        "distinct_products",
        "sub_commodities",
        "manufacturers",
        "brands",
        "national_products",
        "private_products",
        "transaction_line_count",
        "purchasing_households",
        "household_coverage_pct",
        "products_actually_bought",
    ],
    "08_household_coverage": ["metric", "value", "provenance"],
    "09_campaign_feasibility": [
        "campaign",
        "campaign_type",
        "start_day",
        "end_day",
        "duration_days",
        "nominal_start_week",
        "nominal_end_week",
        "start_week",
        "end_week",
        "recipient_households",
        "nonrecipient_households_descriptive",
        "pre_weeks_calendar",
        "campaign_weeks_calendar",
        "post_weeks_calendar",
        "recipient_active_household_weeks_before",
        "recipient_active_household_weeks_during",
        "recipient_active_household_weeks_after",
        "overlapping_campaigns",
        "recipient_households_with_overlapping_campaign",
        "recipient_pct_with_overlapping_campaign",
        "avg_campaigns_per_household",
        "max_campaigns_per_household",
        "coupon_pool_pairs",
        "coupon_pool_products",
        "redemption_rows",
        "redeeming_households",
        "ends_after_transaction_window",
        "zero_post_support",
        "recipient_overlap_warning",
        "interpretation_warning",
    ],
    "10_coupon_audit": ["metric", "value", "provenance"],
    "11_causal_data_audit": ["metric", "value", "provenance"],
    "11b_causal_psw_coverage": ["metric", "value", "provenance"],
    "12_time_panel_feasibility": ["metric", "value", "provenance"],
    "12b_household_week_activity": [
        "household_key",
        "active_weeks",
        "first_week",
        "last_week",
        "window_length",
    ],
    "13_design_feasibility": [
        "design",
        "classification",
        "supporting_evidence",
        "unresolved_questions",
        "major_threats",
    ],
}

#: Artifacts that must contain at least one row.
NON_EMPTY_ARTIFACTS: tuple[str, ...] = tuple(EXPECTED_ARTIFACTS)

#: Exact aggregate row-key contracts for every public artifact. These hashes
#: freeze only the ordered key universe, not empirical values. Together with
#: exact schemas they prevent an allowlisted filename from silently changing to
#: observation-level or otherwise undeclared rows.
PUBLIC_ARTIFACT_KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    "01_table_inventory": ("table_name",),
    "02_column_profiles": ("table_name", "column_name"),
    "03_grain_keys": ("table_name", "candidate_key"),
    "04_referential_integrity": ("relationship",),
    "05_transaction_audit": ("metric",),
    "06_price_reconciliation": ("check_name",),
    "07_product_hierarchy": ("metric",),
    "07b_product_categories": ("commodity",),
    "08_household_coverage": ("metric",),
    "09_campaign_feasibility": ("campaign",),
    "10_coupon_audit": ("metric",),
    "11_causal_data_audit": ("metric",),
    "11b_causal_psw_coverage": ("metric",),
    "12_time_panel_feasibility": ("metric",),
    "13_design_feasibility": ("design",),
}
EXPECTED_PUBLIC_KEY_SHA256: dict[str, str] = {
    "01_table_inventory": "1205a66a9b3dedf3781d9fd26f10aecc1a49c5f19e1e461903b8a82d31ac8daa",
    "02_column_profiles": "8477e8fff82995b3d1b5b5c44744f56503e83e3bbf60ed976ea6d4af5871c47b",
    "03_grain_keys": "e15f9c86a569ef0ec2754df4de3f2d124dbe5059be6986fe4f8a121f45906d7f",
    "04_referential_integrity": "a2da65c17a0bb013e65a160d73bd343bcc02928ef0479f6241ace9804c518064",
    "05_transaction_audit": "97fc152a892358cd24233858576660edcf6a59b7fad81a0cbe72e30c8037e8fb",
    "06_price_reconciliation": "1da49afc11f411a8980f9533262d5ee47fe1311116f21e54cb4caa059dc77cee",
    "07_product_hierarchy": "8c12bbc08b57ea79c5bc91545e28ba70036b215290dc1eac65a9accad8ac2a87",
    "07b_product_categories": "0597d7ddb6b9677dcc6daa056a10a8f124fb433d15a8cc153b56b9fa909b3bec",
    "08_household_coverage": "39a9940b23fe801b1e6107ff4badf5c9e4333403f4fa2e32ff51aa9de5ace124",
    "09_campaign_feasibility": "548c98fc39f13d736aa55b37f78af3ef46ec6e223b1a86758b4fa35350e6844c",
    "10_coupon_audit": "47f6362826348d1b600c9dcfffd9759889f41c577e4cdfc4d00a4decc03a9cac",
    "11_causal_data_audit": "d99a7395a9268df2881423c90016db8def5dbec3815e98a051da432e940e91de",
    "11b_causal_psw_coverage": "ba75ace51192326ecf6807c28da2a87bac08d590799ce2cbbf599e9cbabded4d",
    "12_time_panel_feasibility": "69eb633cd65a16ba8b97aa73c7fb1ec9f0429e3454fe7da37500c2fa0291f64a",
    "13_design_feasibility": "2e378ca764fb5355d1667ce39c08546aa935d8c7e391367287477466fc7febb7",
}


def _artifact_key_sha256(frame: Any, columns: tuple[str, ...]) -> str:
    rows = sorted(
        tuple(str(value) for value in row)
        for row in frame.loc[:, columns].itertuples(index=False, name=None)
    )
    canonical = "".join("\0".join(row) + "\n" for row in rows).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass
class InvariantResult:
    """Outcome of a single invariant check."""

    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'}  {self.name}: {self.detail}"


def check_expected_tables_discovered(discovered: list[str]) -> InvariantResult:
    """All eight raw tables must be discoverable. A miss means path discovery broke."""
    expected = {
        "transaction_data",
        "product",
        "campaign_table",
        "campaign_desc",
        "coupon",
        "coupon_redempt",
        "causal_data",
        "hh_demographic",
    }
    missing = expected - set(discovered)
    return InvariantResult(
        "raw_tables_discovered",
        not missing,
        f"{len(discovered)}/8 found" + (f"; MISSING {sorted(missing)}" if missing else ""),
    )


def check_artifact_schemas(artifacts: dict[str, Any]) -> list[InvariantResult]:
    """Every expected artifact exists with its required columns and >=1 row."""
    results: list[InvariantResult] = []
    for name, required in EXPECTED_ARTIFACTS.items():
        if name not in artifacts:
            results.append(InvariantResult(f"artifact_present:{name}", False, "artifact missing"))
            continue
        columns = [str(c) for c in artifacts[name].columns]
        if columns != required:
            results.append(
                InvariantResult(
                    f"artifact_schema:{name}",
                    False,
                    f"exact schema expected {required}, got {columns}",
                )
            )
            continue
        if name in NON_EMPTY_ARTIFACTS and len(artifacts[name]) == 0:
            results.append(InvariantResult(f"artifact_non_empty:{name}", False, "zero rows"))
            continue
        results.append(
            InvariantResult(
                f"artifact_schema:{name}", True, f"{len(artifacts[name])} rows, exact schema ok"
            )
        )
        if name in EXPECTED_PUBLIC_KEY_SHA256:
            actual_hash = _artifact_key_sha256(artifacts[name], PUBLIC_ARTIFACT_KEY_COLUMNS[name])
            expected_hash = EXPECTED_PUBLIC_KEY_SHA256[name]
            results.append(
                InvariantResult(
                    f"artifact_keys:{name}",
                    actual_hash == expected_hash,
                    "aggregate key hash matches closed publication contract"
                    if actual_hash == expected_hash
                    else f"aggregate key hash expected {expected_hash}, got {actual_hash}",
                )
            )
    return results


def check_row_counts_consistent(artifacts: dict[str, Any]) -> list[InvariantResult]:
    """Internal consistency: per-table row counts agree across artifacts.

    This is a check on OUR aggregation, not on the data. If the inventory says
    transaction_data has N rows and the grain report says M, one of the two
    queries is wrong.
    """
    results: list[InvariantResult] = []
    if "01_table_inventory" not in artifacts or "02_column_profiles" not in artifacts:
        return [InvariantResult("row_count_consistency", False, "required artifacts missing")]

    inventory = dict(
        zip(
            artifacts["01_table_inventory"]["table_name"],
            artifacts["01_table_inventory"]["row_count"],
            strict=True,
        )
    )
    profiles = artifacts["02_column_profiles"]
    for table, group in profiles.groupby("table_name"):
        counts = set(group["row_count"].tolist())
        if len(counts) != 1:
            results.append(
                InvariantResult(
                    f"row_count_consistent:{table}",
                    False,
                    f"column profiles disagree on row count: {sorted(counts)}",
                )
            )
            continue
        profiled = int(counts.pop())
        actual = int(inventory.get(table, -1))
        results.append(
            InvariantResult(
                f"row_count_consistent:{table}",
                profiled == actual,
                f"inventory={actual:,} profiles={profiled:,}",
            )
        )
    return results


def check_no_null_join_keys(artifacts: dict[str, Any]) -> list[InvariantResult]:
    """Referential-integrity rows must not have null keys.

    A null in a key column means the integrity query itself is wrong.
    """
    results: list[InvariantResult] = []
    if "04_referential_integrity" not in artifacts:
        return [InvariantResult("ri_no_null_keys", False, "required artifact missing")]
    rel = artifacts["04_referential_integrity"]
    for column in ("relationship", "source_distinct_ids", "parent_distinct_ids"):
        nulls = int(rel[column].isna().sum())
        results.append(
            InvariantResult(
                f"ri_no_nulls:{column}", nulls == 0, f"{nulls} null value(s) in {column}"
            )
        )
    return results


def check_raw_unchanged(before: dict[str, Any], after: dict[str, Any]) -> list[InvariantResult]:
    """Raw file size and mtime must be identical before and after the audit.

    This is the immutability guarantee. A violation means the audit wrote to
    data/raw/, which is a hard failure regardless of what else succeeded.
    """
    results: list[InvariantResult] = []
    names = sorted(set(before) | set(after))
    for name in names:
        if name not in before or name not in after:
            results.append(
                InvariantResult(f"raw_unchanged:{name}", False, "file appeared/disappeared")
            )
            continue
        same = before[name] == after[name]
        results.append(
            InvariantResult(
                f"raw_unchanged:{name}",
                same,
                "size+mtime identical" if same else f"{before[name]} -> {after[name]}",
            )
        )
    return results


def check_manifest_valid(verify_result: tuple[bool, str]) -> InvariantResult:
    """The raw manifest must verify before any audit work proceeds."""
    ok, detail = verify_result
    return InvariantResult("raw_manifest_valid", ok, detail)


def _metric_values(frame: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in frame.itertuples():
        try:
            values[str(row.metric)] = float(row.value)
        except (TypeError, ValueError):
            continue
    return values


def check_semantic_consistency(artifacts: dict[str, Any]) -> list[InvariantResult]:
    """Check declared grains and cross-artifact arithmetic, never empirical levels."""
    results: list[InvariantResult] = []

    for name, key in {
        "05_transaction_audit": "metric",
        "06_price_reconciliation": "check_name",
        "07_product_hierarchy": "metric",
        "07b_product_categories": "commodity",
        "08_household_coverage": "metric",
        "09_campaign_feasibility": "campaign",
        "10_coupon_audit": "metric",
        "11_causal_data_audit": "metric",
        "11b_causal_psw_coverage": "metric",
        "12_time_panel_feasibility": "metric",
        "13_design_feasibility": "design",
    }.items():
        if name not in artifacts or key not in artifacts[name].columns:
            continue
        duplicates = int(artifacts[name][key].duplicated().sum())
        results.append(
            InvariantResult(
                f"declared_grain_unique:{name}",
                duplicates == 0,
                f"{duplicates} duplicate {key} value(s)",
            )
        )

    if "07b_product_categories" in artifacts:
        categories = artifacts["07b_product_categories"]
        impossible = int(
            (categories["products_actually_bought"] > categories["products_in_assortment"]).sum()
        )
        results.append(
            InvariantResult(
                "category_products_bought_within_assortment",
                impossible == 0,
                f"{impossible} row(s) exceed their commodity assortment",
            )
        )

    if "09_campaign_feasibility" in artifacts:
        campaigns = artifacts["09_campaign_feasibility"]
        period_columns = [
            "pre_weeks_calendar",
            "campaign_weeks_calendar",
            "post_weeks_calendar",
        ]
        negative = int((campaigns[period_columns] < 0).any(axis=1).sum())
        totals = campaigns[period_columns].sum(axis=1)
        inconsistent = int((totals != totals.iloc[0]).sum()) if len(totals) else 0
        overlap_excess = int(
            (
                campaigns["recipient_households_with_overlapping_campaign"]
                > campaigns["recipient_households"]
            ).sum()
        )
        results.extend(
            [
                InvariantResult(
                    "campaign_support_nonnegative",
                    negative == 0,
                    f"{negative} campaign(s) have negative support",
                ),
                InvariantResult(
                    "campaign_support_partitions_observed_window",
                    inconsistent == 0,
                    f"{inconsistent} campaign(s) have inconsistent support totals",
                ),
                InvariantResult(
                    "campaign_overlap_numerator_within_recipients",
                    overlap_excess == 0,
                    f"{overlap_excess} campaign(s) exceed recipient denominator",
                ),
            ]
        )

    if "04_referential_integrity" in artifacts:
        ri = artifacts["04_referential_integrity"]
        bad_bounds = int(
            ((ri["unmatched_ids"] < 0) | (ri["unmatched_ids"] > ri["source_distinct_ids"])).sum()
        )
        expected_rate = (
            100.0 * (ri["source_distinct_ids"] - ri["unmatched_ids"]) / ri["source_distinct_ids"]
        )
        bad_rate = int(((ri["match_rate_pct"] - expected_rate).abs() > 1e-5).sum())
        results.extend(
            [
                InvariantResult(
                    "ri_unmatched_within_source",
                    bad_bounds == 0,
                    f"{bad_bounds} relationship(s) have impossible unmatched counts",
                ),
                InvariantResult(
                    "ri_match_rate_arithmetic",
                    bad_rate == 0,
                    f"{bad_rate} relationship(s) have inconsistent match rates",
                ),
            ]
        )

    if "04_referential_integrity" in artifacts and "10_coupon_audit" in artifacts:
        relationship = "coupon_redempt.(COUPON_UPC,CAMPAIGN) -> coupon.(COUPON_UPC,CAMPAIGN)"
        ri_rows = artifacts["04_referential_integrity"]
        source = int(
            ri_rows.loc[ri_rows["relationship"] == relationship, "source_distinct_ids"].iloc[0]
        )
        coupon = _metric_values(artifacts["10_coupon_audit"])
        expected = int(coupon["redempt_distinct_campaign_upc_pairs"])
        results.append(
            InvariantResult(
                "coupon_ri_uses_source_key_grain",
                source == expected,
                f"RI source={source:,} coupon source keys={expected:,}",
            )
        )

    if "11_causal_data_audit" in artifacts:
        causal = _metric_values(artifacts["11_causal_data_audit"])
        partition = (
            causal["causal_duplicate_psw_same_placement_groups"]
            + causal["causal_duplicate_psw_display_only_conflicts"]
            + causal["causal_duplicate_psw_mailer_only_conflicts"]
            + causal["causal_duplicate_psw_both_conflicts"]
        )
        checks = {
            "causal_duplicate_partition": partition == causal["causal_duplicate_psw_groups"],
            "causal_duplicate_excess_arithmetic": (
                causal["causal_duplicate_psw_physical_rows"] - causal["causal_duplicate_psw_groups"]
                == causal["causal_duplicate_psw_excess_rows"]
            ),
            "causal_display_conflict_arithmetic": (
                causal["causal_duplicate_psw_display_only_conflicts"]
                + causal["causal_duplicate_psw_both_conflicts"]
                == causal["causal_duplicate_psw_display_conflicts_total"]
            ),
            "causal_mailer_conflict_arithmetic": (
                causal["causal_duplicate_psw_mailer_only_conflicts"]
                + causal["causal_duplicate_psw_both_conflicts"]
                == causal["causal_duplicate_psw_mailer_conflicts_total"]
            ),
        }
        results.extend(
            InvariantResult(name, passed, "duplicate/conflict arithmetic")
            for name, passed in checks.items()
        )

    if "11b_causal_psw_coverage" in artifacts:
        coverage = _metric_values(artifacts["11b_causal_psw_coverage"])
        matched = coverage["purchased_psw_matched_to_causal"]
        placement_partition = (
            coverage["matched_psw_display_only"]
            + coverage["matched_psw_mailer_only"]
            + coverage["matched_psw_both_display_and_mailer"]
            + coverage["matched_psw_with_no_placement"]
        )
        checks = {
            "coverage_unique_key_partition": (
                matched + coverage["purchased_psw_unmatched"] == coverage["purchased_unique_psw"]
            ),
            "coverage_line_partition": (
                coverage["transaction_lines_matched_to_causal"]
                + coverage["transaction_lines_unmatched"]
                == coverage["transaction_lines_total"]
            ),
            "coverage_placement_partition": placement_partition == matched,
            "coverage_any_placement_within_matched": (
                coverage["matched_psw_with_any_placement"] <= matched
            ),
            "coverage_any_display_within_matched": (
                coverage["matched_psw_with_any_display"] <= matched
            ),
            "coverage_any_mailer_within_matched": (
                coverage["matched_psw_with_any_mailer"] <= matched
            ),
        }
        results.extend(
            InvariantResult(name, passed, "coverage arithmetic at unique PSW grain")
            for name, passed in checks.items()
        )

    return results


#: Convenience mapping for reporting.
HARD_INVARIANT_DESCRIPTIONS: dict[str, str] = {
    "raw_manifest_valid": "Raw data matches its committed SHA-256 manifest before audit work starts.",
    "raw_tables_discovered": "All eight vendor CSVs are found by exact filename.",
    "raw_unchanged:*": "No audit step modified a raw file (size and mtime unchanged).",
    "artifact_present:*": "A required audit artifact was produced.",
    "artifact_schema:*": "A required artifact has its exact ordered public schema.",
    "artifact_keys:*": "A public artifact matches its frozen aggregate row-key contract.",
    "artifact_non_empty:*": "A required artifact contains at least one row.",
    "row_count_consistent:*": "Row counts agree across independent audit queries.",
    "ri_no_nulls:*": "Referential-integrity key columns contain no nulls.",
}
