"""Phase 4 aggregate design-artifact allowlist and immutable-input verification."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

from complete_journey.econometrics.samples import SAMPLES, require
from complete_journey.paths import PROJECT_ROOT

OUTPUTS = {
    "summary": "outputs/tables/panels/phase4_preflight_summary.csv",
    "contract": "outputs/tables/panels/phase4_sample_contract.json",
    "provenance": "outputs/tables/panels/phase4_preflight_provenance.json",
}
FIELDS = ("sample", "domain", "metric", "value", "unit", "denominator", "scope")
SCOPE = "DESIGN_PREFLIGHT_NOT_ESTIMATION"
PHASE3_HASHES = {
    "outputs/figures/descriptive/final/f1_weekly_retailer_activity.png": "75c11ea0d3e1c7aafc04afb3cba1dcb51466015b67f5fb0d87ad2e97c8598f3f",
    "outputs/figures/descriptive/final/f2_participation_and_intensity.png": "d9abe325d8b7105b677bf52b0a8de3654de7456a8b4c685298e15d8ea801a5d0",
    "outputs/figures/descriptive/final/f3_household_heterogeneity.png": "cd1571bad1a747d5943786ea99ae9d7a541db7e33a3ef3e6b46afc25a8037849",
    "outputs/figures/descriptive/final/f4_household_activity_concentration.png": "25b76cad77f603b6461670517105ac99c5374352c8989eb32e71771f534da2b9",
    "outputs/figures/descriptive/final/f5_product_structure.png": "eace606e760dfbf0a72979c1a764be4b0858e4bce2e3e6e52223c6aaf2406739",
    "outputs/figures/descriptive/final/f6_commodity_penetration_intensity.png": "38f5750f23dd42dc4ac3ac63b0076368cd081d89bab9e1d6130497956b21d0a8",
    "outputs/figures/descriptive/final/f7_national_private_composition.png": "56bd33f0cc24dd7ef84e8bdb5b25585674d1ab27bf0c60f0245c75b4ea9d199e",
    "outputs/figures/descriptive/final/f8_campaign_landscape.png": "2d6e18ab9150c8745d9ab210e1ce3015adbfbf17e6ea1060235aca831ccf0b28",
    "outputs/figures/descriptive/final/f9_representative_customer_journeys.png": "69adf6dffb43288446a450cdd8ec9510e527fd2784345075a70ad67391e1cf14",
    "outputs/tables/descriptive/final/t1_core_kpis.csv": "ad8cb92db9d70edd55004fa16d142a5960e652671008da3177e5dc3d124c12ea",
    "outputs/tables/descriptive/final/t2_commodity_diagnostics.csv": "f030f4383aa60940f7fa6a54a2a73465c3143675eefec0e1b2bf6062e7ca446a",
    "outputs/tables/descriptive/final/t3_campaign_reconnaissance.csv": "8c474a858620484296f4973295b98234b7503cde694dd96a079efec77976d3bb",
}
FORBIDDEN = (
    "household_key",
    "coefficient",
    "standard_error",
    "std_error",
    "p_value",
    "pvalue",
    "t_stat",
    "fitted_value",
)
CONCENTRATION_KEYS = (
    "contributors",
    "contributor_share",
    "max_share",
    "effective_count",
    "top01_count",
    "top01_share",
    "top05_count",
    "top05_share",
    "top10_count",
    "top10_share",
)
METRICS = {
    "sample": (
        "cells",
        "households",
        "response_weeks",
        "first_response_week",
        "last_response_week",
    ),
    "history": (
        "mean",
        "zeros",
        "ones",
        "variance_ddof0",
        "households_both_states",
        "households_constant0",
        "households_constant1",
        "adjacent_0_to_1",
        "adjacent_1_to_0",
        "switching_households",
        "switching_household_share",
        "switches_mean",
        "switches_max",
        "switches_min",
        "switches_p10",
        "switches_p25",
        "switches_p50",
        "switches_p75",
        "switches_p90",
    ),
    "regressor": (
        "raw_centered_ss",
        "raw_uncentered_ss",
        "household_ss",
        "household_variance_ddof0",
        "twfe_mean",
        "twfe_ss",
        "twfe_variance_ddof0",
        "twfe_min",
        "twfe_max",
        "twfe_raw_centered_ss_share",
        "independent_vector_max_abs_difference",
        "independent_ss_abs_difference",
        "vector_absolute_tolerance",
        "ss_absolute_tolerance",
        *(
            f"{prefix}_{key}"
            for prefix in ("household_within", "household_twfe", "week_twfe")
            for key in CONCENTRATION_KEYS
        ),
    ),
    "outcome": (
        "purchase_zeros",
        "purchase_ones",
        "purchase_mean",
        "purchase_households_varying",
        "purchase_households_constant0",
        "purchase_households_constant1",
        "purchase_weeks_varying",
        *(
            f"{prefix}_{key}"
            for prefix in ("revenue", "trips")
            for key in (
                "zeros",
                "zero_share",
                "positive",
                "positive_share",
                "negative",
                "mean",
                "median",
                "max",
                "p90",
                "p95",
                "p99",
            )
        ),
        "log1p_revenue_finite_cells",
        *(f"revenue_household_total_{key}" for key in CONCENTRATION_KEYS),
        "revenue_largest_cell_share",
        "revenue_top01_cells_count",
        "revenue_top01_cells_share",
    ),
    "transition_support_counts": tuple(
        f"previous{a}_current{b}_count" for a in (0, 1) for b in (0, 1)
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def verify_inputs(root: Path = PROJECT_ROOT) -> dict:
    """Hashes/footers and exact inventories; no rebuilding of accepted artifacts."""
    contract_path = root / "outputs/tables/panels/contracts.json"
    document = json.loads(contract_path.read_text())
    verified = {
        "outputs/tables/panels/contracts.json": sha256(contract_path),
        "data/raw/manifest_raw.csv": sha256(root / "data/raw/manifest_raw.csv"),
    }
    for table in document["tables"]:
        path = root / table["parquet_path"]
        require(
            path.is_file() and sha256(path) == table["parquet_sha256"],
            f"accepted Parquet hash mismatch: {table['table']}",
        )
        require(
            pq.ParquetFile(path).metadata.num_rows == table["actual_row_count"],
            f"accepted footer row count: {table['table']}",
        )
        verified[table["parquet_path"]] = table["parquet_sha256"]
    for relative, expected in PHASE3_HASHES.items():
        require(sha256(root / relative) == expected, f"Phase 3 final drift: {relative}")
        verified[relative] = expected
    manifest_path = root / "data/raw/manifest_raw.csv"
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_paths = {row["relative_path"] for row in rows}
    dataset = root / "data/raw/dunnhumby_The-Complete-Journey"
    actual_paths = {
        str(p.relative_to(root / "data/raw"))
        for p in dataset.rglob("*")
        if p.is_file() and p.name != ".DS_Store"
    }
    require(
        actual_paths == expected_paths and len(rows) == len(expected_paths),
        "raw inventory mismatch",
    )
    for row in rows:
        require(
            not Path(row["relative_path"]).is_absolute()
            and ".." not in Path(row["relative_path"]).parts,
            "unsafe raw manifest path",
        )
        relative = "data/raw/" + row["relative_path"]
        path = root / relative
        require(
            path.stat().st_size == int(row["size_bytes"]) and sha256(path) == row["sha256"],
            f"raw integrity mismatch: {relative}",
        )
        verified[relative] = row["sha256"]
    return {
        "raw_files": len(rows),
        "parquet_artifacts": len(document["tables"]),
        "phase3_final_artifacts": len(PHASE3_HASHES),
        "hashes": verified,
    }


def ignored(root: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", relative], cwd=root, capture_output=True
    )
    require(result.returncode in (0, 1), "git ignore probe failed")
    return result.returncode == 0


PHASE2_PANEL_OUTPUTS = (
    "contracts.json",
    "provenance.json",
    "run_metadata.json",
    "sample_flag_counts.csv",
    "predictive_feature_inventory.csv",
    "qa_summary.csv",
    "panel_eligibility_support_bucket_by_early_late.csv",
    "panel_eligibility_gap_thresholds.csv",
    "panel_eligibility_active_weeks_distribution.csv",
    "panel_eligibility_halves_by_active_weeks_thresholds.csv",
    "panel_eligibility_active_weeks_thresholds.csv",
    "panel_eligibility_thirds_by_active_weeks_thresholds.csv",
    "panel_eligibility_support_halves_crosstab.csv",
    "panel_eligibility_support_distributions.csv",
    "panel_eligibility_observed_span_by_active_weeks.csv",
)


RUN_D_PATHS = (
    "outputs/tables/panels/exploratory/phase4_run_d_sensitivity_matrix.csv",
    "outputs/tables/panels/exploratory/phase4_run_d_diagnostics.csv",
    "outputs/figures/panels/exploratory/phase4_run_d_sensitivity_coefficients.png",
    "outputs/figures/panels/exploratory/phase4_run_d_covariance_comparison.png",
)

RUN_C_PATHS = (
    "outputs/tables/panels/exploratory/phase4_run_c_core_results.csv",
    "outputs/tables/panels/exploratory/phase4_run_c_diagnostics.csv",
    "outputs/figures/panels/exploratory/phase4_run_c_adjustment_ladder.png",
)


def assert_policy(root: Path = PROJECT_ROOT) -> None:
    register = root / "docs/phases/PHASE_04_RESULT_REGISTER.csv"
    require(register.is_file() or root != PROJECT_ROOT, "researcher promotion register missing")
    if register.exists():
        with register.open(newline="") as handle:
            reviewed = list(csv.DictReader(handle))
        from complete_journey.econometrics.final_contract import (
            APPROVED_WORDING,
            RESULT_STATUSES,
            phase4_ledger_prefix,
        )

        require(
            len(reviewed) == len(RESULT_STATUSES)
            and {r["result_id"] for r in reviewed} == set(RESULT_STATUSES),
            "unapproved promoted result register universe",
        )
        require(
            {r["result_id"]: r["status"] for r in reviewed} == RESULT_STATUSES,
            "result promoted or demoted contrary to researcher approval",
        )
        require(
            all(
                r.get("statement") == APPROVED_WORDING[r["result_id"]]
                and all(
                    r.get(k) for k in ("evidence", "interpretation_boundary", "researcher_notes")
                )
                for r in reviewed
            ),
            "approved result wording/evidence/boundary/promotion note missing",
        )
        require(
            all(
                set(r)
                == {
                    "result_id",
                    "outcome",
                    "statement",
                    "evidence",
                    "diagnostic_status",
                    "interpretation_boundary",
                    "status",
                    "researcher_notes",
                }
                for r in reviewed
            ),
            "promotion register schema/identifier mismatch",
        )
        ledger = root / "docs/DECISIONS.md"
        require(ledger.is_file(), "P4-003 decision ledger missing")
        phase4_ledger_prefix(ledger.read_bytes())
    from complete_journey.econometrics.final_contract import FINAL_PATHS

    for relative in (*OUTPUTS.values(), *FINAL_PATHS):
        require(not ignored(root, relative), f"approved aggregate not allowlisted: {relative}")
    for relative in (
        "outputs/models/panel/model.pkl",
        "outputs/figures/panel/result.png",
        "outputs/tables/panel/results.csv",
        "outputs/tables/panels/phase4_customer_rows.csv",
        "outputs/diagnostics/panel/customer_rows.csv",
        "data/warehouse/panel/customer_rows.csv",
    ):
        require(
            ignored(root, relative), f"unauthorized result/customer artifact trackable: {relative}"
        )
    for relative in (*RUN_C_PATHS, *RUN_D_PATHS):
        require(
            ignored(root, relative), f"Run C exploratory artifact must stay ignored: {relative}"
        )
    allowed = (
        set(OUTPUTS.values())
        | set(RUN_C_PATHS)
        | set(RUN_D_PATHS)
        | set(FINAL_PATHS)
        | {"outputs/tables/panels/" + name for name in PHASE2_PANEL_OUTPUTS}
    )
    for tree in ("models", "figures", "tables"):
        for scope in ("panel", "panels"):
            directory = root / "outputs" / tree / scope
            for path in directory.rglob("*"):
                if path.is_file() and path.name != ".gitkeep":
                    require(
                        str(path.relative_to(root)) in allowed,
                        f"Run B must contain no model/result artifact: {path.relative_to(root)}",
                    )
    for path in (root / "outputs").rglob("phase4_*"):
        if path.is_file():
            require(
                str(path.relative_to(root))
                in set(OUTPUTS.values()) | set(RUN_C_PATHS) | set(RUN_D_PATHS) | set(FINAL_PATHS),
                "unapproved Phase 4 output artifact",
            )


def expected_keys() -> set[tuple[str, str, str]]:
    return {
        (spec.name, domain, metric)
        for spec in SAMPLES
        for domain, metrics in METRICS.items()
        if domain not in ("outcome", "transition_support_counts") or spec.name == "S0_baseline"
        for metric in metrics
    }


def unit_and_denominator(row: dict) -> tuple[str, str]:
    metric, domain = row["metric"], row["domain"]
    spec = next(s for s in SAMPLES if s.name == row["sample"])
    if "variance" in metric or "_ss" in metric and not metric.endswith("share"):
        unit = "squared_indicator_units"
    elif "difference" in metric or "tolerance" in metric:
        unit = "numerical_check_units"
    elif (
        "share" in metric
        or metric.endswith("mean")
        and domain == "history"
        or metric == "purchase_mean"
    ):
        unit = (
            "fraction"
            if "share" in metric or metric in ("mean", "purchase_mean")
            else "switches_per_household"
        )
    elif (
        domain == "outcome"
        and metric.startswith("revenue_")
        and metric.rsplit("_", 1)[-1] in ("mean", "median", "max", "p90", "p95", "p99")
    ):
        unit = "retailer_currency_units"
    elif (
        domain == "outcome"
        and metric.startswith("trips_")
        and metric.rsplit("_", 1)[-1] in ("mean", "median", "max", "p90", "p95", "p99")
    ):
        unit = "recorded_baskets"
    elif metric in ("twfe_mean", "twfe_min", "twfe_max"):
        unit = "residualized_indicator_units"
    else:
        unit = "count"
    basis = "declared household-week grid"
    if metric.endswith("_share"):
        if metric == "twfe_raw_centered_ss_share":
            basis = "total raw centered history SS"
        elif metric.endswith("contributor_share"):
            basis = (
                "all response weeks" if metric.startswith("week_") else "all included households"
            )
        elif metric.startswith("household_within_"):
            basis = "total household-demeaned history SS"
        elif metric.startswith(("household_twfe_", "week_twfe_")):
            basis = "total two-way-residualized history SS"
        elif metric.startswith("revenue_") and "zero" not in metric and "positive" not in metric:
            basis = "total unconditional retailer SALES_VALUE"
        elif metric == "switching_household_share":
            basis = "all included households"
        else:
            basis = "all declared household-week cells"
    elif "effective_count" in metric:
        basis = "inverse-Herfindahl of normalized contributions; not an inferential sample size"
    elif "variance" in metric:
        basis = "all declared household-week cells; population variance ddof=0"
    elif metric.startswith("switches_"):
        basis = "adjacent history-state changes per included household"
    denominator = f"{basis}; {spec.households} households; {spec.last_response - spec.first_response + 1} response weeks; {spec.cells} household-weeks"
    return unit, denominator


def summary_bytes(rows: list[dict]) -> bytes:
    keys = [(r["sample"], r["domain"], r["metric"]) for r in rows]
    require(
        len(keys) == len(set(keys)) and set(keys) == expected_keys(),
        "aggregate metric/key universe mismatch",
    )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda r: (r["sample"], r["domain"], r["metric"])):
        require(
            set(row) == {"sample", "domain", "metric", "value"},
            "unknown or identifier/result column",
        )
        require(
            isinstance(row["value"], (int, float)) and math.isfinite(row["value"]),
            "nonfinite/nonaggregate metric",
        )
        unit, denominator = unit_and_denominator(row)
        writer.writerow(
            {
                **row,
                "value": format(row["value"], ".17g"),
                "unit": unit,
                "denominator": denominator,
                "scope": SCOPE,
            }
        )
    payload = handle.getvalue().encode()
    reject_results(payload)
    return payload


def reject_results(payload: bytes) -> None:
    lowered = payload.lower()
    require(
        not any(word.encode() in lowered for word in FORBIDDEN),
        "identifier or model-result field in preflight payload",
    )


def source_identity(root: Path) -> dict:
    paths = sorted(
        [p for p in (root / "src/complete_journey/econometrics").glob("*.py")]
        + [
            root / "scripts/run_panel.py",
            root / "pyproject.toml",
            root / "uv.lock",
            root / ".gitignore",
            root / "docs/DECISIONS.md",
        ]
    )
    records = {str(p.relative_to(root)): sha256(p) for p in paths}
    return {"files": records, "state_sha256": hashlib.sha256(canonical_json(records)).hexdigest()}


def validate_metadata(value, expected, path: str = "metadata") -> None:
    """Closed recursive schema: exact keys, list shape, scalar types and literals."""
    require(type(value) is type(expected), f"preflight JSON schema type: {path}")
    if isinstance(expected, dict):
        require(value.keys() == expected.keys(), f"preflight JSON schema keys: {path}")
        for key in expected:
            validate_metadata(value[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        require(len(value) == len(expected), f"preflight JSON schema list: {path}")
        for index, (actual, declared) in enumerate(zip(value, expected, strict=True)):
            validate_metadata(actual, declared, f"{path}[{index}]")
    else:
        require(value == expected, f"preflight JSON schema literal: {path}")


def contract_schema() -> dict:
    """Declared aggregate/input and fixed toy metadata; no fitted numeric fields."""
    from complete_journey.econometrics.covariance import covariance_options
    from complete_journey.econometrics.samples import LAGS, sample_contract

    fits = []
    for entity, time, effects_df in (
        (False, False, 0),
        (False, True, 15),
        (True, False, 11),
        (True, True, 26),
    ):
        kinds = (
            ("two_way", "household", "dk4", "dk8") if entity and time else ("two_way", "household")
        )
        for kind in kinds:
            clusters = (
                {"entity": 12, "time": 16, "intersection": 192}
                if kind == "two_way"
                else {"entity": 12}
                if kind == "household"
                else {}
            )
            fits.append(
                {
                    "kind": kind,
                    "entity_effects": entity,
                    "time_effects": time,
                    "observations": 192,
                    "regressors_including_constant": 2,
                    "effects_df": effects_df,
                    "df_model": 2 + effects_df,
                    "df_resid": 190 - effects_df,
                    "cluster_counts": clusters,
                    "options": covariance_options(kind, effects=entity or time),
                    "covariance_implementation": "DriscollKraay"
                    if kind.startswith("dk")
                    else "ClusteredCovariance",
                    "constant_retained": True,
                    "constant_outcome_rows_retained": True,
                    "manual_covariance_passed": True,
                    "native_student_t_passed": True,
                    "native_variances_positive": True,
                    "covariance_repaired_or_clipped": False,
                    "full_covariance_positive_semidefinite": not (
                        entity and time and kind == "two_way"
                    ),
                }
            )
    software = {
        "evidence_status": "SYNTHETIC NON-EVIDENCE",
        "passed": True,
        "linearmodels_version": "7.0",
        "dk_status": "CLEANLY IMPLEMENTABLE",
        "fixture": {
            "source": "fixed internal synthetic toy only",
            "balanced": True,
            "observations": 192,
            "entities": 12,
            "periods": 16,
            "constant_outcome_entities": 2,
        },
        "fit_checks": fits,
        "singleton_checks": {
            "default_singletons_true_retains": True,
            "explicit_singletons_false_drops_recursive_two_core": True,
            "frozen_policy": "singletons=True; retain, do not silently delete",
            "observations_after_explicit_deletion": 192,
            "observations_before": 195,
            "observations_retained_by_default": 195,
            "recursive_drop_count": 3,
            "passed": True,
        },
        "absorbed_variable_checks": {
            "check_rank": True,
            "drop_absorbed": False,
            "passed": True,
            "refused_cases": ["entity_invariant", "time_invariant", "additive_two_way"],
        },
        "full_covariance_validity_review": {
            "requires_researcher_review": False,
            "researcher_decision": "P4-002",
            "indefinite_toy_covariance_count": 1,
            "status": "SCALAR_BOUNDARY_CLARIFIED_FULL_MATRIX_FLAGGED",
            "joint_tests_permitted": False,
            "repairs_permitted": False,
            "scalar_history_variance_rule": "finite and strictly positive; stop otherwise",
            "description": "Native two-way TWFE full covariance is indefinite on the toy, despite positive "
            "native diagonal variances and exact manual agreement. No repair or clipping. "
            "P4-002 permits native scalar history inference only; no joint tests or "
            "matrix repair. Run C still requires separate researcher authorization.",
        },
        "inference_reference": "native Student t(df_resid), not cluster count minus one",
        "cluster_normalization": "n/(n-k-extra_df); g/(g-1)*(n-1)/n separately for each dimension and intersection",
        "intercept_normalization": "balanced FE residualization plus restored grand means; explicit constant retained",
        "dk_normalization": "time-summed scores; Bartlett weight 1-j/(bandwidth+1); n/(n-k-extra_df)",
        "checks": dict.fromkeys(
            (
                "binary_outcome_constructor",
                "manual_cluster_covariance",
                "manual_dk_covariance",
                "actual_cluster_counts",
                "effects_df_counted_explicitly",
                "native_student_t_df_resid",
                "constant_retention",
                "singleton_semantics",
                "absorbed_variable_refusal",
                "same_solution_and_rows_across_covariances",
            ),
            True,
        ),
    }
    return {
        "scope": SCOPE,
        "protocol": "P4-001-RUN-B",
        "samples": sample_contract(),
        "structural_gate": "PASSED",
        "concentration_gate": "AGGREGATE_SUPPORT_FOR_RESEARCHER_REVIEW_NO_POSTHOC_CUTOFF",
        "input_checks": {
            "households": 2500,
            "rows": 255000,
            "weeks": 102,
            "integral_values_verified_without_integer_cast": [
                "units",
                "qa_quantity_zero_lines",
                "qa_retail_disc_positive_lines",
                "qa_implied_paid_negative_lines",
                "lag1_units",
                "trips",
                "lag1_trips",
            ],
            "populations": {
                "sample_full": 2500,
                "sample_sustained_support": 2115,
                "sample_higher_support": 1807,
            },
            "lag_checks": {
                key: {"pairs": 252500, "first_week_nulls": 2500, "max_abs_difference": 0.0}
                for key in LAGS
            },
        },
        "covariance_options": {
            kind: covariance_options(kind) for kind in ("two_way", "household", "dk4", "dk8")
        },
        "synthetic_software_checks": software,
        "synthetic_influence_checks": dict.fromkeys(
            (
                "additive_contributions_reconcile",
                "deletion_matches_dummy_refit",
                "fixed_projection_is_not_refit",
                "household_denominators_reconcile",
                "partial_leverage_sums_to_one",
                "synthetic_only",
            ),
            True,
        ),
    }


def validate_contract(contract: dict) -> None:
    validate_metadata(contract, contract_schema(), "contract")


def validate_provenance(provenance: dict, root: Path, *, allow_prior_source: bool = False) -> None:
    """Closed provenance shape; only exact source/input paths and SHA-256 leaves."""
    require(
        type(provenance) is dict
        and set(provenance) == {"scope", "protocol", "source", "inputs", "outputs"},
        "preflight JSON schema provenance keys",
    )
    validate_metadata(provenance["scope"], SCOPE, "provenance.scope")
    validate_metadata(provenance["protocol"], "P4-001-RUN-B", "provenance.protocol")
    source = provenance["source"]
    inputs = provenance["inputs"]
    require(
        type(source) is dict and set(source) == {"files", "state_sha256"},
        "preflight JSON schema source",
    )
    require(
        type(inputs) is dict
        and set(inputs) == {"raw_files", "parquet_artifacts", "phase3_final_artifacts", "hashes"},
        "preflight JSON schema inputs",
    )
    for key, count in (("raw_files", 9), ("parquet_artifacts", 12), ("phase3_final_artifacts", 12)):
        validate_metadata(inputs[key], count, "inputs." + key)
    contract = json.loads((root / "outputs/tables/panels/contracts.json").read_text())
    with (root / "data/raw/manifest_raw.csv").open(newline="") as handle:
        raw_paths = {"data/raw/" + row["relative_path"] for row in csv.DictReader(handle)}
    input_paths = (
        raw_paths
        | set(PHASE3_HASHES)
        | {table["parquet_path"] for table in contract["tables"]}
        | {"outputs/tables/panels/contracts.json", "data/raw/manifest_raw.csv"}
    )
    source_paths = set(source_identity(root)["files"])
    if allow_prior_source:
        for revision in (
            "78cb04b1d2aea99df5e346ab26f402e3d1c53706",
            "b28148f1b77aef31c57f616aadfe66457b868f19",
            "1f22d89a54b4582fb36dafeeac808f99af4a1ed9",
        ):
            historical = json.loads(
                subprocess.check_output(
                    ["git", "show", revision + ":" + OUTPUTS["provenance"]], cwd=root
                )
            )["source"]
            if source == historical:
                source_paths = set(historical["files"])
                break
    for mapping, paths in (
        (source["files"], source_paths),
        (inputs["hashes"], input_paths),
        (provenance["outputs"], {OUTPUTS["summary"], OUTPUTS["contract"]}),
    ):
        require(type(mapping) is dict and set(mapping) == paths, "preflight JSON schema hash paths")
        for digest in mapping.values():
            validate_digest(digest)
    validate_digest(source["state_sha256"])
    require(
        source["state_sha256"] == hashlib.sha256(canonical_json(source["files"])).hexdigest(),
        "preflight JSON schema source digest",
    )


def validate_digest(value) -> None:
    require(
        type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "preflight JSON schema SHA-256 leaf",
    )


def publish_preflight(
    rows: list[dict], contract: dict, inputs: dict, root: Path = PROJECT_ROOT
) -> dict:
    """Only three exact aggregate paths; validate every payload before any write."""
    assert_policy(root)
    source_before = source_identity(root)
    summary = summary_bytes(rows)
    require(summary == summary_bytes(rows), "nondeterministic aggregate serialization")
    contract_payload = canonical_json(contract)
    reject_results(contract_payload)
    validate_contract(contract)
    provenance = {
        "scope": SCOPE,
        "protocol": "P4-001-RUN-B",
        "source": source_before,
        "inputs": inputs,
        "outputs": {
            OUTPUTS["summary"]: hashlib.sha256(summary).hexdigest(),
            OUTPUTS["contract"]: hashlib.sha256(contract_payload).hexdigest(),
        },
    }
    provenance_payload = canonical_json(provenance)
    reject_results(provenance_payload)
    validate_provenance(provenance, root)
    require(source_before == source_identity(root), "source changed during preflight")
    for key, payload in (
        ("summary", summary),
        ("contract", contract_payload),
        ("provenance", provenance_payload),
    ):
        path = root / OUTPUTS[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        if key in ("summary", "contract") and path.exists():
            require(
                path.read_bytes() == payload, "accepted Run B evidence differs: STOP, no rewrite"
            )
        else:
            path.write_bytes(payload)
        require(path.read_bytes() == payload, "published bytes differ")
    return {key: sha256(root / path) for key, path in OUTPUTS.items()}


def validate_existing_summary(
    root: Path = PROJECT_ROOT, *, require_current_source: bool = True, inputs: dict | None = None
) -> None:
    path = root / OUTPUTS["summary"]
    present = [(root / relative).exists() for relative in OUTPUTS.values()]
    require(not any(present) or all(present), "partial preflight generation")
    if not path.exists():
        return
    reject_results(path.read_bytes())
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == FIELDS, "tracked preflight schema drift")
        rows = list(reader)
    restored = [
        {
            k: float(row[k]) if k == "value" else row[k]
            for k in ("sample", "domain", "metric", "value")
        }
        for row in rows
    ]
    require(summary_bytes(restored) == path.read_bytes(), "preflight content/semantics drift")
    contract_path, provenance_path = root / OUTPUTS["contract"], root / OUTPUTS["provenance"]
    require(contract_path.is_file() and provenance_path.is_file(), "partial preflight generation")
    for artifact in (contract_path, provenance_path):
        reject_results(artifact.read_bytes())
    contract = json.loads(contract_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    validate_contract(contract)
    validate_provenance(provenance, root, allow_prior_source=True)
    require(contract["scope"] == SCOPE and provenance["scope"] == SCOPE, "preflight scope drift")
    require(
        contract["protocol"] == provenance["protocol"] == "P4-001-RUN-B", "preflight protocol drift"
    )
    from complete_journey.econometrics.samples import sample_contract

    require(contract["samples"] == sample_contract(), "frozen sample contract drift")
    for kind in ("summary", "contract"):
        require(
            provenance["outputs"][OUTPUTS[kind]] == sha256(root / OUTPUTS[kind]),
            "preflight output/provenance hash mismatch",
        )
    if require_current_source:
        from complete_journey.econometrics.archives import (
            ACCEPTED_PREFLIGHT_PROVENANCE,
            accepted_source,
        )

        expected_source = source_identity(root)
        if sha256(provenance_path) == ACCEPTED_PREFLIGHT_PROVENANCE:
            expected_source = accepted_source(root)
        require(
            provenance["source"] == expected_source,
            "preflight source drift; accepted historical provenance must not be rewritten",
        )
        if inputs is not None:
            require(
                provenance["inputs"] == inputs,
                "preflight input identity drift; regenerate with --preflight",
            )
