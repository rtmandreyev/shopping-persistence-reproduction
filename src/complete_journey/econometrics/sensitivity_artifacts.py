"""Four exact ignored Run D outputs, independent of immutable Run C artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from complete_journey.econometrics import sensitivity
from complete_journey.econometrics.artifacts import (
    RUN_C_PATHS,
    RUN_D_PATHS,
    assert_policy,
    sha256,
    source_identity,
)
from complete_journey.econometrics.core_artifacts import (
    DIAGNOSTIC_FIELDS,
    csv_bytes,
    validate_saved,
    verify_frozen_preflight,
)
from complete_journey.econometrics.covariance import covariance_options
from complete_journey.econometrics.samples import SAMPLES, PanelContractError, require
from complete_journey.paths import PROJECT_ROOT

SCOPE = "RUN_D_EXPLORATORY_UNREVIEWED"
OUTCOMES = ("retailer_purchase", "weekly_sales_value", "trips")
VARIANTS = ("baseline", "S1", "S2", "full_week_pair", "post_formation", "household", "dk4", "dk8")
EXPECTED_KEYS = {(job.outcome, job.variant) for job in sensitivity.JOBS}
MATRIX_FIELDS = frozenset(
    "outcome check_type variant population response_first response_last response_weeks households nobs transformation covariance coefficient standard_error ci_lower ci_upper t_statistic p_value df_resid reference_distribution household_clusters week_clusters debiased group_debias auto_df count_effects focal_variance full_covariance_psd covariance_min_eigenvalue covariance_psd_tolerance absolute_coefficient_change signed_coefficient_change relative_percent_change se_ratio ci_width_ratio sign_stable independent_coefficient independent_abs_difference manual_focal_variance manual_variance_abs_difference interpretation_note status row_key_sha256 model_id sample covariance_type household_fe week_fe kernel bandwidth independent_abs_tolerance independent_rel_tolerance manual_variance_abs_tolerance manual_variance_rel_tolerance reference_baseline_coefficient reference_baseline_standard_error reference_baseline_ci_lower reference_baseline_ci_upper".split()
)
DEPENDENCE_METRICS = (
    "residual_adjacent_pooled_correlation",
    "aggregate_score_correlation_lag1",
    "aggregate_score_correlation_lag4",
    "aggregate_score_correlation_lag8",
)
METADATA_KEYS = (
    "source_state_sha256",
    "matrix_sha256",
    "sensitivity_figure_sha256",
    "covariance_figure_sha256",
    "diagnostics_data_sha256",
    "run_c_artifact_hashes",
)
DEPENDENCE_DENOMINATOR = "baseline S0; response weeks 2–102; same Run C diagnostic definition"
METADATA_DENOMINATOR = "current Run D source and immutable Run C outputs"


def _empty(value):
    return value is None or (type(value) is str and value == "")


def _number(value, field):
    require(not isinstance(value, (bool, np.bool_)), f"invalid Run D numeric {field}")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PanelContractError(f"invalid Run D numeric {field}") from exc
    require(np.isfinite(parsed), f"nonfinite Run D numeric {field}")
    return parsed


def _flag(value, field):
    require(
        type(value) is bool or (type(value) is str and value in ("True", "False")),
        f"invalid Run D boolean {field}",
    )
    return value is True or value == "True"


def _hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_matrix(rows):
    """Authorize semantics, not just byte identity; never modify caller values."""
    require(len(rows) == len(EXPECTED_KEYS), "Run D closed result universe mismatch")
    require(
        all(set(row) == MATRIX_FIELDS for row in rows),
        "undeclared, missing or identifier sensitivity schema field",
    )
    require(
        {(r["outcome"], r["variant"]) for r in rows} == EXPECTED_KEYS,
        "Run D closed result universe mismatch",
    )
    jobs = {(job.outcome, job.variant): job for job in sensitivity.JOBS}
    baselines = {r["outcome"]: r for r in rows if r["variant"] == "baseline"}
    for row in rows:
        job = jobs[row["outcome"], row["variant"]]
        spec = SAMPLES[job.sample_index]
        weeks = spec.last_response - spec.first_response + 1
        options = covariance_options(job.covariance, effects=True)
        expected_text = {
            "check_type": job.check_type,
            "model_id": job.model_id,
            "sample": spec.name,
            "population": spec.population,
            "transformation": job.transformation,
            "covariance": job.covariance,
            "covariance_type": sensitivity.COVARIANCE_LABELS[job.covariance],
            "reference_distribution": "native Student t(df_resid)",
            "status": "UNREVIEWED",
        }
        for field, expected in expected_text.items():
            require(row[field] == expected, f"Run D specification {field} mismatch")
        expected_counts = {
            "response_first": spec.first_response,
            "response_last": spec.last_response,
            "response_weeks": weeks,
            "households": spec.households,
            "nobs": spec.cells,
            "df_resid": spec.cells - spec.households - weeks,
            "household_clusters": spec.households if options.get("cluster_entity") else None,
            "week_clusters": weeks if options.get("cluster_time") else None,
            "bandwidth": options.get("bandwidth"),
        }
        for field, expected in expected_counts.items():
            require(
                _empty(row[field]) if expected is None else _number(row[field], field) == expected,
                f"Run D specification {field} mismatch",
            )
        kernel = options.get("kernel")
        require(
            _empty(row["kernel"]) if kernel is None else row["kernel"] == kernel,
            "Run D specification kernel mismatch",
        )
        for field in (
            "household_fe",
            "week_fe",
            "debiased",
            "auto_df",
            "count_effects",
            "group_debias",
        ):
            expected = True if field in ("household_fe", "week_fe") else options.get(field)
            require(
                _empty(row[field]) if expected is None else _flag(row[field], field) is expected,
                f"Run D specification {field} mismatch",
            )
        for field, expected in (
            ("independent_abs_tolerance", sensitivity.ABS_TOL),
            ("independent_rel_tolerance", sensitivity.REL_TOL),
            ("manual_variance_abs_tolerance", sensitivity.MANUAL_ABS_TOL),
            ("manual_variance_rel_tolerance", sensitivity.MANUAL_REL_TOL),
        ):
            require(_number(row[field], field) == expected, f"Run D frozen {field} mismatch")
        numeric = {
            field: _number(row[field], field)
            for field in (
                "coefficient",
                "standard_error",
                "ci_lower",
                "ci_upper",
                "t_statistic",
                "p_value",
                "focal_variance",
                "independent_coefficient",
                "independent_abs_difference",
                "manual_focal_variance",
                "manual_variance_abs_difference",
                "covariance_min_eigenvalue",
                "covariance_psd_tolerance",
            )
        }
        beta, se = numeric["coefficient"], numeric["standard_error"]
        require(
            se > 0
            and numeric["ci_upper"] > numeric["ci_lower"]
            and numeric["focal_variance"] > 0
            and numeric["manual_focal_variance"] > 0,
            "Run D scalar uncertainty and native/manual focal variance must be positive finite",
        )
        require(0 <= numeric["p_value"] <= 1, "Run D p-value outside probability bounds")
        for actual, independent, difference, manual in (
            ("coefficient", "independent_coefficient", "independent_abs_difference", False),
            ("focal_variance", "manual_focal_variance", "manual_variance_abs_difference", True),
        ):
            require(
                numeric[difference] >= 0
                and numeric[difference] == abs(numeric[actual] - numeric[independent])
                and sensitivity._close(numeric[actual], numeric[independent], manual=manual),
                f"Run D {difference} reconciliation mismatch",
            )
        df = expected_counts["df_resid"]
        require(df > 0, "Run D nonpositive residual degrees of freedom")
        quantile = stats.t.ppf(0.975, df)
        for field, expected in (
            ("standard_error", np.sqrt(numeric["focal_variance"])),
            ("ci_lower", beta - quantile * se),
            ("ci_upper", beta + quantile * se),
            ("t_statistic", beta / se),
            # Preserve native CDF rounding to zero, not an invented tail value.
            ("p_value", 2 * (1 - stats.t.cdf(abs(beta / se), df))),
        ):
            require(sensitivity._close(numeric[field], expected), f"Run D native {field} mismatch")
        tolerance = numeric["covariance_psd_tolerance"]
        require(
            tolerance > 0
            and _flag(row["full_covariance_psd"], "full_covariance_psd")
            == (numeric["covariance_min_eigenvalue"] >= -tolerance),
            "Run D covariance PSD flag mismatch; no repair permitted",
        )
        require(_hash(row["row_key_sha256"]), "invalid Run D row key hash")
        require(
            type(row["interpretation_note"]) is str and bool(row["interpretation_note"]),
            "missing Run D interpretation note",
        )
        baseline = baselines[row["outcome"]]
        reference = {}
        for field in ("coefficient", "standard_error", "ci_lower", "ci_upper"):
            reference[field] = _number(baseline[field], field)
            require(
                _number(row[f"reference_baseline_{field}"], f"reference_baseline_{field}")
                == reference[field],
                f"Run D baseline reference {field} mismatch",
            )
        if job.sample_index == 0:
            require(
                row["row_key_sha256"] == baseline["row_key_sha256"],
                "Run D baseline row key changed",
            )
        if job.check_type == "covariance":
            require(
                sensitivity._close(beta, reference["coefficient"]),
                "Run D covariance-only point estimate changed",
            )
        require(
            _flag(row["sign_stable"], "sign_stable")
            == bool(np.sign(beta) == np.sign(reference["coefficient"])),
            "Run D sign comparison mismatch",
        )
        delta = beta - reference["coefficient"]
        base_width = reference["ci_upper"] - reference["ci_lower"]
        require(
            reference["standard_error"] > 0 and base_width > 0, "invalid Run D baseline uncertainty"
        )
        comparisons = {
            "absolute_coefficient_change": abs(delta),
            "signed_coefficient_change": delta,
            "relative_percent_change": None
            if abs(reference["coefficient"]) <= sensitivity.ABS_TOL
            else 100 * delta / reference["coefficient"],
            "se_ratio": se / reference["standard_error"],
            "ci_width_ratio": (numeric["ci_upper"] - numeric["ci_lower"]) / base_width,
        }
        for field, expected in comparisons.items():
            if job.variant == "log1p" or expected is None:
                require(_empty(row[field]), f"Run D noncomparable {field} must be empty")
            else:
                require(
                    _number(row[field], field) == expected, f"Run D comparison {field} mismatch"
                )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate Run D metadata hash key")
        result[key] = value
    return result


def validate_diagnostics(rows, *, include_metadata=True):
    """Exact diagnostic schema and closed twelve-metric/six-metadata universe."""
    require(
        all(tuple(row) == DIAGNOSTIC_FIELDS for row in rows),
        "Run D exact diagnostic schema mismatch",
    )
    expected = {(o, "M3", metric) for o in OUTCOMES for metric in DEPENDENCE_METRICS}
    if include_metadata:
        expected |= {("ALL", "META", key) for key in METADATA_KEYS}
    require(
        len(rows) == len(expected)
        and {(r["outcome"], r["model_id"], r["diagnostic"]) for r in rows} == expected,
        "Run D closed diagnostic universe mismatch",
    )
    for row in rows:
        require(row["scope"] == SCOPE, "Run D diagnostic scope mismatch")
        if row["model_id"] == "M3":
            require(
                row["unit"] == "correlation" and row["denominator"] == DEPENDENCE_DENOMINATOR,
                "Run D diagnostic correlation metadata mismatch",
            )
            require(
                -1 <= _number(row["value"], "correlation") <= 1, "Run D correlation outside bounds"
            )
        else:
            require(
                row["unit"] == "metadata" and row["denominator"] == METADATA_DENOMINATOR,
                "Run D diagnostic metadata mismatch",
            )
            if row["diagnostic"] != "run_c_artifact_hashes":
                require(_hash(row["value"]), "invalid Run D metadata hash")
            else:
                try:
                    hashes = json.loads(row["value"], object_pairs_hook=_unique_object)
                except (TypeError, ValueError) as exc:
                    raise PanelContractError("invalid Run D core hash metadata JSON") from exc
                require(
                    type(hashes) is dict
                    and set(hashes) == set(RUN_C_PATHS)
                    and all(_hash(value) for value in hashes.values()),
                    "Run D exact core hash metadata mismatch",
                )


def figure(rows, covariance=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    variants = (
        ("baseline", "household", "dk4", "dk8")
        if covariance
        else ("baseline", "S1", "S2", "full_week_pair", "post_formation")
    )
    labels = (
        ("Two-way", "Household", "DK4", "DK8")
        if covariance
        else ("S0", "S1", "S2", "Full-week\npair", "Post-\nformation")
    )
    units = (
        "Probability points",
        "Retailer currency units / household-week",
        "Recorded baskets / household-week",
    )
    for ax, outcome, unit in zip(axes, OUTCOMES, units, strict=True):
        lookup = {r["variant"]: r for r in rows if r["outcome"] == outcome}
        selected = [lookup[v] for v in variants]
        beta = np.array([r["coefficient"] for r in selected], dtype=float)
        lo, hi = (
            np.array([r["ci_lower"] for r in selected], dtype=float),
            np.array([r["ci_upper"] for r in selected], dtype=float),
        )
        ax.errorbar(
            range(len(variants)), beta, yerr=np.vstack([beta - lo, hi - beta]), fmt="o", capsize=5
        )
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(range(len(variants)), labels)
        ax.set_title(outcome, loc="left")
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Adjusted descriptive associations — UNREVIEWED\n"
        + (
            "Same S0 TWFE coefficient; different native 95% intervals"
            if covariance
            else "Predeclared population/calendar checks; primary two-way-clustered 95% intervals"
        )
    )
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, metadata={"Software": "complete-journey Run D"})
    plt.close(fig)
    return buffer.getvalue()


def save(rows, dependence, root=PROJECT_ROOT):
    assert_policy(root)
    verify_frozen_preflight(root)
    require(validate_saved(root) is not None, "Run C baseline missing")
    validate_matrix(rows)
    require(root != PROJECT_ROOT, "accepted Run D outputs are immutable in Run E")
    source = source_identity(root)["state_sha256"]
    core_hashes = {p: sha256(root / p) for p in RUN_C_PATHS}
    matrix = csv_bytes(rows, tuple(rows[0]))
    require(
        set(dependence) == set(OUTCOMES)
        and all(set(metrics) == set(DEPENDENCE_METRICS) for metrics in dependence.values()),
        "Run D closed dependence universe mismatch",
    )
    diagnostics = [
        {
            "outcome": outcome,
            "model_id": "M3",
            "diagnostic": metric,
            "value": value,
            "unit": "correlation",
            "denominator": "baseline S0; response weeks 2–102; same Run C diagnostic definition",
            "scope": SCOPE,
        }
        for outcome, metrics in dependence.items()
        for metric, value in metrics.items()
    ]
    validate_diagnostics(diagnostics, include_metadata=False)
    sensitivity_image, covariance_image = figure(rows), figure(rows, covariance=True)
    metadata = {
        "source_state_sha256": source,
        "matrix_sha256": hashlib.sha256(matrix).hexdigest(),
        "sensitivity_figure_sha256": hashlib.sha256(sensitivity_image).hexdigest(),
        "covariance_figure_sha256": hashlib.sha256(covariance_image).hexdigest(),
        "diagnostics_data_sha256": hashlib.sha256(
            csv_bytes(diagnostics, DIAGNOSTIC_FIELDS)
        ).hexdigest(),
        "run_c_artifact_hashes": json.dumps(core_hashes, sort_keys=True),
    }
    diagnostics.extend(
        {
            "outcome": "ALL",
            "model_id": "META",
            "diagnostic": key,
            "value": value,
            "unit": "metadata",
            "denominator": "current Run D source and immutable Run C outputs",
            "scope": SCOPE,
        }
        for key, value in metadata.items()
    )
    validate_diagnostics(diagnostics)
    payloads = (
        matrix,
        csv_bytes(diagnostics, DIAGNOSTIC_FIELDS),
        sensitivity_image,
        covariance_image,
    )
    require(
        source == source_identity(root)["state_sha256"]
        and core_hashes == {p: sha256(root / p) for p in RUN_C_PATHS},
        "source or Run C artifacts changed during Run D render",
    )
    for relative, payload in zip(RUN_D_PATHS, payloads, strict=True):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        require(path.read_bytes() == payload, "Run D output bytes differ")
    return {p: sha256(root / p) for p in RUN_D_PATHS}


def validate_existing(root=PROJECT_ROOT):
    assert_policy(root)
    verify_frozen_preflight(root)
    present = [(root / p).is_file() for p in RUN_D_PATHS]
    require(not any(present) or all(present), "partial Run D output generation")
    if not any(present):
        return None
    from complete_journey.econometrics.archives import ACCEPTED_D_HASHES, accepted_source

    for relative, digest in zip(RUN_D_PATHS, ACCEPTED_D_HASHES, strict=True):
        require(sha256(root / relative) == digest, "accepted Run D artifact bytes changed")
    with (root / RUN_D_PATHS[0]).open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames is not None
            and len(reader.fieldnames) == len(MATRIX_FIELDS)
            and set(reader.fieldnames) == MATRIX_FIELDS,
            "Run D exact matrix CSV schema mismatch",
        )
        rows = list(reader)
    validate_matrix(rows)
    with (root / RUN_D_PATHS[1]).open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames is not None and tuple(reader.fieldnames) == DIAGNOSTIC_FIELDS,
            "Run D exact diagnostic CSV schema mismatch",
        )
        diagnostics = list(reader)
    validate_diagnostics(diagnostics)
    meta = {r["diagnostic"]: r["value"] for r in diagnostics if r["model_id"] == "META"}
    require(
        meta["source_state_sha256"] == accepted_source(root)["state_sha256"],
        "stale Run D generating source",
    )
    for metric, relative in zip(
        ("matrix_sha256", "sensitivity_figure_sha256", "covariance_figure_sha256"),
        (RUN_D_PATHS[0], RUN_D_PATHS[2], RUN_D_PATHS[3]),
        strict=True,
    ):
        require(meta[metric] == sha256(root / relative), "Run D matrix/figure drift")
    data = [r for r in diagnostics if r["model_id"] != "META"]
    require(
        meta["diagnostics_data_sha256"]
        == hashlib.sha256(csv_bytes(data, DIAGNOSTIC_FIELDS)).hexdigest(),
        "Run D diagnostic data drift",
    )
    require(
        json.loads(meta["run_c_artifact_hashes"]) == {p: sha256(root / p) for p in RUN_C_PATHS},
        "Run C artifact bytes drifted",
    )
    return rows
