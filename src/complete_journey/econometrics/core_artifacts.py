"""Three exact Git-ignored Run C artifacts. No final output or model serialization."""

from __future__ import annotations

import csv
import io
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from complete_journey.econometrics.artifacts import (
    OUTPUTS,
    RUN_C_PATHS,
    assert_policy,
    sha256,
    source_identity,
)
from complete_journey.econometrics.core import CORE_KEYS, R2_DEFINITIONS, adjustment_changes
from complete_journey.econometrics.core_diagnostics import SCOPE
from complete_journey.econometrics.samples import require
from complete_journey.paths import PROJECT_ROOT

ACCEPTED_PREFLIGHT = {
    "summary": "0297ead4f887d72ae553b125accddd149f6c8948df6c10069f8f9863191b6625",
    "contract": "8ba9da8ab9cadb15965186fb4efa90c45d2aaae1b104bff7aa158e57a71a08b1",
}
RESULT_FIELDS = tuple(
    "outcome model_id sample status coefficient coefficient_percentage_points standard_error ci_lower ci_upper t_statistic p_value nobs households response_weeks household_fe week_fe covariance_type household_clusters week_clusters df_resid reference_distribution debiased group_debias auto_df count_effects focal_variance full_covariance_psd covariance_min_eigenvalue covariance_psd_tolerance r2 r2_within r2_between r2_overall r2_inclusive independent_coefficient independent_abs_difference independent_abs_tolerance independent_rel_tolerance row_key_sha256 source_state_sha256".split()
)
DIAGNOSTIC_FIELDS = ("outcome", "model_id", "diagnostic", "value", "unit", "denominator", "scope")


def verify_frozen_preflight(root=PROJECT_ROOT):
    for kind, expected in ACCEPTED_PREFLIGHT.items():
        require(
            sha256(root / OUTPUTS[kind]) == expected, f"accepted Run B {kind} bytes changed: STOP"
        )


def csv_bytes(rows, fields):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        require(set(row) == set(fields), "exploratory schema/identifier field mismatch")
        writer.writerow(
            {
                k: ""
                if v is None
                else format(float(v), ".17g")
                if isinstance(v, (float, np.floating))
                else v
                for k, v in row.items()
            }
        )
    payload = handle.getvalue().encode()
    require(
        b"household_key" not in payload.lower() and b"household_id" not in payload.lower(),
        "identifier leak in exploratory artifact",
    )
    return payload


RUN_C_COMMIT = "b28148f1b77aef31c57f616aadfe66457b868f19"
ACCEPTED_CORE_BYTES = (
    "cad9e708c607516e3e3b31452d64169c7a9eba429345e09262296fe6afc56ed0",
    "2da35e6fa2e8da9417b474788705dc0e408b62aee0f49c7ed5d9cf276c53ad13",
    "97be8434ed6cb2b1e115e7fa2395c55999992d9ccb2e1b18f5ffb19266dd7dfd",
)


def accepted_core_source(root=PROJECT_ROOT):
    """Retain historical source, permitting only exact researcher-authorized deltas."""
    from complete_journey.econometrics.archives import accepted_source

    return accepted_source(root, RUN_C_COMMIT)["state_sha256"]


def validate_results(rows, source):
    require(
        len(rows) == 9 and {(r["outcome"], r["model_id"]) for r in rows} == set(CORE_KEYS),
        "wrong or duplicate core-result universe",
    )
    require(len({r["row_key_sha256"] for r in rows}) == 1, "core rows differ")
    for row in rows:
        require(set(row) == set(RESULT_FIELDS), "result schema mismatch")
        require(
            row["status"] == "UNREVIEWED" and row["sample"] == "S0_baseline",
            "unauthorized result status/sample",
        )
        require(row["source_state_sha256"] == source, "result-generating source changed")
        require(
            tuple(
                int(row[k])
                for k in (
                    "nobs",
                    "households",
                    "response_weeks",
                    "household_clusters",
                    "week_clusters",
                )
            )
            == (252500, 2500, 101, 2500, 101),
            "wrong core sample/clusters",
        )
        numeric = [
            float(row[k])
            for k in (
                "coefficient",
                "standard_error",
                "ci_lower",
                "ci_upper",
                "t_statistic",
                "p_value",
                "focal_variance",
            )
        ]
        require(np.isfinite(numeric).all() and numeric[-1] > 0, "invalid scalar inference")
        require(
            float(row["independent_abs_difference"])
            <= float(row["independent_abs_tolerance"])
            + float(row["independent_rel_tolerance"]) * abs(float(row["independent_coefficient"])),
            "independent check failed",
        )


def figure_bytes(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, outcome, title, unit in zip(
        axes,
        ("retailer_purchase", "weekly_sales_value"),
        ("A. Recorded retailer participation", "B. Retailer SALES_VALUE"),
        ("Probability points", "Retailer currency units per household-week"),
        strict=True,
    ):
        selected = [r for r in rows if r["outcome"] == outcome]
        beta = np.array([r["coefficient"] for r in selected])
        lo, hi = (
            np.array([r["ci_lower"] for r in selected]),
            np.array([r["ci_upper"] for r in selected]),
        )
        ax.errorbar(range(4), beta, yerr=np.vstack([beta - lo, hi - beta]), fmt="o", capsize=5)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(range(4), ["M0\nPooled", "M1\nWeek FE", "M2\nHousehold FE", "M3\nTWFE"])
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Adjusted descriptive associations — UNREVIEWED\nNative 95% two-way-clustered intervals; S0 responses 2–102"
    )
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, metadata={"Software": "complete-journey Run C"})
    plt.close(fig)
    return buffer.getvalue()


def write_exploratory(rows, diagnostics, root=PROJECT_ROOT):
    import hashlib

    assert_policy(root)
    verify_frozen_preflight(root)
    source = source_identity(root)["state_sha256"]
    rows = [{**r, "source_state_sha256": source} for r in rows]
    validate_results(rows, source)
    results_payload = csv_bytes(rows, RESULT_FIELDS)
    image = figure_bytes(rows)
    for change in adjustment_changes(rows):
        for metric in ("change", "absolute_change", "relative_percent_change"):
            diagnostics.append(
                {
                    "outcome": change["outcome"],
                    "model_id": change["earlier"] + "->" + change["later"],
                    "diagnostic": metric,
                    "value": change[metric],
                    "unit": "percent" if metric.startswith("relative") else "native outcome units",
                    "denominator": "earlier-model coefficient for relative percent; identical core rows",
                    "scope": SCOPE,
                }
            )
    metadata = {
        "diagnostics_data_sha256": hashlib.sha256(
            csv_bytes(diagnostics, DIAGNOSTIC_FIELDS)
        ).hexdigest(),
        "source_state_sha256": source,
        "results_sha256": hashlib.sha256(results_payload).hexdigest(),
        "figure_sha256": hashlib.sha256(image).hexdigest(),
        "r2_definitions": json.dumps(R2_DEFINITIONS, sort_keys=True),
    }
    diagnostics.extend(
        {
            "outcome": "ALL",
            "model_id": "META",
            "diagnostic": k,
            "value": v,
            "unit": "metadata",
            "denominator": "exact generating source and local artifacts",
            "scope": SCOPE,
        }
        for k, v in metadata.items()
    )
    diagnostic_payload = csv_bytes(diagnostics, DIAGNOSTIC_FIELDS)
    require(
        source_identity(root)["state_sha256"] == source, "source changed during exploratory render"
    )
    # Every payload is constructed/validated before writing any artifact.
    for relative, payload in zip(
        RUN_C_PATHS, (results_payload, diagnostic_payload, image), strict=True
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        require(path.read_bytes() == payload, "exploratory bytes differ")
    return {p: sha256(root / p) for p in RUN_C_PATHS}


def validate_saved(root=PROJECT_ROOT):
    """Integrity/source checks allow recovery without automatically refitting models."""
    assert_policy(root)
    verify_frozen_preflight(root)
    present = [(root / p).is_file() for p in RUN_C_PATHS]
    require(not any(present) or all(present), "partial Run C generation: STOP")
    if not any(present):
        return None
    for relative, digest in zip(RUN_C_PATHS, ACCEPTED_CORE_BYTES, strict=True):
        require(sha256(root / relative) == digest, "accepted Run C artifact bytes changed: STOP")
    with (root / RUN_C_PATHS[0]).open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames) == RESULT_FIELDS, "saved result columns changed")
        rows = list(reader)
    source = accepted_core_source(root)
    validate_results(rows, source)
    with (root / RUN_C_PATHS[1]).open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames) == DIAGNOSTIC_FIELDS, "saved diagnostic columns changed")
        diagnostics = list(reader)
    keys = [(r["outcome"], r["model_id"], r["diagnostic"]) for r in diagnostics]
    require(
        len(keys) == len(set(keys)) and all(r["scope"] == SCOPE for r in diagnostics),
        "duplicate or wrong-scope diagnostics",
    )
    metadata = {r["diagnostic"]: r["value"] for r in diagnostics if r["model_id"] == "META"}
    require(
        set(metadata)
        == {
            "source_state_sha256",
            "results_sha256",
            "figure_sha256",
            "r2_definitions",
            "diagnostics_data_sha256",
        },
        "missing diagnostic audit metadata",
    )
    require(
        metadata["source_state_sha256"] == source,
        "saved diagnostics source changed",
    )
    require(
        metadata["results_sha256"] == sha256(root / RUN_C_PATHS[0])
        and metadata["figure_sha256"] == sha256(root / RUN_C_PATHS[2]),
        "saved result/figure bytes changed",
    )
    import hashlib

    data_rows = [r for r in diagnostics if r["model_id"] != "META"]
    require(
        metadata["diagnostics_data_sha256"]
        == hashlib.sha256(csv_bytes(data_rows, DIAGNOSTIC_FIELDS)).hexdigest(),
        "saved diagnostic values changed",
    )
    require(
        metadata["r2_definitions"] == json.dumps(R2_DEFINITIONS, sort_keys=True),
        "R2 definitions changed",
    )
    return rows
