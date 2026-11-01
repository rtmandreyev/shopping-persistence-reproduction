"""Single frozen Run C reveal, ignored aggregate archive, and verified read-only reuse."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from complete_journey.choice import artifacts
from complete_journey.choice.contracts import (
    BASELINE,
    CORE_OUTPUTS,
    DESIGN_COLUMNS,
    LOCAL_PATHS,
    PREPARATION_EVIDENCE_SHA256,
    PREPARATION_VALIDATION_PATH,
    require,
)
from complete_journey.paths import PROJECT_ROOT

SCOPE = "RUN_C_UNPROMOTED_CORE"
SPECIFICATION = {
    "population": "baseline_pure_SPAGHETTI_DRY",
    "history_reference": "HN",
    "models": ["M0", "M1"],
    "covariance": "household_week",
    "equal_event_weights": True,
    "sensitivities_run": False,
    "joint_test_authorized": False,
    "covariance_repair": False,
}
REPORT_KEYS = {
    "scope",
    "status",
    "specification",
    "foundation",
    "results",
    "models",
    "comparison",
    "primary_diagnostics",
}
DIAGNOSTIC_FIELDS = ("domain", "metric", "value", "scope")
COVARIANCE_FIELDS = {
    "finite",
    "symmetric",
    "symmetry_max_absolute_difference",
    "eigenvalues",
    "minimum_eigenvalue",
    "maximum_eigenvalue",
    "negative_eigenvalues",
    "psd",
    "indefinite",
    "eigenvalue_tolerance",
    "component_counts",
    "component_corrections",
    "reference_df",
}


def covariance_summary(fit) -> dict:
    covariance = fit.covariance
    finite = bool(np.isfinite(covariance).all())
    symmetric = bool(np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12))
    values = np.linalg.eigvalsh((covariance + covariance.T) / 2) if finite and symmetric else None
    tolerance = (
        float(np.finfo(float).eps * len(covariance) * np.max(np.abs(values)))
        if values is not None
        else None
    )
    return {
        "finite": finite,
        "symmetric": symmetric,
        "symmetry_max_absolute_difference": float(np.max(np.abs(covariance - covariance.T)))
        if finite
        else None,
        "eigenvalues": values.tolist() if values is not None else [None] * len(covariance),
        "minimum_eigenvalue": float(values.min()) if values is not None else None,
        "maximum_eigenvalue": float(values.max()) if values is not None else None,
        "negative_eigenvalues": int((values < -tolerance).sum()) if values is not None else None,
        "psd": bool(values is not None and values.min() >= -tolerance),
        "indefinite": bool(
            values is not None and values.min() < -tolerance and values.max() > tolerance
        ),
        "eigenvalue_tolerance": tolerance,
        "component_counts": list(fit.component_counts),
        "component_corrections": list(fit.component_corrections),
        "reference_df": fit.reference_df,
    }


def comparison(rows: list[dict]) -> dict:
    indexed = {(r["model"], r["term"]): r for r in rows}
    result = {}
    for term in ("H0", "HB", "HP", "log1p_prior_pure"):
        before = indexed[("M0", term)]["coefficient"]
        after = indexed[("M1", term)]["coefficient"]
        delta = after - before
        result[term] = {
            "signed_change": delta,
            "absolute_change": abs(delta),
            "percentage_point_change": 100 * delta,
            "relative_change": delta / before if abs(before) > 1e-10 else None,
            "relative_denominator_minimum": 1e-10,
            "same_sign": bool(np.sign(before) == np.sign(after)),
        }
    return result


def validate_verification(summary: dict) -> None:
    from complete_journey.choice.core_diagnostics import _validate_closed

    terms = ("H0", "HB", "HP", "log1p_prior_pure")
    point = tuple(
        {
            "term": term,
            "main": float,
            "independent": float,
            "absolute_difference": float,
            "atol": 1e-10,
            "rtol": 1e-8,
            "agrees": True,
        }
        for term in terms
    )
    _validate_closed(summary["point_checks"], point)
    covariance = {
        "agrees": True,
        "maximum_absolute_difference": float,
        "atol": 1e-10,
        "rtol": 1e-8,
        "components": tuple(
            {
                "component": name,
                "sign": sign,
                "clusters": int,
                "correction": float,
                "repeated_cells": int,
                "events_in_repeated_cells": int,
            }
            for name, sign in (("household", 1), ("week", 1), ("household_week_intersection", -1))
        ),
        "focal": tuple(
            {
                "term": term,
                "main_variance": float,
                "independent_variance": float,
                "variance_absolute_difference": float,
                "variance_agrees": True,
                "se_agrees": bool | None,
                "main_se": float | None,
                "independent_se": float | None,
                "se_absolute_difference": float | None,
            }
            for term in terms
        ),
    }
    _validate_closed(summary["covariance_check"], covariance)
    statistics = {
        "sse": float,
        "centered_sst": float,
        "full_design_rank": int,
        "r_squared": float | None,
        "adjusted_r_squared": float | None,
        "r_squared_definition": "1-SSE/SST; SST=sum((Y-mean(Y))**2); total fitted values",
        "adjusted_r_squared_definition": "1-(1-R_squared)*(N-1)/(N-K); K=full_equivalent_design_rank",
        "constant_outcome_convention": "R_squared_and_adjusted_R_squared_None_when_centered_SST_zero",
    }
    _validate_closed(summary["fit_statistics"], statistics)
    cov = summary["covariance"]
    for name in (
        "symmetry_max_absolute_difference",
        "minimum_eigenvalue",
        "maximum_eigenvalue",
        "eigenvalue_tolerance",
    ):
        require(
            cov[name] is None or type(cov[name]) is float and np.isfinite(cov[name]),
            "covariance scalar metadata type",
        )
    require(
        all(v is None or type(v) is float and np.isfinite(v) for v in cov["eigenvalues"]),
        "covariance eigenvalue scalar metadata",
    )
    require(
        type(cov["component_corrections"]) is list
        and len(cov["component_corrections"]) == 3
        and all(type(v) is float and v > 0 for v in cov["component_corrections"]),
        "covariance correction metadata",
    )


def validate_report(report: dict) -> None:
    from complete_journey.choice.core import validate_result_rows
    from complete_journey.choice.core_diagnostics import (
        validate_foundation,
        validate_model_diagnostics,
        validate_primary,
    )

    require(type(report) is dict and set(report) == REPORT_KEYS, "closed core report keys")
    require(
        report["scope"] == SCOPE
        and report["status"] == "UNREVIEWED"
        and artifacts.canonical_json(report["specification"])
        == artifacts.canonical_json(SPECIFICATION),
        "core scope/specification/promotion refusal",
    )
    validate_result_rows(report["results"])
    rows = report["results"]
    require(
        len(rows) == 8
        and {(r["model"], r["term"]) for r in rows}
        == {(m, t) for m in ("M0", "M1") for t in ("H0", "HB", "HP", "log1p_prior_pure")},
        "core result universe",
    )
    require(
        all(
            (r["nobs"], r["households"], r["stores"], r["weeks"]) == (6544, 1560, 163, 102)
            for r in rows
        ),
        "frozen core sample metadata",
    )
    validate_foundation(report["foundation"])
    foundation = report["foundation"]
    require(
        (
            foundation["events"],
            foundation["households"],
            foundation["national"],
            foundation["private"],
        )
        == (6544, 1560, 3048, 3496),
        "foundation accepted sample mismatch",
    )
    require(
        [r["events"] for r in foundation["history_table"]] == [1585, 1433, 2064, 1462],
        "foundation accepted history mismatch",
    )
    require(
        type(report["models"]) is dict and set(report["models"]) == {"M0", "M1"},
        "core model universe",
    )
    for model, summary in report["models"].items():
        require(
            type(summary) is dict
            and set(summary)
            == {
                "covariance",
                "point_checks",
                "covariance_check",
                "fit_diagnostics",
                "fit_statistics",
            },
            "core model diagnostic keys",
        )
        cov = summary["covariance"]
        require(
            type(cov) is dict and set(cov) == COVARIANCE_FIELDS, "closed covariance diagnostics"
        )
        require(
            type(cov["eigenvalues"]) is list
            and len(cov["eigenvalues"]) == {"M0": 5, "M1": 169}[model],
            "native eigenvalue metadata shape",
        )
        require(
            cov["component_counts"] == [1560, 102, 6422] and cov["reference_df"] == 101,
            "frozen covariance cluster/reference metadata",
        )
        require(
            all(type(cov[k]) is bool for k in ("finite", "symmetric", "psd", "indefinite")),
            "covariance boolean metadata",
        )
        require(
            type(summary["point_checks"]) is list and len(summary["point_checks"]) == 4,
            "point verification shape",
        )
        validate_verification(summary)
        validate_model_diagnostics(summary["fit_diagnostics"])
    require(
        artifacts.canonical_json(report["comparison"])
        == artifacts.canonical_json(comparison(rows)),
        "context comparison drift",
    )
    validate_primary(report["primary_diagnostics"])
    # All declared nested validators refuse customer aliases; JSON also refuses nonfinite values.
    artifacts.canonical_json(report)


def result_csv(report: dict) -> bytes:
    from complete_journey.choice.core import RESULT_FIELDS

    validate_report(report)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["results"])
    return handle.getvalue().encode()


def _flatten(value, prefix="") -> dict:
    if type(value) is dict:
        return {
            p: v
            for k, child in value.items()
            for p, v in _flatten(child, f"{prefix}.{k}" if prefix else k).items()
        }
    if type(value) is list:
        return {
            p: v
            for i, child in enumerate(value, 1)
            for p, v in _flatten(child, f"{prefix}.{i}").items()
        }
    require(value is None or type(value) in (str, int, float, bool), "unknown diagnostic scalar")
    return {prefix: value}


def diagnostic_csv(report: dict) -> bytes:
    validate_report(report)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_FIELDS, lineterminator="\n")
    writer.writeheader()
    for domain in ("foundation", "models", "comparison", "primary_diagnostics"):
        for metric, value in sorted(_flatten(report[domain]).items()):
            writer.writerow({"domain": domain, "metric": metric, "value": value, "scope": SCOPE})
    return handle.getvalue().encode()


def covariance_archive(fits: dict) -> bytes:
    """Native full matrices, ordinal axes only; fixed ZIP metadata, no identifiers."""
    handle = io.BytesIO()
    with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for model in ("M0", "M1"):
            array = io.BytesIO()
            np.save(array, fits[model].covariance, allow_pickle=False)
            entry = zipfile.ZipInfo(f"{model}_covariance.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, array.getvalue())
    return handle.getvalue()


def figure_bytes(report: dict) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    validate_report(report)
    rows = {(r["model"], r["term"]): r for r in report["results"]}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = {"M0": "#5473b8", "M1": "#b65535"}
    for model, offset in (("M0", 0.13), ("M1", -0.13)):
        ys = np.arange(3) + offset
        values = [rows[(model, t)]["percentage_points"] for t in ("H0", "HB", "HP")]
        ax.scatter(
            values,
            ys,
            label=f"{model}"
            + (" main: store/calendar adjustment" if model == "M1" else " history/depth"),
            color=colors[model],
            s=45,
        )
        for y, value, term in zip(ys, values, ("H0", "HB", "HP"), strict=True):
            row = rows[(model, term)]
            if row["ci_lower"] is not None:
                ax.hlines(
                    y,
                    100 * row["ci_lower"],
                    100 * row["ci_upper"],
                    color=colors[model],
                    linewidth=1.5,
                )
            ax.annotate(
                f"{value:.2f}",
                (value, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color=colors[model],
            )
    ax.set_yticks(range(3), ["H0 vs HN", "HB vs HN", "HP vs HN"])
    ax.invert_yaxis()
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Linear-projection contrast (percentage points)")
    ax.set_title("Descriptive purchased-composition contrasts", loc="left", fontsize=13)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2, frameon=False, fontsize=8)
    any_gated = any(r["ci_lower"] is None for r in report["results"])
    note = (
        "Intervals withheld where covariance is indefinite/invalid; scalar inference is researcher-gated."
        if any_gated
        else "95% intervals use household × week CR1 and t(101); dependence review remains pending."
    )
    fig.text(
        0.12,
        0.055,
        "UNREVIEWED • Recorded pure-type SPAGHETTI DRY events • HN reference",
        fontsize=9,
    )
    fig.text(0.12, 0.025, note, fontsize=8)
    fig.subplots_adjust(bottom=0.28, left=0.14, right=0.95, top=0.86)
    handle = io.BytesIO()
    fig.savefig(
        handle,
        format="png",
        dpi=160,
        metadata={"Software": "complete_journey Phase 5 UNPROMOTED Run C"},
    )
    plt.close(fig)
    return handle.getvalue()


def read_cached(root: Path = PROJECT_ROOT, required: bool = False) -> dict | None:
    paths = {k: root / p for k, p in CORE_OUTPUTS.items()}
    if not any(p.exists() for p in paths.values()):
        require(not required, "Run C archive absent")
        return None
    require(
        all(p.is_file() for p in paths.values()), "partial Run C archive; do not silently refit"
    )
    artifacts.assert_policy(root)
    artifacts.verify_preparation_evidence(root)
    manifest = json.loads(paths["provenance"].read_bytes())
    require(
        type(manifest) is dict
        and set(manifest)
        == {
            "scope",
            "source",
            "inputs",
            "preparation_evidence",
            "preparation_provenance_sha256",
            "preparation_validation_sha256",
            "artifacts",
        },
        "closed core provenance",
    )
    require(
        manifest["scope"] == SCOPE,
        "Run C generating source changed; no silent empirical refit",
    )
    if manifest["source"] != artifacts.source_identity(root):
        # Synthetic/new archives retain the strict current-source route. Only all
        # seven accepted hashes admit historical generation under the pinned source.
        accepted = artifacts.verify_accepted_core(root)
        require(manifest == accepted["provenance"], "accepted Run C manifest changed during read")
    require(
        manifest["inputs"] == artifacts.verify_inputs(root)
        and manifest["preparation_evidence"] == PREPARATION_EVIDENCE_SHA256,
        "core input/evidence drift",
    )
    require(
        manifest["preparation_provenance_sha256"]
        == artifacts.sha256(root / "outputs/tables/choice/provenance.json"),
        "core preparation provenance mismatch",
    )
    require(
        manifest["preparation_validation_sha256"]
        == artifacts.sha256(root / PREPARATION_VALIDATION_PATH),
        "core preparation revalidation hash mismatch",
    )
    expected = {p for k, p in CORE_OUTPUTS.items() if k != "provenance"}
    require(
        type(manifest["artifacts"]) is dict and set(manifest["artifacts"]) == expected,
        "core artifact universe",
    )
    for relative, digest in manifest["artifacts"].items():
        require(artifacts.sha256(root / relative) == digest, "core artifact hash drift")
    report = json.loads(paths["report"].read_bytes())
    validate_report(report)
    require(
        paths["results"].read_bytes() == result_csv(report)
        and paths["diagnostics"].read_bytes() == diagnostic_csv(report),
        "core CSV schema/value drift",
    )
    with np.load(paths["covariance"], allow_pickle=False) as archive:
        require(
            set(archive.files) == {"M0_covariance", "M1_covariance"},
            "native covariance archive keys",
        )
        for model, k in (("M0", 5), ("M1", 169)):
            require(
                archive[f"{model}_covariance"].shape == (k, k), "native covariance archive shape"
            )
    return report


def run_core(root: Path = PROJECT_ROOT) -> tuple[dict, bool]:
    """No sample/specification/covariance switches; existing valid archive is reused."""
    from complete_journey.choice.core import fit_projection, result_rows
    from complete_journey.choice.core_diagnostics import foundation, model_diagnostics, primary
    from complete_journey.choice.preflight import check, reconcile_samples

    check(root)
    cached = read_cached(root)
    if cached is not None:
        return cached, True
    identity = artifacts.source_identity(root)
    inputs = artifacts.verify_inputs(root)
    pure = pd.read_parquet(root / LOCAL_PATHS["pure"])
    require(reconcile_samples(pure)["baseline"] == BASELINE, "frozen core baseline drift")
    factual = foundation(pure)  # Descriptive foundation precedes any model fit.
    frame = pure.loc[:, list(DESIGN_COLUMNS)].copy()
    outcome = pure.private_only.to_numpy(dtype=float)
    fits = {model: fit_projection(frame, outcome, model) for model in ("M0", "M1")}
    rows = [row for model in ("M0", "M1") for row in result_rows(fits[model])]
    report = {
        "scope": SCOPE,
        "status": "UNREVIEWED",
        "specification": SPECIFICATION,
        "foundation": factual,
        "results": rows,
        "models": {
            m: {
                "covariance": covariance_summary(f),
                "point_checks": f.point_checks,
                "covariance_check": f.covariance_check,
                "fit_diagnostics": model_diagnostics(frame, f),
                "fit_statistics": f.covariance_diagnostics["fit_statistics"],
            }
            for m, f in fits.items()
        },
        "comparison": comparison(rows),
        "primary_diagnostics": primary(frame, fits["M1"]),
    }
    validate_report(report)
    payloads = {
        CORE_OUTPUTS["results"]: result_csv(report),
        CORE_OUTPUTS["diagnostics"]: diagnostic_csv(report),
        CORE_OUTPUTS["report"]: artifacts.canonical_json(report),
        CORE_OUTPUTS["covariance"]: covariance_archive(fits),
        CORE_OUTPUTS["figure"]: figure_bytes(report),
    }
    require(
        artifacts.source_identity(root) == identity and artifacts.verify_inputs(root) == inputs,
        "core source/input moved during computation",
    )
    artifacts.verify_preparation_evidence(root)
    artifacts.assert_policy(root)
    manifest = {
        "scope": SCOPE,
        "source": identity,
        "inputs": inputs,
        "preparation_evidence": PREPARATION_EVIDENCE_SHA256,
        "preparation_provenance_sha256": artifacts.sha256(
            root / "outputs/tables/choice/provenance.json"
        ),
        "preparation_validation_sha256": artifacts.sha256(root / PREPARATION_VALIDATION_PATH),
        "artifacts": {p: hashlib.sha256(b).hexdigest() for p, b in payloads.items()},
    }
    payloads[CORE_OUTPUTS["provenance"]] = artifacts.canonical_json(manifest)
    # The entire scope is validated before any final local file is replaced.
    require(set(payloads) == set(CORE_OUTPUTS.values()), "Run C writer path universe")
    scratch = tempfile.gettempdir()  # honours TMPDIR; portable fallback (post-audit revision)
    with tempfile.TemporaryDirectory(prefix="phase5-core-", dir=scratch) as temp:
        staged = {}
        for i, (relative, payload) in enumerate(sorted(payloads.items())):
            p = Path(temp) / str(i)
            p.write_bytes(payload)
            staged[relative] = p
        for relative, p in staged.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(p, target)
    verified = read_cached(root, required=True)
    require(verified == report, "Run C exact readback failed")
    return verified, False
