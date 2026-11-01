"""Bounded Run D R1–R4; ignored aggregate outputs and strict read-only reuse."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from complete_journey.choice import artifacts
from complete_journey.choice.contracts import (
    ACCEPTED_RUN_C_SHA256,
    CORE_OUTPUTS,
    LOCAL_PATHS,
    RUN_D_OUTPUTS,
    require,
)
from complete_journey.paths import PROJECT_ROOT

SCOPE = "RUN_D_UNPROMOTED_ROBUSTNESS"
TABLE_FIELDS = (
    "variant",
    "term",
    "coefficient",
    "percentage_points",
    "native_se",
    "focal_variance",
    "ci_lower",
    "ci_upper",
    "t_statistic",
    "p_value",
    "inference_status",
    "covariance_psd",
    "covariance_indefinite",
    "reference_df",
)
COMPARISON_FIELDS = (*TABLE_FIELDS, "se_ratio_primary", "se_ratio_R3")


def csv_bytes(rows: list[dict], fields: tuple[str, ...]) -> bytes:
    require(all(set(row) == set(fields) for row in rows), "closed Run D CSV keys")
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


def sensitivity_rows(report: dict) -> list[dict]:
    return [
        row
        for variant in ("R1", "R2", "R3", "R4")
        for row in report["variants"][variant]["results"]
    ]


def covariance_rows(report: dict, baseline: dict) -> list[dict]:
    base = {r["term"]: r for r in baseline["results"] if r["model"] == "M1"}
    r3 = {r["term"]: r for r in report["variants"]["R3"]["results"]}
    rows = []
    for variant in ("PRIMARY", "R3", "R4"):
        selected = (
            list(base.values()) if variant == "PRIMARY" else report["variants"][variant]["results"]
        )
        for source in selected:
            row = {name: (variant if name == "variant" else source[name]) for name in TABLE_FIELDS}
            term = row["term"]
            require(row["coefficient"] == base[term]["coefficient"], "covariance-only point drift")
            for field, reference in (("se_ratio_primary", base), ("se_ratio_R3", r3)):
                denominator = reference[term]["native_se"]
                row[field] = (
                    row["native_se"] / denominator
                    if row["native_se"] is not None and denominator
                    else None
                )
            rows.append(row)
    require(len(rows) == 12, "baseline covariance comparison row count")
    return rows


def figure_bytes(report: dict, baseline: dict, comparison: bool) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = (
        covariance_rows(report, baseline)
        if comparison
        else [
            *[{**r, "variant": "BASELINE"} for r in baseline["results"] if r["model"] == "M1"],
            *report["variants"]["R2"]["results"],
        ]
    )
    variants = ("PRIMARY", "R3", "R4") if comparison else ("BASELINE", "R2")
    colors = ("#777777", "#2474ad", "#bc7730")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    gated = []
    for offset, variant in enumerate(variants):
        selected = {r["term"]: r for r in rows if r["variant"] == variant}
        label = {
            "PRIMARY": "Household × week",
            "R3": "Household × store",
            "R4": "Household only",
            "BASELINE": "Baseline all weeks",
            "R2": "Responses weeks 35–102",
        }[variant]
        blocked = any(selected[t]["ci_lower"] is None for t in ("H0", "HB", "HP"))
        if blocked:
            label += " — inference gated"
            gated.append(variant)
        positions = np.arange(3) + (offset - (len(variants) - 1) / 2) * 0.20
        values = [100 * selected[t]["coefficient"] for t in ("H0", "HB", "HP")]
        ax.scatter(
            values,
            positions,
            label=label,
            color=colors[offset],
            marker=("o", "s", "^")[offset],
            s=40,
        )
        for y, term in zip(positions, ("H0", "HB", "HP"), strict=True):
            row = selected[term]
            if row["ci_lower"] is not None:
                require(
                    row["inference_status"] == "PERMITTED" and row["covariance_psd"],
                    "gated figure interval refused",
                )
                ax.hlines(
                    y,
                    100 * row["ci_lower"],
                    100 * row["ci_upper"],
                    color=colors[offset],
                    linewidth=1.5,
                )
    ax.set_yticks(range(3), ("H0 vs HN", "HB vs HN", "HP vs HN"))
    ax.invert_yaxis()
    ax.axvline(0, color="#999999", linewidth=0.7)
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlabel("Descriptive linear-projection contrast (percentage points)")
    ax.set_title(
        "Baseline covariance comparison" if comparison else "Categorical history robustness",
        loc="left",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), frameon=False, fontsize=9)
    fig.text(
        0.12,
        0.075,
        "EXPLORATORY / UNREVIEWED • SPAGHETTI DRY pure events • HN reference",
        fontsize=9,
    )
    fig.text(
        0.12,
        0.035,
        "Intervals shown only for full-PSD valid covariance; gated variants show points only.",
        fontsize=8,
    )
    if not comparison:
        fig.text(
            0.12,
            0.005,
            "R1 prior-share coefficient has different units and is not plotted on this axis.",
            fontsize=8,
        )
    fig.subplots_adjust(left=0.14, right=0.96, top=0.90, bottom=0.34)
    handle = io.BytesIO()
    fig.savefig(
        handle, format="png", dpi=160, metadata={"Software": "complete_journey Run D UNREVIEWED"}
    )
    plt.close(fig)
    return handle.getvalue()


def covariance_bytes(matrices: dict[str, np.ndarray]) -> bytes:
    import zipfile

    require(set(matrices) == {"R1", "R2", "R3", "R4"}, "Run D native matrix universe")
    handle = io.BytesIO()
    with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for variant in ("R1", "R2", "R3", "R4"):
            array = io.BytesIO()
            np.save(array, matrices[variant], allow_pickle=False)
            entry = zipfile.ZipInfo(f"{variant}_covariance.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, array.getvalue())
    return handle.getvalue()


def validate_native(report: dict, matrices: dict[str, np.ndarray]) -> None:
    """Reconcile ordinal native matrices with report eigenvalues/focal variances."""
    indices = {
        "R1": {
            "prior_private_share_filled": 166,
            "has_prior_pure_history": 167,
            "log1p_prior_pure": 165,
        },
        "R2": {"H0": 148, "HB": 149, "HP": 150, "log1p_prior_pure": 147},
        "R3": {"H0": 166, "HB": 167, "HP": 168, "log1p_prior_pure": 165},
        "R4": {"H0": 166, "HB": 167, "HP": 168, "log1p_prior_pure": 165},
    }
    require(set(matrices) == set(indices), "Run D native matrix universe")
    for variant, k in (("R1", 168), ("R2", 151), ("R3", 169), ("R4", 169)):
        matrix = matrices[variant]
        require(
            matrix.shape == (k, k) and np.isfinite(matrix).all(), "Run D native matrix shape/finite"
        )
        require(
            np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-10), "Run D native matrix symmetry"
        )
        summary = report["variants"][variant]["covariance"]
        eigen = np.linalg.eigvalsh((matrix + matrix.T) / 2)
        require(
            np.allclose(eigen, summary["eigenvalues"], atol=1e-10, rtol=1e-8),
            "Run D native eigenvalue/report drift",
        )
        for term, position in indices[variant].items():
            require(
                np.isclose(
                    matrix[position, position],
                    summary["focal_variances"][term]["variance"],
                    atol=1e-10,
                    rtol=1e-8,
                ),
                "Run D native focal/report drift",
            )


def read_cached(root: Path = PROJECT_ROOT, required: bool = False) -> dict | None:
    from complete_journey.choice.robustness import validate_report
    from complete_journey.choice.run_c import read_cached as read_core

    paths = {k: root / p for k, p in RUN_D_OUTPUTS.items()}
    if not any(p.exists() for p in paths.values()):
        require(not required, "Run D archive absent")
        return None
    require(all(p.is_file() for p in paths.values()), "partial Run D archive; no silent rerun")
    artifacts.assert_policy(root)
    artifacts.verify_accepted_core(root)
    manifest = json.loads(paths["provenance"].read_bytes())
    require(
        type(manifest) is dict
        and set(manifest) == {"scope", "status", "source", "inputs", "accepted_run_c", "artifacts"},
        "closed Run D provenance",
    )
    require(
        manifest["scope"] == SCOPE and manifest["status"] == "UNREVIEWED",
        "Run D provenance metadata",
    )
    if manifest["source"] != artifacts.source_identity(root):
        # New/synthetic archives keep the strict current-source branch. Only all
        # seven accepted pins admit historical generation, without reattribution.
        accepted = artifacts.verify_accepted_robustness(root)
        require(manifest == accepted["provenance"], "accepted Run D manifest changed during read")
    require(
        manifest["inputs"] == artifacts.verify_inputs(root)
        and manifest["accepted_run_c"] == ACCEPTED_RUN_C_SHA256,
        "Run D evidence drift",
    )
    expected = {p for k, p in RUN_D_OUTPUTS.items() if k != "provenance"}
    require(
        type(manifest["artifacts"]) is dict and set(manifest["artifacts"]) == expected,
        "Run D artifact universe",
    )
    for p, digest in manifest["artifacts"].items():
        require(artifacts.sha256(root / p) == digest, "Run D artifact hash drift")
    report = json.loads(paths["report"].read_bytes())
    validate_report(report)
    base = read_core(root, required=True)
    require(
        paths["sensitivity"].read_bytes() == csv_bytes(sensitivity_rows(report), TABLE_FIELDS),
        "Run D sensitivity CSV drift",
    )
    require(
        paths["covariance_table"].read_bytes()
        == csv_bytes(covariance_rows(report, base), COMPARISON_FIELDS),
        "Run D comparison CSV drift",
    )
    with np.load(paths["covariance"], allow_pickle=False) as native:
        require(
            set(native.files) == {f"{v}_covariance" for v in ("R1", "R2", "R3", "R4")},
            "Run D native matrix keys",
        )
        validate_native(report, {v: native[f"{v}_covariance"] for v in ("R1", "R2", "R3", "R4")})
    return report


def run_robustness(root: Path = PROJECT_ROOT) -> tuple[dict, bool]:
    from complete_journey.choice.preflight import check
    from complete_journey.choice.robustness import compute, validate_report
    from complete_journey.choice.run_c import read_cached as read_core

    check(root)
    artifacts.verify_accepted_core(root)
    cached = read_cached(root)
    if cached is not None:
        return cached, True
    identity = artifacts.source_identity(root)
    inputs = artifacts.verify_inputs(root)
    baseline = read_core(root, required=True)
    with np.load(root / CORE_OUTPUTS["covariance"], allow_pickle=False) as archive:
        native = archive["M1_covariance"].copy()
    pure = pd.read_parquet(root / LOCAL_PATHS["pure"])
    report, matrices = compute(pure, baseline, native)
    validate_report(report)
    validate_native(report, matrices)
    payloads = {
        RUN_D_OUTPUTS["report"]: artifacts.canonical_json(report),
        RUN_D_OUTPUTS["sensitivity"]: csv_bytes(sensitivity_rows(report), TABLE_FIELDS),
        RUN_D_OUTPUTS["covariance_table"]: csv_bytes(
            covariance_rows(report, baseline), COMPARISON_FIELDS
        ),
        RUN_D_OUTPUTS["covariance"]: covariance_bytes(matrices),
        RUN_D_OUTPUTS["history_figure"]: figure_bytes(report, baseline, False),
        RUN_D_OUTPUTS["covariance_figure"]: figure_bytes(report, baseline, True),
    }
    require(
        artifacts.source_identity(root) == identity and artifacts.verify_inputs(root) == inputs,
        "Run D source/input changed",
    )
    artifacts.verify_accepted_core(root)
    artifacts.assert_policy(root)
    manifest = {
        "scope": SCOPE,
        "status": "UNREVIEWED",
        "source": identity,
        "inputs": inputs,
        "accepted_run_c": dict(ACCEPTED_RUN_C_SHA256),
        "artifacts": {p: hashlib.sha256(b).hexdigest() for p, b in payloads.items()},
    }
    payloads[RUN_D_OUTPUTS["provenance"]] = artifacts.canonical_json(manifest)
    require(set(payloads) == set(RUN_D_OUTPUTS.values()), "Run D writer exact ignored scope")
    scratch = tempfile.gettempdir()  # honours TMPDIR; portable fallback (post-audit revision)
    with tempfile.TemporaryDirectory(prefix="phase5-run-d-", dir=scratch) as temp:
        staged = {}
        for i, (p, b) in enumerate(sorted(payloads.items())):
            path = Path(temp) / str(i)
            path.write_bytes(b)
            staged[p] = path
        for p, path in staged.items():
            target = root / p
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
    verified = read_cached(root, required=True)
    require(verified == report, "Run D exact readback")
    return verified, False
