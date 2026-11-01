"""Run E publication from accepted archives; no empirical estimation interfaces."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import t

from complete_journey.choice import artifacts
from complete_journey.choice.contracts import CORE_OUTPUTS, require
from complete_journey.choice.final_contracts import (
    ACCEPTED_RUN_E_PROVENANCE_SHA256,
    AUTHORIZED_SOURCE_REVISIONS,
    CLOSURE_EXEMPT_SOURCES,
    D2_PATH,
    D2_SHA256,
    FINAL_OUTPUTS,
    FOCAL_TERMS,
    INFERENCE_SCOPE,
    P5_002_ENTRY,
    PUBLICATION_PROVENANCE,
    RUN_E_LEDGER_SHA256,
    authorized_scope,
)
from complete_journey.paths import PROJECT_ROOT

T6_FIELDS = (
    "section",
    "history_term",
    "reference",
    "estimate",
    "estimate_unit",
    "se",
    "ci_low",
    "ci_high",
    "t",
    "p",
    "covariance",
    "inference_status",
    "reference_df",
    "nobs",
    "households",
    "stores",
    "weeks",
    "store_effects",
    "calendar_controls",
    "interpretation_note",
)
T7_FIELDS = (
    "robustness_domain",
    "variant",
    "term",
    "coefficient",
    "coefficient_unit",
    "se",
    "ci_low",
    "ci_high",
    "covariance",
    "inference_status",
    "full_covariance_status",
    "focal_covariance_status",
    "focal_dimension",
    "focal_min_eigenvalue",
    "sample",
    "interpretation_note",
)
T6_UNIVERSE = tuple(("RAW", h) for h in ("HN", "H0", "HB", "HP")) + tuple(
    (model, term) for model in ("M0", "M1") for term in FOCAL_TERMS
)
T7_UNIVERSE = tuple(
    (v, term)
    for v in ("R1", "R2", "R3", "R4")
    for term in (
        ("prior_private_share_filled", "has_prior_pure_history", "log1p_prior_pure")
        if v == "R1"
        else FOCAL_TERMS
    )
) + tuple((f"D2_{v}", block) for v in ("A", "B", "C") for block in ("history", "history_depth"))
# Rendering vocabulary is narrower than the approved limitation's ledger wording.
BANNED_RENDER = (
    "loyalty",
    "state dependence",
    "causal effect",
    "effect",
    "household_key",
    "basket_id",
)


def focal_inference(
    row: dict, model="M1", sample="baseline_pure_SPAGHETTI_DRY", covariance="household_week"
) -> dict:
    """Only P5-002 baseline focal scalars from native archived values, never a fit."""
    require(authorized_scope(model, sample, covariance, FOCAL_TERMS), "P5-002 scope violation")
    require(
        row["model"] == model and row["term"] in FOCAL_TERMS and row["reference_df"] == 101,
        "P5-002 row scope violation",
    )
    require(
        row["covariance"] == covariance and row["nobs"] == 6544,
        "P5-002 sample/covariance violation",
    )
    require(
        row["covariance_indefinite"] and not row["covariance_psd"],
        "full covariance status must remain indefinite",
    )
    beta, se, variance = row["coefficient"], row["native_se"], row["focal_variance"]
    require(
        all(type(v) is float and np.isfinite(v) for v in (beta, se, variance))
        and se > 0
        and variance > 0,
        "positive finite native focal values required",
    )
    require(np.isclose(se**2, variance, atol=1e-10, rtol=1e-8), "native SE/variance mismatch")
    critical = float(t.ppf(0.975, 101))
    statistic = float(beta / se)
    p = float(2 * t.sf(abs(statistic), 101))
    # No finite-precision literal zero is published; a textual tail bound would be needed.
    require(p > 0, "tail rounded to zero; stop instead of publishing literal zero")
    return {
        "estimate": beta,
        "se": se,
        "ci_low": float(beta - critical * se),
        "ci_high": float(beta + critical * se),
        "t": statistic,
        "p": p,
        "covariance": "household × response week",
        "inference_status": "P5-002 focal scalar authorized",
        "reference_df": 101,
    }


def validate_geometry(core: dict, robust: dict, d2: dict, native: np.ndarray) -> None:
    require(d2["scope"] == "RUN_D2_COVARIANCE_GEOMETRY_ONLY", "accepted D2 scope")
    mapping = d2["parameter_map"]
    require(
        mapping["validated_against_accepted_metadata"] is True and mapping["dimension"] == 169,
        "accepted D2 parameter map",
    )
    require(native.shape == (169, 169) and np.isfinite(native).all(), "native baseline matrix")
    require(np.allclose(native, native.T, atol=1e-12, rtol=1e-10), "native baseline symmetry")
    eig = np.linalg.eigvalsh((native + native.T) / 2)
    tolerance = np.finfo(float).eps * 169 * np.max(abs(eig))
    require(eig.min() < -tolerance, "full baseline covariance must remain indefinite")
    rows = {r["term"]: r for r in core["results"] if r["model"] == "M1"}
    require(set(rows) == set(FOCAL_TERMS), "baseline focal universe")
    for term in FOCAL_TERMS:
        row = rows[term]
        require(
            row["coefficient"]
            == d2["accepted_coefficients"][term]
            == robust["baseline_verification"]["coefficients"][term],
            "accepted coefficient drift",
        )
        position = mapping["focal_indices"][term]
        require(native[position, position] == row["focal_variance"], "native focal diagonal drift")
    for terms, block in ((FOCAL_TERMS[:3], "history"), (FOCAL_TERMS, "history_depth")):
        indices = [mapping["focal_indices"][term] for term in terms]
        matrix = native[np.ix_(indices, indices)]
        require(
            np.linalg.eigvalsh(matrix).min() > 0, "P5-002 focal block must be positive definite"
        )
        require(
            np.array_equal(matrix, np.array(d2["covariances"]["A"]["blocks"][block]["matrix"])),
            "D2 native focal matrix drift",
        )
    require(
        all(
            r["variance_status"] == "POSITIVE" and r["native_variance"] > 0
            for r in d2["covariances"]["A"]["pairwise"]
        )
        and len(d2["covariances"]["A"]["pairwise"]) == 6,
        "D2 contrast geometry drift",
    )


def load_evidence(root: Path = PROJECT_ROOT):
    from complete_journey.choice.run_c import read_cached as read_core
    from complete_journey.choice.run_d import read_cached as read_robustness

    artifacts.validate_researcher_decision(root)
    artifacts.assert_policy(root)
    core = read_core(root, required=True)
    robust = read_robustness(root, required=True)
    require(core is not None and robust is not None, "accepted C/D archives required; no refit")
    require(
        (root / D2_PATH).is_file() and artifacts.sha256(root / D2_PATH) == D2_SHA256,
        "exact accepted D2 capture required; no diagnostic rerun",
    )
    d2 = json.loads((root / D2_PATH).read_bytes())
    with np.load(root / CORE_OUTPUTS["covariance"], allow_pickle=False) as archive:
        native = archive["M1_covariance"].copy()
    validate_geometry(core, robust, d2, native)
    return core, robust, d2


def source_state(root: Path = PROJECT_ROOT):
    state = artifacts.source_identity(root)
    files = dict(state["files"])
    files["scripts/publish_choice.py"] = artifacts.sha256(root / "scripts/publish_choice.py")
    return {
        "files": files,
        "state_sha256": hashlib.sha256(artifacts.canonical_json(files)).hexdigest(),
    }


def evidence_state(root: Path = PROJECT_ROOT):
    from complete_journey.choice.contracts import (
        ACCEPTED_RUN_C_SHA256,
        ACCEPTED_RUN_D_SHA256,
        OUTPUTS,
        PREPARATION_EVIDENCE_SHA256,
        PROTECTED_INPUTS,
    )

    paths = (
        set(PROTECTED_INPUTS)
        | set(PREPARATION_EVIDENCE_SHA256)
        | set(ACCEPTED_RUN_C_SHA256)
        | set(ACCEPTED_RUN_D_SHA256)
        | set(OUTPUTS.values())
        | {D2_PATH, ".gitignore", "docs/phases/PHASE_05_RESULT_REGISTER.csv"}
    )
    return {p: artifacts.sha256(root / p) for p in sorted(paths)}


def _rows(core, robust, d2):
    table6 = []
    history = {r["category"]: r for r in core["foundation"]["history_table"]}
    for h in ("HN", "H0", "HB", "HP"):
        table6.append(
            {
                "section": "RAW",
                "history_term": h,
                "reference": "none",
                "estimate": 100 * history[h]["private_share"],
                "estimate_unit": "percent",
                "se": None,
                "ci_low": None,
                "ci_high": None,
                "t": None,
                "p": None,
                "covariance": "none",
                "inference_status": "raw descriptive share",
                "reference_df": None,
                "nobs": history[h]["events"],
                "households": None,
                "stores": None,
                "weeks": None,
                "store_effects": False,
                "calendar_controls": "none",
                "interpretation_note": "Denominator: recorded pure events in this history category; persistent household heterogeneity unresolved; not a transition rate.",
            }
        )
    rows = {(r["model"], r["term"]): r for r in core["results"]}
    for model in ("M0", "M1"):
        for term in FOCAL_TERMS:
            r = rows[(model, term)]
            scalars = (
                focal_inference(r)
                if model == "M1"
                else {
                    "estimate": r["coefficient"],
                    "se": r["native_se"],
                    "ci_low": r["ci_lower"],
                    "ci_high": r["ci_upper"],
                    "t": r["t_statistic"],
                    "p": r["p_value"],
                    "covariance": "household × response week",
                    "inference_status": "accepted PSD Model 0 exploratory scalar inference",
                    "reference_df": r["reference_df"],
                }
            )
            require(
                model != "M0" or r["covariance_psd"] and r["inference_status"] == "PERMITTED",
                "M0 accepted PSD gate",
            )
            note = (
                "Secondary recorded history-depth adjustment; no behavioral interpretation."
                if term == "log1p_prior_pure"
                else "Descriptive purchased-composition contrast; persistent household heterogeneity unresolved."
            )
            if model == "M1":
                note += " Full covariance remains indefinite; P5-002 only authorizes baseline focal scalars."
            table6.append(
                {
                    "section": model,
                    "history_term": term,
                    "reference": "not applicable" if term == "log1p_prior_pure" else "HN",
                    **scalars,
                    "estimate_unit": "probability points",
                    "nobs": r["nobs"],
                    "households": r["households"],
                    "stores": r["stores"],
                    "weeks": r["weeks"],
                    "store_effects": model == "M1",
                    "calendar_controls": "thirds" if model == "M1" else "none",
                    "interpretation_note": note,
                }
            )
    table7 = []
    for variant in ("R1", "R2", "R3", "R4"):
        data = robust["variants"][variant]
        for r in data["results"]:
            if variant == "R1":
                unit = (
                    "probability points per unit recorded prior Private share"
                    if r["term"] == "prior_private_share_filled"
                    else "probability points"
                )
                note = "Alternate representation; share slope units differ from categorical contrasts; persistent household heterogeneity unresolved."
            elif variant == "R2":
                unit = "probability points"
                note = "Responses weeks 35–102; original weeks 1–34 history retained; not a preferred population; persistent household heterogeneity unresolved."
            elif variant == "R3":
                unit = "probability points"
                note = "Baseline covariance-only transparency sensitivity; diagnostic SE only; primary remains household × response week."
            else:
                unit = "probability points"
                note = "Diagnostic benchmark only; omits shared-store/common-week cross-household dependence; PSD does not establish coverage."
            if r["term"] == "log1p_prior_pure":
                note += " Depth is secondary adjustment."
            elif variant != "R1":
                note += " HN reference."
            status = (
                "gated; not in P5-002 scope"
                if variant in ("R1", "R2")
                else "diagnostic SE only"
                if variant == "R3"
                else "diagnostic benchmark interval"
            )
            table7.append(
                {
                    "robustness_domain": "history representation"
                    if variant == "R1"
                    else "response calendar"
                    if variant == "R2"
                    else "covariance comparison",
                    "variant": variant,
                    "term": r["term"],
                    "coefficient": r["coefficient"],
                    "coefficient_unit": unit,
                    "se": r["native_se"],
                    "ci_low": r["ci_lower"] if variant == "R4" else None,
                    "ci_high": r["ci_upper"] if variant == "R4" else None,
                    "covariance": data["covariance"]["kind"],
                    "inference_status": status,
                    "full_covariance_status": "PSD" if r["covariance_psd"] else "indefinite",
                    "focal_covariance_status": "positive definite (D2)"
                    if variant in ("R3", "R4")
                    else "not authorized for scalar publication",
                    "focal_dimension": None,
                    "focal_min_eigenvalue": None,
                    "sample": "weeks 35–102; N=4807" if variant == "R2" else "baseline; N=6544",
                    "interpretation_note": note,
                }
            )
    for variant in ("A", "B", "C"):
        for block in ("history", "history_depth"):
            data = d2["covariances"][variant]
            g = data["blocks"][block]
            table7.append(
                {
                    "robustness_domain": "focal covariance geometry",
                    "variant": f"D2_{variant}",
                    "term": block,
                    "coefficient": None,
                    "coefficient_unit": "not applicable",
                    "se": None,
                    "ci_low": None,
                    "ci_high": None,
                    "covariance": data["kind"],
                    "inference_status": "geometry only; not an inference decision",
                    "full_covariance_status": "PSD" if variant == "C" else "indefinite",
                    "focal_covariance_status": "positive definite",
                    "focal_dimension": len(g["terms"]),
                    "focal_min_eigenvalue": g["minimum_eigenvalue"],
                    "sample": "baseline; N=6544",
                    "interpretation_note": "Predeclared focal block independently reconciled; all six predeclared pairwise variances positive; no repair/joint inference; original-coordinate nuisance loading is not a causal explanation.",
                }
            )
    return table6, table7


def validate_tables(table6, table7, core, robust, d2):
    require(
        type(table6) is list and type(table7) is list and len(table6) == 12 and len(table7) == 21,
        "final table row count",
    )
    require(
        tuple((r.get("section"), r.get("history_term")) for r in table6) == T6_UNIVERSE,
        "T6 row universe",
    )
    require(
        tuple((r.get("variant"), r.get("term")) for r in table7) == T7_UNIVERSE, "T7 row universe"
    )
    for rows, fields in ((table6, T6_FIELDS), (table7, T7_FIELDS)):
        require(
            all(type(r) is dict and set(r) == set(fields) for r in rows),
            "closed identifier-free final schema",
        )
        for row in rows:
            require(
                all(type(v) in (str, int, float, bool, type(None)) for v in row.values()),
                "final scalar-only schema",
            )
            require(
                all(type(v) is not float or np.isfinite(v) for v in row.values()),
                "nonfinite final value",
            )
            if "p" in row and row["p"] is not None:
                require(type(row["p"]) is float and row["p"] > 0, "literal zero p refused")
    expected6, expected7 = _rows(core, robust, d2)
    require(
        artifacts.canonical_json({"T6": table6, "T7": table7})
        == artifacts.canonical_json({"T6": expected6, "T7": expected7}),
        "final table accepted-value/scope drift",
    )


def table_bytes(rows, fields):
    require(all(set(r) == set(fields) for r in rows), "CSV closed fields")
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


def figure_bytes(table6):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    raw = {r["history_term"]: r for r in table6 if r["section"] == "RAW"}
    model = {r["history_term"]: r for r in table6 if r["section"] == "M1"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    left, right = axes
    history = ("HN", "H0", "HB", "HP")
    values = [raw[h]["estimate"] for h in history]
    left.bar(history, values, color=("#7e8e9e", "#4b78aa", "#4b78aa", "#4b78aa"), width=0.65)
    left.set_ylim(0, 100)
    left.set_ylabel("Raw Private-only share (%)")
    left.set_xlabel("Denominator: pure events in each recorded history category", fontsize=9)
    left.set_title("A. Raw descriptive composition", loc="left", fontsize=12)
    for x, value in enumerate(values):
        left.text(x, value + 2, f"{value:.2f}%", ha="center", fontsize=10)
    labels = ("H0", "HB", "HP")
    y = np.arange(3)
    estimates = np.array([100 * model[h]["estimate"] for h in labels])
    lower = np.array([100 * model[h]["ci_low"] for h in labels])
    upper = np.array([100 * model[h]["ci_high"] for h in labels])
    require(
        all(
            model[h]["inference_status"] == "P5-002 focal scalar authorized"
            and model[h]["covariance"] == "household × response week"
            for h in labels
        ),
        "F11 primary interval scope",
    )
    right.errorbar(
        estimates,
        y,
        xerr=np.stack((estimates - lower, upper - estimates)),
        fmt="o",
        color="#4b78aa",
        capsize=4,
    )
    right.set_yticks(y, [f"{h} vs HN" for h in labels])
    right.invert_yaxis()
    right.axvline(0, color="#888888", linewidth=0.8)
    right.set_xlabel("Adjusted descriptive contrast (percentage points)")
    right.set_title("B. Model 1 purchased-composition contrasts", loc="left", fontsize=12)
    right.grid(axis="x", alpha=0.18)
    captions = (
        "HN reference. Model 1 adjusts for recorded history depth, recorded-store indicators and calendar thirds.",
        "95% intervals: primary household × response week, Student-t(101), P5-002 restricted focal scalar inference.",
        "The full 169×169 covariance remains indefinite. Contrasts are descriptive purchased-composition associations.",
        "Persistent household heterogeneity remains unresolved; neither raw shares nor contrasts identify a behavioral mechanism.",
    )
    for i, text in enumerate(captions):
        fig.text(0.08, 0.18 - i * 0.035, text, fontsize=9, ha="left")
    text = [a.get_title(loc=loc) for a in axes for loc in ("left", "center", "right")]
    text += [a.get_xlabel() for a in axes] + [a.get_ylabel() for a in axes]
    text += [
        t0.get_text() for a in axes for t0 in [*a.texts, *a.get_xticklabels(), *a.get_yticklabels()]
    ]
    text += [t0.get_text() for t0 in fig.texts]
    require(
        not any(word in " ".join(text).lower() for word in BANNED_RENDER),
        "F11 refused wording or identifier",
    )
    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.32, wspace=0.36)
    handle = io.BytesIO()
    fig.savefig(
        handle, format="png", dpi=160, metadata={"Software": "complete_journey Phase 5 Run E F11"}
    )
    plt.close(fig)
    return handle.getvalue()


def payloads(core, robust, d2):
    table6, table7 = _rows(core, robust, d2)
    validate_tables(table6, table7, core, robust, d2)
    return {
        FINAL_OUTPUTS["T6"]: table_bytes(table6, T6_FIELDS),
        FINAL_OUTPUTS["T7"]: table_bytes(table7, T7_FIELDS),
        FINAL_OUTPUTS["F11"]: figure_bytes(table6),
    }


def check_publication(root: Path = PROJECT_ROOT):
    core, robust, d2 = load_evidence(root)
    expected = payloads(core, robust, d2)
    require(
        all((root / p).is_file() and (root / p).read_bytes() == b for p, b in expected.items()),
        "final artifact reproduction drift",
    )
    path = root / PUBLICATION_PROVENANCE
    require(path.is_file(), "publication provenance absent")
    manifest = json.loads(path.read_bytes())
    require(
        type(manifest) is dict
        and set(manifest)
        == {"scope", "inference_scope", "source", "inputs", "artifacts", "schemas", "row_counts"},
        "closed publication provenance",
    )
    scope = {**dict(INFERENCE_SCOPE), "terms": list(FOCAL_TERMS)}
    require(
        manifest["scope"] == "RUN_E_ACCEPTED_PUBLICATION" and manifest["inference_scope"] == scope,
        "publication scope drift",
    )
    if manifest["source"] != source_state(root) or manifest["inputs"] != evidence_state(root):
        _check_historical_publication(root, path, manifest)
    require(
        manifest["schemas"] == {"T6": list(T6_FIELDS), "T7": list(T7_FIELDS)}
        and manifest["row_counts"] == {"T6": 12, "T7": 21},
        "publication schema/row metadata",
    )
    hashes = {p: hashlib.sha256(b).hexdigest() for p, b in expected.items()}
    require(manifest["artifacts"] == hashes, "publication hash metadata")
    return manifest


def _check_historical_publication(root: Path, path: Path, manifest: dict) -> None:
    """Revalidate the immutable historical Run E record after Phase 5 closure (P7-002).

    The recorded generating source stays historical and is never compared as if
    current. Acceptance instead requires: the pinned record bytes; an internally
    canonical historical source manifest; a closed ledger whose exact P5-002 prefix
    is the recorded Run E ledger; every other recorded input unchanged; every current
    source file either historical or an exact authorized revision; and (checked by
    the caller) byte-identical F11/T6/T7 rebuilt by the current implementation.
    """
    require(
        artifacts.sha256(path) == ACCEPTED_RUN_E_PROVENANCE_SHA256,
        "publication source/input drift; no stale-source acceptance",
    )
    artifacts.validate_source_manifest(manifest["source"])
    require(artifacts.ledger_state(root) == "closed", "publication source/input drift")
    ledger = (root / "docs/DECISIONS.md").read_bytes()
    entry = P5_002_ENTRY.encode()
    recorded = manifest["inputs"].get("docs/DECISIONS.md")
    require(
        recorded == RUN_E_LEDGER_SHA256
        and manifest["source"]["files"].get("docs/DECISIONS.md") == recorded
        and hashlib.sha256(ledger[: ledger.index(entry) + len(entry)]).hexdigest() == recorded,
        "publication ledger is not the recorded Run E prefix",
    )
    current_inputs = evidence_state(root)
    require(
        set(current_inputs) == set(manifest["inputs"])
        and all(
            current_inputs[p] == d
            for p, d in manifest["inputs"].items()
            if p != "docs/DECISIONS.md"
        ),
        "publication input drift",
    )
    historical = manifest["source"]["files"]
    current = source_state(root)["files"]
    require(set(current) == set(historical), "publication source universe drift")
    for relative, digest in current.items():
        require(
            relative in CLOSURE_EXEMPT_SOURCES
            or digest == historical[relative]
            or AUTHORIZED_SOURCE_REVISIONS.get(relative) == digest,
            f"publication source drift: {relative}",
        )


def publish(root: Path = PROJECT_ROOT):
    # The historical Run E record is immutable once Phase 5 is closed (P5-003, P7-002).
    ledger = root / "docs/DECISIONS.md"
    require(
        not ledger.is_file() or artifacts.ledger_state(root) != "closed",
        "Phase 5 closed; Run E record immutable",
    )
    # Snapshot before loading records; preserve prior evidence and exact current generation.
    before = evidence_state(root)
    source = source_state(root)
    core, robust, d2 = load_evidence(root)
    first = payloads(core, robust, d2)
    second = payloads(core, robust, d2)
    require(first == second, "non-deterministic publication")
    require(
        evidence_state(root) == before and source_state(root) == source,
        "source/evidence changed during render",
    )
    artifacts.assert_policy(root)
    manifest = {
        "scope": "RUN_E_ACCEPTED_PUBLICATION",
        "inference_scope": {**dict(INFERENCE_SCOPE), "terms": list(FOCAL_TERMS)},
        "source": source,
        "inputs": before,
        "artifacts": {p: hashlib.sha256(b).hexdigest() for p, b in first.items()},
        "schemas": {"T6": list(T6_FIELDS), "T7": list(T7_FIELDS)},
        "row_counts": {"T6": 12, "T7": 21},
    }
    intended = {**first, PUBLICATION_PROVENANCE: artifacts.canonical_json(manifest)}
    require(
        set(intended) == set(FINAL_OUTPUTS.values()) | {PUBLICATION_PROVENANCE},
        "exact publication path universe",
    )
    scratch = tempfile.gettempdir()  # honours TMPDIR; portable fallback (post-audit revision)
    # Stage the whole validated batch, then rollback all replacements on any error.
    old = {p: (root / p).read_bytes() if (root / p).is_file() else None for p in intended}
    with tempfile.TemporaryDirectory(prefix="phase5-publish-", dir=scratch) as temp:
        staged = {}
        for i, (p, b) in enumerate(sorted(intended.items())):
            target = Path(temp) / str(i)
            target.write_bytes(b)
            staged[p] = target
        changed = []
        try:
            for p, stage in staged.items():
                if old[p] == intended[p]:
                    continue
                target = root / p
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage, target)
                changed.append(p)
            return check_publication(root)
        except Exception:
            for p in reversed(changed):
                target = root / p
                if old[p] is None:
                    target.unlink(missing_ok=True)
                else:
                    restore = Path(temp) / ("restore_" + str(len(changed)))
                    restore.write_bytes(old[p])
                    os.replace(restore, target)
            raise
