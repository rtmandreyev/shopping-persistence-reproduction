"""Bounded, unpromoted Run D mathematics; no IO, selection search or matrix repair.

compute enforces the accepted empirical boundary. Private numerical seams accept
constructed inputs for mandatory SYNTHETIC_ONLY_NON_EVIDENCE tests. Validation is
outcome-free and uses a static closed contract, never a fitted/caller-derived schema.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t

from complete_journey.choice import core
from complete_journey.choice import covariance as cluster
from complete_journey.choice.contracts import BASELINE, LATER
from complete_journey.choice.core_diagnostics import (
    _SIGNED_SCHEMA,
    _signed_concentration,
    _validate_closed,
    foundation,
)
from complete_journey.choice.design import (
    DESIGN_COLUMNS,
    HISTORY_COLUMNS,
    DesignMatrix,
    _full_rank,
    _residualize,
    _singular_rank,
    _validate_frame,
    build_matrix,
)

ATOL = 1e-10
RTOL = 1e-8
SCOPE = "RUN_D_UNPROMOTED_ROBUSTNESS"
VARIANTS = ("R1", "R2", "R3", "R4")
R1_TERMS = ("prior_private_share_filled", "has_prior_pure_history", "log1p_prior_pure")
TERMS = {"R1": R1_TERMS, **dict.fromkeys(("R2", "R3", "R4"), core.FOCAL_TERMS)}
KINDS = {
    "R1": "household_week",
    "R2": "household_week",
    "R3": "household_store",
    "R4": "household_only",
}
COUNTERS = (
    "prior_national_pure",
    "prior_private_pure",
    "prior_mixed",
    "prior_total_group",
    "prior_pure",
)
RESULT_FIELDS = (
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


def _agree(left, right, label: str) -> None:
    if not np.allclose(left, right, atol=ATOL, rtol=RTOL):
        raise ValueError(f"Run D {label} disagreement")


def _prepare(pure: pd.DataFrame) -> pd.DataFrame:
    """Validate factual history before any response-window selection; never rebuild it."""
    if not isinstance(pure, pd.DataFrame) or pure.columns.has_duplicates:
        raise ValueError("Run D requires unique-column factual pure responses")
    required = {*DESIGN_COLUMNS, *COUNTERS, "prior_private_share", "composition", "private_only"}
    if not required.issubset(pure.columns):
        raise ValueError("Run D missing factual count/share/history fields")
    foundation(pure)  # Frozen independent factual consistency checks, including same-week equality.
    total = pure.prior_pure.to_numpy(dtype=float)
    exists = total > 0
    filled = pure.prior_private_share.to_numpy(dtype=float).copy()
    filled[~exists] = 0
    if not np.isfinite(filled).all() or np.any((filled < 0) | (filled > 1)):
        raise ValueError("Run D invalid factual prior Private share")
    for name, expected in (
        ("has_prior_pure_history", exists.astype(float)),
        ("prior_private_share_filled", filled),
    ):
        if name in pure:
            _agree(pure[name].to_numpy(dtype=float), expected, f"{name} factual consistency")
    prepared = pure.copy()
    prepared["has_prior_pure_history"] = exists.astype(float)
    prepared["prior_private_share_filled"] = filled
    return prepared


def _later(prepared: pd.DataFrame) -> pd.DataFrame:
    """Only select response rows; weeks 1–34 remain in their factual prior counters."""
    return prepared.loc[prepared.week_no.ge(35)].copy()


def _sample(pure: pd.DataFrame) -> dict:
    return {
        "rows": int(len(pure)),
        "households": int(pure.household_key.nunique()),
        "stores": int(pure.store_id.nunique()),
        "weeks": int(pure.week_no.nunique()),
        **{h: int(pure.history_category.eq(h).sum()) for h in ("H0", "HN", "HB", "HP")},
        "mixed_only": int(((pure.prior_pure == 0) & (pure.prior_mixed > 0)).sum()),
        "national": int(pure.private_only.eq(0).sum()),
        "private": int(pure.private_only.eq(1).sum()),
    }


def _sample_gate(pure: pd.DataFrame, later: pd.DataFrame) -> None:
    if _sample(pure) != BASELINE or _sample(later) != LATER:
        raise ValueError("Run D accepted baseline/later empirical sample counts disagree")
    if sorted(pure.week_no.unique()) != list(range(1, 103)) or sorted(
        later.week_no.unique()
    ) != list(range(35, 103)):
        raise ValueError("Run D accepted response-week universe disagrees")


def _r1_design(prepared: pd.DataFrame) -> DesignMatrix:
    _validate_frame(prepared.loc[:, list(DESIGN_COLUMNS)])
    stores = sorted(prepared.store_id.unique())
    thirds = sorted(prepared.calendar_third.unique())
    context = [prepared.store_id.eq(s).to_numpy(dtype=float) for s in stores]
    context += [prepared.calendar_third.eq(c).to_numpy(dtype=float) for c in thirds[1:]]
    depth = prepared.log1p_prior_pure.to_numpy(dtype=float)
    history = prepared.loc[:, list(R1_TERMS[:2])].to_numpy(dtype=float)
    nuisance = np.column_stack((*context, depth))
    nuisance_names = (*(f"store[{s}]" for s in stores), *thirds[1:], "log1p_prior_pure")
    values = np.column_stack((nuisance, history))
    _full_rank(nuisance, "R1 context/depth")
    _full_rank(values, "R1")
    return DesignMatrix(
        values, (*nuisance_names, *R1_TERMS[:2]), nuisance, history, nuisance_names, "R1"
    )


@dataclass(frozen=True)
class _Fit:
    design: DesignMatrix
    coefficients: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    influence: np.ndarray
    point_checks: list[dict]


def _independent_points(prepared: pd.DataFrame, y: np.ndarray, variant: str) -> np.ndarray:
    """Absorb stores by explicit means, then direct LS of all within-store terms."""
    if variant == "R1":
        focal = prepared.loc[:, list(R1_TERMS)].to_numpy(dtype=float)
    else:
        focal = np.column_stack(
            [prepared.history_category.eq(h).to_numpy(dtype=float) for h in HISTORY_COLUMNS]
            + [prepared.log1p_prior_pure.to_numpy(dtype=float)]
        )
    thirds = sorted(prepared.calendar_third.unique())
    indicators = [prepared.calendar_third.eq(c).to_numpy(dtype=float) for c in thirds[1:]]
    within = np.column_stack((y, focal, *indicators))
    for store in pd.unique(prepared.store_id):
        mask = prepared.store_id.eq(store).to_numpy()
        within[mask] -= within[mask].mean(axis=0)
    controls = within[:, 1:]
    beta, _, rank, _ = np.linalg.lstsq(
        controls, within[:, 0], rcond=np.finfo(float).eps * max(controls.shape)
    )
    if rank != controls.shape[1]:
        raise ValueError("Run D independent within-store rank disagreement")
    return beta[: len(TERMS[variant])]


def _fit(prepared: pd.DataFrame, variant: str) -> _Fit:
    """Numerical synthetic seam; exactly R1 or frozen categorical M1 for R2."""
    if variant not in ("R1", "R2"):
        raise ValueError("Run D point fitting permits only R1/R2")
    design = (
        _r1_design(prepared)
        if variant == "R1"
        else build_matrix(prepared.loc[:, list(DESIGN_COLUMNS)], "M1")
    )
    x = design.values
    y = prepared.private_only.to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all() or len(y) <= x.shape[1]:
        raise ValueError("Run D fitting requires binary Y and N > K")
    q, r = np.linalg.qr(x, mode="reduced")
    beta = np.linalg.solve(r, q.T @ y)
    fitted = x @ beta
    residual = y - fitted
    influence = np.linalg.solve(r, q.T).T * residual[:, None]
    independent = _independent_points(prepared, y, variant)
    checks = []
    for term, other in zip(TERMS[variant], independent, strict=True):
        main = float(beta[design.column_names.index(term)])
        _agree(main, other, f"{variant} point {term}")
        checks.append(
            {
                "term": term,
                "main": main,
                "independent": float(other),
                "absolute_difference": float(abs(main - other)),
                "atol": ATOL,
                "rtol": RTOL,
                "agrees": True,
            }
        )
    return _Fit(design, beta, fitted, residual, influence, checks)


def _independent_covariance(
    prepared: pd.DataFrame, fit, kind: str
) -> tuple[np.ndarray, list[dict]]:
    """SVD bread and dictionary sums of actual keys; independently written CR1."""
    x = fit.design.values
    n, k = x.shape
    scores = np.linalg.pinv(x, rcond=np.finfo(float).eps * max(x.shape)).T * fit.residual[:, None]
    columns = ["household_key"]
    if kind != "household_only":
        columns.append("week_no" if kind == "household_week" else "store_id")
    keys = list(prepared[columns].itertuples(index=False, name=None))
    selections = (("household", (0,), 1),)
    if len(columns) == 2:
        axis = "week" if kind == "household_week" else "store"
        selections += ((axis, (1,), 1), (f"household_{axis}_intersection", (0, 1), -1))
    result = np.zeros((k, k))
    facts = []
    for name, selection, sign in selections:
        sums, sizes = {}, {}
        for key, score in zip(keys, scores, strict=True):
            group = tuple(key[j] for j in selection)
            sums[group] = sums.get(group, np.zeros(k)) + score
            sizes[group] = sizes.get(group, 0) + 1
        g = len(sums)
        if g <= 1 or n <= k:
            raise ValueError("Run D independent CR1 requires G > 1 and N > K")
        correction = float(g / (g - 1) * (n - 1) / (n - k))
        meat = np.zeros((k, k))
        for score in sums.values():
            meat += np.outer(score, score)
        result += sign * correction * meat
        facts.append(
            {
                "component": name,
                "sign": sign,
                "clusters": g,
                "correction": correction,
                "repeated_cells": sum(v > 1 for v in sizes.values()),
                "events_in_repeated_cells": sum(v for v in sizes.values() if v > 1),
            }
        )
    return result, facts


def _covariance(prepared: pd.DataFrame, fit, kind: str, terms: tuple[str, ...]):
    codes = cluster._cluster_codes(prepared, kind)
    n, k = fit.design.values.shape
    counts = [int(np.unique(c).size) for c in codes]
    components = [cluster._component_covariance(fit.influence, c, k) for c in codes]
    native = (
        components[0] if len(components) == 1 else components[0] + components[1] - components[2]
    )
    independent, facts = _independent_covariance(prepared, fit, kind)
    _agree(native, independent, f"{kind} full covariance")
    focal_checks = []
    for term in terms:
        j = fit.design.column_names.index(term)
        left, right = float(native[j, j]), float(independent[j, j])
        _agree(left, right, f"{term} focal variance")
        left_se = float(np.sqrt(left)) if np.isfinite(left) and left > 0 else None
        right_se = float(np.sqrt(right)) if np.isfinite(right) and right > 0 else None
        if (left_se is None) != (right_se is None):
            raise ValueError("Run D independent focal variance sign disagreement")
        if left_se is not None:
            _agree(left_se, right_se, f"{term} focal SE")
        focal_checks.append(
            {
                "term": term,
                "main_variance": left,
                "independent_variance": right,
                "variance_agrees": True,
                "main_se": left_se,
                "independent_se": right_se,
                "se_agrees": True if left_se is not None else None,
            }
        )
    check = {
        "agrees": True,
        "maximum_absolute_difference": float(np.max(abs(native - independent))),
        "atol": ATOL,
        "rtol": RTOL,
        "components": facts,
        "focal": focal_checks,
    }
    summary = _covariance_summary(
        native,
        fit.design.column_names,
        terms,
        kind,
        counts,
        [cluster._cr1(g, n, k) for g in counts],
    )
    return native, summary, check


def _covariance_summary(native, names, terms, kind, counts, corrections) -> dict:
    finite = bool(np.isfinite(native).all())
    symmetric = bool(finite and np.allclose(native, native.T, atol=1e-12, rtol=1e-10))
    eigen = np.linalg.eigvalsh((native + native.T) / 2) if symmetric else None
    tolerance = (
        float(np.finfo(float).eps * len(native) * np.max(abs(eigen))) if eigen is not None else None
    )
    psd = bool(eigen is not None and eigen.min() >= -tolerance)
    indefinite = bool(eigen is not None and eigen.min() < -tolerance and eigen.max() > tolerance)
    focal = {}
    for term in terms:
        variance = float(native[names.index(term), names.index(term)])
        positive = bool(np.isfinite(variance) and variance > 0)
        focal[term] = {
            "variance": variance,
            "finite_positive": positive,
            "native_se": float(np.sqrt(variance)) if positive else None,
        }
    positive = all(f["finite_positive"] for f in focal.values())
    status = (
        "STOP_FOCAL_VARIANCE"
        if not positive
        else "PERMITTED"
        if psd
        else "PROVISIONAL_RESEARCHER_GATED"
    )
    return {
        "kind": kind,
        "finite": finite,
        "symmetric": symmetric,
        "eigenvalues": eigen.tolist() if eigen is not None else [None] * len(native),
        "eigenvalue_tolerance": tolerance,
        "psd": psd,
        "indefinite": indefinite,
        "focal_variances": focal,
        "focal_variances_finite_positive": positive,
        "component_counts": counts,
        "component_corrections": corrections,
        "reference_distribution": "Student_t",
        "reference_df": min(counts[:2]) - 1,
        "inference_status": status,
        "covariance_repaired": False,
        "joint_tests": False,
        "scalar_exception": False,
        "se_ratios_vs_baseline": None,
        "se_ratios_vs_R3": None,
    }


def _rows(fit, variant: str, summary: dict, accepted: dict | None = None) -> list[dict]:
    permitted = summary["inference_status"] == "PERMITTED"
    df = summary["reference_df"]
    critical = float(t.ppf(0.975, df)) if permitted else None
    rows = []
    for term in TERMS[variant]:
        beta = float(fit.coefficients[fit.design.column_names.index(term)])
        if accepted is not None:
            _agree(beta, accepted[term], "accepted archived coefficient reuse")
            beta = accepted[
                term
            ]  # Presentation only; native QR/residual/influence arrays untouched.
        focal = summary["focal_variances"][term]
        se = focal["native_se"]
        statistic = float(beta / se) if permitted else None
        rows.append(
            {
                "variant": variant,
                "term": term,
                "coefficient": beta,
                "percentage_points": 100 * beta,
                "native_se": se,
                "focal_variance": focal["variance"],
                "ci_lower": float(beta - critical * se) if permitted else None,
                "ci_upper": float(beta + critical * se) if permitted else None,
                "t_statistic": statistic,
                "p_value": float(2 * t.sf(abs(statistic), df)) if permitted else None,
                "inference_status": summary["inference_status"],
                "covariance_psd": summary["psd"],
                "covariance_indefinite": summary["indefinite"],
                "reference_df": df,
            }
        )
    return rows


def _fitted(fit) -> dict:
    v = fit.fitted
    return {
        "minimum": float(v.min()),
        "maximum": float(v.max()),
        "below_zero_count": int(np.count_nonzero(v < 0)),
        "below_zero_share": float(np.mean(v < 0)),
        "above_one_count": int(np.count_nonzero(v > 1)),
        "above_one_share": float(np.mean(v > 1)),
        "full_fitted_including_context": True,
        "clipped": False,
    }


def _concentration(prepared: pd.DataFrame, fit, terms: tuple[str, ...]) -> dict:
    x, y = fit.design.values, prepared.private_only.to_numpy(dtype=float)
    report = {}
    for term in terms:
        position = fit.design.column_names.index(term)
        controls = np.delete(x, position, axis=1)
        q, _ = np.linalg.qr(controls, mode="reduced")
        rx = x[:, position] - q @ (q.T @ x[:, position])
        partial_y = y - q @ (q.T @ y)
        contributions = rx * partial_y
        denominator = float(rx @ rx)
        numerator = float(contributions.sum())
        target = float(fit.coefficients[position] * denominator)
        _agree(numerator, target, f"{term} signed FWL numerator")
        report[term] = {
            "fwl_denominator": denominator,
            "fwl_numerator": numerator,
            "coefficient_times_denominator": target,
            "numerator_reconciles": True,
            "households": _signed_concentration(prepared, "household_key", contributions),
            "stores": _signed_concentration(prepared, "store_id", contributions),
        }
    return report


def _specification(prepared, fit, variant: str) -> dict:
    thirds = sorted(prepared.calendar_third.unique())
    history_rank = _singular_rank(_residualize(fit.design.history, fit.design.nuisance))[1]
    return {
        "model": "continuous_history_M1" if variant == "R1" else "M1",
        "population": "later_pure_SPAGHETTI_DRY"
        if variant == "R2"
        else "baseline_pure_SPAGHETTI_DRY",
        "sample": _sample(prepared),
        "first_response_week": int(prepared.week_no.min()),
        "last_response_week": int(prepared.week_no.max()),
        "calendar_reference": thirds[0],
        "calendar_indicators": thirds[1:],
        "design_columns": int(fit.design.values.shape[1]),
        "design_rank": _singular_rank(fit.design.values)[1],
        "history_rank": history_rank,
        "focal_terms": list(TERMS[variant]),
        "history_reference": None if variant == "R1" else "HN",
        "history_prepared_before_response_subset": True,
        "factual_no_history_share_is_null": True,
        "store_effects": True,
        "global_intercept": False,
        "household_effects": False,
        "equal_event_weights": True,
        "covariance": KINDS[variant],
        "coefficient_source": "accepted_Run_C_archive" if variant in ("R3", "R4") else "native_QR",
        "native_arrays_reused_from_baseline": variant in ("R3", "R4"),
    }


def _verify_baseline(
    fit, baseline_report: dict, baseline_covariance: np.ndarray
) -> tuple[dict, dict]:
    core.validate_result_rows(baseline_report["results"])
    archived = [r for r in baseline_report["results"] if r["model"] == "M1"]
    if len(archived) != 4 or {r["term"] for r in archived} != set(core.FOCAL_TERMS):
        raise ValueError("Run D accepted baseline focal universe disagrees")
    if (
        not isinstance(baseline_covariance, np.ndarray)
        or baseline_covariance.shape != fit.covariance.shape
    ):
        raise ValueError("Run D accepted baseline covariance shape disagrees")
    _agree(fit.covariance, baseline_covariance, "accepted baseline covariance archive")
    values, checks = {}, []
    for row in archived:
        term = row["term"]
        j = fit.design.column_names.index(term)
        main = float(fit.coefficients[j])
        accepted = row["coefficient"]
        _agree(main, accepted, f"accepted baseline {term}")
        _agree(fit.covariance[j, j], row["focal_variance"], f"accepted baseline variance {term}")
        native_se = float(np.sqrt(fit.covariance[j, j])) if fit.covariance[j, j] > 0 else None
        if (native_se is None) != (row["native_se"] is None):
            raise ValueError("Run D accepted baseline SE sign disagrees")
        if native_se is not None:
            _agree(native_se, row["native_se"], f"accepted baseline SE {term}")
        values[term] = accepted
        checks.append(
            {
                "term": term,
                "recomputed": main,
                "archived": accepted,
                "absolute_difference": abs(main - accepted),
                "agrees": True,
            }
        )
    checks.sort(key=lambda c: core.FOCAL_TERMS.index(c["term"]))
    return values, {
        "model": "M1",
        "coefficients": values,
        "native_se": {row["term"]: row["native_se"] for row in archived},
        "point_checks": checks,
        "covariance_agrees": True,
        "covariance_maximum_absolute_difference": float(
            np.max(abs(fit.covariance - baseline_covariance))
        ),
        "archive_coefficients_reused_for_R3_R4": True,
        "atol": ATOL,
        "rtol": RTOL,
    }


def _changes(rows: list[dict], baseline: dict) -> dict:
    result = {}
    for row in rows:
        term = row["term"]
        before, after = baseline[term], row["coefficient"]
        delta = after - before
        result[term] = {
            "baseline_coefficient": before,
            "later_coefficient": after,
            "signed_change": delta,
            "absolute_change": abs(delta),
            "percentage_point_change": 100 * delta,
            "relative_change": delta / before if abs(before) > ATOL else None,
            "relative_denominator_minimum": ATOL,
            "same_sign": bool(np.sign(before) == np.sign(after)),
        }
    return result


def _ratios(summary: dict, reference: dict) -> dict:
    return {
        term: float(f["native_se"] / reference["focal_variances"][term]["native_se"])
        if f["native_se"] is not None
        and reference["focal_variances"][term]["native_se"] is not None
        else None
        for term, f in summary["focal_variances"].items()
    }


def compute(
    pure: pd.DataFrame, baseline_report: dict, baseline_covariance: np.ndarray
) -> tuple[dict, dict]:
    """Return closed empirical report and native R1/R2/R3/R4 covariance matrices.

    baseline_report is the accepted Run C report with its results list;
    baseline_covariance is the archived full M1 covariance (169 by 169).
    No empirical estimation is performed merely by importing or validating.
    """
    prepared = _prepare(pure)
    later = _later(prepared)
    _sample_gate(prepared, later)
    baseline = core.fit_projection(
        prepared.loc[:, list(DESIGN_COLUMNS)], prepared.private_only.to_numpy(dtype=float), "M1"
    )
    accepted, verification = _verify_baseline(baseline, baseline_report, baseline_covariance)
    variants, covariances = {}, {}
    baseline_summary = _covariance_summary(
        baseline.covariance,
        baseline.design.column_names,
        core.FOCAL_TERMS,
        "household_week",
        list(baseline.component_counts),
        list(baseline.component_corrections),
    )
    for variant in VARIANTS:
        sample = later if variant == "R2" else prepared
        fit = _fit(sample, variant) if variant in ("R1", "R2") else baseline
        matrix, summary, check = _covariance(sample, fit, KINDS[variant], TERMS[variant])
        if variant in ("R3", "R4"):
            summary["se_ratios_vs_baseline"] = _ratios(summary, baseline_summary)
        if variant == "R4":
            summary["se_ratios_vs_R3"] = _ratios(summary, variants["R3"]["covariance"])
        rows = _rows(fit, variant, summary, accepted if variant in ("R3", "R4") else None)
        variants[variant] = {
            "results": rows,
            "covariance": summary,
            "point_checks": fit.point_checks,
            "covariance_check": check,
            "fitted": _fitted(fit) if variant in ("R1", "R2") else None,
            "concentration": _concentration(
                sample, fit, R1_TERMS if variant == "R1" else HISTORY_COLUMNS
            )
            if variant in ("R1", "R2")
            else None,
            "specification": _specification(sample, fit, variant),
        }
        covariances[variant] = matrix
    report = {
        "scope": SCOPE,
        "status": "UNREVIEWED",
        "variants": variants,
        "baseline_comparison": _changes(variants["R2"]["results"], accepted),
        "baseline_verification": verification,
    }
    validate_report(report)
    return report, covariances


# Static schema builders take only literal variant metadata; no data, report or fit.
def _point_schema(terms):
    return tuple(
        {
            "term": term,
            "main": float,
            "independent": float,
            "absolute_difference": float,
            "atol": ATOL,
            "rtol": RTOL,
            "agrees": True,
        }
        for term in terms
    )


def _check_schema(terms, kind):
    axis = "week" if kind == "household_week" else "store"
    components = (
        (("household", 1),)
        if kind == "household_only"
        else (("household", 1), (axis, 1), (f"household_{axis}_intersection", -1))
    )
    return {
        "agrees": True,
        "maximum_absolute_difference": float,
        "atol": ATOL,
        "rtol": RTOL,
        "components": tuple(
            {
                "component": name,
                "sign": sign,
                "clusters": int,
                "correction": float,
                "repeated_cells": int,
                "events_in_repeated_cells": int,
            }
            for name, sign in components
        ),
        "focal": tuple(
            {
                "term": term,
                "main_variance": float,
                "independent_variance": float,
                "variance_agrees": True,
                "main_se": float | None,
                "independent_se": float | None,
                "se_agrees": bool | None,
            }
            for term in terms
        ),
    }


def _summary_schema(variant):
    terms = TERMS[variant]
    k = 168 if variant == "R1" else 151 if variant == "R2" else 169
    components = 1 if variant == "R4" else 3
    return {
        "kind": KINDS[variant],
        "finite": bool,
        "symmetric": bool,
        "eigenvalues": tuple(float | None for _ in range(k)),
        "eigenvalue_tolerance": float | None,
        "psd": bool,
        "indefinite": bool,
        "focal_variances": {
            term: {"variance": float, "finite_positive": bool, "native_se": float | None}
            for term in terms
        },
        "focal_variances_finite_positive": bool,
        "component_counts": tuple(int for _ in range(components)),
        "component_corrections": tuple(float for _ in range(components)),
        "reference_distribution": "Student_t",
        "reference_df": int,
        "inference_status": str,
        "covariance_repaired": False,
        "joint_tests": False,
        "scalar_exception": False,
        "se_ratios_vs_baseline": dict.fromkeys(terms, float | None)
        if variant in ("R3", "R4")
        else None,
        "se_ratios_vs_R3": dict.fromkeys(terms, float | None) if variant == "R4" else None,
    }


def _spec_schema(variant):
    later = variant == "R2"
    k = 168 if variant == "R1" else 151 if later else 169
    return {
        "model": "continuous_history_M1" if variant == "R1" else "M1",
        "population": "later_pure_SPAGHETTI_DRY" if later else "baseline_pure_SPAGHETTI_DRY",
        "sample": dict(LATER if later else BASELINE),
        "first_response_week": 35 if later else 1,
        "last_response_week": 102,
        "calendar_reference": "THIRD_2" if later else "THIRD_1",
        "calendar_indicators": ("THIRD_3",) if later else ("THIRD_2", "THIRD_3"),
        "design_columns": k,
        "design_rank": k,
        "history_rank": 2 if variant == "R1" else 3,
        "focal_terms": tuple(TERMS[variant]),
        "history_reference": None if variant == "R1" else "HN",
        "history_prepared_before_response_subset": True,
        "factual_no_history_share_is_null": True,
        "store_effects": True,
        "global_intercept": False,
        "household_effects": False,
        "equal_event_weights": True,
        "covariance": KINDS[variant],
        "coefficient_source": "accepted_Run_C_archive" if variant in ("R3", "R4") else "native_QR",
        "native_arrays_reused_from_baseline": variant in ("R3", "R4"),
    }


def _row_schema(variant, term):
    return {
        "variant": variant,
        "term": term,
        "coefficient": float,
        "percentage_points": float,
        "native_se": float | None,
        "focal_variance": float,
        "ci_lower": float | None,
        "ci_upper": float | None,
        "t_statistic": float | None,
        "p_value": float | None,
        "inference_status": str,
        "covariance_psd": bool,
        "covariance_indefinite": bool,
        "reference_df": int,
    }


REPORT_SCHEMA = {
    "scope": SCOPE,
    "status": "UNREVIEWED",
    "variants": {
        v: {
            "results": tuple(_row_schema(v, term) for term in TERMS[v]),
            "covariance": _summary_schema(v),
            "point_checks": _point_schema(TERMS[v]),
            "covariance_check": _check_schema(TERMS[v], KINDS[v]),
            "fitted": {
                "minimum": float,
                "maximum": float,
                "below_zero_count": int,
                "below_zero_share": float,
                "above_one_count": int,
                "above_one_share": float,
                "full_fitted_including_context": True,
                "clipped": False,
            }
            if v in ("R1", "R2")
            else None,
            "concentration": {
                term: {
                    "fwl_denominator": float,
                    "fwl_numerator": float,
                    "coefficient_times_denominator": float,
                    "numerator_reconciles": True,
                    "households": _SIGNED_SCHEMA,
                    "stores": _SIGNED_SCHEMA,
                }
                for term in (R1_TERMS if v == "R1" else HISTORY_COLUMNS)
            }
            if v in ("R1", "R2")
            else None,
            "specification": _spec_schema(v),
        }
        for v in VARIANTS
    },
    "baseline_comparison": {
        term: {
            "baseline_coefficient": float,
            "later_coefficient": float,
            "signed_change": float,
            "absolute_change": float,
            "percentage_point_change": float,
            "relative_change": float | None,
            "relative_denominator_minimum": ATOL,
            "same_sign": bool,
        }
        for term in core.FOCAL_TERMS
    },
    "baseline_verification": {
        "model": "M1",
        "coefficients": dict.fromkeys(core.FOCAL_TERMS, float),
        "native_se": dict.fromkeys(core.FOCAL_TERMS, float | None),
        "point_checks": tuple(
            {
                "term": term,
                "recomputed": float,
                "archived": float,
                "absolute_difference": float,
                "agrees": True,
            }
            for term in core.FOCAL_TERMS
        ),
        "covariance_agrees": True,
        "covariance_maximum_absolute_difference": float,
        "archive_coefficients_reused_for_R3_R4": True,
        "atol": ATOL,
        "rtol": RTOL,
    },
}


def validate_report(report: dict) -> None:
    """Closed recursive empirical validation; no estimator calls or numerical fitting."""
    _validate_closed(report, REPORT_SCHEMA)
    accepted = report["baseline_verification"]["coefficients"]
    for check in report["baseline_verification"]["point_checks"]:
        _agree(check["recomputed"], check["archived"], "baseline verification")
        _agree(
            check["absolute_difference"],
            abs(check["recomputed"] - check["archived"]),
            "baseline check difference",
        )
        if check["archived"] != accepted[check["term"]]:
            raise ValueError("Run D baseline archived coefficient alias")
    for variant, data in report["variants"].items():
        cov = data["covariance"]
        sample = data["specification"]["sample"]
        counts = cov["component_counts"]
        second = sample["stores"] if variant == "R3" else sample["weeks"]
        if counts[0] != sample["households"] or (
            variant != "R4"
            and (counts[1] != second or not max(counts[:2]) <= counts[2] <= sample["rows"])
        ):
            raise ValueError("Run D cluster-count universe")
        if cov["reference_df"] != min(counts[:2]) - 1:
            raise ValueError("Run D reference df convention")
        n, k = sample["rows"], data["specification"]["design_rank"]
        expected = [float(g / (g - 1) * (n - 1) / (n - k)) for g in counts]
        _agree(cov["component_corrections"], expected, "component corrections")
        eigen = cov["eigenvalues"]
        if not cov["finite"] or not cov["symmetric"] or any(v is None for v in eigen):
            raise ValueError("Run D requires preserved finite symmetric native covariance")
        tolerance = float(np.finfo(float).eps * k * max(abs(v) for v in eigen))
        _agree(cov["eigenvalue_tolerance"], tolerance, "eigenvalue tolerance")
        psd = min(eigen) >= -tolerance
        indefinite = min(eigen) < -tolerance and max(eigen) > tolerance
        positive = all(v["variance"] > 0 for v in cov["focal_variances"].values())
        status = (
            "STOP_FOCAL_VARIANCE"
            if not positive
            else "PERMITTED"
            if psd
            else "PROVISIONAL_RESEARCHER_GATED"
        )
        if (
            cov["psd"],
            cov["indefinite"],
            cov["focal_variances_finite_positive"],
            cov["inference_status"],
        ) != (psd, indefinite, positive, status):
            raise ValueError("Run D covariance validity/scalar gate")
        for row, point, focal_check in zip(
            data["results"], data["point_checks"], data["covariance_check"]["focal"], strict=True
        ):
            term = row["term"]
            focal = cov["focal_variances"][term]
            variance = focal["variance"]
            se = float(np.sqrt(variance)) if variance > 0 else None
            if focal["finite_positive"] != (variance > 0) or focal["native_se"] != se:
                raise ValueError("Run D focal variance/diagnostic SE")
            if (
                row["native_se"] != se
                or row["focal_variance"] != variance
                or (
                    row["inference_status"],
                    row["covariance_psd"],
                    row["covariance_indefinite"],
                    row["reference_df"],
                )
                != (status, psd, indefinite, cov["reference_df"])
            ):
                raise ValueError("Run D row/summary covariance disagreement")
            _agree(row["percentage_points"], 100 * row["coefficient"], "row coefficient units")
            _agree(row["coefficient"], point["main"], "row point check")
            _agree(point["main"], point["independent"], "independent point")
            _agree(
                point["absolute_difference"],
                abs(point["main"] - point["independent"]),
                "point difference",
            )
            _agree(focal_check["main_variance"], variance, "focal check variance")
            _agree(
                focal_check["main_variance"],
                focal_check["independent_variance"],
                "independent variance",
            )
            if (focal_check["main_se"] is None) != (se is None) or (
                focal_check["independent_se"] is None
            ) != (se is None):
                raise ValueError("Run D independent focal SE availability")
            if se is not None:
                _agree(focal_check["main_se"], se, "focal check SE")
                _agree(focal_check["independent_se"], se, "independent focal SE")
            if focal_check["se_agrees"] is not (True if se is not None else None):
                raise ValueError("Run D independent focal SE status")
            inference = (row["ci_lower"], row["ci_upper"], row["t_statistic"], row["p_value"])
            if status != "PERMITTED" and any(v is not None for v in inference):
                raise ValueError("Run D gated scalar inference must be withheld")
            if status == "PERMITTED":
                if any(v is None for v in inference):
                    raise ValueError("Run D permitted scalar inference missing")
                statistic = row["coefficient"] / se
                critical = float(t.ppf(0.975, cov["reference_df"]))
                _agree(
                    inference,
                    (
                        row["coefficient"] - critical * se,
                        row["coefficient"] + critical * se,
                        statistic,
                        float(2 * t.sf(abs(statistic), cov["reference_df"])),
                    ),
                    "Student t scalars",
                )
            if variant in ("R3", "R4") and row["coefficient"] != accepted[term]:
                raise ValueError("Run D covariance-only archived coefficient exact invariance")
        for component, g, correction in zip(
            data["covariance_check"]["components"], counts, expected, strict=True
        ):
            if (
                component["clusters"] != g
                or component["repeated_cells"] > g
                or component["events_in_repeated_cells"] > n
            ):
                raise ValueError("Run D independent cluster accounting")
            _agree(component["correction"], correction, "independent correction")
        if variant in ("R3", "R4"):
            reference_se = report["baseline_verification"]["native_se"]
            ratios = {}
            for term, focal in cov["focal_variances"].items():
                before, after = reference_se[term], focal["native_se"]
                if before is not None and before <= 0:
                    raise ValueError("Run D archived diagnostic SE must be positive")
                ratios[term] = (
                    float(after / before) if after is not None and before is not None else None
                )
            if cov["se_ratios_vs_baseline"] != ratios:
                raise ValueError("Run D baseline SE ratio accounting")
        if data["fitted"] is not None:
            fitted = data["fitted"]
            if fitted["minimum"] > fitted["maximum"]:
                raise ValueError("Run D fitted bounds")
            for bound in ("below_zero", "above_one"):
                if fitted[f"{bound}_count"] > n:
                    raise ValueError("Run D fitted count")
                _agree(fitted[f"{bound}_share"], fitted[f"{bound}_count"] / n, "fitted count/share")
            for term, concentration in data["concentration"].items():
                _agree(
                    concentration["fwl_numerator"],
                    concentration["coefficient_times_denominator"],
                    "concentration numerator",
                )
                coefficient = next(r["coefficient"] for r in data["results"] if r["term"] == term)
                _agree(
                    concentration["fwl_numerator"],
                    coefficient * concentration["fwl_denominator"],
                    "concentration coefficient",
                )
                for axis, count in (
                    ("households", sample["households"]),
                    ("stores", sample["stores"]),
                ):
                    if concentration[axis]["observed_groups"] != count:
                        raise ValueError("Run D concentration group universe")
                    _agree(
                        concentration[axis]["group_net_sum"],
                        concentration["fwl_numerator"],
                        "signed group-net numerator",
                    )
    expected_changes = _changes(report["variants"]["R2"]["results"], accepted)
    if report["baseline_comparison"] != expected_changes:
        raise ValueError("Run D baseline/later term-change accounting")
    expected_ratios = _ratios(
        report["variants"]["R4"]["covariance"], report["variants"]["R3"]["covariance"]
    )
    if report["variants"]["R4"]["covariance"]["se_ratios_vs_R3"] != expected_ratios:
        raise ValueError("Run D R4/R3 SE ratios")
