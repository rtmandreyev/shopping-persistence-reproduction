"""Frozen Run C projections and household × response-week covariance only.

The caller owns the accepted empirical sample gate. This module never selects
rows, reads data, or exposes alternative weights/specifications/covariances.
Full native covariance is retained even when scalar inference is blocked.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t

from complete_journey.choice import covariance as cluster
from complete_journey.choice.design import DesignMatrix, build_matrix

FOCAL_TERMS = ("H0", "HB", "HP", "log1p_prior_pure")
AGREEMENT_ATOL = 1e-10
AGREEMENT_RTOL = 1e-8
RESULT_FIELDS = (
    "model",
    "term",
    "coefficient",
    "percentage_points",
    "native_se",
    "ci_lower",
    "ci_upper",
    "t_statistic",
    "p_value",
    "inference_status",
    "focal_variance",
    "nobs",
    "households",
    "stores",
    "weeks",
    "store_effects",
    "calendar_third_effects",
    "household_effects",
    "weights",
    "history_reference",
    "covariance",
    "household_clusters",
    "week_clusters",
    "intersection_clusters",
    "reference_distribution",
    "reference_df",
    "covariance_psd",
    "covariance_indefinite",
    "r_squared",
    "adjusted_r_squared",
)
# Backward-compatible descriptive name; both are the same deterministic tuple.
RESULT_COLUMNS = RESULT_FIELDS
_RESULT_INTS = frozenset(
    (
        "nobs",
        "households",
        "stores",
        "weeks",
        "household_clusters",
        "week_clusters",
        "intersection_clusters",
        "reference_df",
    )
)
_RESULT_BOOLS = frozenset(
    (
        "store_effects",
        "calendar_third_effects",
        "household_effects",
        "covariance_psd",
        "covariance_indefinite",
    )
)
_RESULT_STRINGS = frozenset(
    (
        "model",
        "term",
        "inference_status",
        "weights",
        "history_reference",
        "covariance",
        "reference_distribution",
    )
)
_RESULT_OPTIONAL = frozenset(
    (
        "native_se",
        "ci_lower",
        "ci_upper",
        "t_statistic",
        "p_value",
        "r_squared",
        "adjusted_r_squared",
    )
)
# Static closed row schema, requiring exact built-in scalar types (not bool as int).
RESULT_ROW_SCHEMA = {
    name: (int,)
    if name in _RESULT_INTS
    else (bool,)
    if name in _RESULT_BOOLS
    else (str,)
    if name in _RESULT_STRINGS
    else (float, type(None))
    if name in _RESULT_OPTIONAL
    else (float,)
    for name in RESULT_FIELDS
}


def validate_result_rows(rows: list[dict]) -> None:
    """Validate four rows for one model or eight for both; no IO or fitting.

    The empirical 6,544-row gate is the caller's separate contract. This closed
    scalar validator also supports synthetic inputs and requires both models,
    when present, to describe exactly the same response population.
    """
    if type(rows) is not list or len(rows) not in (4, 8):
        raise ValueError("closed Run C result rows require four or eight rows")
    groups = {}
    inferential = ("ci_lower", "ci_upper", "t_statistic", "p_value")
    for row in rows:
        if type(row) is not dict or set(row) != set(RESULT_FIELDS):
            raise ValueError("closed Run C result keys violated")
        for field, types in RESULT_ROW_SCHEMA.items():
            value = row[field]
            if type(value) not in types:
                raise ValueError(f"closed Run C result scalar type violated: {field}")
            if type(value) is float and field != "focal_variance" and not np.isfinite(value):
                raise ValueError(f"nonfinite Run C result scalar: {field}")
        if row["model"] not in ("M0", "M1") or row["term"] not in FOCAL_TERMS:
            raise ValueError("Run C result model/term metadata violated")
        for field, value in {
            "weights": "none",
            "history_reference": "HN",
            "covariance": "household_week",
            "reference_distribution": "Student_t",
            "household_effects": False,
        }.items():
            if row[field] != value:
                raise ValueError(f"Run C result frozen metadata violated: {field}")
        expected_context = row["model"] == "M1"
        if (
            row["store_effects"] != expected_context
            or row["calendar_third_effects"] != expected_context
        ):
            raise ValueError("Run C result context metadata violated")
        if any(row[field] <= 0 for field in _RESULT_INTS):
            raise ValueError("Run C result counts/df must be positive")
        if any(row[field] > row["nobs"] for field in _RESULT_INTS if field != "nobs"):
            raise ValueError("Run C result counts exceed response population")
        if (
            row["household_clusters"] != row["households"]
            or row["week_clusters"] != row["weeks"]
            or row["reference_df"] != min(row["households"], row["weeks"]) - 1
        ):
            raise ValueError("Run C result cluster/df metadata violated")
        if row["intersection_clusters"] < max(row["households"], row["weeks"]):
            raise ValueError("Run C result intersection count violated")
        if not np.isclose(
            row["percentage_points"],
            100 * row["coefficient"],
            atol=AGREEMENT_ATOL,
            rtol=AGREEMENT_RTOL,
        ):
            raise ValueError("Run C result coefficient units violated")
        variance = row["focal_variance"]
        positive = bool(np.isfinite(variance) and variance > 0)
        if positive:
            if row["native_se"] is None or not np.isclose(
                row["native_se"], np.sqrt(variance), atol=AGREEMENT_ATOL, rtol=AGREEMENT_RTOL
            ):
                raise ValueError("Run C result diagnostic SE violated")
        elif row["native_se"] is not None:
            raise ValueError("Run C result invalid variance has an SE")
        status = row["inference_status"]
        if status not in ("PERMITTED", "PROVISIONAL_RESEARCHER_GATED", "STOP_FOCAL_VARIANCE"):
            raise ValueError("Run C result inference status violated")
        if row["covariance_psd"] and row["covariance_indefinite"]:
            raise ValueError("Run C result covariance flags conflict")
        if status == "PERMITTED":
            if (
                not positive
                or not row["covariance_psd"]
                or any(row[field] is None for field in inferential)
            ):
                raise ValueError("Run C result unauthorized scalar inference")
            if not 0 <= row["p_value"] <= 1 or row["ci_lower"] > row["ci_upper"]:
                raise ValueError("Run C result interval/tail area violated")
        elif any(row[field] is not None for field in inferential):
            raise ValueError("Run C result gated inference must be withheld")
        groups.setdefault(row["model"], []).append(row)
    if len(groups) * len(FOCAL_TERMS) != len(rows):
        raise ValueError("Run C result duplicated or missing model rows")
    population = (
        "nobs",
        "households",
        "stores",
        "weeks",
        "household_clusters",
        "week_clusters",
        "intersection_clusters",
        "reference_df",
    )
    for group in groups.values():
        if len(group) != 4 or {row["term"] for row in group} != set(FOCAL_TERMS):
            raise ValueError("Run C result duplicated or missing focal terms")
        first = group[0]
        for row in group:
            if any(
                row[field] != first[field]
                for field in (
                    *population,
                    "inference_status",
                    "covariance_psd",
                    "covariance_indefinite",
                    "r_squared",
                    "adjusted_r_squared",
                )
            ):
                raise ValueError("Run C result model-level metadata disagree")
        all_positive = all(
            np.isfinite(row["focal_variance"]) and row["focal_variance"] > 0 for row in group
        )
        if (first["inference_status"] == "STOP_FOCAL_VARIANCE") != (not all_positive):
            raise ValueError("Run C result model focal variance gate violated")
        if first["inference_status"] == "PROVISIONAL_RESEARCHER_GATED" and first["covariance_psd"]:
            raise ValueError("Run C result PSD inference status conflicts")
    if len(groups) == 2 and any(
        groups["M0"][0][field] != groups["M1"][0][field] for field in population
    ):
        raise ValueError("Run C result response populations disagree")


@dataclass(frozen=True)
class CoreFit:
    """Internal arrays, not an identifier-bearing publication interface."""

    model: str
    design: DesignMatrix
    coefficients: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    influence: np.ndarray
    q: np.ndarray
    covariance: np.ndarray
    point_checks: list[dict]
    covariance_check: dict
    covariance_diagnostics: dict
    component_counts: tuple[int, int, int]
    component_corrections: tuple[float, float, float]
    reference_df: int


def _independent_points(frame: pd.DataFrame, y: np.ndarray, design: DesignMatrix) -> np.ndarray:
    """Direct M0 LS; M1 store demeaning then explicit third-only FWL.

    Independently construct all four focal terms for M1, rather than reusing
    design residualization or repeating the main full-matrix estimator.
    """
    if design.model == "M0":
        independent, _, rank, _ = np.linalg.lstsq(
            design.values, y, rcond=np.finfo(float).eps * max(design.values.shape)
        )
        if rank != design.values.shape[1]:
            raise ValueError("independent M0 rank disagreement")
        return independent[[design.column_names.index(term) for term in FOCAL_TERMS]]
    terms = np.column_stack(
        [frame["history_category"].eq(term).to_numpy(dtype=float) for term in FOCAL_TERMS[:3]]
        + [frame["log1p_prior_pure"].to_numpy(dtype=float)]
    )
    thirds = sorted(frame["calendar_third"].unique())
    third_terms = (
        np.column_stack(
            [frame["calendar_third"].eq(level).to_numpy(dtype=float) for level in thirds[1:]]
        )
        if len(thirds) > 1
        else np.empty((len(frame), 0))
    )
    within = np.column_stack((y, terms, third_terms))
    stores = frame["store_id"].to_numpy()
    for store in pd.unique(stores):
        mask = stores == store
        within[mask] -= within[mask].mean(axis=0)
    residualized = within[:, :5]
    if third_terms.shape[1]:
        context = within[:, 5:]
        tq, tr = np.linalg.qr(context, mode="reduced")
        if np.linalg.matrix_rank(tr) != context.shape[1]:
            raise ValueError("independent M1 third rank disagreement")
        residualized = residualized - tq @ (tq.T @ residualized)
    focal = residualized[:, 1:]
    independent, _, rank, _ = np.linalg.lstsq(
        focal, residualized[:, 0], rcond=np.finfo(float).eps * max(focal.shape)
    )
    if rank != len(FOCAL_TERMS):
        raise ValueError("independent M1 FWL rank disagreement")
    return independent


def _point_checks(
    frame: pd.DataFrame, y: np.ndarray, design: DesignMatrix, beta: np.ndarray
) -> list[dict]:
    independent = _independent_points(frame, y, design)
    checks = []
    for term, value in zip(FOCAL_TERMS, independent, strict=True):
        main = float(beta[design.column_names.index(term)])
        agrees = bool(np.isclose(main, value, atol=AGREEMENT_ATOL, rtol=AGREEMENT_RTOL))
        checks.append(
            {
                "term": term,
                "main": main,
                "independent": float(value),
                "absolute_difference": float(abs(main - value)),
                "atol": AGREEMENT_ATOL,
                "rtol": AGREEMENT_RTOL,
                "agrees": agrees,
            }
        )
        if not agrees:
            raise ValueError(f"independent point estimates disagree for {design.model} {term}")
    return checks


def _independent_covariance(
    frame: pd.DataFrame, x: np.ndarray, residual: np.ndarray
) -> tuple[np.ndarray, list[dict]]:
    """SVD bread and explicit dictionary score sums; no shared grouping/CR1 helper."""
    n, k = x.shape
    bread = np.linalg.pinv(x, rcond=np.finfo(float).eps * max(x.shape))
    scores = bread.T * residual[:, None]
    keys = list(frame[["household_key", "week_no"]].itertuples(index=False, name=None))
    result = np.zeros((k, k))
    facts = []
    for name, selection, sign in (
        ("household", (0,), 1),
        ("week", (1,), 1),
        ("household_week_intersection", (0, 1), -1),
    ):
        sums = {}
        sizes = {}
        for key, score in zip(keys, scores, strict=True):
            group = tuple(key[position] for position in selection)
            sums[group] = sums.get(group, np.zeros(k)) + score
            sizes[group] = sizes.get(group, 0) + 1
        g = len(sums)
        if g <= 1 or n <= k:
            raise ValueError("independent CR1 requires G > 1 and N > K")
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
                "repeated_cells": sum(size > 1 for size in sizes.values()),
                "events_in_repeated_cells": sum(size for size in sizes.values() if size > 1),
            }
        )
    return result, facts


def _covariance_check(
    frame: pd.DataFrame, design: DesignMatrix, residual: np.ndarray, native: np.ndarray
) -> dict:
    independent, facts = _independent_covariance(frame, design.values, residual)
    agrees = bool(
        np.isfinite(native).all()
        and np.isfinite(independent).all()
        and np.allclose(native, independent, atol=AGREEMENT_ATOL, rtol=AGREEMENT_RTOL)
    )
    if not agrees:
        raise ValueError(
            "independent household-week covariance implementations disagree or are nonfinite"
        )
    focal = []
    for term in FOCAL_TERMS:
        index = design.column_names.index(term)
        main, other = float(native[index, index]), float(independent[index, index])
        positive = main > 0 and other > 0
        variance_agrees = bool(np.isclose(main, other, atol=AGREEMENT_ATOL, rtol=AGREEMENT_RTOL))
        se_agrees = (
            bool(
                np.isclose(np.sqrt(main), np.sqrt(other), atol=AGREEMENT_ATOL, rtol=AGREEMENT_RTOL)
            )
            if positive
            else None
        )
        if not variance_agrees or se_agrees is False:
            raise ValueError(
                f"independent focal covariance variance/SE implementations disagree for {term}"
            )
        focal.append(
            {
                "term": term,
                "main_variance": main,
                "independent_variance": other,
                "variance_absolute_difference": float(abs(main - other)),
                "variance_agrees": variance_agrees,
                "se_agrees": se_agrees,
                "main_se": float(np.sqrt(main)) if main > 0 else None,
                "independent_se": float(np.sqrt(other)) if other > 0 else None,
                "se_absolute_difference": float(abs(np.sqrt(main) - np.sqrt(other)))
                if positive
                else None,
            }
        )
    return {
        "agrees": agrees,
        "maximum_absolute_difference": float(np.max(np.abs(native - independent))),
        "atol": AGREEMENT_ATOL,
        "rtol": AGREEMENT_RTOL,
        "components": facts,
        "focal": focal,
    }


def _covariance_diagnostics(native: np.ndarray, names: tuple[str, ...]) -> dict:
    finite = bool(np.isfinite(native).all())
    symmetric = bool(finite and np.allclose(native, native.T, atol=1e-12, rtol=1e-10))
    exact = bool(np.array_equal(native, native.T))
    eigenvalues = None
    tolerance = None
    psd = indefinite = False
    negative_count = material_negative_count = None
    minimum = maximum = None
    if finite and symmetric:
        # Numerical diagnosis only: the returned native covariance is untouched.
        eigen = np.linalg.eigvalsh((native + native.T) / 2)
        tolerance = float(np.finfo(float).eps * len(names) * np.max(np.abs(eigen)))
        eigenvalues = eigen.tolist()
        minimum, maximum = float(eigen.min()), float(eigen.max())
        negative_count = int(np.count_nonzero(eigen < 0))
        material_negative_count = int(np.count_nonzero(eigen < -tolerance))
        psd = bool(minimum >= -tolerance)
        indefinite = bool(minimum < -tolerance and maximum > tolerance)
    focal = {}
    for term in FOCAL_TERMS:
        variance = float(native[names.index(term), names.index(term)])
        status = (
            "POSITIVE"
            if np.isfinite(variance) and variance > 0
            else (
                "NONFINITE"
                if not np.isfinite(variance)
                else "ZERO"
                if variance == 0
                else "NEGATIVE"
            )
        )
        focal[term] = {
            "variance": variance,
            "finite_positive": status == "POSITIVE",
            "status": status,
        }
    positive = all(item["finite_positive"] for item in focal.values())
    status = (
        "STOP_FOCAL_VARIANCE"
        if not positive
        else ("PERMITTED" if finite and symmetric and psd else "PROVISIONAL_RESEARCHER_GATED")
    )
    return {
        "finite": finite,
        "symmetric": symmetric,
        "exactly_symmetric": exact,
        "symmetry_maximum_absolute_difference": float(np.max(np.abs(native - native.T)))
        if finite
        else None,
        "eigenvalues": eigenvalues,
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": maximum,
        "negative_eigenvalue_count": negative_count,
        "material_negative_eigenvalue_count": material_negative_count,
        "eigenvalue_tolerance": tolerance,
        "full_matrix_psd": psd,
        "full_matrix_indefinite": indefinite,
        "focal_variances": focal,
        "focal_variances_finite_positive": positive,
        "inference_status": status,
        "covariance_repaired": False,
    }


def fit_projection(frame: pd.DataFrame, outcome: np.ndarray, model: str) -> CoreFit:
    """Fit exactly M0/M1 by QR, on every supplied row with equal event weights."""
    design = build_matrix(frame, model)
    if not isinstance(outcome, np.ndarray) or outcome.shape != (len(frame),):
        raise ValueError("outcome must be a row-aligned ndarray vector")
    if not np.issubdtype(outcome.dtype, np.number) and outcome.dtype != np.dtype(bool):
        raise ValueError("outcome must be finite and binary")
    if (
        np.iscomplexobj(outcome)
        or not np.isfinite(outcome).all()
        or not np.isin(outcome, (0, 1)).all()
    ):
        raise ValueError("outcome must be finite and binary")
    y = outcome.astype(float, copy=False)
    x = design.values
    n, k = x.shape
    if n <= k:
        raise ValueError("fit requires N > full equivalent design rank K")
    q, r = np.linalg.qr(x, mode="reduced")
    coefficients = np.linalg.solve(r, q.T @ y)
    fitted = x @ coefficients  # Includes every store/third term; never clipped.
    residual = y - fitted
    influence = np.linalg.solve(r, q.T).T * residual[:, None]
    point_checks = _point_checks(frame, y, design, coefficients)
    household = pd.factorize(frame["household_key"], sort=True)[0]
    week = pd.factorize(frame["week_no"], sort=True)[0]
    intersection = pd.MultiIndex.from_arrays((household, week)).factorize(sort=True)[0]
    codes = (household, week, intersection)
    counts = tuple(int(np.unique(component).size) for component in codes)
    corrections = tuple(cluster._cr1(g, n, k) for g in counts)
    components = [cluster._component_covariance(influence, component, k) for component in codes]
    native = components[0] + components[1] - components[2]
    check = _covariance_check(frame, design, residual, native)
    diagnostics = _covariance_diagnostics(native, design.column_names)
    diagnostics["structure"] = {
        "nobs": n,
        "households": counts[0],
        "stores": int(frame["store_id"].nunique()),
        "weeks": counts[1],
    }
    sse = float(residual @ residual)
    centered = y - y.mean()
    sst = float(centered @ centered)
    r_squared = float(1 - sse / sst) if sst > 0 else None
    adjusted = float(1 - (1 - r_squared) * (n - 1) / (n - k)) if r_squared is not None else None
    diagnostics["fit_statistics"] = {
        "sse": sse,
        "centered_sst": sst,
        "full_design_rank": k,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted,
        "r_squared_definition": "1-SSE/SST; SST=sum((Y-mean(Y))**2); total fitted values",
        "adjusted_r_squared_definition": "1-(1-R_squared)*(N-1)/(N-K); K=full_equivalent_design_rank",
        "constant_outcome_convention": "R_squared_and_adjusted_R_squared_None_when_centered_SST_zero",
    }
    return CoreFit(
        model,
        design,
        coefficients,
        fitted,
        residual,
        influence,
        q,
        native,
        point_checks,
        check,
        diagnostics,
        counts,
        corrections,
        min(counts[:2]) - 1,
    )


def result_rows(fit: CoreFit) -> list[dict]:
    """Four closed, identifier-free scalar rows; eight for the two frozen models.

    Native positive-finite SEs remain diagnostics when inference is gated. CI,
    t and p are all absent unless every focal variance is positive and the full
    matrix is finite, symmetric and PSD. This does not settle coverage/publication.
    """
    diagnostic = fit.covariance_diagnostics
    structure = diagnostic["structure"]
    statistics = diagnostic["fit_statistics"]
    permitted = diagnostic["inference_status"] == "PERMITTED"
    critical = float(t.ppf(0.975, fit.reference_df)) if permitted else None
    rows = []
    for term in FOCAL_TERMS:
        position = fit.design.column_names.index(term)
        beta = float(fit.coefficients[position])
        variance = float(fit.covariance[position, position])
        se = float(np.sqrt(variance)) if np.isfinite(variance) and variance > 0 else None
        statistic = float(beta / se) if permitted else None
        row = {
            "model": fit.model,
            "term": term,
            "coefficient": beta,
            "percentage_points": 100 * beta,
            "native_se": se,
            "ci_lower": float(beta - critical * se) if permitted else None,
            "ci_upper": float(beta + critical * se) if permitted else None,
            "t_statistic": statistic,
            "p_value": float(2 * t.sf(abs(statistic), fit.reference_df)) if permitted else None,
            "inference_status": diagnostic["inference_status"],
            "focal_variance": variance,
            **structure,
            "store_effects": fit.model == "M1",
            "calendar_third_effects": fit.model == "M1",
            "household_effects": False,
            "weights": "none",
            "history_reference": "HN",
            "covariance": "household_week",
            "household_clusters": fit.component_counts[0],
            "week_clusters": fit.component_counts[1],
            "intersection_clusters": fit.component_counts[2],
            "reference_distribution": "Student_t",
            "reference_df": fit.reference_df,
            "covariance_psd": diagnostic["full_matrix_psd"],
            "covariance_indefinite": diagnostic["full_matrix_indefinite"],
            "r_squared": statistics["r_squared"],
            "adjusted_r_squared": statistics["adjusted_r_squared"],
        }
        if tuple(row) != RESULT_COLUMNS:
            raise ValueError("closed Run C result schema violated")
        rows.append(row)
    return rows
