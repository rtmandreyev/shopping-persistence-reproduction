"""Frozen, outcome-free Phase 5 matrices and support diagnostics.

Only the six-column projection below may cross this boundary. No fitting or
response-vector interface lives here. Rank uses eps * max(shape) * s_max;
1e-12 is used only for roundoff/nonzero support accounting, not numerical rank.
"""

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

DESIGN_COLUMNS = (
    "household_key",
    "store_id",
    "week_no",
    "history_category",
    "log1p_prior_pure",
    "calendar_third",
)
HISTORY_COLUMNS = ("H0", "HB", "HP")
THIRDS = ("THIRD_1", "THIRD_2", "THIRD_3")
NONZERO_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DesignMatrix:
    """In-memory matrices; store labels are never included in support_report."""

    values: np.ndarray
    column_names: tuple[str, ...]
    nuisance: np.ndarray
    history: np.ndarray
    nuisance_column_names: tuple[str, ...]
    model: str


def _validate_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("design requires a DataFrame")
    if tuple(frame.columns) != DESIGN_COLUMNS:
        raise ValueError("design requires exactly the ordered outcome-free DESIGN_COLUMNS")
    if frame.empty or frame.isna().any().any():
        raise ValueError("design requires nonempty, nonmissing values")
    for name in ("household_key", "store_id"):
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series):
            if not np.isfinite(series.to_numpy(dtype=float)).all():
                raise ValueError("nonfinite design identifiers")
        elif not series.map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError("design identifiers must be finite numeric values or nonempty strings")
        try:
            sorted(series.unique())
        except TypeError as error:
            raise ValueError("design identifiers must have sortable levels") from error
    for name in ("week_no", "log1p_prior_pure"):
        if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(
            frame[name]
        ):
            raise ValueError("week and depth must be numeric")
    week = frame["week_no"].to_numpy(dtype=float)
    depth = frame["log1p_prior_pure"].to_numpy(dtype=float)
    if not np.isfinite(week).all() or not np.isfinite(depth).all():
        raise ValueError("nonfinite week or depth")
    if np.any((week < 1) | (week > 102) | (week != np.floor(week))):
        raise ValueError("week must be an integer in 1..102")
    if not frame["history_category"].isin(("H0", "HN", "HB", "HP")).all():
        raise ValueError("unknown history category")
    expected_third = np.asarray(THIRDS)[((week - 1) // 34).astype(int)]
    if not np.array_equal(frame["calendar_third"].to_numpy(), expected_third):
        raise ValueError("calendar third must match the frozen week boundaries")
    h0 = frame["history_category"].eq("H0").to_numpy()
    if np.any(depth < 0) or np.any(depth[h0] != 0) or np.any(depth[~h0] <= 0):
        raise ValueError("depth must be zero for H0 and positive for prior pure history")


def _singular_rank(values: np.ndarray) -> tuple[np.ndarray, int, float]:
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("matrix must be two-dimensional and finite")
    singular = np.linalg.svd(values, compute_uv=False)
    tolerance = (
        float(np.finfo(float).eps * max(values.shape) * singular[0]) if singular.size else 0.0
    )
    return singular, int(np.count_nonzero(singular > tolerance)), tolerance


def _full_rank(values: np.ndarray, label: str) -> None:
    _, rank, _ = _singular_rank(values)
    if rank != values.shape[1]:
        raise ValueError(f"rank deficient {label}: rank {rank}, columns {values.shape[1]}")


def _residualize(values: np.ndarray, context: np.ndarray) -> np.ndarray:
    _full_rank(context, "projection context")
    q, _ = np.linalg.qr(context, mode="reduced")
    return values - q @ (q.T @ values)


def build_matrix(frame: pd.DataFrame, model: str = "M0") -> DesignMatrix:
    """Return the frozen full-rank matrix, never receiving an outcome or weights.

    M0: intercept, H0, HB, HP, depth. M1: all observed stores (no global
    intercept), observed thirds after the first observed third, depth, H0/HB/HP.
    Observed store/third levels are sorted, including for later-calendar samples.
    """
    _validate_frame(frame)
    if model not in ("M0", "M1"):
        raise ValueError("model must be M0 or M1")
    history = np.column_stack(
        [frame["history_category"].eq(name).to_numpy(dtype=float) for name in HISTORY_COLUMNS]
    )
    depth = frame["log1p_prior_pure"].to_numpy(dtype=float)[:, None]
    if model == "M0":
        nuisance = np.column_stack((np.ones(len(frame)), depth))
        nuisance_names = ("intercept", "log1p_prior_pure")
        values = np.column_stack((nuisance[:, 0], history, depth))
        names = ("intercept", *HISTORY_COLUMNS, "log1p_prior_pure")
    else:
        stores = sorted(frame["store_id"].unique())
        thirds = sorted(frame["calendar_third"].unique())
        context = [frame["store_id"].eq(level).to_numpy(dtype=float) for level in stores]
        context += [frame["calendar_third"].eq(level).to_numpy(dtype=float) for level in thirds[1:]]
        nuisance = np.column_stack((*context, depth))
        nuisance_names = (
            *(f"store[{level}]" for level in stores),
            *thirds[1:],
            "log1p_prior_pure",
        )
        values = np.column_stack((nuisance, history))
        names = (*nuisance_names, *HISTORY_COLUMNS)
    _full_rank(nuisance, f"{model} context/depth")
    _full_rank(values, model)
    return DesignMatrix(values, names, nuisance, history, nuisance_names, model)


def _matrix_report(values: np.ndarray) -> dict:
    singular, rank, tolerance = _singular_rank(values)
    normalized = values / np.linalg.norm(values, axis=0)
    unit_singular, _, _ = _singular_rank(normalized)
    return {
        "rows": int(values.shape[0]),
        "columns": int(values.shape[1]),
        "rank": rank,
        "rank_tolerance": tolerance,
        "singular_values": singular.tolist(),
        "condition_number": float(singular[0] / singular[-1]),
        "unit_column_condition_number": float(unit_singular[0] / unit_singular[-1]),
    }


def _store_demean(values: np.ndarray, codes: np.ndarray, groups: int) -> np.ndarray:
    sums = np.zeros((groups, values.shape[1]))
    np.add.at(sums, codes, values)
    means = sums / np.bincount(codes, minlength=groups)[:, None]
    return values - means[codes]


def _independent_residualization(frame: pd.DataFrame, history: np.ndarray) -> np.ndarray:
    """Independent FWL route: absorb stores first, then QR depth/third.

    In particular this does not project on the full context a second time under
    a different solver name. The remaining controls must themselves be demeaned.
    """
    codes, levels = pd.factorize(frame["store_id"], sort=True)
    thirds = sorted(frame["calendar_third"].unique())
    controls = np.column_stack(
        [frame["log1p_prior_pure"].to_numpy(dtype=float)]
        + [frame["calendar_third"].eq(level).to_numpy(dtype=float) for level in thirds[1:]]
    )
    within_history = _store_demean(history, codes, len(levels))
    within_controls = _store_demean(controls, codes, len(levels))
    return _residualize(within_history, within_controls)


def _concentration(squares: np.ndarray, codes: np.ndarray, groups: int) -> dict:
    sums = np.bincount(codes, weights=squares, minlength=groups)
    total = float(sums.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("no finite residualized support")
    shares = np.sort(sums / total)[::-1]
    fractions = (0.01, 0.05, 0.10)
    return {
        "observed_groups": int(groups),
        "nonzero_support_groups": int(np.count_nonzero(sums > NONZERO_TOLERANCE)),
        "largest_share": float(shares[0]),
        "effective_groups_inverse_hhi": float(1 / np.sum(shares**2)),
        "top_fraction_shares": {
            f"{fraction:.2f}": float(shares[: ceil(fraction * groups)].sum())
            for fraction in fractions
        },
    }


def support_report(frame: pd.DataFrame) -> dict:
    """Identifier-free, nested design-only diagnostics for the passed population.

    Residual variances use N. Shares retain raw centered indicator SS. Concentration
    is squared residual support, including zero-support groups in top-fraction
    denominators; joint support sums H0/HB/HP squared residuals under HN coding.
    No support threshold, sample selection, or inference decision is made here.
    """
    matrices = {model: build_matrix(frame, model) for model in ("M0", "M1")}
    main = matrices["M1"]
    residual = _residualize(main.history, main.nuisance)
    _full_rank(residual, "residualized history")
    independent = _independent_residualization(frame, main.history)
    difference = np.abs(residual - independent)
    if not np.allclose(residual, independent, rtol=1e-10, atol=1e-10):
        raise ValueError("independent residualization routes disagree")
    terms = {}
    partial_residuals = []
    for position, name in enumerate(HISTORY_COLUMNS):
        raw = main.history[:, position]
        values = residual[:, position]
        ss = float(values @ values)
        centered = raw - raw.mean()
        raw_ss = float(centered @ centered)
        other = np.delete(residual, position, axis=1)
        partial = _residualize(values[:, None], other)[:, 0]
        partial_residuals.append(partial)
        terms[name] = {
            "residual_ss": ss,
            "raw_centered_ss": raw_ss,
            "remaining_centered_ss_share": ss / raw_ss,
            "mean": float(values.mean()),
            "variance_n": float(values.var(ddof=0)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "nonzero_rows": int(np.count_nonzero(np.abs(values) > NONZERO_TOLERANCE)),
            "ss_partialling_other_history": float(partial @ partial),
        }
    concentration = {}
    partial_concentration = {}
    q_history, _ = np.linalg.qr(residual, mode="reduced")
    for axis in ("household_key", "store_id", "week_no"):
        codes, levels = pd.factorize(frame[axis], sort=True)
        concentration[axis] = {
            name: _concentration(residual[:, position] ** 2, codes, len(levels))
            for position, name in enumerate(HISTORY_COLUMNS)
        }
        concentration[axis]["joint"] = _concentration(
            np.sum(residual**2, axis=1), codes, len(levels)
        )
        concentration[axis]["joint_subspace_leverage"] = _concentration(
            np.sum(q_history**2, axis=1), codes, len(levels)
        )
        partial_concentration[axis] = {
            name: _concentration(partial_residuals[position] ** 2, codes, len(levels))
            for position, name in enumerate(HISTORY_COLUMNS)
        }
    return {
        "models": {name: _matrix_report(matrix.values) for name, matrix in matrices.items()},
        "context_depth": _matrix_report(main.nuisance),
        "residualized_history": {**_matrix_report(residual), "terms": terms},
        "concentration": concentration,
        "partial_history_concentration": partial_concentration,
        "independent_residualization": {
            "maximum_absolute_difference": float(difference.max()),
            "frobenius_difference": float(np.linalg.norm(difference)),
            "agrees": True,
        },
        "conventions": {
            "history_reference": "HN",
            "variance_denominator": "N",
            "rank_tolerance": "eps * max(shape) * largest_singular_value",
            "nonzero_tolerance": NONZERO_TOLERANCE,
            "top_fraction_denominator": "all_observed_groups_ceiling",
            "effective_count": "inverse_hhi",
        },
    }
