"""Synthetic-only future OLS/LPM and cluster-covariance verification.

Run B exposes only run_synthetic_preflight(), which takes no data or outcome.
Private fitting seams require a tagged synthetic fixture, not an event table.
No real response is read. Internal numerical arrays never enter its closed report.

CR1 is applied separately to each meat, including the true intersection:
G/(G-1) * (N-1)/(N-K), with K the full equivalent design rank. Two-way
covariance is A + B - intersection, not A + B - event-level HC covariance.
Future univariate Student t reference df is min(G_A,G_B)-1, or G_household-1
for the one-component benchmark. No intervals, tail areas, joint tests, scalar
exceptions, eigenvalue clipping or covariance repair are implemented here.
An indefinite synthetic matrix is reported mechanically, not resolved and not
permission to publish real-data scalar inference.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.api import OLS as _OLS
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups

from complete_journey.choice.design import (
    DESIGN_COLUMNS,
    HISTORY_COLUMNS,
    DesignMatrix,
    _full_rank,
    _residualize,
    build_matrix,
)

_SYNTHETIC_TOKEN = object()
_COVARIANCE_KINDS = ("household_week", "household_store", "household_only")
_AGREEMENT_RTOL = 1e-8
_AGREEMENT_ATOL = 1e-10

# Static, recursively closed report contract, inspectable without fitting anything.
# dict means exactly those keys; tuple means a JSON list of exactly that length;
# type leaves require exact built-in types (bool is NOT accepted as int).
_CONVENTIONS = {
    "fit": "unweighted_QR_total_fitted_no_clipping",
    "cr1": "per_component_G/(G-1)*(N-1)/(N-K)_full_design_rank",
    "two_way": "A_plus_B_minus_true_intersection",
    "reference": "future_univariate_Student_t_min_G_minus_one",
    "covariance_repair": False,
    "joint_tests": False,
    "scalar_exception_decision": False,
    "real_data_inference_authorized": False,
}
_VALIDITY_SCHEMA = {
    "finite": bool,
    "symmetric": bool,
    "focal_variances_finite_positive": bool,
    "full_matrix_psd": bool,
    "full_matrix_indefinite": bool,
}
REPORT_SCHEMA = {
    "scope": str,
    "conventions": {name: type(value) for name, value in _CONVENTIONS.items()},
    "structure": {
        "households": int,
        "stores": int,
        "weeks": int,
        "household_week_cells": int,
        "repeated_household_week_cells": int,
        "events_in_repeated_household_week_cells": int,
        "household_store_pairs": int,
        "multi_store_households": int,
    },
    "models": {
        model: {
            "rows": int,
            "columns": int,
            "rank": int,
            "binary_outcome": bool,
            "equal_event_weights": bool,
            "qr_direct_agrees": bool,
            "qr_fwl_agrees": bool,
            "unclipped_below_zero_count": int,
            "unclipped_above_one_count": int,
            "covariance": {
                kind: {
                    "component_cluster_counts": (int,)
                    if kind == "household_only"
                    else (int, int, int),
                    "reference_df": int,
                    "manual_agrees": bool,
                    "statsmodels_agrees": bool,
                    "validity": dict(_VALIDITY_SCHEMA),
                }
                for kind in _COVARIANCE_KINDS
            },
        }
        for model in ("M0", "M1")
    },
}


def _schema_leaf_paths(schema: object, prefix: str = "") -> tuple[str, ...]:
    if isinstance(schema, dict):
        return tuple(
            path
            for name, child in schema.items()
            for path in _schema_leaf_paths(child, f"{prefix}.{name}" if prefix else name)
        )
    if isinstance(schema, tuple):
        return tuple(
            path
            for index, child in enumerate(schema, start=1)
            for path in _schema_leaf_paths(child, f"{prefix}.{index}")
        )
    return () if schema is str else (prefix,)


# Dotted numeric/boolean scalar-leaf paths; list positions are one-based, matching
# preflight's numeric flattener. String metadata remains in REPORT_SCHEMA only.
# The ordered universe is derived only from the static schema, never from output/Y.
SYNTHETIC_METRICS = _schema_leaf_paths(REPORT_SCHEMA)


def _validate_report_schema(value: object, schema: object, path: str = "report") -> None:
    if isinstance(schema, dict):
        if type(value) is not dict or set(value) != set(schema):
            raise ValueError(f"closed synthetic report keys violated at {path}")
        for name, child in schema.items():
            _validate_report_schema(value[name], child, f"{path}.{name}")
    elif isinstance(schema, tuple):
        if type(value) is not list or len(value) != len(schema):
            raise ValueError(f"closed synthetic report list violated at {path}")
        for index, child in enumerate(schema):
            _validate_report_schema(value[index], child, f"{path}.{index}")
    elif type(value) is not schema:
        raise ValueError(f"closed synthetic report scalar type violated at {path}")


def validate_synthetic_report(report: dict) -> None:
    """Validate closed keys/types and frozen metadata without fitting or reading data."""
    _validate_report_schema(report, REPORT_SCHEMA)
    if report["scope"] != "SYNTHETIC_ONLY_NON_EVIDENCE" or report["conventions"] != _CONVENTIONS:
        raise ValueError("synthetic report metadata does not match the frozen conventions")


@dataclass(frozen=True)
class _SyntheticFixture:
    frame: pd.DataFrame
    outcome: np.ndarray
    _token: object = field(repr=False)


@dataclass(frozen=True)
class _SyntheticFit:
    fixture: _SyntheticFixture
    design: DesignMatrix
    coefficients: np.ndarray
    residual: np.ndarray
    fitted: np.ndarray
    influence: np.ndarray
    direct_agrees: bool
    fwl_agrees: bool
    _token: object = field(repr=False)


def _make_synthetic_fixture() -> _SyntheticFixture:
    """Fixed design, not selected using significance or covariance results.

    Deliberate repeated household-week cells and household movers exercise both
    intersection operators. A nonlinear binary depth threshold with a long depth
    tail exercises unbounded linear fits. This is not a factual history ledger.
    """
    rng = np.random.default_rng(5021)
    rows = []
    prior_counts = (1, 2, 3, 5, 8, 13, 21, 1000)
    for household in range(24):
        for position, week in enumerate((2, 8, 16, 28, 36, 45, 57, 66, 72, 81, 91, 101)):
            repeats = 2 if household % 3 == 0 and position % 4 == 0 else 1
            for repeat in range(repeats):
                category = ("H0", "HN", "HB", "HP")[int(rng.integers(0, 4))]
                prior = 0 if category == "H0" else prior_counts[int(rng.integers(0, 8))]
                rows.append(
                    (
                        household,
                        (household + position + repeat) % 5,
                        week,
                        category,
                        float(np.log1p(prior)),
                        f"THIRD_{(week - 1) // 34 + 1}",
                    )
                )
    frame = pd.DataFrame(rows, columns=DESIGN_COLUMNS)
    # A fixed, intentionally nonlinear synthetic binary rule; never a real label.
    outcome = (frame["log1p_prior_pure"].to_numpy() >= np.log1p(5)).astype(float)
    return _SyntheticFixture(frame, outcome, _SYNTHETIC_TOKEN)


def _require_synthetic(fixture: _SyntheticFixture) -> None:
    if not isinstance(fixture, _SyntheticFixture) or fixture._token is not _SYNTHETIC_TOKEN:
        raise TypeError("only a tagged synthetic fixture is accepted; no event-table fitting API")
    y = fixture.outcome
    if not isinstance(y, np.ndarray) or y.shape != (len(fixture.frame),):
        raise ValueError("synthetic outcome must be a row-aligned vector")
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("synthetic LPM outcome must be finite and binary")


def _agree(left: np.ndarray, right: np.ndarray, label: str) -> bool:
    if not np.allclose(left, right, rtol=_AGREEMENT_RTOL, atol=_AGREEMENT_ATOL):
        raise ValueError(f"synthetic {label} implementations disagree")
    return True


def _fit_synthetic(fixture: _SyntheticFixture, model: str) -> _SyntheticFit:
    _require_synthetic(fixture)
    design = build_matrix(fixture.frame, model)
    x, y = design.values, fixture.outcome
    if x.shape[0] <= x.shape[1]:
        raise ValueError("synthetic fit requires N > full equivalent design rank K")
    q, r = np.linalg.qr(x, mode="reduced")
    coefficients = np.linalg.solve(r, q.T @ y)
    fitted = x @ coefficients  # Total fitted values, including all store effects; never clipped.
    residual = y - fitted
    direct, _, direct_rank, _ = np.linalg.lstsq(x, y, rcond=np.finfo(float).eps * max(x.shape))
    if direct_rank != x.shape[1]:
        raise ValueError("rank deficient direct synthetic fit")
    direct_agrees = _agree(coefficients, direct, "QR/direct least-squares")
    _agree(fitted, x @ direct, "total fitted values")
    h_residual = _residualize(design.history, design.nuisance)
    y_residual = _residualize(y[:, None], design.nuisance)[:, 0]
    _full_rank(h_residual, "synthetic FWL history")
    hq, hr = np.linalg.qr(h_residual, mode="reduced")
    fwl = np.linalg.solve(hr, hq.T @ y_residual)
    indices = [design.column_names.index(name) for name in HISTORY_COLUMNS]
    fwl_agrees = _agree(coefficients[indices], fwl, "FWL history")
    # (X'X)^-1 X' from triangular QR solves, without normal-equation inversion.
    influence = np.linalg.solve(r, q.T).T * residual[:, None]
    return _SyntheticFit(
        fixture,
        design,
        coefficients,
        residual,
        fitted,
        influence,
        direct_agrees,
        fwl_agrees,
        _SYNTHETIC_TOKEN,
    )


def _cr1(groups: int, n: int, k: int) -> float:
    if groups <= 1:
        raise ValueError("CR1 requires more than one observed cluster per component")
    if n <= k:
        raise ValueError("CR1 requires N > full equivalent design rank K")
    return float(groups / (groups - 1) * (n - 1) / (n - k))


def _cluster_codes(frame: pd.DataFrame, kind: str) -> tuple[np.ndarray, ...]:
    if kind not in _COVARIANCE_KINDS:
        raise ValueError("unknown synthetic covariance configuration")
    household = pd.factorize(frame["household_key"], sort=True)[0]
    if kind == "household_only":
        return (household,)
    second_name = "week_no" if kind == "household_week" else "store_id"
    second = pd.factorize(frame[second_name], sort=True)[0]
    # Factorize TRUE joint keys; repeated cells share a score-summing cluster.
    joint = pd.MultiIndex.from_arrays((household, second)).factorize(sort=True)[0]
    return household, second, joint


def _require_fit(fit: _SyntheticFit) -> None:
    if not isinstance(fit, _SyntheticFit) or fit._token is not _SYNTHETIC_TOKEN:
        raise TypeError("cluster verification requires an internal synthetic fit")
    _require_synthetic(fit.fixture)


def _component_covariance(influence: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    n = len(codes)
    groups = int(codes.max()) + 1
    sums = np.zeros((groups, influence.shape[1]))
    np.add.at(sums, codes, influence)
    return _cr1(groups, n, k) * (sums.T @ sums)


def _synthetic_covariance(fit: _SyntheticFit, kind: str) -> np.ndarray:
    _require_fit(fit)
    components = _cluster_codes(fit.fixture.frame, kind)
    k = fit.design.values.shape[1]
    covariance = _component_covariance(fit.influence, components[0], k)
    if len(components) == 3:
        covariance = (
            covariance
            + _component_covariance(fit.influence, components[1], k)
            - _component_covariance(fit.influence, components[2], k)
        )
    return covariance


def _manual_synthetic_covariance(fit: _SyntheticFit, kind: str) -> np.ndarray:
    """Independent loop aggregation and SVD pseudoinverse bread for testing."""
    _require_fit(fit)
    x = fit.design.values
    n, k = x.shape
    pinv = np.linalg.pinv(x, rcond=np.finfo(float).eps * max(x.shape))
    contributions = pinv.T * fit.residual[:, None]
    columns = ["household_key"]
    if kind == "household_week":
        columns.append("week_no")
    elif kind == "household_store":
        columns.append("store_id")
    elif kind != "household_only":
        raise ValueError("unknown synthetic covariance configuration")
    keys = list(fit.fixture.frame[columns].itertuples(index=False, name=None))
    covariance = np.zeros((k, k))
    selections = ((0,),) if len(columns) == 1 else ((0,), (1,), (0, 1))
    for selection in selections:
        sums = {}
        for key, contribution in zip(keys, contributions, strict=True):
            group = tuple(key[position] for position in selection)
            sums[group] = sums.get(group, np.zeros(k)) + contribution
        meat = np.zeros((k, k))
        for contribution in sums.values():
            meat += np.outer(contribution, contribution)
        sign = -1 if len(selection) == 2 else 1
        covariance += sign * _cr1(len(sums), n, k) * meat
    return covariance


def _native_synthetic_covariance(fit: _SyntheticFit, kind: str) -> np.ndarray:
    _require_fit(fit)
    native = _OLS(fit.fixture.outcome, fit.design.values, hasconst=True).fit(method="pinv")
    _agree(fit.coefficients, native.params, "statsmodels least-squares")
    codes = _cluster_codes(fit.fixture.frame, kind)
    if len(codes) == 1:
        return cov_cluster(native, codes[0], use_correction=True)
    return cov_cluster_2groups(native, codes[0], codes[1], use_correction=True)[0]


def _validity(covariance: np.ndarray, focal: list[int]) -> dict[str, bool]:
    finite = bool(np.isfinite(covariance).all())
    symmetric = bool(np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12))
    variances = np.diag(covariance)[focal]
    positive = bool(np.all(np.isfinite(variances) & (variances > 0)))
    psd = False
    indefinite = False
    if finite and symmetric:
        # The symmetric average is only for numerical eigendiagnosis; the returned
        # covariance is never modified. The rank-scale tolerance handles roundoff.
        eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2)
        tolerance = np.finfo(float).eps * covariance.shape[0] * np.max(np.abs(eigenvalues))
        psd = bool(eigenvalues.min() >= -tolerance)
        indefinite = bool(eigenvalues.min() < -tolerance and eigenvalues.max() > tolerance)
    return {
        "finite": finite,
        "symmetric": symmetric,
        "focal_variances_finite_positive": positive,
        "full_matrix_psd": psd,
        "full_matrix_indefinite": indefinite,
    }


def run_synthetic_preflight() -> dict:
    """Closed identifier-free metadata/count/boolean report; no result estimates.

    Numerical solver/covariance disagreement raises. Matrix indefiniteness and
    focal-variance validity are statuses, never automatic publication permission.
    """
    fixture = _make_synthetic_fixture()
    models = {}
    for model in ("M0", "M1"):
        fit = _fit_synthetic(fixture, model)
        focal = [fit.design.column_names.index(name) for name in HISTORY_COLUMNS]
        configurations = {}
        for kind in _COVARIANCE_KINDS:
            covariance = _synthetic_covariance(fit, kind)
            manual = _manual_synthetic_covariance(fit, kind)
            native = _native_synthetic_covariance(fit, kind)
            codes = _cluster_codes(fixture.frame, kind)
            counts = [int(np.unique(component).size) for component in codes]
            configurations[kind] = {
                "component_cluster_counts": counts,
                "reference_df": min(counts[:2]) - 1,
                "manual_agrees": _agree(covariance, manual, "manual covariance"),
                "statsmodels_agrees": _agree(covariance, native, "statsmodels covariance"),
                "validity": _validity(covariance, focal),
            }
        models[model] = {
            "rows": int(len(fixture.frame)),
            "columns": int(fit.design.values.shape[1]),
            "rank": int(fit.design.values.shape[1]),
            "binary_outcome": bool(np.isin(fixture.outcome, (0, 1)).all()),
            "equal_event_weights": True,
            "qr_direct_agrees": fit.direct_agrees,
            "qr_fwl_agrees": fit.fwl_agrees,
            "unclipped_below_zero_count": int(np.count_nonzero(fit.fitted < -1e-12)),
            "unclipped_above_one_count": int(np.count_nonzero(fit.fitted > 1 + 1e-12)),
            "covariance": configurations,
        }
    cells = fixture.frame.groupby(["household_key", "week_no"]).size()
    mobility = fixture.frame.groupby("household_key")["store_id"].nunique()
    report = {
        "scope": "SYNTHETIC_ONLY_NON_EVIDENCE",
        "conventions": dict(_CONVENTIONS),
        "structure": {
            "households": int(fixture.frame["household_key"].nunique()),
            "stores": int(fixture.frame["store_id"].nunique()),
            "weeks": int(fixture.frame["week_no"].nunique()),
            "household_week_cells": int(len(cells)),
            "repeated_household_week_cells": int((cells > 1).sum()),
            "events_in_repeated_household_week_cells": int(cells[cells > 1].sum()),
            "household_store_pairs": int(
                len(fixture.frame[["household_key", "store_id"]].drop_duplicates())
            ),
            "multi_store_households": int((mobility > 1).sum()),
        },
        "models": models,
    }
    validate_synthetic_report(report)
    return report
