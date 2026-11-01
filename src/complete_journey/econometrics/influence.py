"""Future single-regressor influence diagnostics; no data loading or exclusion rule.

Section N of the frozen Phase 4 design requires baseline estimation first and
researcher approval for any exclusion sensitivity. These balanced-array helpers
neither choose a household nor alter a baseline sample. Outcome-linked diagnostics
refuse computation unless estimation_authorized=True. Run B may use only the
internally fixed synthetic verifier, never supply real outcomes to these helpers.
"""

import numpy as np


def _panel(values: np.ndarray) -> np.ndarray:
    panel = np.asarray(values, dtype=float)
    if panel.ndim != 2 or min(panel.shape) < 2:
        raise ValueError("Expected a complete household-by-week array, at least 2 by 2")
    if not np.isfinite(panel).all():
        raise ValueError("Panel values must be finite; no silent row deletion")
    return panel


def _denominator(value: float, tolerance: float, label: str) -> float:
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("denominator_tol must be finite and nonnegative")
    if not np.isfinite(value) or value <= tolerance:
        raise ValueError(f"{label} denominator is nonfinite, zero or numerically unsupported")
    return value


def balanced_twfe_residualize(values: np.ndarray) -> np.ndarray:
    """Double demean a finite balanced panel; not an unbalanced-panel projector.

    Rows are households, columns are common weeks; no weights or other controls.
    The input is not modified. This helper is independent of real-data preflight.
    """
    panel = _panel(values)
    return panel - panel.mean(axis=1, keepdims=True) - panel.mean(axis=0) + panel.mean()


def residualized_x_leverage(
    residualized_x: np.ndarray, *, denominator_tol: float = 1e-12
) -> dict[str, float | np.ndarray]:
    """Return squared-x denominator contributions and partial slope leverage.

    The caller supplies an already residualized regressor. Cell leverage is
    x_tilde**2 / sum(x_tilde**2), not the full FE-model hat diagonal. Household
    leverage is its row sum; both total leverages equal one. This uses no outcome.
    denominator_tol is a numerical guard, not a household exclusion threshold.
    """
    x = _panel(residualized_x)
    cell_denominator = np.square(x)
    denominator = _denominator(float(cell_denominator.sum()), denominator_tol, "Baseline")
    return {
        "denominator": denominator,
        "cell_denominator_contribution": cell_denominator,
        "household_denominator_contribution": cell_denominator.sum(axis=1),
        "cell_partial_leverage": cell_denominator / denominator,
        "household_partial_leverage": cell_denominator.sum(axis=1) / denominator,
    }


def household_slope_influence(
    x: np.ndarray,
    y: np.ndarray,
    household_index: int,
    *,
    estimation_authorized: bool = False,
    denominator_tol: float = 1e-12,
) -> dict[str, float | np.ndarray]:
    """Preserve baseline and diagnose one explicitly supplied household deletion.

    N_h = sum_t(x_tilde_ht * y_tilde_ht), D_h = sum_t(x_tilde_ht**2).
    N_h / D is an additive baseline slope contribution: its sum equals beta.
    (N - N_h) / (D - D_h) is a FIXED-PROJECTION deletion quantity, not a TWFE
    refit. Actual leave-household-out slope residualizes BOTH raw arrays again
    on retained households, since removing a household changes the week means.
    Changes below are deleted-sample minus baseline, not exclusion decisions.

    Complete balanced arrays only. An explicit authorization is a caller guard,
    not proof of researcher approval; no sample-selection policy is implemented.
    """
    if estimation_authorized is not True:
        raise PermissionError("Slope diagnostics require explicit estimation_authorized=True")
    x_raw, y_raw = _panel(x), _panel(y)
    if x_raw.shape != y_raw.shape:
        raise ValueError("Regressor and outcome must have identical household-week shapes")
    if x_raw.shape[0] < 3:
        raise ValueError("Household deletion must retain at least two households")
    if (
        isinstance(household_index, bool)
        or not isinstance(household_index, (int, np.integer))
        or not 0 <= household_index < x_raw.shape[0]
    ):
        raise ValueError("household_index must be an in-range row index")

    x_res = balanced_twfe_residualize(x_raw)
    y_res = balanced_twfe_residualize(y_raw)
    result = residualized_x_leverage(x_res, denominator_tol=denominator_tol)
    denominator = float(result["denominator"])
    household_numerator = (x_res * y_res).sum(axis=1)
    if not np.isfinite(household_numerator).all():
        raise ValueError("Household numerator contributions must be finite")
    numerator = float(household_numerator.sum())
    baseline = numerator / denominator

    keep = np.arange(x_raw.shape[0]) != household_index
    # Sum retained squares directly: avoid cancellation in D - D_h.
    fixed_denominator = _denominator(
        float(np.square(x_res[keep]).sum()), denominator_tol, "Fixed-projection deletion"
    )
    fixed_slope = float((x_res[keep] * y_res[keep]).sum()) / fixed_denominator
    deleted_x_res = balanced_twfe_residualize(x_raw[keep])
    deleted_y_res = balanced_twfe_residualize(y_raw[keep])
    deleted_denominator = _denominator(
        float(np.square(deleted_x_res).sum()), denominator_tol, "Re-residualized deletion"
    )
    deleted_slope = float((deleted_x_res * deleted_y_res).sum()) / deleted_denominator
    if not np.isfinite([baseline, fixed_slope, deleted_slope]).all():
        raise ValueError("Slope diagnostics must be finite")
    result.update(
        numerator=numerator,
        household_numerator_contribution=household_numerator,
        household_additive_slope_contribution=household_numerator / denominator,
        baseline_slope=baseline,
        fixed_projection_deleted_denominator=fixed_denominator,
        fixed_projection_deleted_slope=fixed_slope,
        fixed_projection_slope_change=fixed_slope - baseline,
        leave_household_out_denominator=deleted_denominator,
        leave_household_out_slope=deleted_slope,
        leave_household_out_slope_change=deleted_slope - baseline,
    )
    return result


def verify_influence_fixture() -> dict[str, bool]:
    """Check fixed internal toy arrays; return boolean metadata only.

    No external arrays, data paths or identifiers are accepted, read or returned.
    Synthetic authorization here does not authorize estimation on real data.
    """
    x = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, 1]])
    y = np.array([[2, 5, 13, 8], [1, 9, 4, 10], [20, 3, 19, 2], [2, 4, 5, 11]])
    diagnostics = household_slope_influence(x, y, 0, estimation_authorized=True)
    kept_x, kept_y = x[1:], y[1:]
    households, weeks = kept_x.shape
    # Independent explicit household/week dummy regression, synthetic only.
    design = np.column_stack(
        (
            np.repeat(np.eye(households), weeks, axis=0),
            np.tile(np.eye(weeks)[:, 1:], (households, 1)),
            kept_x.ravel(),
        )
    )
    reference = np.linalg.lstsq(design, kept_y.ravel(), rcond=None)[0][-1]
    return {
        "synthetic_only": True,
        "partial_leverage_sums_to_one": bool(
            np.isclose(np.sum(diagnostics["cell_partial_leverage"]), 1.0)
        ),
        "household_denominators_reconcile": bool(
            np.isclose(
                np.sum(diagnostics["household_denominator_contribution"]),
                diagnostics["denominator"],
            )
        ),
        "additive_contributions_reconcile": bool(
            np.isclose(
                np.sum(diagnostics["household_additive_slope_contribution"]),
                diagnostics["baseline_slope"],
            )
        ),
        "deletion_matches_dummy_refit": bool(
            np.isclose(diagnostics["leave_household_out_slope"], reference)
        ),
        "fixed_projection_is_not_refit": bool(
            not np.isclose(
                diagnostics["fixed_projection_deleted_slope"],
                diagnostics["leave_household_out_slope"],
            )
        ),
    }
