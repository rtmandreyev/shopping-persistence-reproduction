"""Phase 4 M0-M3 specification on balanced or unbalanced panels (diagnostic statistics).

The accepted Phase 4 ``core.fit_model`` refuses unbalanced calendars, and operator O is
unbalanced. This adapter applies the same specification: unweighted least squares of
y on the history regressor(s), with pooled / week / household / two-way effects. It
computes point estimates by exact one-way demeaning or converged alternating
projections. It is verified against ``fit_model`` on balanced panels and against
linearmodels PanelOLS on unbalanced ones (tests). Coefficients are diagnostic
statistics, never structural parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from complete_journey.prediction import config
from complete_journey.prediction.panel import require

MODELS = {"M0": (False, False), "M1": (False, True), "M2": (True, False), "M3": (True, True)}


def _demean(values: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    out = values.astype(float).copy()
    if not groups:
        return out - out.mean(axis=0)
    for _ in range(config.TWFE_MAX_ITER):
        previous = out.copy()
        for codes in groups:
            counts = np.bincount(codes)
            for j in range(out.shape[1]):
                out[:, j] -= (np.bincount(codes, weights=out[:, j]) / counts)[codes]
        if len(groups) == 1 or np.max(np.abs(out - previous)) <= config.TWFE_TOL:
            return out
    raise RuntimeError("two-way alternating projections did not converge")


def fe_coefficients(
    frame: pd.DataFrame, outcome: str, regressors: list[str], model: str
) -> dict[str, float]:
    """Slopes of ``outcome`` on ``regressors`` under M0/M1/M2/M3 (no covariance)."""
    require(model in MODELS, f"unknown ladder model {model}")
    require(len(frame) > len(regressors) + 1, "too few rows for the ladder")
    entity, time = MODELS[model]
    groups = []
    if entity:
        groups.append(pd.factorize(frame["household_key"])[0])
    if time:
        groups.append(pd.factorize(frame["week_no"])[0])
    data = frame[[outcome, *regressors]].to_numpy(dtype=float)
    resid = _demean(data, groups)
    y, X = resid[:, 0], resid[:, 1:]
    require(np.linalg.matrix_rank(X) == X.shape[1], "history regressor has no residual support")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return dict(zip(regressors, (float(b) for b in beta), strict=True))
