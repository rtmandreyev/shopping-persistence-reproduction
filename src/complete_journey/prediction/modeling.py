"""Temporal splits, the B0/B1/B2 ladder and frozen probability metrics (P7-003).

Every fit goes through ``require_fit_allowed``, so real outcomes cannot be fitted or
scored in Run B. All preprocessing lives inside scikit-learn pipelines fitted on the
training rows only; validation and test rows are transformed, never fitted.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from complete_journey.prediction import config
from complete_journey.prediction.panel import (
    CanonicalPanel,
    require,
    require_fit_allowed,
    require_frames_allowed,
)

# --- Temporal split -----------------------------------------------------------------


def split_label(response_week: pd.Series, splits=config.SPLITS) -> pd.Series:
    """Map response weeks to train/validation/test; weeks outside every split are NaN."""
    label = pd.Series(pd.NA, index=response_week.index, dtype="object")
    for name, (lo, hi) in splits.items():
        label[response_week.between(lo, hi)] = name
    return label


def temporal_split(rows: pd.DataFrame, splits=config.SPLITS) -> dict[str, pd.DataFrame]:
    labels = split_label(rows["response_week"], splits)
    parts = {name: rows.loc[labels == name] for name in splits}
    require(sum(len(p) for p in parts.values()) <= len(rows), "split overlap")
    for (a, pa), (b, pb) in itertools.combinations(parts.items(), 2):
        require(
            pa.empty or pb.empty or pa["response_week"].max() < pb["response_week"].min(),
            f"split {a} is not strictly before {b}",
        )
    return parts


# --- Metrics ------------------------------------------------------------------------


def _check(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    require(y.shape == p.shape and y.size > 0, "metric inputs must be non-empty and aligned")
    require(np.isin(y, (0.0, 1.0)).all(), "metric outcomes must be binary")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "probabilities in [0, 1]")
    return y, p


def log_loss(y, p) -> float:
    """Mean natural-log loss; probabilities clipped at PROBABILITY_CLIP."""
    y, p = _check(y, p)
    p = np.clip(p, config.PROBABILITY_CLIP, 1 - config.PROBABILITY_CLIP)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2))


def roc_auc(y, p) -> float | None:
    """Descriptive only; ``None`` (explicitly undefined) when one class is absent."""
    y, p = _check(y, p)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def calibration(y, p) -> dict:
    """Calibration-in-the-large (intercept with logit offset) and slope; reliability table."""
    import statsmodels.api as sm

    y, p = _check(y, p)
    out = {"intercept": None, "slope": None, "reliability": []}
    lp = np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
    if len(np.unique(y)) == 2 and np.ptp(lp) > 0:
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        intercept = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit()
        out["slope"] = float(slope.params[1])
        out["intercept"] = float(intercept.params[0])
    edges = np.quantile(p, np.linspace(0, 1, config.CALIBRATION_BINS + 1))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, config.CALIBRATION_BINS - 1)
    for b in range(config.CALIBRATION_BINS):
        mask = bins == b
        if mask.any():
            out["reliability"].append(
                {
                    "bin": b,
                    "n": int(mask.sum()),
                    "mean_p": float(p[mask].mean()),
                    "mean_y": float(y[mask].mean()),
                }
            )
    return out


def recency_gain(loss_tier0: float, loss_tier1: float) -> float:
    """Frozen sign: Tier 0 loss minus Tier 1 loss; positive = recency improves prediction."""
    return float(loss_tier0 - loss_tier1)


# --- Models -------------------------------------------------------------------------


class PrevalenceModel:
    """B0: the training base rate; the same probability for every row."""

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        require(y.size > 0, "B0 needs training rows")
        self.p_ = float(y.mean())
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p_), np.full(n, self.p_)])


def make_b1(C: float) -> Pipeline:
    """B1: L2 logistic (l1_ratio=0) after train-fitted median imputation and scaling."""
    return Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=C,
                    l1_ratio=0.0,
                    solver=config.B1_SOLVER,
                    tol=config.B1_TOL,
                    max_iter=config.B1_MAX_ITER,
                ),
            ),
        ]
    )


def make_b2(params: dict) -> HistGradientBoostingClassifier:
    """B2: the single tree ensemble; NaN handled natively; fixed seed, no early stopping."""
    return HistGradientBoostingClassifier(
        max_iter=config.B2_MAX_ITER, early_stopping=False, random_state=config.SEED, **params
    )


def b2_grid() -> list[dict]:
    keys = list(config.B2_GRID)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*config.B2_GRID.values())
    ]


@dataclass
class FittedModel:
    name: str
    params: dict
    estimator: object
    features: tuple[str, ...]
    validation_log_loss: dict

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        require_frames_allowed(frame)  # scoring real rows is guarded like fitting
        X = frame[list(self.features)].to_numpy(dtype=float)
        return self.estimator.predict_proba(X)[:, 1]


def _xy(frame, features):
    return frame[list(features)].to_numpy(dtype=float), frame["y"].to_numpy(dtype=float)


def fit_fixed_b1(panel: CanonicalPanel, train: pd.DataFrame, features) -> FittedModel:
    """B1 with the frozen fixed penalty (VI-6 and Tier 0/1): no tuning."""
    require_fit_allowed(panel)
    require_frames_allowed(train, panel=panel)
    X, y = _xy(train, features)
    require(len(np.unique(y)) == 2, "B1 training rows need both outcomes")
    model = make_b1(config.B1_FIXED_C).fit(X, y)
    return FittedModel("B1", {"C": config.B1_FIXED_C}, model, tuple(features), {})


def fit_selected(
    panel: CanonicalPanel, name: str, train: pd.DataFrame, validation: pd.DataFrame, features
) -> FittedModel:
    """Select on validation log loss over the frozen grid, then refit on train+validation.

    B1 uses the Tier 2 C grid; B2 uses its 12-point grid. Test rows are never passed.
    """
    require_fit_allowed(panel)
    require_frames_allowed(train, validation, panel=panel)
    require(name in config.MODEL_LADDER, f"model outside the frozen ladder: {name}")
    Xt, yt = _xy(train, features)
    Xv, yv = _xy(validation, features)
    if name == "B0":
        model = PrevalenceModel().fit(Xt, yt)
        score = log_loss(yv, model.predict_proba(Xv)[:, 1])
        final = PrevalenceModel().fit(np.vstack([Xt, Xv]), np.concatenate([yt, yv]))
        return FittedModel("B0", {}, final, tuple(features), {"base_rate": score})
    if name == "B1":
        candidates = [({"C": c}, lambda c=c: make_b1(c)) for c in config.B1_TIER2_GRID]
    else:
        candidates = [(params, lambda params=params: make_b2(params)) for params in b2_grid()]
    scores = {}
    for params, build in candidates:
        fitted = build().fit(Xt, yt)
        scores[str(params)] = log_loss(yv, fitted.predict_proba(Xv)[:, 1])
    best = min(candidates, key=lambda c: (scores[str(c[0])], str(c[0])))
    final = best[1]().fit(np.vstack([Xt, Xv]), np.concatenate([yt, yv]))
    return FittedModel(name, best[0], final, tuple(features), scores)


def score(model: FittedModel, frame: pd.DataFrame) -> dict:
    """Frozen metric set on one evaluation frame."""
    p = model.predict(frame)
    y = frame["y"].to_numpy(dtype=float)
    return {
        config.PRIMARY_METRIC: log_loss(y, p),
        config.SECONDARY_METRIC: brier(y, p),
        "roc_auc": roc_auc(y, p),
        "prevalence": float(y.mean()),
        "calibration": calibration(y, p),
        "n": int(len(y)),
    }
