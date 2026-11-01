"""Run C only: nine frozen unweighted descriptive projections, never sensitivities."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS

from complete_journey.econometrics.covariance import covariance_options
from complete_journey.econometrics.samples import SAMPLES, construct_sample, require

HISTORY = "lag1_retailer_purchase"
MODEL_EFFECTS = {"M0": (False, False), "M1": (False, True), "M2": (True, False), "M3": (True, True)}
CORE_KEYS = tuple(
    (outcome, model)
    for outcome in ("retailer_purchase", "weekly_sales_value")
    for model in MODEL_EFFECTS
) + (("trips", "M3"),)
ABS_TOL = 1e-9
REL_TOL = 1e-9
PSD_TOL = 1e-12
R2_DEFINITIONS = {
    "r2": "1-SSE/SST on the model's weighted/effect-transformed data; explicit constant gives centered SST; all weights here equal one",
    "r2_within": "1-SSE/SST for entity-demeaned outcome and slope predictions; does not include fitted time effects",
    "r2_between": "1-SSE/SST for entity-average outcome and slope predictions with centered SST when a constant is present",
    "r2_overall": "1-SSE/SST for raw outcome and constant-plus-history predictions, without absorbed effects",
    "r2_inclusive": "FE models only: 1-SSE/SST including estimated effects with centered raw-outcome SST; unavailable for pooled",
}


def independent_slope(frame: pd.DataFrame, outcome: str, model_id: str) -> float:
    """FWL via grouped means, independent of linearmodels' transformed design."""
    entity, time = MODEL_EFFECTS[model_id]
    data = frame[[HISTORY, outcome]].astype(float)
    if entity and time:
        transformed = (
            data
            - data.groupby(frame.household_key).transform("mean")
            - data.groupby(frame.week_no).transform("mean")
            + data.mean()
        )
    elif entity:
        transformed = data - data.groupby(frame.household_key).transform("mean")
    elif time:
        transformed = data - data.groupby(frame.week_no).transform("mean")
    else:
        # Independent two-column least squares for the pooled comparator.
        design = np.column_stack([np.ones(len(data)), data[HISTORY]])
        return float(np.linalg.lstsq(design, data[outcome], rcond=None)[0][1])
    x, y = transformed[HISTORY].to_numpy(), transformed[outcome].to_numpy()
    denominator = float(x @ x)
    require(denominator > 1e-12, "independent projection has absorbed history")
    return float(x @ y / denominator)


def covariance_boundary(matrix: pd.DataFrame) -> dict:
    """Inspect native matrix; never repair. Tolerance classifies numerical PSD only."""
    native = matrix.to_numpy(copy=True)
    require(np.isfinite(native).all(), "native covariance contains nonfinite entries")
    variance = float(matrix.loc[HISTORY, HISTORY])
    require(
        np.isfinite(variance) and variance > 0,
        "P4-002: focal slope variance must be finite and strictly positive",
    )
    require(
        np.allclose(native, native.T, atol=1e-12, rtol=1e-12), "native covariance is asymmetric"
    )
    eigenvalues = np.linalg.eigvalsh(native)
    tolerance = PSD_TOL * max(1.0, float(np.max(np.abs(eigenvalues))))
    return {
        "focal_variance": variance,
        "full_covariance_psd": bool(eigenvalues.min() >= -tolerance),
        "covariance_min_eigenvalue": float(eigenvalues.min()),
        "covariance_psd_tolerance": tolerance,
    }


def fit_model(frame: pd.DataFrame, outcome: str, model_id: str) -> tuple[object, dict]:
    """Bounded helper also exercised on synthetic complete calendars in tests."""
    require((outcome, model_id) in CORE_KEYS, "outcome/model is outside frozen Run C")
    require(not frame.duplicated(["household_key", "week_no"]).any(), "duplicate model row")
    households, weeks = frame.household_key.nunique(), frame.week_no.nunique()
    require(len(frame) == households * weeks, "model calendar must be balanced")
    require(frame[HISTORY].isin([0, 1]).all(), "history must be binary and available")
    require(np.isfinite(frame[outcome]).all(), "model outcome contains unavailable values")
    if outcome == "retailer_purchase":
        require(frame[outcome].isin([0, 1]).all(), "participation must be binary")
    indexed = frame.set_index(["household_key", "week_no"])
    x = pd.DataFrame(
        {"constant": 1.0, HISTORY: indexed[HISTORY].astype(float)}, index=indexed.index
    )
    y = indexed[outcome].astype(float)
    entity, time = MODEL_EFFECTS[model_id]
    if model_id == "M0":
        model = PooledOLS(y, x, check_rank=True)
    else:
        model = PanelOLS(
            y,
            x,
            entity_effects=entity,
            time_effects=time,
            singletons=True,
            drop_absorbed=False,
            check_rank=True,
        )
    options = covariance_options("two_way", effects=model_id != "M0")
    fitted = model.fit(**options)
    require(
        fitted.nobs == len(frame) and fitted.resids.index.equals(indexed.index),
        "estimator changed observation rows/order",
    )
    require(
        fitted.model.exog.vars == ["constant", HISTORY],
        "unexpected control or regressor orientation",
    )
    require(np.all(fitted.model.weights.values2d == 1), "unexpected observation weights")
    require(fitted._cov_type == "Clustered", "wrong covariance object")
    boundary = covariance_boundary(fitted.cov)
    beta = float(fitted.params[HISTORY])
    independent = independent_slope(frame, outcome, model_id)
    require(
        abs(beta - independent) <= ABS_TOL + REL_TOL * abs(independent),
        "independent coefficient reconciliation failed",
    )
    interval = fitted.conf_int(level=0.95).loc[HISTORY]
    row_hash = hashlib.sha256(
        pd.util.hash_pandas_object(indexed.index, index=False).values.tobytes()
    ).hexdigest()
    result = {
        "outcome": outcome,
        "model_id": model_id,
        "sample": "S0_baseline",
        "status": "UNREVIEWED",
        "coefficient": beta,
        "coefficient_percentage_points": 100 * beta if outcome == "retailer_purchase" else None,
        "standard_error": float(fitted.std_errors[HISTORY]),
        "ci_lower": float(interval.iloc[0]),
        "ci_upper": float(interval.iloc[1]),
        "t_statistic": float(fitted.tstats[HISTORY]),
        "p_value": float(fitted.pvalues[HISTORY]),
        "nobs": fitted.nobs,
        "households": households,
        "response_weeks": weeks,
        "household_fe": entity,
        "week_fe": time,
        "covariance_type": "two-way household/response-week clustered",
        "household_clusters": households,
        "week_clusters": weeks,
        "df_resid": fitted.df_resid,
        "reference_distribution": "native Student t(df_resid)",
        "debiased": options["debiased"],
        "group_debias": options["group_debias"],
        "auto_df": options.get("auto_df"),
        "count_effects": options.get("count_effects"),
        **boundary,
        "r2": float(fitted.rsquared),
        "r2_within": float(fitted.rsquared_within),
        "r2_between": float(fitted.rsquared_between),
        "r2_overall": float(fitted.rsquared_overall),
        "r2_inclusive": float(fitted.rsquared_inclusive) if model_id != "M0" else None,
        "independent_coefficient": independent,
        "independent_abs_difference": abs(beta - independent),
        "independent_abs_tolerance": ABS_TOL,
        "independent_rel_tolerance": REL_TOL,
        "row_key_sha256": row_hash,
    }
    require(
        np.isfinite(
            [
                result[k]
                for k in (
                    "coefficient",
                    "standard_error",
                    "ci_lower",
                    "ci_upper",
                    "t_statistic",
                    "p_value",
                )
            ]
        ).all(),
        "nonfinite scalar inference",
    )
    return fitted, result


def estimate_core(mart: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    """Production target fixed to S0 responses 2–102; no configurable search surface."""
    frame = construct_sample(mart, SAMPLES[0])
    records, fits = [], {}
    for outcome, model_id in CORE_KEYS:
        fitted, record = fit_model(frame, outcome, model_id)
        records.append(record)
        fits[outcome, model_id] = fitted
    require(
        len(records) == 9 and len({r["row_key_sha256"] for r in records}) == 1,
        "core ladder has different rows or duplicate models",
    )
    return frame, records, fits


def adjustment_changes(records: list[dict]) -> list[dict]:
    lookup = {(r["outcome"], r["model_id"]): r["coefficient"] for r in records}
    return [
        {
            "outcome": outcome,
            "earlier": earlier,
            "later": later,
            "change": lookup[outcome, later] - lookup[outcome, earlier],
            "absolute_change": abs(lookup[outcome, later] - lookup[outcome, earlier]),
            "relative_percent_change": 100
            * (lookup[outcome, later] - lookup[outcome, earlier])
            / lookup[outcome, earlier]
            if abs(lookup[outcome, earlier]) > ABS_TOL
            else None,
        }
        for outcome in ("retailer_purchase", "weekly_sales_value")
        for earlier, later in (("M0", "M1"), ("M0", "M2"), ("M1", "M3"), ("M2", "M3"))
    ]
