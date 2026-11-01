"""Run C aggregate diagnostics; no exclusion, refit, estimator or covariance change."""

from __future__ import annotations

import numpy as np
import pandas as pd

from complete_journey.econometrics.influence import (
    balanced_twfe_residualize,
    residualized_x_leverage,
)
from complete_journey.econometrics.preflight import concentration
from complete_journey.econometrics.samples import require

SCOPE = "RUN_C_EXPLORATORY_UNREVIEWED"


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a.ravel(), b.ravel()
    da, db = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((da @ da) * (db @ db)))
    return float(da @ db / denominator) if denominator > 1e-12 else None


def core_diagnostics(frame: pd.DataFrame, fits: dict) -> list[dict]:
    """Balanced sorted frame. Scores, not week-mean residuals, assess time dependence."""
    households, weeks = frame.household_key.nunique(), frame.week_no.nunique()
    require(len(frame) == households * weeks, "diagnostic calendar is not balanced")
    indexed = frame.set_index(["household_key", "week_no"])
    shape = (households, weeks)
    x = frame.lag1_retailer_purchase.to_numpy(dtype=float).reshape(shape)
    xt = balanced_twfe_residualize(x)
    rows = []

    def add(outcome, model, metric, value, unit, denominator):
        rows.append(
            {
                "outcome": outcome,
                "model_id": model,
                "diagnostic": metric,
                "value": value,
                "unit": unit,
                "denominator": denominator,
                "scope": SCOPE,
            }
        )

    for (outcome, model), fitted in fits.items():
        require(fitted.resids.index.equals(indexed.index), "diagnostic/model row mismatch")
        residual = fitted.resids.to_numpy(dtype=float).reshape(shape)
        if outcome == "retailer_purchase":
            # Total fitted outcome, INCLUDING absorbed FE, rather than xb alone.
            prediction = frame[outcome].to_numpy().reshape(shape) - residual
            for name, value, unit in (
                ("fitted_min", prediction.min(), "probability points"),
                ("fitted_max", prediction.max(), "probability points"),
                ("fitted_below_zero_count", (prediction < 0).sum(), "cells"),
                ("fitted_above_one_count", (prediction > 1).sum(), "cells"),
                ("fitted_below_zero_share", (prediction < 0).mean(), "fraction"),
                ("fitted_above_one_share", (prediction > 1).mean(), "fraction"),
            ):
                add(
                    outcome,
                    model,
                    name,
                    float(value),
                    unit,
                    f"all {len(frame)} fitted household-weeks; absorbed FE included",
                )
        if model != "M3":
            continue
        add(
            outcome,
            model,
            "residual_adjacent_pooled_correlation",
            correlation(residual[:, :-1], residual[:, 1:]),
            "correlation",
            f"{households * (weeks - 1)} within-household adjacent residual pairs; global centering",
        )
        correlations = [correlation(row[:-1], row[1:]) for row in residual]
        valid = np.array([v for v in correlations if v is not None])
        add(
            outcome,
            model,
            "residual_household_correlation_defined",
            len(valid),
            "households",
            f"{households}; undefined constant-residual cases excluded only from this diagnostic",
        )
        for q in (0.1, 0.5, 0.9):
            add(
                outcome,
                model,
                f"residual_household_correlation_p{int(q * 100)}",
                float(np.quantile(valid, q)) if len(valid) else None,
                "correlation",
                f"{len(valid)} defined within-household correlations",
            )
        score = (xt * residual).sum(axis=0)
        add(
            outcome,
            model,
            "aggregate_score_sum",
            float(score.sum()),
            "indicator times outcome units",
            "sum of time-aggregated FWL scores; first-order condition approximately zero",
        )
        add(
            outcome,
            model,
            "aggregate_score_variance",
            float(score.var()),
            "squared score units",
            f"{weeks} response weeks; ddof=0",
        )
        for lag in (1, 4, 8):
            add(
                outcome,
                model,
                f"aggregate_score_correlation_lag{lag}",
                correlation(score[lag:], score[:-lag]) if weeks > lag else None,
                "correlation",
                f"{max(weeks - lag, 0)} paired weekly sums of x_tilde times residual; Pearson centering",
            )
        if outcome != "weekly_sales_value":
            continue
        y = frame[outcome].to_numpy(dtype=float).reshape(shape)
        yt = balanced_twfe_residualize(y)
        leverage = residualized_x_leverage(xt)
        nh, dh = (xt * yt).sum(axis=1), leverage["household_denominator_contribution"]
        numerator, denominator = float(nh.sum()), float(dh.sum())
        require(
            np.isclose(
                numerator / denominator,
                fitted.params["lag1_retailer_purchase"],
                atol=1e-9,
                rtol=1e-9,
            ),
            "influence numerator/denominator mismatch",
        )
        add(
            outcome,
            model,
            "household_numerator_total",
            numerator,
            "indicator times retailer currency units",
            "sum across all household FWL numerator contributions",
        )
        add(
            outcome,
            model,
            "household_denominator_total",
            denominator,
            "squared indicator units",
            "sum across all household FWL denominator contributions",
        )
        add(
            outcome,
            model,
            "household_numerator_gross_absolute",
            float(np.abs(nh).sum()),
            "indicator times retailer currency units",
            "sum of absolute household contributions; cancellation preserved",
        )
        add(
            outcome,
            model,
            "household_numerator_positive_count",
            int((nh > 0).sum()),
            "households",
            f"all {households}",
        )
        add(
            outcome,
            model,
            "household_numerator_negative_count",
            int((nh < 0).sum()),
            "households",
            f"all {households}",
        )
        add(
            outcome,
            model,
            "household_max_abs_additive_slope_contribution",
            float(np.max(np.abs(nh)) / denominator),
            "retailer currency units",
            "max absolute N_h divided by total baseline D; not deletion or refit",
        )
        add(
            outcome,
            model,
            "household_max_abs_contribution_relative_net",
            float(np.max(np.abs(nh)) / abs(numerator)) if abs(numerator) > 1e-12 else None,
            "ratio",
            "max abs(N_h) / abs(sum N_h); can exceed 1 under cancellation",
        )
        for name, values, basis in (
            (
                "numerator_abs",
                np.abs(nh),
                "gross sum of absolute household numerator contributions",
            ),
            ("denominator", dh, "total baseline regressor SS"),
            (
                "residual_sse_household",
                np.square(residual).sum(axis=1),
                "total revenue residual SSE",
            ),
        ):
            for key, value in concentration(values).items():
                add(
                    outcome,
                    model,
                    "household_" + name + "_" + key,
                    value,
                    "fraction" if "share" in key else "count",
                    basis,
                )
        r = residual.ravel()
        for q in (0.01, 0.5, 0.9, 0.95, 0.99):
            add(
                outcome,
                model,
                f"revenue_residual_p{int(q * 100)}",
                float(np.quantile(r, q)),
                "retailer currency units",
                "all baseline residual cells; linear-interpolated quantiles",
            )
        add(
            outcome,
            model,
            "revenue_residual_min",
            float(r.min()),
            "retailer currency units",
            "all baseline residual cells",
        )
        add(
            outcome,
            model,
            "revenue_residual_max",
            float(r.max()),
            "retailer currency units",
            "all baseline residual cells",
        )
        squared = np.square(r)
        add(
            outcome,
            model,
            "revenue_residual_sse",
            float(squared.sum()),
            "squared retailer currency units",
            "all baseline residual cells",
        )
        add(
            outcome,
            model,
            "revenue_largest_residual_cell_sse_share",
            float(squared.max() / squared.sum()),
            "fraction",
            "total revenue residual SSE",
        )
        for fraction, label in ((0.001, "001"), (0.01, "01")):
            count = int(np.ceil(len(r) * fraction))
            add(
                outcome,
                model,
                "revenue_residual_top" + label + "_cells_count",
                count,
                "cells",
                f"ceil({fraction} times all baseline cells)",
            )
            add(
                outcome,
                model,
                "revenue_residual_top" + label + "_cells_sse_share",
                float(np.sort(squared)[-count:].sum() / squared.sum()),
                "fraction",
                "total revenue residual SSE; no cell deletion",
            )

    purchase, lag = frame.retailer_purchase.to_numpy(), frame.lag1_retailer_purchase.to_numpy()
    rates = {}
    for previous in (0, 1):
        mask = lag == previous
        require(mask.any(), "raw transition state lacks support")
        rates[previous] = float(purchase[mask].mean())
        add(
            "retailer_purchase",
            "RAW",
            f"unadjusted_transition_rate_previous{previous}",
            rates[previous],
            "probability",
            f"{int(mask.sum())} cells with previous participation={previous}; unadjusted descriptive",
        )
    add(
        "retailer_purchase",
        "RAW",
        "unadjusted_descriptive_transition_difference",
        rates[1] - rates[0],
        "probability points",
        "two conditional participation rates; NOT TWFE coefficient",
    )
    return rows
