"""Design-support diagnostics only: no real outcome/history slope or inference."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from complete_journey.econometrics.samples import SampleSpec, construct_sample, require

RESIDUAL_ATOL = 1e-12
SS_ATOL = 1e-8
TOP_FRACTIONS = (0.01, 0.05, 0.10)


def double_demean(x: np.ndarray) -> np.ndarray:
    """Balanced household-by-response-week additive projection, regressor only."""
    x = np.asarray(x, dtype=float)
    require(x.ndim == 2 and np.isfinite(x).all(), "residualization needs a finite balanced matrix")
    return x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + x.mean()


def alternating_projection(x: np.ndarray) -> np.ndarray:
    """Independent grouped alternating projections, not the double-demeaning helper."""
    x = np.asarray(x, dtype=float)
    require(x.ndim == 2 and np.isfinite(x).all(), "independent projection input")
    n, t = x.shape
    entity = np.repeat(np.arange(n), t)
    week = np.tile(np.arange(t), n)
    residual = pd.Series(x.ravel().copy())
    for _ in range(20):
        old = residual.to_numpy(copy=True)
        residual -= residual.groupby(entity, sort=False).transform("mean")
        residual -= residual.groupby(week, sort=False).transform("mean")
        if float(np.max(np.abs(residual.to_numpy() - old))) <= RESIDUAL_ATOL:
            return residual.to_numpy().reshape(n, t)
    raise RuntimeError("independent demeaning did not converge")


def concentration(contributions: np.ndarray) -> dict[str, float]:
    """Equal-count top shares and inverse-Herfindahl effective contributor count."""
    values = np.asarray(contributions, dtype=float)
    require(values.ndim == 1 and len(values) > 0, "concentration vector shape")
    require(
        np.isfinite(values).all() and (values >= 0).all(), "invalid concentration contributions"
    )
    total = float(values.sum())
    require(total > 0, "concentration denominator is zero")
    ordered = np.sort(values)[::-1]
    result = {
        "contributors": float((values > RESIDUAL_ATOL).sum()),
        "contributor_share": float((values > RESIDUAL_ATOL).mean()),
        "max_share": float(values.max() / total),
        "effective_count": float(total**2 / np.square(values).sum()),
    }
    for fraction in TOP_FRACTIONS:
        count = math.ceil(len(values) * fraction)
        label = f"top{round(fraction * 100):02d}"
        result[f"{label}_count"] = float(count)
        result[f"{label}_share"] = float(ordered[:count].sum() / total)
    return result


def raw_history_support(x: np.ndarray) -> dict[str, float]:
    require(np.isin(x, [0, 1]).all(), "history support requires binary values")
    switches = np.diff(x, axis=1)
    switch_counts = np.count_nonzero(switches, axis=1)
    both = (x.min(axis=1) == 0) & (x.max(axis=1) == 1)
    result = {
        "mean": float(x.mean()),
        "zeros": float((x == 0).sum()),
        "ones": float((x == 1).sum()),
        "variance_ddof0": float(x.var(ddof=0)),
        "households_both_states": float(both.sum()),
        "households_constant0": float((x.max(axis=1) == 0).sum()),
        "households_constant1": float((x.min(axis=1) == 1).sum()),
        "adjacent_0_to_1": float((switches == 1).sum()),
        "adjacent_1_to_0": float((switches == -1).sum()),
        "switching_households": float((switch_counts > 0).sum()),
        "switching_household_share": float((switch_counts > 0).mean()),
        "switches_mean": float(switch_counts.mean()),
        "switches_max": float(switch_counts.max()),
    }
    for label, q in (
        ("min", 0),
        ("p10", 0.1),
        ("p25", 0.25),
        ("p50", 0.5),
        ("p75", 0.75),
        ("p90", 0.9),
    ):
        result[f"switches_{label}"] = float(np.quantile(switch_counts, q))
    return result


def regressor_support(x: np.ndarray) -> dict[str, float]:
    """Only x enters; outcome residuals/cross-products are deliberately absent."""
    centered = x - x.mean()
    within = x - x.mean(axis=1, keepdims=True)
    tw = double_demean(x)
    independent = alternating_projection(x)
    error = float(np.max(np.abs(tw - independent)))
    ss_error = float(abs(np.square(tw).sum() - np.square(independent).sum()))
    require(
        error <= RESIDUAL_ATOL and ss_error <= SS_ATOL, "independent residualization disagreement"
    )
    require(float(np.max(np.abs(tw.mean(axis=0)))) <= RESIDUAL_ATOL, "week orthogonality failure")
    require(
        float(np.max(np.abs(tw.mean(axis=1)))) <= RESIDUAL_ATOL, "household orthogonality failure"
    )
    raw_ss = float(np.square(centered).sum())
    hh_ss = float(np.square(within).sum())
    tw_ss = float(np.square(tw).sum())
    # Numerical non-absorption only. No economic concentration cutoff is selected.
    require(
        raw_ss > RESIDUAL_ATOL and hh_ss > RESIDUAL_ATOL and tw_ss > RESIDUAL_ATOL,
        "chosen history is numerically absorbed; STOP before Run C",
    )
    result = {
        "raw_centered_ss": raw_ss,
        "raw_uncentered_ss": float(np.square(x).sum()),
        "household_ss": hh_ss,
        "household_variance_ddof0": float(within.var(ddof=0)),
        "twfe_mean": float(tw.mean()),
        "twfe_ss": tw_ss,
        "twfe_variance_ddof0": float(tw.var(ddof=0)),
        "twfe_min": float(tw.min()),
        "twfe_max": float(tw.max()),
        "twfe_raw_centered_ss_share": tw_ss / raw_ss,
        "independent_vector_max_abs_difference": error,
        "independent_ss_abs_difference": ss_error,
        "vector_absolute_tolerance": RESIDUAL_ATOL,
        "ss_absolute_tolerance": SS_ATOL,
    }
    for prefix, contributions in (
        ("household_within", np.square(within).sum(axis=1)),
        ("household_twfe", np.square(tw).sum(axis=1)),
        ("week_twfe", np.square(tw).sum(axis=0)),
    ):
        result.update(
            {f"{prefix}_{key}": value for key, value in concentration(contributions).items()}
        )
    return result


def outcome_support(frame: pd.DataFrame, spec: SampleSpec) -> dict[str, float]:
    """Marginal support/scale, with no joint history association calculations."""
    n, t = spec.households, spec.last_response - spec.first_response + 1
    p = frame.retailer_purchase.to_numpy().reshape(n, t)
    result = {
        "purchase_zeros": float((p == 0).sum()),
        "purchase_ones": float((p == 1).sum()),
        "purchase_mean": float(p.mean()),
        "purchase_households_varying": float((np.ptp(p, axis=1) > 0).sum()),
        "purchase_households_constant0": float((p.max(axis=1) == 0).sum()),
        "purchase_households_constant1": float((p.min(axis=1) == 1).sum()),
        "purchase_weeks_varying": float((np.ptp(p, axis=0) > 0).sum()),
    }
    for field, prefix in (("weekly_sales_value", "revenue"), ("trips", "trips")):
        y = frame[field].to_numpy(dtype=float)
        require(np.isfinite(y).all(), f"nonfinite {field}")
        result.update(
            {
                f"{prefix}_zeros": float((y == 0).sum()),
                f"{prefix}_zero_share": float((y == 0).mean()),
                f"{prefix}_positive": float((y > 0).sum()),
                f"{prefix}_positive_share": float((y > 0).mean()),
                f"{prefix}_negative": float((y < 0).sum()),
                f"{prefix}_mean": float(y.mean()),
                f"{prefix}_median": float(np.median(y)),
                f"{prefix}_max": float(y.max()),
            }
        )
        for label, q in (("p90", 0.9), ("p95", 0.95), ("p99", 0.99)):
            result[f"{prefix}_{label}"] = float(np.quantile(y, q))
        if prefix == "revenue":
            require((y > -1).all(), "log1p revenue domain invalid; STOP/ASK")
            log_y = np.log1p(y)
            require(np.isfinite(log_y).all(), "nonfinite log1p revenue")
            result["log1p_revenue_finite_cells"] = float(len(log_y))
            require((y >= 0).all(), "negative revenue concentration requires researcher decision")
            totals = y.reshape(n, t).sum(axis=1)
            result.update(
                {
                    f"revenue_household_total_{key}": value
                    for key, value in concentration(totals).items()
                }
            )
            # Aggregate extreme-cell scales, not a household exclusion candidate.
            ordered = np.sort(y)[::-1]
            result["revenue_largest_cell_share"] = float(ordered[0] / y.sum())
            result["revenue_top01_cells_count"] = float(math.ceil(len(y) * 0.01))
            result["revenue_top01_cells_share"] = float(
                ordered[: math.ceil(len(y) * 0.01)].sum() / y.sum()
            )
    return result


def transition_counts(frame: pd.DataFrame) -> dict[str, float]:
    """Allowed 2x2 support COUNTS only; no rates/differences or significance."""
    previous = frame.lag1_retailer_purchase.to_numpy()
    current = frame.retailer_purchase.to_numpy()
    result = {
        f"previous{a}_current{b}_count": float(((previous == a) & (current == b)).sum())
        for a in (0, 1)
        for b in (0, 1)
    }
    require(sum(result.values()) == len(frame), "transition counts do not exhaust sample")
    require(
        result["previous0_current0_count"] + result["previous0_current1_count"]
        == (previous == 0).sum(),
        "transition row margin mismatch",
    )
    require(
        result["previous0_current0_count"] + result["previous1_current0_count"]
        == (current == 0).sum(),
        "transition column margin mismatch",
    )
    return result


def preflight_records(mart: pd.DataFrame, specs: tuple[SampleSpec, ...]) -> list[dict]:
    rows = []
    for spec in specs:
        frame = construct_sample(mart, spec)
        x = frame.lag1_retailer_purchase.to_numpy(dtype=float).reshape(spec.households, -1)
        domains = {
            "sample": {
                "cells": float(len(frame)),
                "households": float(spec.households),
                "response_weeks": float(x.shape[1]),
                "first_response_week": float(spec.first_response),
                "last_response_week": float(spec.last_response),
            },
            "history": raw_history_support(x),
            "regressor": regressor_support(x),
        }
        if spec.name == "S0_baseline":
            domains["outcome"] = outcome_support(frame, spec)
            domains["transition_support_counts"] = transition_counts(frame)
        for domain, metrics in domains.items():
            for metric, value in sorted(metrics.items()):
                rows.append(
                    {"sample": spec.name, "domain": domain, "metric": metric, "value": value}
                )
    return rows
