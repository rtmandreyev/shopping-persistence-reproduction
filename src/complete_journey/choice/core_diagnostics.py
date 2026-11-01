"""Aggregate-only Run C accounting and post-fit descriptive diagnostics.

No fitting, covariance construction, files, selection, or identifier values are
returned here. Numerator concentration uses signed FWL rx * partial_y event
contributions; score concentration uses signed rx * full_residual contributions.
They answer different questions: point-estimate accounting versus residual-weighted
influence. Both concentrate absolute *group nets*, not absolute event contributions
or shares of a potentially zero signed total. All observed groups enter the fixed
ceil(1/5/10 percent) denominators; effective counts are inverse share HHI.

Residual pair products are descriptive unordered event-pair summaries, not a
covariance estimator or a tested materiality criterion. Store effects mechanically
constrain store means and shared-store pair products. Weekly score persistence can
include repeated-household dependence; it does not prove uncovered cross-household
dependence. No automatic exclusion, dominance, or inference decision is made.
"""

from __future__ import annotations

from math import ceil
from types import UnionType
from typing import Any, get_args

import numpy as np
import pandas as pd

from complete_journey.choice.design import (
    DESIGN_COLUMNS,
    HISTORY_COLUMNS,
    _validate_frame,
)

ATOL = 1e-10
RTOL = 1e-8
NONZERO_TOLERANCE = 1e-12
WEEKLY_LAGS = (1, 4, 8)
HISTORY_CATEGORIES = ("H0", "HN", "HB", "HP")
GAP_BINS = ("1", "2-4", "5-8", ">8")
FRACTIONS = (0.01, 0.05, 0.10)


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > NONZERO_TOLERANCE else None


def _binary(values: np.ndarray) -> None:
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("diagnostics require a binary private_only response")


def foundation(pure: pd.DataFrame) -> dict:
    """Account for all pure responses without recomputing or inventing chronology.

    Required fields are DESIGN_COLUMNS plus composition/private_only. When factual
    event keys, line counts, or history counters are supplied, their exact semantics
    are checked. A pure response projection alone cannot reconstruct mixed history.
    Constant-type households include singletons; repeated constants require >=2
    pure events. Both definitions are exposed separately, never silently conflated.
    """
    if not isinstance(pure, pd.DataFrame) or pure.columns.has_duplicates:
        raise ValueError("foundation requires a unique-column pure DataFrame")
    required = {*DESIGN_COLUMNS, "composition", "private_only"}
    if not required.issubset(pure.columns):
        raise ValueError("foundation missing factual pure response fields")
    _validate_frame(pure.loc[:, list(DESIGN_COLUMNS)])
    y = pure.private_only.to_numpy(dtype=float)
    _binary(y)
    expected_composition = np.where(y == 1, "PRIVATE_ONLY", "NATIONAL_ONLY")
    if not np.array_equal(pure.composition.to_numpy(), expected_composition):
        raise ValueError("composition/private_only pure semantics disagree")
    if "basket_id" in pure and (
        pure.basket_id.isna().any() or pure.duplicated(["household_key", "basket_id"]).any()
    ):
        raise ValueError("foundation duplicate/missing factual event key")
    _validate_optional_facts(pure, y)
    grouped = pure.groupby("household_key", sort=False).private_only.agg(["size", "sum"])
    n = grouped["size"].to_numpy(dtype=int)
    private = grouped["sum"].to_numpy(dtype=float)
    single = n == 1
    national_constant = private == 0
    private_constant = private == n
    switcher = ~(national_constant | private_constant)
    count = len(pure)
    result = {
        "events": count,
        "national": int(np.count_nonzero(y == 0)),
        "private": int(np.count_nonzero(y == 1)),
        "private_share": float(y.mean()),
        "households": len(grouped),
        "single_event_households": int(single.sum()),
        "repeated_households": int((~single).sum()),
        "constant_national_households": int(national_constant.sum()),
        "constant_private_households": int(private_constant.sum()),
        "repeated_constant_national_households": int((national_constant & ~single).sum()),
        "repeated_constant_private_households": int((private_constant & ~single).sum()),
        "switcher_households": int(switcher.sum()),
        "single_event_events": int(n[single].sum()),
        "single_event_share": float(n[single].sum() / count),
        "repeated_events": int(n[~single].sum()),
        "repeated_event_share": float(n[~single].sum() / count),
        "switcher_events": int(n[switcher].sum()),
        "switcher_event_share": float(n[switcher].sum() / count),
        "history_table": [],
    }
    for category in HISTORY_CATEGORIES:
        current = y[pure.history_category.eq(category).to_numpy()]
        result["history_table"].append(
            {
                "category": category,
                "events": len(current),
                "national": int(np.count_nonzero(current == 0)),
                "private": int(np.count_nonzero(current == 1)),
                "private_share": float(current.mean()) if len(current) else None,
            }
        )
    validate_foundation(result)
    return result


def _nonnegative_integers(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for name in columns:
        values = frame[name].to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values != np.floor(values))
        ):
            raise ValueError("invalid nonnegative integer factual counters")


def _validate_optional_facts(pure: pd.DataFrame, y: np.ndarray) -> None:
    lines = ("national_lines", "private_lines")
    if any(name in pure for name in lines):
        if not all(name in pure for name in lines):
            raise ValueError("incomplete factual line counters")
        _nonnegative_integers(pure, lines)
        if not (
            np.array_equal(pure.private_lines.gt(0).to_numpy(), y == 1)
            and np.array_equal(pure.national_lines.gt(0).to_numpy(), y == 0)
        ):
            raise ValueError("pure response/line presence semantics disagree")
    counters = (
        "prior_national_pure",
        "prior_private_pure",
        "prior_mixed",
        "prior_total_group",
        "prior_pure",
    )
    if any(name in pure for name in counters):
        if not all(name in pure for name in counters):
            raise ValueError("incomplete factual history counters")
        _nonnegative_integers(pure, counters)
        national = pure.prior_national_pure.to_numpy()
        private = pure.prior_private_pure.to_numpy()
        total = national + private
        category = np.select(
            (total == 0, private == 0, national == 0), ("H0", "HN", "HP"), default="HB"
        )
        if not (
            np.array_equal(pure.prior_pure.to_numpy(), total)
            and np.array_equal(
                pure.prior_total_group.to_numpy(), total + pure.prior_mixed.to_numpy()
            )
            and np.array_equal(pure.history_category.to_numpy(), category)
            and np.allclose(pure.log1p_prior_pure.to_numpy(), np.log1p(total), atol=ATOL, rtol=RTOL)
        ):
            raise ValueError("factual history/category/depth semantics disagree")
        if "prior_mixed_only" in pure and not np.array_equal(
            pure.prior_mixed_only.to_numpy(), (total == 0) & pure.prior_mixed.gt(0).to_numpy()
        ):
            raise ValueError("prior mixed-only semantics disagree")
        if "prior_private_share" in pure:
            share = pure.prior_private_share.to_numpy(dtype=float)
            if not (
                np.isnan(share[total == 0]).all()
                and np.allclose(
                    share[total > 0], private[total > 0] / total[total > 0], atol=ATOL, rtol=RTOL
                )
            ):
                raise ValueError("prior Private share semantics disagree")
        # Same-week equality checks do not impose any within-week basket ordering.
        if pure.groupby(["household_key", "week_no"])[list(counters)].nunique().gt(1).any().any():
            raise ValueError("same-week factual history counters disagree")


def _fit_arrays(frame: pd.DataFrame, fit: Any) -> tuple[np.ndarray, ...]:
    """Validate row alignment and full fitted convention on a duck-typed core fit."""
    _validate_frame(frame)
    model = fit.model
    if model not in ("M0", "M1") or fit.design.model != model:
        raise ValueError("diagnostic fit model must be M0 or M1 and agree with design")
    x = np.asarray(fit.design.values, dtype=float)
    names = tuple(fit.design.column_names)
    n = len(frame)
    history = [frame.history_category.eq(h).to_numpy(dtype=float) for h in HISTORY_COLUMNS]
    depth = frame.log1p_prior_pure.to_numpy(dtype=float)
    if model == "M0":
        expected_names = ("intercept", *HISTORY_COLUMNS, "log1p_prior_pure")
        expected = np.column_stack((np.ones(n), *history, depth))
    else:
        stores = sorted(frame.store_id.unique())
        thirds = sorted(frame.calendar_third.unique())
        expected_names = (
            *(f"store[{s}]" for s in stores),
            *thirds[1:],
            "log1p_prior_pure",
            *HISTORY_COLUMNS,
        )
        expected = np.column_stack(
            [frame.store_id.eq(s).to_numpy(dtype=float) for s in stores]
            + [frame.calendar_third.eq(t).to_numpy(dtype=float) for t in thirds[1:]]
            + [depth, *history]
        )
    if names != expected_names or x.shape != expected.shape or not np.array_equal(x, expected):
        raise ValueError("diagnostic frame/design alignment or frozen specification disagreement")
    beta = np.asarray(fit.coefficients, dtype=float)
    fitted = np.asarray(fit.fitted, dtype=float)
    u = np.asarray(fit.residual, dtype=float)
    q = np.asarray(fit.q, dtype=float)
    k = x.shape[1]
    if beta.shape != (k,) or fitted.shape != (n,) or u.shape != (n,) or q.shape != (n, k):
        raise ValueError("diagnostic fit array shapes disagree")
    if not all(np.isfinite(v).all() for v in (x, beta, fitted, u, q)):
        raise ValueError("nonfinite diagnostic fit arrays")
    y = fitted + u
    # Round only to validate an exactly binary source reconstructed through floats.
    rounded = np.rint(y)
    _binary(rounded)
    if not np.allclose(y, rounded, atol=ATOL, rtol=0):
        raise ValueError("diagnostics require a binary reconstructed response")
    if not np.allclose(fitted, x @ beta, atol=ATOL, rtol=RTOL):
        raise ValueError("diagnostics require FULL fitted values including context")
    if not np.allclose(q.T @ q, np.eye(k), atol=ATOL, rtol=RTOL):
        raise ValueError("diagnostic Q must be orthonormal")
    if not np.allclose(q @ (q.T @ x), x, atol=ATOL, rtol=RTOL):
        raise ValueError("diagnostic Q must span the full frozen design")
    if not np.allclose(q.T @ u, 0, atol=ATOL, rtol=0):
        raise ValueError("diagnostic residual is not the fitted projection residual")
    return x, beta, rounded, rounded - u, u, q


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    quantiles = ("minimum", "median", "p90", "p95", "p99", "maximum")
    result: dict = {"count": int(len(values))}
    result.update(dict.fromkeys(quantiles))
    result.update({"mean": None, "standard_deviation_n": None, "root_mean_square": None})
    if len(values):
        result.update(
            {
                name: float(value)
                for name, value in zip(
                    quantiles, np.quantile(values, (0, 0.5, 0.9, 0.95, 0.99, 1)), strict=True
                )
            }
        )
        result.update(
            mean=float(values.mean()),
            standard_deviation_n=float(values.std(ddof=0)),
            root_mean_square=float(np.sqrt(np.mean(values**2))),
        )
    return result


def _group_sums(frame: pd.DataFrame, columns: list[str], values: np.ndarray) -> np.ndarray:
    internal = frame.loc[:, columns].copy()
    internal["_value"] = values
    return internal.groupby(columns, sort=False, observed=True)._value.sum().to_numpy(dtype=float)


def _mass_concentration(mass: np.ndarray) -> dict:
    total = float(mass.sum())
    groups = len(mass)
    shares = np.sort(mass / total)[::-1] if total > NONZERO_TOLERANCE else None
    return {
        "observed_groups": groups,
        "mass_sum": total,
        "largest_share": float(shares[0]) if shares is not None else None,
        "top_fraction_shares": {
            f"{f:.2f}": float(shares[: ceil(f * groups)].sum()) if shares is not None else None
            for f in FRACTIONS
        },
        "effective_groups_inverse_hhi": float(1 / np.sum(shares**2))
        if shares is not None
        else None,
    }


def _signed_concentration(frame: pd.DataFrame, axis: str, values: np.ndarray) -> dict:
    nets = _group_sums(frame, [axis], values)
    absolute_mass = np.abs(nets)
    result = _mass_concentration(absolute_mass)
    result.update(
        group_net_sum=float(nets.sum()),
        sum_absolute_group_nets=float(absolute_mass.sum()),
        gross_absolute_event_sum=float(np.abs(values).sum()),
        cancellation_ratio=_ratio(abs(float(nets.sum())), float(absolute_mass.sum())),
        largest_absolute_group_net=float(absolute_mass.max()),
    )
    return result


def model_diagnostics(frame: pd.DataFrame, fit: Any) -> dict:
    """Full fitted/leverage aggregates for M0 or M1; exact ordered six-field frame."""
    _, _, y, fitted, u, q = _fit_arrays(frame, fit)
    n, k = q.shape
    h = np.sum(q**2, axis=1)
    threshold = 2 * k / n  # Conventional descriptive rule, not an exclusion threshold.
    singleton = frame.store_id.map(frame.store_id.value_counts()).eq(1).to_numpy()
    applicable = fit.model == "M1"
    selected = singleton if applicable else np.zeros(n, dtype=bool)
    has_singleton = bool(selected.any())
    trace = float(h.sum())
    result = {
        "fitted": {
            "minimum": float(fitted.min()),
            "maximum": float(fitted.max()),
            "below_zero_count": int(np.count_nonzero(fitted < 0)),
            "below_zero_share": float(np.mean(fitted < 0)),
            "above_one_count": int(np.count_nonzero(fitted > 1)),
            "above_one_share": float(np.mean(fitted > 1)),
            "full_fitted_including_context": True,
            "clipped": False,
        },
        "leverage": {
            **_distribution(h),
            "trace": trace,
            "design_columns": k,
            "trace_absolute_difference_from_k": abs(trace - k),
            "trace_agrees_with_k": bool(np.isclose(trace, k, atol=ATOL, rtol=RTOL)),
            "conventional_threshold_2k_over_n": float(threshold),
            "above_conventional_threshold_count": int(np.count_nonzero(h > threshold)),
            "above_conventional_threshold_share": float(np.mean(h > threshold)),
            "concentration": {
                axis: _mass_concentration(_group_sums(frame, [column], h))
                for axis, column in (("households", "household_key"), ("stores", "store_id"))
            },
        },
        "singleton_stores": {
            "applicable": applicable,
            "stores": int(selected.sum()),
            "rows": int(selected.sum()),
            "leverage_minimum": float(h[selected].min()) if has_singleton else None,
            "leverage_maximum": float(h[selected].max()) if has_singleton else None,
            "maximum_absolute_leverage_difference_from_one": float(np.max(abs(h[selected] - 1)))
            if has_singleton
            else None,
            "maximum_absolute_residual": float(np.max(abs(u[selected]))) if has_singleton else None,
            "maximum_absolute_full_fitted_minus_response": float(
                np.max(abs(fitted[selected] - y[selected]))
            )
            if has_singleton
            else None,
            "all_leverage_near_one": bool(np.allclose(h[selected], 1, atol=ATOL, rtol=RTOL))
            if has_singleton
            else None,
            "all_residual_near_zero": bool(np.allclose(u[selected], 0, atol=ATOL, rtol=0))
            if has_singleton
            else None,
            "all_full_fitted_agree_with_response": bool(
                np.allclose(fitted[selected], y[selected], atol=ATOL, rtol=RTOL)
            )
            if has_singleton
            else None,
        },
    }
    validate_model_diagnostics(result)
    return result


def _weekly_scores(frame: pd.DataFrame, score: np.ndarray) -> dict:
    first, last = int(frame.week_no.min()), int(frame.week_no.max())
    weeks = np.arange(first, last + 1)
    sequence = np.zeros(len(weeks))
    np.add.at(sequence, frame.week_no.to_numpy(dtype=int) - first, score)
    centered = sequence - sequence.mean()
    norm = float(centered @ centered)
    acf = {}
    for lag in WEEKLY_LAGS:
        acf[f"lag_{lag}"] = (
            float(centered[lag:] @ centered[:-lag] / norm)
            if len(sequence) > lag and norm > NONZERO_TOLERANCE
            else None
        )
    observed = int(frame.week_no.nunique())
    return {
        "first_week": first,
        "last_week": last,
        "calendar_weeks": len(weeks),
        "observed_weeks": observed,
        "zero_filled_unobserved_weeks": len(weeks) - observed,
        "score_sum": float(sequence.sum()),
        "centered_sum_squares": norm,
        "values": [
            {"week_no": int(w), "score": float(s)} for w, s in zip(weeks, sequence, strict=True)
        ],
        "acf": acf,
    }


def _lag_pair_summary(gap: np.ndarray, before: np.ndarray, after: np.ndarray) -> dict:
    n = len(gap)
    left = before - before.mean() if n else before
    right = after - after.mean() if n else after
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return {
        "pairs": n,
        "gap": _distribution(gap),
        "pooled_pair_pearson": float(left @ right / denominator)
        if denominator > NONZERO_TOLERANCE
        else None,
        "mean_residual_product": float(np.mean(before * after)) if n else None,
    }


def _within_household(internal: pd.DataFrame) -> dict:
    weekly = internal.groupby(["household_key", "week_no"], sort=True).u.mean().reset_index()
    weekly["previous_week"] = weekly.groupby("household_key").week_no.shift()
    weekly["previous_residual"] = weekly.groupby("household_key").u.shift()
    paired = weekly.loc[weekly.previous_week.notna()]
    gap = (paired.week_no - paired.previous_week).to_numpy(dtype=float)
    before, after = paired.previous_residual.to_numpy(), paired.u.to_numpy()
    selectors = (gap == 1, (gap >= 2) & (gap <= 4), (gap >= 5) & (gap <= 8), gap > 8)
    return {
        "household_week_cells": len(weekly),
        "households_with_pairs": int(paired.household_key.nunique()),
        "overall": _lag_pair_summary(gap, before, after),
        "gap_bins": {
            name: _lag_pair_summary(gap[select], before[select], after[select])
            for name, select in zip(GAP_BINS, selectors, strict=True)
        },
    }


def _pair_totals(internal: pd.DataFrame, columns: list[str]) -> tuple[int, float]:
    grouped = internal.groupby(columns, sort=False, observed=True).u.agg(["size", "sum"])
    count = grouped["size"].to_numpy(dtype=np.int64)
    sums = grouped["sum"].to_numpy(dtype=float)
    # Ordered n^2 formulas divided by two give unordered distinct-event pairs.
    pairs = int(np.sum(count * (count - 1) // 2))
    product = float((np.sum(sums**2) - np.sum(internal.u.to_numpy() ** 2)) / 2)
    return pairs, product


def _shared_store_pairs(internal: pd.DataFrame) -> dict:
    all_count, all_product = _pair_totals(internal, ["store_id"])
    hh_count, hh_product = _pair_totals(internal, ["store_id", "household_key"])
    sw_count, sw_product = _pair_totals(internal, ["store_id", "week_no"])
    hsw_count, hsw_product = _pair_totals(internal, ["store_id", "household_key", "week_no"])
    cross_count, cross_product = all_count - hh_count, all_product - hh_product
    same_count, same_product = sw_count - hsw_count, sw_product - hsw_product
    different_count, different_product = cross_count - same_count, cross_product - same_product
    return {
        name: {
            "pairs": count,
            "residual_product_sum": product,
            "mean_residual_product": float(product / count) if count else None,
        }
        for name, count, product in (
            ("all_weeks", cross_count, cross_product),
            ("same_week", same_count, same_product),
            ("different_weeks", different_count, different_product),
        )
    }


def _residual_dependence(frame: pd.DataFrame, u: np.ndarray) -> dict:
    internal = frame.loc[:, ["household_key", "store_id", "week_no"]].copy()
    internal["u"] = u
    store_means = internal.groupby("store_id", sort=False).u.mean().to_numpy()
    return {
        "within_household": _within_household(internal),
        "store_means": {
            **_distribution(store_means),
            "maximum_absolute_mean": float(np.max(abs(store_means))),
            "mechanically_constrained_by_store_effects": True,
        },
        "store_week_means": _distribution(
            internal.groupby(["store_id", "week_no"], sort=False).u.mean().to_numpy()
        ),
        "household_store_means": _distribution(
            internal.groupby(["household_key", "store_id"], sort=False).u.mean().to_numpy()
        ),
        "cross_household_shared_store_pairs": _shared_store_pairs(internal),
    }


def primary(frame: pd.DataFrame, fit: Any) -> dict:
    """M1-only focal FWL, residual-score concentration, weekly ACF and residual pairs.

    Each focal regressor AND Y is independently QR-residualized on *all* other
    columns, including the two remaining history terms. The numerator sums to
    beta * sum(rx^2), checked with fixed atol=1e-10/rtol=1e-8. Efficient scores are
    rx * full u (the covariance direction up to a constant scaling). Calendar
    sequences span min..max observed week only, fill internal unobserved weeks with
    zero, and use exactly lags 1, 4, 8. No tests, lag search or materiality threshold.
    """
    if fit.model != "M1":
        raise ValueError("primary diagnostics require M1 only")
    x, beta, y, _, u, _ = _fit_arrays(frame, fit)
    terms = {}
    for name in HISTORY_COLUMNS:
        position = fit.design.column_names.index(name)
        controls = np.delete(x, position, axis=1)
        q_controls, _ = np.linalg.qr(controls, mode="reduced")
        rx = x[:, position] - q_controls @ (q_controls.T @ x[:, position])
        partial_y = y - q_controls @ (q_controls.T @ y)
        numerator_events = rx * partial_y
        denominator = float(rx @ rx)
        numerator = float(numerator_events.sum())
        target = float(beta[position] * denominator)
        if not np.isclose(numerator, target, atol=ATOL, rtol=RTOL):
            raise ValueError("focal FWL numerator does not reconcile with fitted coefficient")
        score = rx * u
        score_sum = float(score.sum())
        if not np.isclose(score_sum, 0, atol=ATOL, rtol=0):
            raise ValueError("focal residual score does not sum to zero")
        terms[name] = {
            "fwl_denominator": denominator,
            "fwl_numerator": numerator,
            "coefficient_times_denominator": target,
            "numerator_absolute_reconciliation_difference": abs(numerator - target),
            "numerator_reconciles": True,
            "fwl_coefficient": _ratio(numerator, denominator),
            "residual_score_sum": score_sum,
            "residual_score_sums_to_zero": True,
            "numerator_concentration": {
                axis: _signed_concentration(frame, column, numerator_events)
                for axis, column in (("households", "household_key"), ("stores", "store_id"))
            },
            "residual_score_concentration": {
                axis: _signed_concentration(frame, column, score)
                for axis, column in (("households", "household_key"), ("stores", "store_id"))
            },
            "weekly_scores": _weekly_scores(frame, score),
        }
    result = {
        "focal_contrasts": terms,
        "residual_dependence": _residual_dependence(frame, u),
        "conventions": {
            "agreement_atol": ATOL,
            "agreement_rtol": RTOL,
            "nonzero_tolerance": NONZERO_TOLERANCE,
            "top_fraction_denominator_all_observed_groups_ceiling": True,
            "concentration_uses_absolute_signed_group_nets": True,
            "weekly_acf_centered_full_sequence_norm": True,
            "internal_missing_calendar_weeks_have_zero_score": True,
            "household_pairs_are_consecutive_observed_week_means": True,
            "shared_store_pairs_exclude_same_household": True,
            "shared_store_pair_products_mechanically_fe_constrained": True,
            "aggregate_acf_proves_uncovered_cross_household_dependence": False,
            "materiality_threshold_tested": False,
            "exclusions_or_covariance_selection": False,
        },
    }
    validate_primary(result)
    return result


# Static recursive contracts: dictionaries have EXACT keys; tuples describe fixed
# lists (the four history rows); a one-item list describes repeated weekly rows.
# Type leaves require native Python types (bool is not int); union leaves permit
# exactly the named native scalar types. Literal leaves pin frozen metadata.
# These declarations and validators never receive a frame, fit, or outcome.
_DISTRIBUTION_SCHEMA = {
    "count": int,
    **dict.fromkeys(
        (
            "minimum",
            "median",
            "p90",
            "p95",
            "p99",
            "maximum",
            "mean",
            "standard_deviation_n",
            "root_mean_square",
        ),
        float | None,
    ),
}
_MASS_SCHEMA = {
    "observed_groups": int,
    "mass_sum": float,
    "largest_share": float | None,
    "top_fraction_shares": dict.fromkeys(("0.01", "0.05", "0.10"), float | None),
    "effective_groups_inverse_hhi": float | None,
}
_SIGNED_SCHEMA = {
    **_MASS_SCHEMA,
    "group_net_sum": float,
    "sum_absolute_group_nets": float,
    "gross_absolute_event_sum": float,
    "cancellation_ratio": float | None,
    "largest_absolute_group_net": float,
}
_PAIR_SCHEMA = {
    "pairs": int,
    "residual_product_sum": float,
    "mean_residual_product": float | None,
}
_LAG_PAIR_SCHEMA = {
    "pairs": int,
    "gap": _DISTRIBUTION_SCHEMA,
    "pooled_pair_pearson": float | None,
    "mean_residual_product": float | None,
}
_WEEKLY_SCHEMA = {
    **dict.fromkeys(
        (
            "first_week",
            "last_week",
            "calendar_weeks",
            "observed_weeks",
            "zero_filled_unobserved_weeks",
        ),
        int,
    ),
    "score_sum": float,
    "centered_sum_squares": float,
    "values": [{"week_no": int, "score": float}],
    "acf": dict.fromkeys(("lag_1", "lag_4", "lag_8"), float | None),
}
FOUNDATION_SCHEMA = {
    **dict.fromkeys(
        (
            "events",
            "national",
            "private",
            "households",
            "single_event_households",
            "repeated_households",
            "constant_national_households",
            "constant_private_households",
            "repeated_constant_national_households",
            "repeated_constant_private_households",
            "switcher_households",
            "single_event_events",
            "repeated_events",
            "switcher_events",
        ),
        int,
    ),
    **dict.fromkeys(
        ("private_share", "single_event_share", "repeated_event_share", "switcher_event_share"),
        float,
    ),
    "history_table": tuple(
        {
            "category": category,
            "events": int,
            "national": int,
            "private": int,
            "private_share": float | None,
        }
        for category in HISTORY_CATEGORIES
    ),
}
MODEL_DIAGNOSTICS_SCHEMA = {
    "fitted": {
        "minimum": float,
        "maximum": float,
        "below_zero_count": int,
        "below_zero_share": float,
        "above_one_count": int,
        "above_one_share": float,
        "full_fitted_including_context": True,
        "clipped": False,
    },
    "leverage": {
        **_DISTRIBUTION_SCHEMA,
        "trace": float,
        "design_columns": int,
        "trace_absolute_difference_from_k": float,
        "trace_agrees_with_k": bool,
        "conventional_threshold_2k_over_n": float,
        "above_conventional_threshold_count": int,
        "above_conventional_threshold_share": float,
        "concentration": {"households": _MASS_SCHEMA, "stores": _MASS_SCHEMA},
    },
    "singleton_stores": {
        "applicable": bool,
        "stores": int,
        "rows": int,
        **dict.fromkeys(
            (
                "leverage_minimum",
                "leverage_maximum",
                "maximum_absolute_leverage_difference_from_one",
                "maximum_absolute_residual",
                "maximum_absolute_full_fitted_minus_response",
            ),
            float | None,
        ),
        **dict.fromkeys(
            (
                "all_leverage_near_one",
                "all_residual_near_zero",
                "all_full_fitted_agree_with_response",
            ),
            bool | None,
        ),
    },
}
_FOCAL_SCHEMA = {
    "fwl_denominator": float,
    "fwl_numerator": float,
    "coefficient_times_denominator": float,
    "numerator_absolute_reconciliation_difference": float,
    "numerator_reconciles": True,
    "fwl_coefficient": float | None,
    "residual_score_sum": float,
    "residual_score_sums_to_zero": True,
    "numerator_concentration": {"households": _SIGNED_SCHEMA, "stores": _SIGNED_SCHEMA},
    "residual_score_concentration": {"households": _SIGNED_SCHEMA, "stores": _SIGNED_SCHEMA},
    "weekly_scores": _WEEKLY_SCHEMA,
}
PRIMARY_SCHEMA = {
    "focal_contrasts": {name: _FOCAL_SCHEMA for name in HISTORY_COLUMNS},
    "residual_dependence": {
        "within_household": {
            "household_week_cells": int,
            "households_with_pairs": int,
            "overall": _LAG_PAIR_SCHEMA,
            "gap_bins": {name: _LAG_PAIR_SCHEMA for name in GAP_BINS},
        },
        "store_means": {
            **_DISTRIBUTION_SCHEMA,
            "maximum_absolute_mean": float,
            "mechanically_constrained_by_store_effects": True,
        },
        "store_week_means": _DISTRIBUTION_SCHEMA,
        "household_store_means": _DISTRIBUTION_SCHEMA,
        "cross_household_shared_store_pairs": {
            "all_weeks": _PAIR_SCHEMA,
            "same_week": _PAIR_SCHEMA,
            "different_weeks": _PAIR_SCHEMA,
        },
    },
    "conventions": {
        "agreement_atol": ATOL,
        "agreement_rtol": RTOL,
        "nonzero_tolerance": NONZERO_TOLERANCE,
        "top_fraction_denominator_all_observed_groups_ceiling": True,
        "concentration_uses_absolute_signed_group_nets": True,
        "weekly_acf_centered_full_sequence_norm": True,
        "internal_missing_calendar_weeks_have_zero_score": True,
        "household_pairs_are_consecutive_observed_week_means": True,
        "shared_store_pairs_exclude_same_household": True,
        "shared_store_pair_products_mechanically_fe_constrained": True,
        "aggregate_acf_proves_uncovered_cross_household_dependence": False,
        "materiality_threshold_tested": False,
        "exclusions_or_covariance_selection": False,
    },
}


def _validate_closed(value: Any, schema: Any) -> None:
    if isinstance(schema, dict):
        if type(value) is not dict or set(value) != set(schema):
            raise ValueError("aggregate diagnostics exact dictionary keys required")
        for key, subschema in schema.items():
            _validate_closed(value[key], subschema)
    elif isinstance(schema, tuple):
        if type(value) is not list or len(value) != len(schema):
            raise ValueError("aggregate diagnostics fixed history list required")
        for child, subschema in zip(value, schema, strict=True):
            _validate_closed(child, subschema)
    elif isinstance(schema, list):
        if type(value) is not list:
            raise ValueError("aggregate diagnostics weekly list required")
        for child in value:
            _validate_closed(child, schema[0])
    elif isinstance(schema, (type, UnionType)):
        types = get_args(schema) if isinstance(schema, UnionType) else (schema,)
        if type(value) not in types:
            raise ValueError(
                "aggregate diagnostics native scalar type required; arrays/aliases refused"
            )
        if type(value) is float and not np.isfinite(value):
            raise ValueError("aggregate diagnostics finite scalar required")
        if type(value) is int and value < 0:
            raise ValueError("aggregate diagnostics nonnegative count/calendar scalar required")
    elif type(value) is not type(schema) or value != schema:
        raise ValueError("aggregate diagnostics frozen literal metadata mismatch")


def validate_foundation(report: dict) -> None:
    """Reject unknown keys, arbitrary arrays, scalar aliases, and history labels."""
    _validate_closed(report, FOUNDATION_SCHEMA)
    if report["national"] + report["private"] != report["events"]:
        raise ValueError("foundation response counts do not reconcile")
    if sum(row["events"] for row in report["history_table"]) != report["events"]:
        raise ValueError("foundation history counts do not reconcile")
    for row in report["history_table"]:
        if row["national"] + row["private"] != row["events"]:
            raise ValueError("foundation history response counts do not reconcile")
        expected = row["private"] / row["events"] if row["events"] else None
        if (row["private_share"] is None) != (expected is None) or (
            expected is not None
            and not np.isclose(row["private_share"], expected, atol=ATOL, rtol=RTOL)
        ):
            raise ValueError("foundation history share/denominator semantics disagree")


def validate_model_diagnostics(report: dict) -> None:
    """Validate only the aggregate contract; never build or fit a design."""
    _validate_closed(report, MODEL_DIAGNOSTICS_SCHEMA)


def validate_primary(report: dict) -> None:
    """Validate closed aggregates and permitted contiguous calendar score lists."""
    _validate_closed(report, PRIMARY_SCHEMA)
    for name in HISTORY_COLUMNS:
        weekly = report["focal_contrasts"][name]["weekly_scores"]
        first, last = weekly["first_week"], weekly["last_week"]
        if not (1 <= first <= last <= 102):
            raise ValueError("weekly scores outside frozen calendar support")
        if (
            weekly["calendar_weeks"] != last - first + 1
            or len(weekly["values"]) != last - first + 1
        ):
            raise ValueError("weekly scores calendar length mismatch")
        if [row["week_no"] for row in weekly["values"]] != list(range(first, last + 1)):
            raise ValueError("weekly scores require every calendar week exactly once")
        if not (
            1 <= weekly["observed_weeks"] <= weekly["calendar_weeks"]
            and weekly["zero_filled_unobserved_weeks"]
            == weekly["calendar_weeks"] - weekly["observed_weeks"]
        ):
            raise ValueError("weekly zero-fill accounting mismatch")
        if not np.isclose(
            sum(row["score"] for row in weekly["values"]), weekly["score_sum"], atol=ATOL, rtol=RTOL
        ):
            raise ValueError("weekly aggregate score sum mismatch")
