"""Frozen covariance options and SYNTHETIC NON-EVIDENCE software preflight.

No caller data are accepted, no data files are read, and no fitted quantities are
returned. Toy fits exist only to test linearmodels 7.0's implementation of the
Phase 4 section K contract; this module is not a general estimation framework.
"""

import warnings

import linearmodels
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS
from linearmodels.panel.utility import AbsorbingEffectError
from linearmodels.shared.exceptions import SingletonWarning
from scipy import stats

_SYNTHETIC_TAG = "SYNTHETIC NON-EVIDENCE"


def covariance_options(kind: str = "two_way", effects: bool = True) -> dict:
    """Return fresh, explicit fit kwargs, without relying on library defaults.

    ``effects=False`` omits PanelOLS-only degree-of-freedom flags so the kwargs
    can be passed to PooledOLS. DK uses no cluster-only flags, including
    ``group_debias``. FE flags explicitly count effects even when nested in a
    cluster. This function configures covariance only, not model construction.
    """
    if type(effects) is not bool:
        raise TypeError("effects must be a bool")
    if kind not in ("two_way", "household", "dk4", "dk8"):
        raise ValueError("kind must be two_way, household, dk4, or dk8")
    options: dict = {"debiased": True}
    if kind in ("two_way", "household"):
        options.update(
            cov_type="clustered",
            cluster_entity=True,
            cluster_time=kind == "two_way",
            group_debias=True,
        )
    else:
        options.update(cov_type="kernel", kernel="bartlett", bandwidth=4 if kind == "dk4" else 8)
    if effects:
        options.update(auto_df=False, count_effects=True)
    return options


def _require(condition: bool, description: str) -> None:
    # Do not include fitted quantities in failure messages, even on a STOP.
    if not condition:
        raise RuntimeError(f"{_SYNTHETIC_TAG}: software semantics mismatch: {description}")


def _equal(actual, expected, description: str) -> None:
    _require(bool(np.allclose(actual, expected, rtol=1e-9, atol=1e-11)), description)


def _toy() -> pd.DataFrame:
    """Fixed balanced internal toy; its regressor is not an observed lag."""
    entities, periods = 12, 16  # Both explicit DK bandwidths are shorter than T.
    rng = np.random.default_rng(4207)
    index = pd.MultiIndex.from_product(
        [range(entities), range(periods)], names=["synthetic_entity", "synthetic_time"]
    )
    entity = np.repeat(np.arange(entities), periods)
    time = np.tile(np.arange(periods), entities)
    x = 2 + rng.normal(size=len(index)) + entity / 10 + np.sin(time / 3)
    y = 3 + 0.4 * x + rng.normal(size=len(index)) + entity / 7 + np.cos(time / 4)
    # Linear FE must retain both always-zero and always-one toy outcomes.
    y[entity == 0] = 0
    y[entity == 1] = 1
    return pd.DataFrame(
        {"synthetic_constant": 1.0, "synthetic_x": x, "synthetic_y": y}, index=index
    )


def _transform(frame: pd.DataFrame, entity_effects: bool, time_effects: bool) -> np.ndarray:
    """Balanced FE transform with the intercept normalization independently built.

    Double demeaning is a - entity_mean - time_mean + grand_mean. With an
    explicit constant, linearmodels adds the grand mean *again*: the constant
    stays one rather than becoming a zero column. This matters for its full
    covariance (especially the normalized intercept), not just the slope.
    """
    values = frame.to_numpy(copy=True)
    grand_mean = values.mean(axis=0)
    if entity_effects:
        values -= frame.groupby(level=0).transform("mean").to_numpy()
    if time_effects:
        values -= frame.groupby(level=1).transform("mean").to_numpy()
    if entity_effects and time_effects:
        values += grand_mean
    if entity_effects or time_effects:
        values += grand_mean
    return values


def _score_meat(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Unnormalized sum of cluster score outer products, independent of library."""
    _, codes = np.unique(groups, return_inverse=True, axis=0)
    totals = np.zeros((int(codes.max()) + 1, scores.shape[1]))
    np.add.at(totals, codes, scores)
    return totals.T @ totals


def _manual_cluster(x, residual, entity, time, df_resid: int, two_way: bool) -> np.ndarray:
    nobs = len(x)
    scores = x * residual[:, None]

    def corrected(groups):
        count = len(np.unique(groups, axis=0))
        correction = count / (count - 1) * (nobs - 1) / nobs
        return correction * _score_meat(scores, groups)

    meat = corrected(entity)
    if two_way:
        meat += corrected(time) - corrected(np.column_stack([entity, time]))
    bread = np.linalg.inv(x.T @ x)
    covariance = (nobs / df_resid) * bread @ meat @ bread
    return (covariance + covariance.T) / 2


def _manual_dk(x, residual, time, df_resid: int, bandwidth: int) -> np.ndarray:
    scores = x * residual[:, None]
    _, codes = np.unique(time, return_inverse=True)
    totals = np.zeros((int(codes.max()) + 1, x.shape[1]))
    np.add.at(totals, codes, scores)
    meat = totals.T @ totals
    for lag in range(1, bandwidth + 1):
        cross = totals[lag:].T @ totals[:-lag]
        meat += (1 - lag / (bandwidth + 1)) * (cross + cross.T)
    # Library's /T and T/n cancel, leaving the same n/df_resid factor.
    bread = np.linalg.inv(x.T @ x)
    covariance = (len(x) / df_resid) * bread @ meat @ bread
    return (covariance + covariance.T) / 2


def check_scalar_variance(value: float) -> None:
    """P4-002: native scalar history inference stops, never repairs invalid variance."""
    _require(
        bool(np.isfinite(value) and value > 0),
        "positive finite scalar history variance; never repair or switch",
    )


def _check_native_reference(result) -> None:
    covariance = result.cov.to_numpy()
    check_scalar_variance(float(covariance[1, 1]))
    _require(bool(np.isfinite(covariance).all()), "finite covariance")
    _require(bool((np.diag(covariance) > 0).all()), "positive native variances; never clip")
    # Two-way inclusion/exclusion is not PSD-preserving (notably for a
    # normalized TWFE intercept). Do not impose a non-frozen PSD repair; native
    # coefficient inference requires finite, positive diagonal variances.
    parameters = result.params.to_numpy()
    errors = np.sqrt(np.diag(covariance))
    quantiles = stats.t.ppf([0.025, 0.975], result.df_resid)
    _equal(
        result.conf_int().to_numpy(),
        parameters[:, None] + errors[:, None] * quantiles,
        "native 95% Student t(df_resid) intervals",
    )
    _equal(
        result.pvalues.to_numpy(),
        2 * (1 - stats.t.cdf(np.abs(parameters / errors), result.df_resid)),
        "native Student t(df_resid) p-values",
    )


def _check_fit(frame, entity_effects: bool, time_effects: bool, kind: str) -> dict:
    effects = entity_effects or time_effects
    x_frame = frame[["synthetic_constant", "synthetic_x"]]
    y_frame = frame[["synthetic_y"]]
    entity = frame.index.get_level_values(0).to_numpy()
    time = frame.index.get_level_values(1).to_numpy()
    nobs = len(frame)
    nentity, ntime = len(np.unique(entity)), len(np.unique(time))
    extra_df = (nentity - 1 if entity_effects else 0) + (ntime - 1 if time_effects else 0)
    df_resid = nobs - x_frame.shape[1] - extra_df
    options = covariance_options(kind, effects=effects)
    if effects:
        model = PanelOLS(
            y_frame,
            x_frame,
            entity_effects=entity_effects,
            time_effects=time_effects,
            singletons=True,
            drop_absorbed=False,
            check_rank=True,
        )
    else:
        model = PooledOLS(y_frame, x_frame, check_rank=True)
    result = model.fit(**options)
    _require(result.nobs == nobs, "all toy observations, including constant outcomes, retained")
    _require(result.resids.index.equals(frame.index), "exact toy row index retained")
    _require(list(result.params.index) == list(x_frame.columns), "explicit constant retained")
    _require(result.df_resid == df_resid, "reported df_resid counts explicit effects")
    _require(result.df_model == 2 + extra_df, "reported df_model counts explicit effects")
    x = _transform(x_frame, entity_effects, time_effects)
    y = _transform(y_frame, entity_effects, time_effects).ravel()
    parameters = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ parameters
    _equal(result.params.to_numpy(), parameters, "manual balanced FE/intercept normalization")
    _equal(result.resids.to_numpy(), residual, "manual residualized toy residuals")
    covariance_impl = result._deferred_cov.__self__
    _equal(covariance_impl._x, x, "actual library covariance uses restored-grand-mean design")
    _equal(covariance_impl._scale, nobs / df_resid, "actual covariance n/(n-k-extra_df) scale")
    _require(covariance_impl._extra_df == extra_df, "actual extra_df explicitly counts effects")
    _require(covariance_impl._debiased is True, "actual debiased flag")
    if kind in ("two_way", "household"):
        _require(covariance_impl._group_debias is True, "actual cluster group correction")
        expected_clusters = np.column_stack([entity, time]) if kind == "two_way" else entity
        actual_clusters = covariance_impl._clusters
        _require(actual_clusters.shape == expected_clusters.shape, "actual cluster dimension flags")
        _equal(
            actual_clusters, expected_clusters, "actual household/response-time cluster identities"
        )
        manual = _manual_cluster(x, residual, entity, time, df_resid, kind == "two_way")
        cluster_counts = {"entity": nentity}
        if kind == "two_way":
            cluster_counts.update(
                time=ntime, intersection=len(np.unique(expected_clusters, axis=0))
            )
    else:
        bandwidth = 4 if kind == "dk4" else 8
        _require(covariance_impl._kernel == "bartlett", "actual DK Bartlett kernel")
        _require(covariance_impl._bandwidth == bandwidth, "actual explicit DK bandwidth")
        manual = _manual_dk(x, residual, time, df_resid, bandwidth)
        cluster_counts = {}  # DK aggregate-time scores are not cluster-based inference.
    _equal(result.cov.to_numpy(), manual, "independently constructed complete covariance")
    _check_native_reference(result)
    return {
        "kind": kind,
        "entity_effects": entity_effects,
        "time_effects": time_effects,
        "observations": nobs,
        "regressors_including_constant": 2,
        "effects_df": extra_df,
        "df_model": int(result.df_model),
        "df_resid": int(result.df_resid),
        "cluster_counts": cluster_counts,
        "options": options,
        "covariance_implementation": type(covariance_impl).__name__,
        "manual_covariance_passed": True,
        "native_student_t_passed": True,
        "constant_outcome_rows_retained": True,
        "constant_retained": True,
        "full_covariance_positive_semidefinite": bool(
            np.linalg.eigvalsh(result.cov.to_numpy()).min() >= -1e-11
        ),
        "native_variances_positive": True,
        "covariance_repaired_or_clipped": False,
    }


def _check_singletons(frame: pd.DataFrame) -> dict:
    # A three-edge appendage cascades out of the two-core: testing only a single
    # unique entity would miss the recursive two-way singleton rule.
    appendage = pd.DataFrame(
        {"synthetic_constant": 1.0, "synthetic_x": [2.2, 3.1, 1.7], "synthetic_y": [1.0, 0.0, 2.0]},
        index=pd.MultiIndex.from_tuples([(12, 0), (12, 16), (13, 16)], names=frame.index.names),
    )
    unbalanced = pd.concat([frame, appendage])
    x = unbalanced[["synthetic_constant", "synthetic_x"]]
    y = unbalanced.synthetic_y
    default_model = PanelOLS(
        y, x, entity_effects=True, time_effects=True, drop_absorbed=False, check_rank=True
    )
    retained = default_model.fit(**covariance_options())
    _require(retained.nobs == len(unbalanced), "singletons=True (default) retains observations")
    _require(retained.resids.index.equals(unbalanced.index), "retained singleton row identities")
    _require(default_model._singleton_index is None, "default performs no singleton deletion")
    _check_native_reference(retained)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SingletonWarning)
        dropped_model = PanelOLS(
            y,
            x,
            entity_effects=True,
            time_effects=True,
            singletons=False,
            drop_absorbed=False,
            check_rank=True,
        )
    _require(
        any(issubclass(w.category, SingletonWarning) for w in caught), "singleton removal warning"
    )
    _require(
        dropped_model.dependent.index.equals(frame.index),
        "singletons=False recursively removes three-edge appendage",
    )
    _require(int(dropped_model._singleton_index.sum()) == 3, "recursive singleton drop count")
    # The frozen policy is retention, not deletion. No clustered fit of a
    # singleton-deleted sample is required or used here; DK verifies its row count.
    dropped = dropped_model.fit(**covariance_options("dk4"))
    _require(dropped.nobs == len(frame), "singleton deletion fit row count")
    _require(dropped.resids.index.equals(frame.index), "singleton deletion fit identities")
    return {
        "passed": True,
        "default_singletons_true_retains": True,
        "explicit_singletons_false_drops_recursive_two_core": True,
        "observations_before": len(unbalanced),
        "observations_retained_by_default": int(retained.nobs),
        "observations_after_explicit_deletion": int(dropped.nobs),
        "recursive_drop_count": 3,
        "frozen_policy": "singletons=True; retain, do not silently delete",
    }


def _check_absorbed_refusal(frame: pd.DataFrame) -> dict:
    entity = frame.index.get_level_values(0).to_numpy(dtype=float)
    time = frame.index.get_level_values(1).to_numpy(dtype=float)
    refused = []
    for name, absorbed, entity_effects, time_effects in (
        ("entity_invariant", entity, True, False),
        ("time_invariant", time, False, True),
        ("additive_two_way", entity + time, True, True),
    ):
        x = frame[["synthetic_constant", "synthetic_x"]].assign(synthetic_absorbed=absorbed)
        model = PanelOLS(
            frame.synthetic_y,
            x,
            entity_effects=entity_effects,
            time_effects=time_effects,
            singletons=True,
            drop_absorbed=False,
            check_rank=True,
        )
        try:
            model.fit(**covariance_options())
        except AbsorbingEffectError:
            refused.append(name)
        else:
            _require(False, f"absorbed variable must raise, not drop: {name}")
    return {"passed": True, "refused_cases": refused, "drop_absorbed": False, "check_rank": True}


def verify_software_semantics() -> dict:
    """Fit only internally generated toys and return metadata-only check results.

    Any semantic mismatch raises a STOP (RuntimeError) rather than changing
    covariance, clipping a variance, omitting observations or altering inference.
    ``passed`` denotes implementation agreement only. Full-matrix indefiniteness
    is explicitly reported under ``full_covariance_validity_review``. P4-002
    permits native scalar history inference only, with positive finite variance,
    no joint tests and no covariance repair; it does not authorize Run C.
    Private covariance fields are intentionally inspected: this is a version-
    specific implementation preflight, not a portable public estimator API.
    """
    _require(linearmodels.__version__ == "7.0", "locked linearmodels version must be 7.0")
    frame = _toy()
    binary_y = (frame.synthetic_y > frame.synthetic_y.median()).astype(float)
    binary_y.loc[0] = 0.0
    binary_y.loc[1] = 1.0
    binary_model = PanelOLS(
        binary_y,
        frame[["synthetic_constant", "synthetic_x"]],
        entity_effects=True,
        time_effects=True,
        singletons=True,
        drop_absorbed=False,
        check_rank=True,
    )
    _require(
        bool(binary_model.dependent.dataframe.iloc[:, 0].isin([0, 1]).all()),
        "binary outcome constructor support",
    )
    _require(
        len(binary_model.dependent.dataframe) == len(frame),
        "binary constructor retains constant outcomes",
    )
    fits = []
    for entity_effects, time_effects in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        for kind in ("two_way", "household"):
            fits.append(_check_fit(frame, entity_effects, time_effects, kind))
    for kind in ("dk4", "dk8"):
        fits.append(_check_fit(frame, True, True, kind))
    # The same independent LS solution is asserted by every covariance fit,
    # so neither the coefficients nor the rows can change across covariance types.
    singleton_checks = _check_singletons(frame)
    absorbed_checks = _check_absorbed_refusal(frame)
    indefinite_covariances = sum(not fit["full_covariance_positive_semidefinite"] for fit in fits)
    return {
        "evidence_status": _SYNTHETIC_TAG,
        "passed": True,  # Implementation-semantic checks, not permission for real estimation.
        "full_covariance_validity_review": {
            "requires_researcher_review": False,
            "researcher_decision": "P4-002",
            "indefinite_toy_covariance_count": indefinite_covariances,
            "status": "SCALAR_BOUNDARY_CLARIFIED_FULL_MATRIX_FLAGGED",
            "joint_tests_permitted": False,
            "repairs_permitted": False,
            "scalar_history_variance_rule": "finite and strictly positive; stop otherwise",
            "description": (
                "Native two-way TWFE full covariance is indefinite on the toy, despite positive "
                "native diagonal variances and exact manual agreement. No repair or clipping. "
                "P4-002 permits native scalar history inference only; no joint tests or "
                "matrix repair. Run C still requires separate researcher authorization."
            ),
        },
        "linearmodels_version": linearmodels.__version__,
        "fixture": {
            "source": "fixed internal synthetic toy only",
            "balanced": True,
            "observations": len(frame),
            "entities": 12,
            "periods": 16,
            "constant_outcome_entities": 2,
        },
        "fit_checks": fits,
        "singleton_checks": singleton_checks,
        "absorbed_variable_checks": absorbed_checks,
        "inference_reference": "native Student t(df_resid), not cluster count minus one",
        "cluster_normalization": "n/(n-k-extra_df); g/(g-1)*(n-1)/n separately for each dimension and intersection",
        "intercept_normalization": "balanced FE residualization plus restored grand means; explicit constant retained",
        "dk_normalization": "time-summed scores; Bartlett weight 1-j/(bandwidth+1); n/(n-k-extra_df)",
        "dk_status": "CLEANLY IMPLEMENTABLE",
        "checks": {
            "binary_outcome_constructor": True,
            "manual_cluster_covariance": True,
            "manual_dk_covariance": True,
            "actual_cluster_counts": True,
            "effects_df_counted_explicitly": True,
            "native_student_t_df_resid": True,
            "constant_retention": True,
            "singleton_semantics": True,
            "absorbed_variable_refusal": True,
            "same_solution_and_rows_across_covariances": True,
        },
    }
