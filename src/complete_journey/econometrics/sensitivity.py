"""Closed Run D register: descriptive TWFE sensitivities and independent audits.

Accepts an already validated mart and the nine validated archived Run C rows.
No file access, output publication, covariance repair, joint test or deletion.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from complete_journey.econometrics.core import CORE_KEYS, HISTORY, covariance_boundary
from complete_journey.econometrics.covariance import covariance_options
from complete_journey.econometrics.samples import SAMPLES, construct_sample, require

# Fixed before any input is inspected; never calibrated to empirical discrepancies.
ABS_TOL = 1e-9
REL_TOL = 1e-9
MANUAL_ABS_TOL = 1e-9
MANUAL_REL_TOL = 1e-9
OUTCOMES = ("retailer_purchase", "weekly_sales_value", "trips")
LOG_TRANSFORMATION = "log(1 + retailer SALES_VALUE)"
LOG_COLUMN = "_run_d_log1p_sales_value"
COVARIANCE_LABELS = {
    "two_way": "two-way household/response-week clustered",
    "household": "household clustered",
    "dk4": "Driscoll-Kraay Bartlett bandwidth 4",
    "dk8": "Driscoll-Kraay Bartlett bandwidth 8",
}


@dataclass(frozen=True)
class SensitivityJob:
    outcome: str
    check_type: str
    variant: str
    sample_index: int
    covariance: str
    transformation: str = "levels"

    @property
    def model_id(self) -> str:
        return "M4" if self.variant == "log1p" else "M3"


# Baselines first, then the twelve population/calendar fits, one functional form,
# and nine covariance fits. There is deliberately no configurable search surface.
JOBS = (
    tuple(SensitivityJob(y, "baseline", "baseline", 0, "two_way") for y in OUTCOMES)
    + tuple(
        SensitivityJob(y, check, variant, sample_index, "two_way")
        for variant, check, sample_index in (
            ("S1", "population", 3),
            ("S2", "population", 4),
            ("full_week_pair", "calendar", 1),
            ("post_formation", "calendar", 2),
        )
        for y in OUTCOMES
    )
    + (
        SensitivityJob(
            "weekly_sales_value", "functional_form", "log1p", 0, "two_way", LOG_TRANSFORMATION
        ),
    )
    + tuple(
        SensitivityJob(y, "covariance", kind, 0, kind)
        for kind in ("household", "dk4", "dk8")
        for y in OUTCOMES
    )
)
require(
    len(JOBS) == 25 and len({(j.outcome, j.variant) for j in JOBS}) == 25,
    "Run D register must contain exactly 25 unique fits",
)


def _close(actual: float, expected: float, *, manual: bool = False) -> bool:
    absolute, relative = (MANUAL_ABS_TOL, MANUAL_REL_TOL) if manual else (ABS_TOL, REL_TOL)
    return bool(
        np.isfinite([actual, expected]).all()
        and abs(actual - expected) <= absolute + relative * abs(expected)
    )


def _independent_twfe(
    frame: pd.DataFrame, outcome_column: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """FWL from original fields via group means, not the estimator's design."""
    data = frame[[HISTORY, outcome_column]].astype(float)
    transformed = (
        data
        - data.groupby(frame.household_key).transform("mean")
        - data.groupby(frame.week_no).transform("mean")
        + data.mean()
    )
    xt, yt = transformed[HISTORY].to_numpy(), transformed[outcome_column].to_numpy()
    denominator = float(xt @ xt)
    require(denominator > 1e-12, "independent Run D projection has absorbed history")
    return xt, yt, float(xt @ yt / denominator)


def _group_scores(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Independent NumPy sums, including household/week intersection groups."""
    _, codes = np.unique(groups, return_inverse=True, axis=0)
    totals = np.zeros((int(codes.max()) + 1, scores.shape[1]))
    np.add.at(totals, codes, scores)
    return totals


def _manual_covariance(
    frame: pd.DataFrame, xt: np.ndarray, yt: np.ndarray, beta: float, kind: str
) -> np.ndarray:
    """Native normalization, rebuilt without native params/residuals/covariance.

    The restored-grand-mean design matters for the normalized intercept. Use
    e = y_tilde - beta*x_tilde; bread = inv(Z'Z). Each cluster correction is
    g/(g-1)*(n-1)/n and the overall FE/debias correction is n/df_resid.
    DK's time normalizations cancel, leaving the same overall correction.
    """
    require(kind in COVARIANCE_LABELS, "undeclared Run D covariance")
    n = len(frame)
    entity, time = frame.household_key.to_numpy(), frame.week_no.to_numpy()
    h, t = frame.household_key.nunique(), frame.week_no.nunique()
    df_resid = n - h - t  # 2 regressors plus (h-1)+(t-1) absorbed effects.
    require(df_resid > 0, "nonpositive Run D residual degrees of freedom")
    z = np.column_stack([np.ones(n), xt + frame[HISTORY].mean()])
    residual = yt - beta * xt
    scores = z * residual[:, None]
    bread = np.linalg.inv(z.T @ z)

    def corrected(groups):
        totals = _group_scores(scores, groups)
        g = len(totals)
        require(g > 1, "cluster correction requires at least two groups")
        return (g / (g - 1) * (n - 1) / n) * (totals.T @ totals)

    if kind in ("two_way", "household"):
        meat = corrected(entity)
        if kind == "two_way":
            meat += corrected(time) - corrected(np.column_stack([entity, time]))
    else:
        bandwidth = 4 if kind == "dk4" else 8
        require(t > bandwidth, "declared DK bandwidth requires a longer response calendar")
        totals = _group_scores(scores, time)
        meat = totals.T @ totals
        for lag in range(1, bandwidth + 1):
            cross = totals[lag:].T @ totals[:-lag]
            meat += (1 - lag / (bandwidth + 1)) * (cross + cross.T)
    covariance = (n / df_resid) * bread @ meat @ bread
    return (covariance + covariance.T) / 2


def _fit_job(frame: pd.DataFrame, job: SensitivityJob) -> tuple[object, dict]:
    """Bounded estimator helper; synthetic tests may supply complete toy panels."""
    require(job in JOBS, "job is outside the closed Run D register")
    require(not frame[["household_key", "week_no"]].isna().any().any(), "null Run D model key")
    require(not frame.duplicated(["household_key", "week_no"]).any(), "duplicate Run D model row")
    households, weeks = frame.household_key.nunique(), frame.week_no.nunique()
    require(len(frame) == households * weeks, "Run D model calendar must be balanced")
    require(
        frame[["household_key", "week_no"]].equals(
            frame.sort_values(["household_key", "week_no"])[["household_key", "week_no"]]
        ),
        "Run D model rows must be household/week ordered",
    )
    first, last = int(frame.week_no.min()), int(frame.week_no.max())
    require(
        tuple(sorted(frame.week_no.unique())) == tuple(range(first, last + 1)),
        "nonconsecutive Run D calendar",
    )
    require(frame[HISTORY].isin([0, 1]).all(), "Run D history must be binary and available")
    require(np.isfinite(frame[job.outcome]).all(), "Run D outcome contains unavailable values")
    if job.outcome == "retailer_purchase":
        require(frame[job.outcome].isin([0, 1]).all(), "Run D participation must be binary")

    # Keep the original revenue untouched. Both native and independent paths use
    # this explicitly assigned local transformed response, not the levels field.
    local = frame
    response = job.outcome
    if job.variant == "log1p":
        require(
            (frame.weekly_sales_value > -1).all(),
            "Run D log1p outcome outside domain; no rescue transformation",
        )
        local = frame.copy()
        local[LOG_COLUMN] = np.log1p(frame.weekly_sales_value)
        response = LOG_COLUMN
    indexed = local.set_index(["household_key", "week_no"])
    x = pd.DataFrame(
        {"constant": 1.0, HISTORY: indexed[HISTORY].astype(float)}, index=indexed.index
    )
    model = PanelOLS(
        indexed[response].astype(float),
        x,
        entity_effects=True,
        time_effects=True,
        singletons=True,
        drop_absorbed=False,
        check_rank=True,
    )
    options = covariance_options(job.covariance, effects=True)
    fitted = model.fit(**options)
    require(
        fitted.nobs == len(frame) and fitted.resids.index.equals(indexed.index),
        "Run D estimator changed observation rows/order",
    )
    require(fitted.model.exog.vars == ["constant", HISTORY], "unexpected Run D regressor")
    require(np.all(fitted.model.weights.values2d == 1), "unexpected Run D weights")
    require(
        fitted.df_resid == len(frame) - households - weeks,
        "Run D effect degrees of freedom changed",
    )
    require(
        fitted._cov_type
        == ("Clustered" if job.covariance in ("two_way", "household") else "Driscoll-Kraay"),
        "wrong Run D covariance object",
    )
    boundary = covariance_boundary(fitted.cov)
    beta = float(fitted.params[HISTORY])
    xt, yt, independent = _independent_twfe(local, response)
    require(_close(beta, independent), "Run D independent coefficient reconciliation failed")
    manual = float(_manual_covariance(local, xt, yt, independent, job.covariance)[1, 1])
    require(
        np.isfinite(manual) and manual > 0,
        "Run D manual focal variance must be finite and strictly positive",
    )
    require(
        _close(boundary["focal_variance"], manual, manual=True),
        "Run D independent focal variance reconciliation failed",
    )
    interval = fitted.conf_int(level=0.95).loc[HISTORY]
    spec = SAMPLES[job.sample_index]
    row = {
        "outcome": job.outcome,
        "model_id": job.model_id,
        "check_type": job.check_type,
        "variant": job.variant,
        "sample": spec.name,
        "population": spec.population,
        "response_first": first,
        "response_last": last,
        "response_weeks": weeks,
        "households": households,
        "nobs": int(fitted.nobs),
        "transformation": job.transformation,
        "covariance": job.covariance,
        "covariance_type": COVARIANCE_LABELS[job.covariance],
        "coefficient": beta,
        "standard_error": float(fitted.std_errors[HISTORY]),
        "ci_lower": float(interval.iloc[0]),
        "ci_upper": float(interval.iloc[1]),
        "t_statistic": float(fitted.tstats[HISTORY]),
        "p_value": float(fitted.pvalues[HISTORY]),
        "df_resid": int(fitted.df_resid),
        "reference_distribution": "native Student t(df_resid)",
        "household_fe": True,
        "week_fe": True,
        "household_clusters": households if job.covariance in ("two_way", "household") else None,
        "week_clusters": weeks if job.covariance == "two_way" else None,
        "debiased": options["debiased"],
        "group_debias": options.get("group_debias"),
        "auto_df": options["auto_df"],
        "count_effects": options["count_effects"],
        "kernel": options.get("kernel"),
        "bandwidth": options.get("bandwidth"),
        **boundary,
        "independent_coefficient": independent,
        "independent_abs_difference": abs(beta - independent),
        "independent_abs_tolerance": ABS_TOL,
        "independent_rel_tolerance": REL_TOL,
        "manual_focal_variance": manual,
        "manual_variance_abs_difference": abs(boundary["focal_variance"] - manual),
        "manual_variance_abs_tolerance": MANUAL_ABS_TOL,
        "manual_variance_rel_tolerance": MANUAL_REL_TOL,
        "status": "UNREVIEWED",
        "row_key_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(indexed.index, index=False).values.tobytes()
        ).hexdigest(),
    }
    require(
        np.isfinite(
            [
                row[k]
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
        "nonfinite Run D scalar inference",
    )
    require(
        row["standard_error"] > 0 and row["ci_upper"] > row["ci_lower"],
        "nonpositive Run D uncertainty width",
    )
    return fitted, row


_REFERENCE_NUMERIC = (
    "standard_error",
    "ci_lower",
    "ci_upper",
    "t_statistic",
    "p_value",
    "focal_variance",
    "covariance_min_eigenvalue",
    "covariance_psd_tolerance",
)
_REFERENCE_COUNTS = (
    "nobs",
    "households",
    "response_weeks",
    "household_clusters",
    "week_clusters",
    "df_resid",
)
_REFERENCE_FLAGS = (
    "household_fe",
    "week_fe",
    "debiased",
    "group_debias",
    "auto_df",
    "count_effects",
    "full_covariance_psd",
)


def _reference_flag(value) -> bool:
    require(value in (True, False, "True", "False"), "invalid archived Run C boolean")
    return value is True or value == "True"


def _verify_baseline(row: dict, archived: dict) -> None:
    """Exact slope and row identity are gates, not tolerance-based reassurance."""
    require(
        row["coefficient"] == float(archived["coefficient"]),
        "Run C baseline coefficient drift: STOP before sensitivities",
    )
    require(
        row["row_key_sha256"] == archived["row_key_sha256"],
        "Run C baseline observation keys drift: STOP",
    )
    for key in _REFERENCE_NUMERIC:
        require(_close(row[key], float(archived[key])), f"Run C baseline {key} drift: STOP")
    for key in _REFERENCE_COUNTS:
        require(row[key] == float(archived[key]), f"Run C baseline {key} drift: STOP")
    for key in _REFERENCE_FLAGS:
        require(row[key] == _reference_flag(archived[key]), f"Run C baseline {key} drift: STOP")
    for key in ("sample", "status", "reference_distribution", "covariance_type"):
        require(row[key] == archived[key], f"Run C baseline {key} drift: STOP")


def _add_comparison(row: dict, baseline: dict) -> None:
    transformed = row["variant"] == "log1p"
    change = row["coefficient"] - baseline["coefficient"]
    row.update(
        reference_baseline_coefficient=baseline["coefficient"],
        reference_baseline_standard_error=baseline["standard_error"],
        reference_baseline_ci_lower=baseline["ci_lower"],
        reference_baseline_ci_upper=baseline["ci_upper"],
        absolute_coefficient_change=None if transformed else abs(change),
        signed_coefficient_change=None if transformed else change,
        relative_percent_change=None
        if transformed or abs(baseline["coefficient"]) <= ABS_TOL
        else 100 * change / baseline["coefficient"],
        # Levels/log uncertainty magnitudes have different units as well.
        se_ratio=None if transformed else row["standard_error"] / baseline["standard_error"],
        ci_width_ratio=None
        if transformed
        else (row["ci_upper"] - row["ci_lower"]) / (baseline["ci_upper"] - baseline["ci_lower"]),
        sign_stable=bool(np.sign(row["coefficient"]) == np.sign(baseline["coefficient"])),
    )
    notes = {
        "baseline": "Reproduced Run C descriptive benchmark; scalar native inference only; no joint tests or covariance repair.",
        "population": "Predeclared activity-selected population changes the descriptive association; not bias correction or a preferred population.",
        "calendar": "Predeclared common-calendar sensitivity; unchanged history and outcomes, not verified enrollment or a clean start.",
        "functional_form": "Qualitative sign and precision comparison only: log(1 + retailer SALES_VALUE) changes units and estimand; not elasticity or percent revenue effect. No levels/log magnitude or uncertainty-width ratio.",
        "covariance": "Same baseline rows and TWFE point specification; covariance changes uncertainty only. Household and two-way clustering and DK protect against different dependence structures/asymptotics; none is automatically correct.",
    }
    row["interpretation_note"] = notes[row["check_type"]]


def _matrix_twfe(values: np.ndarray) -> np.ndarray:
    """Diagnostic matrix route, independent of grouped-mean point auditing."""
    return (
        values
        - values.mean(axis=1, keepdims=True)
        - values.mean(axis=0, keepdims=True)
        + values.mean()
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    # Same global-centering definition as Run C core_diagnostics.correlation.
    a, b = a.ravel(), b.ravel()
    da, db = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((da @ da) * (db @ db)))
    return float(da @ db / denominator) if denominator > 1e-12 else None


def _dependence_confirmation(frame: pd.DataFrame, fits: dict) -> dict:
    """Aggregate only: no identifiers, deletion rule, refit, or bandwidth search."""
    shape = (frame.household_key.nunique(), frame.week_no.nunique())
    xt = _matrix_twfe(frame[HISTORY].to_numpy(dtype=float).reshape(shape))
    indexed = frame.set_index(["household_key", "week_no"])
    result = {}
    for outcome in OUTCOMES:
        fitted = fits[outcome]
        require(fitted.resids.index.equals(indexed.index), "Run D dependence/model row mismatch")
        residual = fitted.resids.to_numpy(dtype=float).reshape(shape)
        score = (xt * residual).sum(axis=0)
        result[outcome] = {
            "residual_adjacent_pooled_correlation": _pearson(residual[:, :-1], residual[:, 1:]),
            **{
                f"aggregate_score_correlation_lag{lag}": _pearson(score[lag:], score[:-lag])
                if shape[1] > lag
                else None
                for lag in (1, 4, 8)
            },
        }
    return result


def run_sensitivities(mart: pd.DataFrame, baseline_rows: list[dict]) -> tuple[list[dict], dict]:
    """Return exactly 25 aggregate rows and three baseline dependence summaries.

    Callers own mart/archive integrity and output policy. All three new baseline
    fits must pass exact archived-coefficient and row-key gates before any other
    job is fitted. Numeric archive fields may be CSV strings. No I/O occurs here.
    """
    require(
        len(baseline_rows) == 9
        and {(r["outcome"], r["model_id"]) for r in baseline_rows} == set(CORE_KEYS),
        "Run D requires the nine unique validated Run C rows",
    )
    archived = {r["outcome"]: r for r in baseline_rows if r["model_id"] == "M3"}
    baseline_frame = construct_sample(mart, SAMPLES[0])
    rows, fits, baselines = [], {}, {}
    for job in JOBS[:3]:
        fitted, row = _fit_job(baseline_frame, job)
        _verify_baseline(row, archived[job.outcome])
        fits[job.outcome] = fitted
        baselines[job.outcome] = row
        _add_comparison(row, row)
        rows.append(row)
    for job in JOBS[3:]:
        frame = (
            baseline_frame
            if job.sample_index == 0
            else construct_sample(mart, SAMPLES[job.sample_index])
        )
        _, row = _fit_job(frame, job)
        baseline = baselines[job.outcome]
        if job.check_type == "covariance":
            require(
                row["row_key_sha256"] == baseline["row_key_sha256"],
                "Run D covariance-only observation keys changed",
            )
            require(
                _close(row["coefficient"], baseline["coefficient"]),
                "Run D covariance-only point estimate changed: STOP",
            )
        if job.variant == "log1p":
            require(
                row["row_key_sha256"] == baseline["row_key_sha256"],
                "Run D transformation changed observation keys",
            )
        _add_comparison(row, baseline)
        rows.append(row)
    require(
        len(rows) == 25
        and {(r["outcome"], r["variant"]) for r in rows} == {(j.outcome, j.variant) for j in JOBS},
        "Run D did not produce the closed 25-row matrix",
    )
    require(len({tuple(r) for r in rows}) == 1, "Run D rows have inconsistent schemas")
    return rows, _dependence_confirmation(baseline_frame, fits)
