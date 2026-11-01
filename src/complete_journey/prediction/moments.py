"""Moment engine for the Phase 7/8 register (docs/MOMENT_REGISTER.md).

The same functions run on real and simulated ``CanonicalPanel`` objects. Run B
exercises them on synthetic fixtures and on already-published Phase 3/4 aggregates
only. Outputs are aggregate records (``schema``) with no household identifiers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from complete_journey.prediction import config, twfe
from complete_journey.prediction.operators import interior_windows, offline_rows
from complete_journey.prediction.panel import CanonicalPanel, outcome_matrix, require

# --- schema -------------------------------------------------------------------------

MOMENT_FIELDS = (
    "moment_id",
    "component",
    "estimate",
    "unit",
    "population",
    "operator",
    "sample_n",
    "households",
    "uncertainty_low",
    "uncertainty_high",
    "uncertainty_method",
    "source",
    "simulation_architecture",
    "simulation_replication",
    "seed",
    "status",
)
FORBIDDEN_FIELDS = ("household_key", "BASKET_ID", "PRODUCT_ID", "STORE_ID")


def record(
    moment_id,
    component,
    estimate,
    unit,
    operator,
    sample_n,
    households,
    panel,
    population="household-week rows",
    status="computed",
) -> dict:
    return {
        "moment_id": moment_id,
        "component": component,
        "estimate": None if estimate is None else float(estimate),
        "unit": unit,
        "population": population,
        "operator": operator,
        "sample_n": int(sample_n),
        "households": int(households),
        "uncertainty_low": None,
        "uncertainty_high": None,
        "uncertainty_method": None,
        "source": panel.origin,
        "simulation_architecture": None,
        "simulation_replication": None,
        "seed": None,
        "status": status,
    }


def validate_records(records: list[dict]) -> list[dict]:
    for r in records:
        require(tuple(r) == MOMENT_FIELDS, "moment record schema mismatch")
        require(not any(f in r for f in FORBIDDEN_FIELDS), "identifier in moment record")
    return records


# --- window helpers -----------------------------------------------------------------


def window_frame(panel: CanonicalPanel, operator: str) -> pd.DataFrame:
    """Household-week cells inside each household's comparison window."""
    frame = panel.recorded
    if operator == config.OPERATOR_OFFLINE:
        w = interior_windows(panel)
        frame = frame.join(w, on="household_key")
        keep = frame["eligible"] & frame["week_no"].between(frame["w_start"], frame["w_end"])
        return frame.loc[keep, ["household_key", "week_no", config.TARGET_SOURCE]]
    lo, hi = (
        config.SENSITIVITY_RESPONSES
        if operator == config.OPERATOR_SENSITIVITY
        else config.RESPONSE_WEEKS
    )
    return frame.loc[
        frame["week_no"].between(lo, hi), ["household_key", "week_no", config.TARGET_SOURCE]
    ]


def household_rates(panel: CanonicalPanel, operator: str) -> pd.Series:
    cells = window_frame(panel, operator)
    return cells.groupby("household_key")[config.TARGET_SOURCE].mean()


def with_lags(panel: CanonicalPanel, operator: str, lags) -> pd.DataFrame:
    """Operator rows on the common sample for the largest lag, with y_{r-k} columns."""
    rows = offline_rows(panel, operator, lag=max(lags)).rows
    keys, matrix = outcome_matrix(panel.recorded)
    pos = pd.Series(np.arange(len(keys)), index=keys).loc[rows["household_key"]].to_numpy()
    r = rows["response_week"].to_numpy() - config.FIRST_WEEK
    out = rows.rename(columns={"response_week": "week_no"}).copy()
    for k in lags:
        out[f"lag{k}"] = matrix[pos, r - k].astype(float)
    return out


# --- calibration and consistency ----------------------------------------------------


def ci1(panel: CanonicalPanel, operator: str = config.OPERATOR_OFFLINE) -> list[dict]:
    rates = household_rates(panel, operator)
    out = [
        record(
            "CI-1",
            "mean",
            rates.mean(),
            "share of weeks",
            operator,
            len(rates),
            len(rates),
            panel,
            "households",
        )
    ]
    for q in config.CI1_QUANTILES:
        out.append(
            record(
                "CI-1",
                f"q{int(q * 100):02d}",
                rates.quantile(q),
                "share of weeks",
                operator,
                len(rates),
                len(rates),
                panel,
                "households",
            )
        )
    return out


def k1_between_share(panel: CanonicalPanel, operator: str = config.OPERATOR_OFFLINE) -> list[dict]:
    cells = window_frame(panel, operator)
    y = cells[config.TARGET_SOURCE].astype(float)
    means = y.groupby(cells["household_key"]).transform("mean")
    total = float(((y - y.mean()) ** 2).sum())
    between = float(((means - y.mean()) ** 2).sum())
    share = between / total if total > 0 else None
    return [
        record(
            "K-1",
            "between_share",
            share,
            "share of SS",
            operator,
            len(cells),
            cells["household_key"].nunique(),
            panel,
        )
    ]


def k2_concentration(panel: CanonicalPanel, operator: str = config.OPERATOR_OFFLINE) -> list[dict]:
    rates = np.sort(household_rates(panel, operator).to_numpy())
    n = len(rates)
    gini = (
        (2 * np.sum(np.arange(1, n + 1) * rates) / (n * rates.sum()) - (n + 1) / n)
        if rates.sum() > 0
        else None
    )
    top = int(np.ceil(0.10 * n))
    top_share = rates[-top:].sum() / rates.sum() if rates.sum() > 0 else None
    return [
        record("K-2", "gini", gini, "index", operator, n, n, panel, "households"),
        record("K-2", "top10_share", top_share, "share", operator, n, n, panel, "households"),
    ]


def e1_week_path(panel: CanonicalPanel, operator: str = config.OPERATOR_OFFLINE) -> pd.Series:
    rows = offline_rows(panel, operator).rows
    return rows.groupby("response_week")["y"].mean()


def e1_records(panel: CanonicalPanel, operator: str = config.OPERATOR_OFFLINE) -> list[dict]:
    """ENVIRONMENT: weekly prevalence on operator rows (target for the common week path)."""
    rows = offline_rows(panel, operator).rows
    grouped = rows.groupby("response_week")
    return validate_records(
        [
            record(
                "E-1",
                f"week_{int(week):03d}",
                g["y"].mean(),
                "probability",
                operator,
                len(g),
                g["household_key"].nunique(),
                panel,
                status="environment",
            )
            for week, g in grouped
        ]
    )


# --- validation ---------------------------------------------------------------------


def vi1_transitions(panel: CanonicalPanel, operator: str = config.OPERATOR_OFFLINE) -> list[dict]:
    frame = with_lags(panel, operator, (1,))
    y, lag = frame["y"].to_numpy(float), frame["lag1"].to_numpy(float)
    require((lag == 1).any() and (lag == 0).any(), "transitions need both lag states")
    p11, p01 = y[lag == 1].mean(), y[lag == 0].mean()
    n, h = len(frame), frame["household_key"].nunique()
    return [
        record("VI-1", "P11", p11, "probability", operator, n, h, panel),
        record("VI-1", "P01", p01, "probability", operator, n, h, panel),
        record("VI-1", "GAP", p11 - p01, "probability points", operator, n, h, panel),
    ]


def vi2_ladder(panel: CanonicalPanel, operator: str = config.OPERATOR_OFFLINE) -> list[dict]:
    frame = with_lags(panel, operator, (1,))
    n, h = len(frame), frame["household_key"].nunique()
    out, values = [], {}
    for model in twfe.MODELS:
        values[model] = twfe.fe_coefficients(frame, "y", ["lag1"], model)["lag1"]
        out.append(
            record("VI-2", f"O-{model}", values[model], "probability points", operator, n, h, panel)
        )
    ratio = (values["M0"] - values["M2"]) / values["M0"] if values["M0"] else None
    out.append(
        record("VI-2", "M0_to_M2_reduction", ratio, "share (descriptive)", operator, n, h, panel)
    )
    return out


def vi3_lag_profile(panel: CanonicalPanel, operator: str = config.OPERATOR_OFFLINE) -> list[dict]:
    frame = with_lags(panel, operator, config.LAG_SET)
    n, h = len(frame), frame["household_key"].nunique()
    out = []
    for k in config.LAG_SET:
        beta = twfe.fe_coefficients(frame, "y", [f"lag{k}"], "M3")[f"lag{k}"]
        out.append(record("VI-3", f"beta_{k}", beta, "probability points", operator, n, h, panel))
    gamma = twfe.fe_coefficients(frame, "y", ["lag1", "lag2"], "M3")["lag2"]
    out.append(record("VI-3", "gamma_2", gamma, "probability points", operator, n, h, panel))
    return out


def spells(values: np.ndarray) -> list[tuple[int, int, bool]]:
    """(state, length, touches_boundary) for maximal runs of a 0/1 sequence."""
    out, start = [], 0
    for j in range(1, len(values) + 1):
        if j == len(values) or values[j] != values[start]:
            out.append((int(values[start]), j - start, start == 0 or j == len(values)))
            start = j
    return out


def vi4_spells(panel: CanonicalPanel, operator: str = config.OPERATOR_OFFLINE) -> list[dict]:
    cells = window_frame(panel, operator).sort_values(["household_key", "week_no"])
    counts = {0: np.zeros(len(config.SPELL_BINS)), 1: np.zeros(len(config.SPELL_BINS))}
    for _, group in cells.groupby("household_key", sort=False):
        for state, length, boundary in spells(group[config.TARGET_SOURCE].to_numpy()):
            if boundary:
                continue  # censored by the window: excluded
            for b, (lo, hi) in enumerate(config.SPELL_BINS):
                if length >= lo and (hi is None or length <= hi):
                    counts[state][b] += 1
    out = []
    for state, name in ((1, "active"), (0, "inactive")):
        total = counts[state].sum()
        for b, (lo, hi) in enumerate(config.SPELL_BINS):
            label = f"{lo}" if lo == hi else (f"{lo}+" if hi is None else f"{lo}-{hi}")
            share = counts[state][b] / total if total else None
            out.append(
                record(
                    "VI-4",
                    f"{name}_{label}",
                    share,
                    "share of interior spells",
                    operator,
                    int(total),
                    cells["household_key"].nunique(),
                    panel,
                    "spells",
                )
            )
    return out


def vi5_half_to_half(panel: CanonicalPanel, operator: str = config.OPERATOR_OFFLINE) -> list[dict]:
    """SECONDARY (stationarity check): Spearman of first- vs second-half window rates."""
    cells = window_frame(panel, operator).sort_values(["household_key", "week_no"])
    first, second = [], []
    for _, group in cells.groupby("household_key", sort=False):
        y = group[config.TARGET_SOURCE].to_numpy(float)
        if len(y) >= config.HALF_MIN_WEEKS:
            mid = len(y) // 2
            first.append(y[:mid].mean())
            second.append(y[mid:].mean())
    rho = spearmanr(first, second).statistic if len(first) > 2 else None
    return [
        record(
            "VI-5",
            "spearman",
            rho,
            "rank correlation",
            operator,
            len(first),
            len(first),
            panel,
            "households",
            status="secondary",
        )
    ]


def moment_vector(panel: CanonicalPanel, operator: str = config.OPERATOR_OFFLINE) -> list[dict]:
    """All register moments except VI-6 (which needs model fitting; see protocol)."""
    records = []
    for fn in (
        ci1,
        k1_between_share,
        k2_concentration,
        vi1_transitions,
        vi2_ladder,
        vi3_lag_profile,
        vi4_spells,
        vi5_half_to_half,
    ):
        records.extend(fn(panel, operator))
    return validate_records(records)


# --- uncertainty and shuffle --------------------------------------------------------


def household_bootstrap(panel: CanonicalPanel, rng: np.random.Generator) -> CanonicalPanel:
    """Resample whole household trajectories with replacement; duplicates get new keys."""
    frame = panel.recorded
    keys = frame["household_key"].unique()
    draw = rng.choice(keys, size=len(keys), replace=True)
    base = config.SYNTHETIC_KEY_OFFSET if panel.origin == "synthetic" else 0
    groups = {k: g for k, g in frame.groupby("household_key", sort=False)}
    parts = []
    for j, key in enumerate(draw):
        part = groups[key].copy()
        part["household_key"] = base + j + 1
        parts.append(part)
    return CanonicalPanel(
        pd.concat(parts, ignore_index=True),
        origin=panel.origin,
        label=f"bootstrap of {panel.label}",
    )


def shuffle_check(
    panel: CanonicalPanel,
    seed: int,
    operator: str = config.OPERATOR_OFFLINE,
    replications: int = config.SHUFFLE_REPLICATIONS,
) -> dict:
    """D-4: calibration moments must be identical across shuffles; report dynamic shifts."""
    rng = np.random.default_rng(seed)
    calib_fns = (ci1, k1_between_share, k2_concentration)
    base_calib = [r["estimate"] for fn in calib_fns for r in fn(panel, operator)]
    base_gap = next(
        r["estimate"] for r in vi1_transitions(panel, operator) if r["component"] == "GAP"
    )
    shifts = []
    for _ in range(replications):
        shuffled = within_window_shuffle(panel, rng, operator)
        calib = [r["estimate"] for fn in calib_fns for r in fn(shuffled, operator)]
        # Identical up to floating summation order (row order changes inside windows).
        require(
            np.allclose(calib, base_calib, rtol=0.0, atol=config.SHUFFLE_ATOL),
            "calibration moment changed under within-window shuffle",
        )
        gap = next(
            r["estimate"] for r in vi1_transitions(shuffled, operator) if r["component"] == "GAP"
        )
        shifts.append(gap - base_gap)
    return {
        "replications": replications,
        "calibration_identical": True,
        "mean_gap_shift": float(np.mean(shifts)),
    }


def within_window_shuffle(
    panel: CanonicalPanel, rng: np.random.Generator, operator: str = config.OPERATOR_OFFLINE
) -> CanonicalPanel:
    """Permute each household's outcomes among the weeks of its comparison window.

    Preserved: household, window, the multiset of outcomes in the window, and every
    week outside the window (including the anchored endpoints for operator O).
    Destroyed: within-household week order and alignment with calendar weeks.
    """
    frame = panel.recorded.copy()
    inside = window_frame(panel, operator).index
    target = frame.columns.get_loc(config.TARGET_SOURCE)
    for _, idx in frame.loc[inside].groupby("household_key").groups.items():
        values = frame.loc[idx, config.TARGET_SOURCE].to_numpy()
        frame.iloc[frame.index.get_indexer(idx), target] = rng.permutation(values)
    return CanonicalPanel(frame, origin=panel.origin, label=f"shuffle of {panel.label}")
