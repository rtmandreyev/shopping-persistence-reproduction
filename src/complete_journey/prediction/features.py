"""Point-in-time feature tiers (P7-003).

* Tier 0: order-free own-history statistics from the first recorded purchase f_i
  through origin t. They are functions of the count k and tenure n only.
* Tier 1: Tier 0 plus bounded recency (y_t, recent-4 share, capped weeks since last).
* Tier 2: Tier 1 plus the audited stored feature-store allowlist. It is real-time
  population P only and real data only; it is never simulated.

Every value at origin t uses weeks <= t. A row whose household has no recorded purchase
at or before t is refused (that would be the C1 pre-entry population).
"""

from __future__ import annotations

import csv
from functools import lru_cache

import numpy as np
import pandas as pd

from complete_journey.paths import PROJECT_ROOT
from complete_journey.prediction import config
from complete_journey.prediction.operators import ROW_COLUMNS
from complete_journey.prediction.panel import (
    ORIGIN_COLUMN,
    CanonicalPanel,
    outcome_matrix,
    require,
)


def _history_arrays(panel: CanonicalPanel):
    keys, matrix = outcome_matrix(panel.recorded)
    weeks = np.arange(config.FIRST_WEEK, config.LAST_WEEK + 1)
    csum = np.cumsum(matrix, axis=1)
    # Point-in-time "last purchase at or before t" (week number; 0 if none yet).
    last = np.maximum.accumulate(np.where(matrix == 1, weeks, 0), axis=1)
    has_any = matrix.any(axis=1)
    first = np.where(has_any, matrix.argmax(axis=1) + config.FIRST_WEEK, 0)
    return keys, matrix, csum, last, first


def build_features(panel: CanonicalPanel, rows: pd.DataFrame, tier: int) -> pd.DataFrame:
    """Return rows with Tier 0 (tier=0) or Tier 1 (tier=1) features at each origin."""
    require(tier in (0, 1), "build_features computes Tier 0/1 only; Tier 2 is attach_tier2")
    require(tuple(rows.columns[:4]) == ROW_COLUMNS, "rows must be operator/population rows")
    keys, matrix, csum, last, first = _history_arrays(panel)
    position = pd.Series(np.arange(len(keys)), index=keys)
    i = position.loc[rows["household_key"]].to_numpy()
    t = rows["origin_week"].to_numpy()
    tj = t - config.FIRST_WEEK
    f = first[i]
    require(((f >= 1) & (f <= t)).all(), "feature row before the household's first purchase")
    n = (t - f + 1).astype(float)
    k = csum[i, tj].astype(float)  # no purchases exist before f
    rate = (k + 1.0) / (n + 2.0)
    out = rows.copy()
    out[ORIGIN_COLUMN] = panel.origin  # data-level origin stamp for the fitting guard
    out["T0_tenure_log"] = np.log1p(n)
    out["T0_rate_logit"] = np.log(rate / (1.0 - rate))
    out["T0_rate_x_tenure"] = out["T0_rate_logit"] * out["T0_tenure_log"]
    if tier == 1:
        out["T1_y_t"] = matrix[i, tj].astype(float)
        start = np.maximum(f, t - config.RECENT_WINDOW + 1)
        # purchases in [start, t] = csum(t) - csum(start - 1); csum(week 0) = 0
        before = np.where(start >= 2, csum[i, np.maximum(start - 2, 0)], 0)
        window = (t - start + 1).astype(float)
        out["T1_recent4_share"] = (csum[i, tj] - before) / window
        since = np.minimum(t - last[i, tj], config.WEEKS_SINCE_CAP)
        out["T1_weeks_since_last"] = np.log1p(since.astype(float))
    return out


# --- Tier 2 allowlist ---------------------------------------------------------------


@lru_cache(maxsize=1)
def inventory() -> tuple[dict, ...]:
    with (PROJECT_ROOT / config.FEATURE_INVENTORY).open(newline="") as handle:
        return tuple(csv.DictReader(handle))


def stored_tier2_allowlist() -> tuple[str, ...]:
    """Stored feature-store columns admitted to Tier 2 (explicit allowlist, not patterns)."""
    return tuple(
        r["name"]
        for r in inventory()
        if r["tier2"] == "yes" and not r["name"].startswith(("T0_", "T1_"))
    )


def validate_store_columns(columns) -> None:
    """Every feature-store column must be classified; unknown columns are never admitted."""
    known = {r["name"] for r in inventory()}
    unknown = sorted(set(columns) - known)
    require(not unknown, f"unclassified feature-store columns (not admitted): {unknown}")


def attach_tier2(rows: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Join allowlisted stored features at as_of_week = origin week (population P only)."""
    validate_store_columns(store.columns)
    allow = list(stored_tier2_allowlist())
    require({"household_key", "as_of_week"} <= set(store.columns), "feature store needs keys")
    picked = store[["household_key", "as_of_week", *allow]].rename(
        columns={"as_of_week": "origin_week"}
    )
    out = rows.merge(picked, on=["household_key", "origin_week"], how="left", validate="1:1")
    require(len(out) == len(rows), "Tier 2 join changed row count")
    return out


def tier2_columns() -> tuple[str, ...]:
    return (*config.TIER1, *stored_tier2_allowlist())


def truncation_invariant(panel: CanonicalPanel, rows: pd.DataFrame, cutoff: int, seed: int) -> bool:
    """Features at origins <= cutoff must not change when weeks > cutoff are scrambled."""
    rng = np.random.default_rng(seed)
    scrambled = panel.recorded.copy()
    future = scrambled["week_no"] > cutoff
    scrambled.loc[future, config.TARGET_SOURCE] = rng.integers(0, 2, future.sum()).astype(np.int8)
    other = CanonicalPanel(scrambled, origin=panel.origin, label="truncation probe")
    early = rows.loc[rows["origin_week"] <= cutoff]
    a = build_features(panel, early, tier=1)
    b = build_features(other, early, tier=1)
    feature_cols = list(config.TIER1)
    return bool(np.array_equal(a[feature_cols].to_numpy(), b[feature_cols].to_numpy()))
