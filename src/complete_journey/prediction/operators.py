"""Observation operators and the real-time predictive population (P7-002 / P7-003).

Two different objects, deliberately different types:

* ``OfflineComparisonRows`` — operators O / S / G for the **offline ABM comparison**.
  O uses each household's first-to-last recorded purchase (future information) and is
  legitimate only as a symmetric operator applied identically to real and simulated
  panels. It is never a predictive population.
* ``RealtimePredictionRows`` — population P for the **real-time benchmark**. Each row
  depends only on information through its origin week (a recorded purchase at or
  before t). It never sees last purchase, future tenure, future activity or sample flags.

Simulated observation (``record_observation``) is applied **after** behaviour is
generated. It masks recording outside a window and anchors the recorded endpoints. It
never changes the behaviour panel, and no agent decision can depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from complete_journey.prediction import config
from complete_journey.prediction.panel import (
    CanonicalPanel,
    household_support,
    outcome_matrix,
    require,
)

ROW_COLUMNS = ("household_key", "origin_week", "response_week", "y")


@dataclass(frozen=True)
class OfflineComparisonRows:
    """Response rows of an offline comparison operator (may use future support)."""

    operator: str
    rows: pd.DataFrame
    uses_future_support: bool


@dataclass(frozen=True)
class RealtimePredictionRows:
    """Point-in-time prediction rows; contains no future-derived column by construction."""

    population: str
    rows: pd.DataFrame


def interior_windows(panel: CanonicalPanel) -> pd.DataFrame:
    """W_i = [f_i+1, min(l_i-1, 101)] and eligibility |W_i| >= MIN_INTERIOR_WEEKS."""
    support = household_support(panel.recorded)
    start = support["f"] + 1
    end = np.minimum(support["l"] - 1, config.RESPONSE_WEEKS[1])
    length = (end - start + 1).clip(lower=0)
    out = pd.DataFrame({"w_start": start, "w_end": end, "w_len": length})
    out["eligible"] = out["w_len"].fillna(0) >= config.MIN_INTERIOR_WEEKS
    return out


def _pairs(panel: CanonicalPanel) -> pd.DataFrame:
    frame = panel.recorded.rename(columns={"week_no": "response_week", config.TARGET_SOURCE: "y"})
    frame = frame[["household_key", "response_week", "y"]].copy()
    frame["origin_week"] = frame["response_week"] - 1
    lo, hi = config.RESPONSE_WEEKS
    frame = frame.loc[frame["response_week"].between(lo, hi)]
    return frame[list(ROW_COLUMNS)].reset_index(drop=True)


def offline_rows(panel: CanonicalPanel, operator: str, lag: int = 1) -> OfflineComparisonRows:
    """Rows (r, r-lag) admitted by an offline operator."""
    require(operator in config.OPERATORS, f"unknown offline operator: {operator}")
    require(lag >= 1, "lag must be positive")
    rows = _pairs(panel)
    if operator == config.OPERATOR_OFFLINE:
        windows = interior_windows(panel)
        rows = rows.join(windows, on="household_key")
        keep = (
            rows["eligible"]
            & (rows["response_week"] <= rows["w_end"])
            & (rows["response_week"] - lag >= rows["w_start"])
        )
        rows = rows.loc[keep, list(ROW_COLUMNS)]
        future = True
    elif operator == config.OPERATOR_SENSITIVITY:
        lo, hi = config.SENSITIVITY_RESPONSES
        rows = rows.loc[rows["response_week"].between(lo, hi) & (rows["response_week"] - lag >= 1)]
        future = False
    else:
        rows = rows.loc[rows["response_week"] - lag >= 1]
        future = False
    return OfflineComparisonRows(operator, rows.reset_index(drop=True), future)


def realtime_rows(panel: CanonicalPanel) -> RealtimePredictionRows:
    """Population P: origin t has a recorded purchase at or before t; response t+1 <= 101."""
    keys, matrix = outcome_matrix(panel.recorded)
    ever = np.maximum.accumulate(matrix, axis=1)  # point-in-time: weeks <= t only
    rows = _pairs(panel)
    week_index = {w: j for j, w in enumerate(range(config.FIRST_WEEK, config.LAST_WEEK + 1))}
    key_index = pd.Series(np.arange(len(keys)), index=keys)
    origin_col = rows["origin_week"].map(week_index).to_numpy()
    eligible = ever[key_index.loc[rows["household_key"]].to_numpy(), origin_col] == 1
    return RealtimePredictionRows(
        config.POPULATION_REALTIME, rows.loc[eligible].reset_index(drop=True)
    )


def empirical_window_pairs(panel: CanonicalPanel) -> pd.DataFrame:
    """E-2: the multiset of (f, l) recorded-support pairs, with household keys dropped."""
    support = household_support(panel.recorded).dropna()
    return support.astype(int).reset_index(drop=True)


def draw_empirical_windows(
    pairs: pd.DataFrame, agent_keys, rng: np.random.Generator
) -> pd.DataFrame:
    """Declared matching rule: assign empirical (f, l) pairs to agents without replacement,
    independently of agent type. With as many agents as pairs, the multiset of window
    lengths equals the empirical one exactly.
    """
    agent_keys = np.asarray(agent_keys)
    require(len(agent_keys) <= len(pairs), "more agents than empirical windows")
    chosen = rng.choice(len(pairs), size=len(agent_keys), replace=False)
    drawn = pairs.iloc[chosen].reset_index(drop=True)
    return pd.DataFrame({"household_key": agent_keys, "f": drawn["f"], "l": drawn["l"]})


def record_observation(
    behavior: pd.DataFrame, windows: pd.DataFrame, label: str = "synthetic world"
) -> CanonicalPanel:
    """Apply endpoint-anchored observation windows **after** behaviour (record only).

    ``windows`` has household_key, f, l (each behaviour household gets one empirical
    window). Recorded y equals behaviour inside (f, l), is 1 at f and l (anchors), and
    0 outside. The behaviour panel is returned unchanged alongside the record.
    """
    require(set(windows.columns) >= {"household_key", "f", "l"}, "windows need f and l")
    require((windows["f"] <= windows["l"]).all(), "window start after end")
    frame = behavior.merge(windows[["household_key", "f", "l"]], on="household_key", how="left")
    require(frame[["f", "l"]].notna().all().all(), "every behaviour household needs a window")
    inside = frame["week_no"].between(frame["f"], frame["l"])
    endpoint = (frame["week_no"] == frame["f"]) | (frame["week_no"] == frame["l"])
    recorded = frame[["household_key", "week_no"]].copy()
    recorded[config.TARGET_SOURCE] = np.where(
        endpoint, 1, np.where(inside, frame[config.TARGET_SOURCE], 0)
    ).astype(np.int8)
    return CanonicalPanel(recorded, origin="synthetic", behavior=behavior.copy(), label=label)
