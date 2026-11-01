"""SYNTHETIC NON-EVIDENCE data-generating processes for Run B preflight (S1–S7).

Behaviour is generated first:
  logit p_it = alpha_i + tau_t + sigma * z_it + delta * y_{i,t-1},
  z_it = rho * z_{i,t-1} + sqrt(1 - rho^2) * e_it.
Observation, if any, is applied afterwards by ``operators.record_observation``.
These are test fixtures with known truth, not Phase 8 architectures, and they carry
no empirical meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from complete_journey.prediction import config
from complete_journey.prediction.operators import record_observation
from complete_journey.prediction.panel import CanonicalPanel

WEEKS = config.LAST_WEEK - config.FIRST_WEEK + 1
BURN_IN = 30


@dataclass(frozen=True)
class DGP:
    name: str
    households: int = 1500
    mu: float = 0.1
    s_alpha: float = 0.0
    sigma: float = 0.0
    rho: float = 0.0
    delta: float = 0.0
    trend: float = 0.0
    windows: str = "full"  # "full" or "heterogeneous"
    notes: str = ""
    truth: dict = field(default_factory=dict)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(spec: DGP, seed: int) -> tuple[CanonicalPanel, dict]:
    rng = np.random.default_rng(seed)
    n, total = spec.households, BURN_IN + WEEKS
    alpha = spec.mu + spec.s_alpha * rng.standard_normal(n)
    tau = np.concatenate([np.zeros(BURN_IN), spec.trend * (np.arange(WEEKS) / (WEEKS - 1) - 0.5)])
    z = rng.standard_normal(n)
    y_prev = np.zeros(n)
    ys, carry = [], []
    for t in range(total):
        z = spec.rho * z + np.sqrt(1 - spec.rho**2) * rng.standard_normal(n)
        index = alpha + tau[t] + spec.sigma * z
        p = _sigmoid(index + spec.delta * y_prev)
        y = (rng.random(n) < p).astype(np.int8)
        if t >= BURN_IN:
            ys.append(y)
            # True one-week carryover at the current latent state (economic units).
            carry.append(_sigmoid(index + spec.delta) - _sigmoid(index))
        y_prev = y
    matrix = np.column_stack(ys)
    keys = config.SYNTHETIC_KEY_OFFSET + np.arange(1, n + 1)
    behavior = pd.DataFrame(
        {
            "household_key": np.repeat(keys, WEEKS),
            "week_no": np.tile(np.arange(config.FIRST_WEEK, config.LAST_WEEK + 1), n),
            config.TARGET_SOURCE: matrix.ravel(),
        }
    )
    if spec.windows == "heterogeneous":
        f = rng.integers(1, 31, n)
        last = rng.integers(70, config.LAST_WEEK + 1, n)
        windows = pd.DataFrame({"household_key": keys, "f": f, "l": last})
        panel = record_observation(behavior, windows, label=spec.name)
    else:
        panel = CanonicalPanel(
            behavior, origin="synthetic", behavior=behavior.copy(), label=spec.name
        )
    truth = {
        "delta_logit": spec.delta,
        "true_carryover_pp": float(np.mean(carry)),
        "rho": spec.rho,
        "sigma": spec.sigma,
        "s_alpha": spec.s_alpha,
        **spec.truth,
    }
    return panel, truth


FIXTURES = {
    "S1": DGP("S1 constant base rate", notes="no heterogeneity, no memory"),
    "S2": DGP("S2 explicit lag", s_alpha=0.6, delta=1.0, notes="true history dependence"),
    "S3": DGP(
        "S3 latent AR state",
        s_alpha=0.6,
        sigma=1.2,
        rho=0.85,
        notes="serially correlated state, no purchase memory",
    ),
    "S4": DGP("S4 time trend", s_alpha=0.6, trend=1.5, notes="distribution shift over calendar"),
    "S5": DGP("S5 heterogeneity", s_alpha=1.2, notes="stable heterogeneity, no memory"),
    "S6": DGP(
        "S6 observation truncation",
        s_alpha=1.2,
        windows="heterogeneous",
        notes="S5 behaviour, heterogeneous observation windows imposed afterwards",
    ),
}

#: S7 is a feature-store fixture: a synthetic future-derived column must be refused.
S7_LEAK_COLUMN = "future_retailer_purchase_next_week"


def s7_feature_store(panel: CanonicalPanel) -> pd.DataFrame:
    """Synthetic store with one legitimate column and one leaking future column."""
    frame = panel.recorded.rename(columns={"week_no": "as_of_week"}).copy()
    frame["current_retailer_purchase"] = frame[config.TARGET_SOURCE]
    frame[S7_LEAK_COLUMN] = frame.groupby("household_key")[config.TARGET_SOURCE].shift(-1)
    return frame.drop(columns=[config.TARGET_SOURCE])
