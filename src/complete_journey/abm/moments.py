"""Phase 8 moment adapter over the frozen Phase 7 engine (design §P–§R).

Nothing is redefined:
- **P1** VI-2 ``O-M3``: ``prediction.moments.vi2_ladder``;
- **P2** VI-3 ``beta_8`` and **P3** VI-3 ``gamma_2``:
  ``prediction.moments.vi3_lag_profile``;
- **P4** VI-6 ``delta_log_loss_recency``: ``prediction.protocol.recency_gain``, with
  B1 retrained inside this world;
- **P5** VI-4 ``active_1``: ``prediction.moments.vi4_spells``.

``ci1_fast`` is a numpy equivalent of the Phase 7 CI-1 function, used inside the
calibration optimizer. Tests prove it equals ``prediction.moments.ci1`` exactly.

Tier metadata travels with every value. Only PRIMARY moments can reach the adequacy
engine, so secondary and diagnostic moments can never acquire veto power.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from complete_journey.abm import config
from complete_journey.abm.config import require
from complete_journey.abm.operators import require_phase8_panel
from complete_journey.prediction import config as p7
from complete_journey.prediction import moments as p7m
from complete_journey.prediction import protocol as p7p
from complete_journey.prediction.panel import CanonicalPanel

PRIMARY, SECONDARY, DIAGNOSTIC = "primary", "secondary", "diagnostic"


@dataclass(frozen=True)
class MomentValues:
    """Moment values with their tier; ``origin`` is "simulated" or "synthetic"."""

    tier: str
    values: dict[str, float]
    origin: str = "simulated"

    def __post_init__(self):
        require(self.tier in (PRIMARY, SECONDARY, DIAGNOSTIC), "unknown moment tier")
        if self.tier == PRIMARY:
            require(tuple(self.values) == config.PRIMARY_IDS, "primary vector must be P1..P5")


def tier_of(moment_id: str, component: str) -> str:
    if (moment_id, component) in {(m[1], m[2]) for m in config.PRIMARY_MOMENTS}:
        return PRIMARY
    if (moment_id, component) in set(config.SECONDARY_MOMENTS):
        return SECONDARY
    return DIAGNOSTIC


def _component(records: list[dict], component: str) -> float:
    value = next(r["estimate"] for r in records if r["component"] == component)
    require(value is not None and np.isfinite(value), f"moment {component} undefined")
    return float(value)


def primary_moments(panel: CanonicalPanel) -> MomentValues:
    """P1..P5 for one world under operator O (Phase 7 code, unchanged)."""
    require_phase8_panel(panel)
    op = p7.OPERATOR_OFFLINE
    ladder = p7m.vi2_ladder(panel, op)
    lags = p7m.vi3_lag_profile(panel, op)
    gain = p7p.recency_gain(panel, op)
    spell = p7m.vi4_spells(panel, op)
    values = {
        "P1": _component(ladder, "O-M3"),
        "P2": _component(lags, "beta_8"),
        "P3": _component(lags, "gamma_2"),
        "P4": float(gain[p7.RECENCY_GAIN]),
        "P5": _component(spell, "active_1"),
    }
    return MomentValues(PRIMARY, values)


#: Secondary summaries carried alongside P1..P5 by Stage 1 tables (reported, no veto).
BUNDLE_SECONDARY = ("GAP", "P11", "P01", "O-M2", "beta_1", "beta_2", "beta_4", "inactive_1")


def moment_bundle(panel: CanonicalPanel) -> tuple[MomentValues, dict[str, float]]:
    """P1..P5 plus the Stage 1 secondary summaries, from a single pass of the frozen
    Phase 7 functions. The values are identical to ``primary_moments``."""
    require_phase8_panel(panel)
    op = p7.OPERATOR_OFFLINE
    trans = p7m.vi1_transitions(panel, op)
    ladder = p7m.vi2_ladder(panel, op)
    lags = p7m.vi3_lag_profile(panel, op)
    gain = p7p.recency_gain(panel, op)
    spell = p7m.vi4_spells(panel, op)
    primary = {
        "P1": _component(ladder, "O-M3"),
        "P2": _component(lags, "beta_8"),
        "P3": _component(lags, "gamma_2"),
        "P4": float(gain[p7.RECENCY_GAIN]),
        "P5": _component(spell, "active_1"),
    }
    secondary = {
        "GAP": _component(trans, "GAP"),
        "P11": _component(trans, "P11"),
        "P01": _component(trans, "P01"),
        "O-M2": _component(ladder, "O-M2"),
        "beta_1": _component(lags, "beta_1"),
        "beta_2": _component(lags, "beta_2"),
        "beta_4": _component(lags, "beta_4"),
        "inactive_1": _component(spell, "inactive_1"),
    }
    return MomentValues(PRIMARY, primary), secondary


def secondary_moments(panel: CanonicalPanel) -> MomentValues:
    """Secondary register components (reported, never veto)."""
    require_phase8_panel(panel)
    op = p7.OPERATOR_OFFLINE
    records = (
        p7m.vi1_transitions(panel, op)
        + p7m.vi2_ladder(panel, op)
        + p7m.vi3_lag_profile(panel, op)
        + p7m.vi4_spells(panel, op)
        + p7m.vi5_half_to_half(panel, op)
        + p7p.vi6_records(panel, op)
    )
    wanted = set(config.SECONDARY_MOMENTS)
    values = {}
    for r in records:
        if (r["moment_id"], r["component"]) in wanted:
            values[f"{r['moment_id']}:{r['component']}"] = r["estimate"]
    return MomentValues(SECONDARY, values)


def ci1_fast(behavior: np.ndarray) -> np.ndarray:
    """CI-1 [mean, q10, q25, q50, q75, q90] under operator O from a (n, 102) matrix.

    It is identical to ``prediction.moments.ci1`` on the same panel, as pinned by tests:
    - interior window [f+1, min(l−1, 101)];
    - eligibility |W| ≥ 8;
    - unweighted household rates;
    - pandas-default linear quantiles.
    """
    y = np.asarray(behavior, dtype=np.int8)
    n, weeks = y.shape
    require(weeks == config.T_WEEKS, "CI-1 needs weeks 1-102")
    any_purchase = y.any(axis=1)
    f = np.where(any_purchase, y.argmax(axis=1) + 1, 0)
    l_week = np.where(any_purchase, weeks - y[:, ::-1].argmax(axis=1), 0)
    start = f + 1
    end = np.minimum(l_week - 1, config.RESPONSE_WEEKS[1])
    length = np.clip(end - start + 1, 0, None)
    eligible = any_purchase & (length >= config.MIN_INTERIOR_WEEKS)
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(y, axis=1)], axis=1)
    rows = np.flatnonzero(eligible)
    require(rows.size > 0, "no eligible household for CI-1")
    s, e = start[rows], end[rows]
    rates = (csum[rows, e] - csum[rows, s - 1]) / (e - s + 1)
    qs = np.quantile(rates, config.CI1_QUANTILES)
    return np.concatenate([[rates.mean()], qs])
