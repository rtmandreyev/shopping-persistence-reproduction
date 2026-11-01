"""Interpretable parameter mappings (design §V, §AD.2).

- **Persistence.** rho or lambda ↔ half-life, via ``config.half_life_to_persistence``
  and ``config.persistence_to_half_life``.
- **Δ_1.** The one-week intervention effect of last week's purchase, in probability
  points (P8-002: a per-point label of the search grid, and the RQ2 truth index on the
  ascending branch). It holds alpha_i, the older history and any latent state fixed, and is
  averaged over operator-O response rows (r with r and r−1 in the envelope interior)
  of a simulated world.
- **Δ_12.** The expected additional purchase-weeks in r … r+11 caused by setting
  y_i,r−1 = 1 rather than 0. It is computed by paired re-simulation with identical
  uniforms and latent paths, over every origin r = 2 … 91 (r + 11 ≤ 102).

The mappings use simulated distributions only. No empirical moment enters them.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from complete_journey.abm import config
from complete_journey.abm.architectures import Heterogeneity, Spec, World, a2, simulate
from complete_journey.abm.config import require
from complete_journey.abm.seeds import Family, SeedKey, draws

DELTA12_ORIGINS = tuple(range(2, 92))  # every r with r + 11 <= 102 (§AD.2)
DELTA12_HORIZON = 12


def response_mask(behavior: np.ndarray, lag: int = 1) -> np.ndarray:
    """(n, 102) mask of response weeks r admitted by operator O for ``lag``.

    Same rule as ``prediction.operators.offline_rows``: r ≤ w_end, r − lag ≥ w_start,
    the household is eligible, and 2 ≤ r ≤ 101.
    """
    y = np.asarray(behavior)
    n, weeks = y.shape
    any_purchase = y.any(axis=1)
    f = np.where(any_purchase, y.argmax(axis=1) + 1, 0)
    last = np.where(any_purchase, weeks - y[:, ::-1].argmax(axis=1), 0)
    start = f + 1
    end = np.minimum(last - 1, config.RESPONSE_WEEKS[1])
    eligible = any_purchase & (np.clip(end - start + 1, 0, None) >= config.MIN_INTERIOR_WEEKS)
    r = np.arange(1, weeks + 1)[None, :]
    lo, hi = config.RESPONSE_WEEKS
    return (
        eligible[:, None]
        & (r <= end[:, None])
        & (r - lag >= start[:, None])
        & (r >= lo)
        & (r <= hi)
    )


def delta1(world: World) -> float:
    """Δ_1 in probability points for A2 (stock), A2-recency or the T4 mixed truth."""
    params = dict(world.spec.params)
    mask = response_mask(world.behavior)
    rows, cols = np.nonzero(mask)  # cols = r - 1 (0-based week index)
    require(rows.size > 0, "no response rows for Δ_1")
    alpha = world.alpha[rows]
    delta, lam = params["delta"], params["lam"]
    latent = 0.0
    if world.latent is not None and "sigma_z" in params:
        latent = params["sigma_z"] * world.latent[rows, cols]
    if world.spec.family == Family.A2:
        h_prev = world.history[rows, cols - 1]  # H_{r-1}
        base = alpha + latent + delta * lam * h_prev
        effect = expit(base + delta * (1.0 - lam)) - expit(base)
    else:
        require(world.spec.family == Family.A2_RECENCY, "Δ_1 is defined for history families")
        last_before = world.last_purchase[rows, cols - 1]  # last purchase before r-1
        r = cols + 1
        term0 = np.where(
            np.isnan(last_before),
            0.0,
            delta * np.power(lam, np.nan_to_num(r - last_before) - 1),
        )
        effect = expit(alpha + latent + delta) - expit(alpha + latent + term0)
    return float(100.0 * effect.mean())


def delta1_peak(lam: float, het: Heterogeneity, key: SeedKey, n: int) -> tuple[float, float]:
    """(delta_peak, max Δ_1) over the predeclared scan (common random numbers).

    Δ_1(delta) of the normalized stock rises, then falls through saturation; P8-002 uses
    the ascending branch only.
    """
    values = [delta1(simulate(a2(d, lam), het, key, n)) for d in config.DELTA_PEAK_SCAN]
    k = int(np.argmax(values))
    rising = np.diff([0.0, *values[: k + 1]])
    require(
        (rising >= -config.DELTA1_MAPPING_TOL_PP).all(),
        f"Δ_1 scan is not monotone below its peak at λ={lam} (ascending branch undefined)",
    )
    return config.DELTA_PEAK_SCAN[k], values[k]


def solve_delta(
    target_pp: float,
    lam: float,
    het: Heterogeneity,
    key: SeedKey,
    n: int,
    tol_pp: float = config.DELTA1_MAPPING_TOL_PP,
) -> tuple[float, float]:
    """Raw delta on the ascending branch whose realized Δ_1 equals ``target_pp``.

    Bisection on [0, delta_peak] with common random numbers. It returns
    (delta, realized Δ_1). If the target exceeds the reachable maximum, it raises
    ``Phase8ContractError`` ("unreachable"): the truth point is infeasible (P8-002) and
    is recorded, never replaced.
    """
    require(target_pp >= 0, "Δ_1 target must be non-negative")
    if target_pp == 0:
        return 0.0, 0.0
    peak, top = delta1_peak(lam, het, key, n)
    require(target_pp <= top + tol_pp, f"Δ_1 = {target_pp} pp unreachable (max {top:.2f} pp)")
    if abs(top - target_pp) <= tol_pp:
        return peak, top

    def realized(d: float) -> float:
        return delta1(simulate(a2(d, lam), het, key, n))

    lo, hi = 0.0, peak
    value, mid = top, peak
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value = realized(mid)
        if abs(value - target_pp) <= tol_pp:
            return mid, value
        lo, hi = (mid, hi) if value < target_pp else (lo, mid)
    require(abs(value - target_pp) <= tol_pp, "Δ_1 bisection did not converge")
    return mid, value


def point_labels(spec: Spec, het: Heterogeneity, key: SeedKey, n: int) -> dict[str, float]:
    """Interpretive Δ_1 (and, for the stock, Δ_12) labels of an A2 grid point (P8-002)."""
    world = simulate(spec, het, key, n)
    labels = {"delta1_pp": delta1(world)}
    if spec.family == Family.A2 and spec.role == "primary":
        labels["delta12_weeks"] = delta12(spec, het, key, n)
    return labels


def delta12(spec: Spec, het: Heterogeneity, key: SeedKey, n: int) -> float:
    """Δ_12 for the geometric-stock A2 by paired re-simulation (expected extra weeks)."""
    require(spec.family == Family.A2 and spec.role == "primary", "Δ_12 implemented for A2")
    world = simulate(spec, het, key, n)
    d = draws(key, n, Family.A2)
    delta, lam = spec.param("delta"), spec.param("lam")
    mask = response_mask(world.behavior)
    diffs = []
    for r in DELTA12_ORIGINS:
        rows = np.flatnonzero(mask[:, r - 1])
        if rows.size == 0:
            continue
        outcome = {}
        for forced in (0, 1):
            h = lam * world.history[rows, r - 2] + (1.0 - lam) * forced  # H_r
            total = np.zeros(rows.size)
            for k in range(DELTA12_HORIZON):
                week = r + k
                idx = config.BURN_IN_WEEKS + week - 1
                p = expit(world.alpha[rows] + delta * h)
                y = d.uniforms[rows, idx] < p
                total += y
                h = lam * h + (1.0 - lam) * y
            outcome[forced] = total
        diffs.append(outcome[1] - outcome[0])
    require(diffs, "no origins available for Δ_12")
    return float(np.concatenate(diffs).mean())
