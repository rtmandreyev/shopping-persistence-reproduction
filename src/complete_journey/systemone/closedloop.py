"""Closed-loop artificial panels from a cached policy table (local only; P9-001).

The DGP for household i in week t:
- Stable profile f_i is the grid level nearest to expit(mu + s*xi_i), with
  (mu, s) = (0.213, 1.540), the Phase 8 A1 fit. It is drawn fresh for every panel.
- Need n_it (need-visible cells only) follows a symmetric 3-state Markov chain:
  stay with q = (2*lambda + 1) / 3, else move to each other level with (1 - q) / 2.
  It starts from the stationary uniform law. In need-hidden cells the need state is
  *absent* from the DGP: nothing is simulated, and nothing can act through it.
- History h_it = (y_{t-1}, ..., y_{t-4}) is updated deterministically each week. It
  starts as 4 iid Bernoulli(f_i) weeks. In history-hidden cells the policy never sees
  h; the bookkeeping only records outcomes.
- y_it ~ Bernoulli(pi(f_i, h_it, n_it)) by exact table lookup. A NaN entry
  (unqueried or undefined state) is a loud error.

There are 26 burn-in weeks, then 102 retained weeks; the operator O is applied
downstream by the frozen Phase 7/8 moment code.

``shuffled=True`` is the local control. Each household's policy sees the history of a
*donor*: a household with the same profile level in an independently simulated
companion panel. The companion panel runs the unshuffled DGP of the same cell, with
its own seeds, own outcomes and (if visible) own need. The household keeps its own
profile and its own need.
- Own-history causal effect: exactly zero. The donor never sees the recipient, and
  nothing in the recipient's own outcomes reaches the history it is shown.
- Recipients remain independent of each other.
- The marginal history content (same profile, same policy) is preserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from complete_journey.systemone import lattice as L
from complete_journey.systemone import settings as S
from complete_journey.systemone.seeds import rng
from complete_journey.systemone.settings import require

H_WEIGHTS = np.array([8, 4, 2, 1])  # h index = 8*h1 + 4*h2 + 2*h3 + h4 (HISTORY_PATTERNS order)


def need_matrix(lam: float) -> np.ndarray:
    require(-0.5 < lam < 1.0, "need lambda must lie in (-0.5, 1)")
    q = (2 * lam + 1) / 3
    m = np.full((3, 3), (1 - q) / 2)
    np.fill_diagonal(m, q)
    return m


def need_summary(lam: float) -> dict:
    q = (2 * lam + 1) / 3
    return {
        "lambda": lam,
        "stay_probability": q,
        "expected_spell_weeks": 1 / (1 - q),
        "half_life_weeks": math.log(0.5) / math.log(lam) if 0 < lam < 1 else None,
        "autocorrelation_lag_1": lam,
        "autocorrelation_lag_4": lam**4,
    }


def assign_profiles(n: int, k: int, g: np.random.Generator) -> np.ndarray:
    grid = np.array(L.grid(k))
    p = 1 / (1 + np.exp(-(S.A1_MU + S.A1_S * g.standard_normal(n))))
    return np.abs(p[:, None] - grid[None, :]).argmin(axis=1)


@dataclass(frozen=True)
class DGP:
    cell: str
    lam: float | None = None  # need persistence (need-visible cells)
    shuffled: bool = False

    def __post_init__(self):
        hv, nv = S.CELL_VISIBILITY[self.cell]
        require((self.lam is not None) == nv, "need lambda is given exactly for need-visible cells")
        require(not self.shuffled or hv, "the shuffled control needs a history-visible cell")

    @property
    def name(self) -> str:
        return (
            f"{self.cell}"
            + (f"-lam{self.lam:g}" if self.lam is not None else "")
            + ("-SHUF" if self.shuffled else "")
        )


def simulate(
    table: np.ndarray,
    dgp: DGP,
    k: int,
    coords: tuple[int, ...],
    n: int = S.N_HOUSEHOLDS,
    t: int = S.T_WEEKS,
    burn: int = S.BURN_IN,
) -> dict:
    """One closed-loop panel. ``coords`` index the Phase 9 seed tree."""
    require(table.shape == (len(L.grid(k)), 16, 3), "policy table has the wrong shape")
    g = rng("closed_loop", *coords)
    prof = assign_profiles(n, k, g)
    f = np.array(L.grid(k))[prof]
    hv, nv = S.CELL_VISIBILITY[dgp.cell]
    hist = (g.random((n, 4)) < f[:, None]).astype(np.int8)  # columns: h1..h4
    need = g.integers(0, 3, n) if nv else np.zeros(n, dtype=int)
    tm = need_matrix(dgp.lam) if nv else None
    if dgp.shuffled:  # independent donor panel, paired by profile level
        gd = rng("shuffle", *coords)
        d_hist = (gd.random((n, 4)) < f[:, None]).astype(np.int8)
        d_need = gd.integers(0, 3, n) if nv else np.zeros(n, dtype=int)
        d_u = gd.random((n, burn + t))
        d_un = gd.random((n, burn + t)) if nv else None
    total = burn + t
    y = np.zeros((n, total), dtype=np.int8)
    u = g.random((n, total))
    un = g.random((n, total)) if nv else None
    seen_log = np.zeros((n, total), dtype=np.int8)
    for w in range(total):
        seen = d_hist if dgp.shuffled else hist
        hidx = (seen @ H_WEIGHTS) if hv else np.zeros(n, dtype=int)
        seen_log[:, w] = hidx
        p = table[prof, hidx, need]
        require(
            not np.isnan(p).any(),
            f"{dgp.name}: closed loop reached an unqueried or undefined state",
        )
        yw = (u[:, w] < p).astype(np.int8)
        y[:, w] = yw
        hist = np.column_stack([yw, hist[:, :3]])
        if nv:
            cum = np.cumsum(tm[need], axis=1)
            need = (un[:, w][:, None] > cum).sum(axis=1)
        if dgp.shuffled:  # the donor evolves on its own history and need only
            pd_ = table[prof, d_hist @ H_WEIGHTS, d_need]
            yd = (d_u[:, w] < pd_).astype(np.int8)
            d_hist = np.column_stack([yd, d_hist[:, :3]])
            if nv:
                cum = np.cumsum(tm[d_need], axis=1)
                d_need = (d_un[:, w][:, None] > cum).sum(axis=1)
    return {
        "behavior": y[:, burn:],
        "profile_index": prof,
        "burn_in": burn,
        "seen_history_index": seen_log,
    }


# --- analytic finite-chain moments (single-household chain; not for shuffled DGPs) -------


def chain(table: np.ndarray, level: int, dgp: DGP):
    """Transition matrix over (h_index, need_index), the same matrix restricted to
    transitions through y = 1, and the per-state purchase probability."""
    require(not dgp.shuffled, "the shuffled DGP is not a single-household chain")
    hv, nv = S.CELL_VISIBILITY[dgp.cell]
    nn = 3 if nv else 1
    nm = need_matrix(dgp.lam) if nv else np.ones((1, 1))
    size = 16 * nn
    p = np.zeros(size)
    tr = np.zeros((size, size))
    tr1 = np.zeros((size, size))
    for h in range(16):
        for ni in range(nn):
            i = h * nn + ni
            pi = table[level, h if hv else 0, ni if nv else 0]
            p[i] = pi
            for y, py in ((1, pi), (0, 1 - pi)):
                h2 = (y << 3) | (h >> 1)
                for nj in range(nn):
                    tr[i, h2 * nn + nj] += py * nm[ni, nj]
                    if y == 1:
                        tr1[i, h2 * nn + nj] += py * nm[ni, nj]
    return tr, tr1, p, nn


def initial_distribution(f: float, nn: int) -> np.ndarray:
    d = np.zeros(16 * nn)
    for h, pat in enumerate(S.HISTORY_PATTERNS):
        ph = np.prod([f if v else 1 - f for v in pat])
        for ni in range(nn):
            d[h * nn + ni] = ph / nn
    return d


def _stationary(tr: np.ndarray, start: np.ndarray | None = None) -> np.ndarray:
    """The unique stationary law when the chain has one; otherwise (several unit
    eigenvalues, e.g. probabilities of exactly 0 or 1 make the chain reducible) the
    Cesàro long-run average from ``start``, which is the law the DGP actually reaches."""
    vals, vecs = np.linalg.eig(tr.T)
    unit = np.abs(vals - 1) < 1e-9
    if unit.sum() == 1:
        st = np.real(vecs[:, np.argmax(unit)])
        st = st / st.sum()
        if st.min() > -1e-10:
            return np.clip(st, 0, None) / np.clip(st, 0, None).sum()
    d = np.full(len(tr), 1 / len(tr)) if start is None else start.copy()
    acc = np.zeros(len(tr))
    for _ in range(20000):
        nxt = d @ tr
        acc += nxt
        if np.abs(nxt - d).sum() < 1e-13:  # aperiodic: the limit law itself
            return nxt / nxt.sum()
        d = nxt
    return acc / acc.sum()  # periodic: the Cesàro long-run average


def analytic(
    table: np.ndarray, level: int, f: float, dgp: DGP, t: int = S.T_WEEKS, burn: int = S.BURN_IN
) -> dict:
    """Exact expected weekly purchase rate and lag-1 co-purchase rate P(y_w = 1,
    y_{w+1} = 1) over the retained weeks, the stationary values, and the total-variation
    distance from stationarity at the end of burn-in."""
    tr, tr1, p, nn = chain(table, level, dgp)
    d = initial_distribution(f, nn)
    for _ in range(burn):
        d = d @ tr
    st = _stationary(tr, initial_distribution(f, nn))
    tv = float(0.5 * np.abs(d - st).sum())
    nxt = tr1 @ p  # P(y_w = 1, y_{w+1} = 1 | state at w)
    rates, joint = [], []
    for _ in range(t):
        rates.append(float(d @ p))
        joint.append(float(d @ nxt))
        d = d @ tr
    return {
        "rate_by_week": np.array(rates),
        "joint11_by_week": np.array(joint[:-1]),
        "stationary_rate": float(st @ p),
        "stationary_joint11": float(st @ nxt),
        "tv_after_burn_in": tv,
        "stationary": st,
        "p": p,
    }


def stationary_distribution(
    table: np.ndarray, level: int, dgp: DGP, k: int
) -> tuple[np.ndarray, int]:
    tr, _, _, nn = chain(table, level, dgp)
    return _stationary(tr, initial_distribution(L.grid(k)[level], nn)), nn


def profile_mass(k: int) -> np.ndarray:
    """Exact mass of each grid level under nearest-level rounding of the A1 law."""
    from scipy.stats import norm

    grid = np.array(L.grid(k))
    edges = np.concatenate([[0.0], (grid[1:] + grid[:-1]) / 2, [1.0]])
    z = (
        np.log(np.clip(edges, 1e-12, 1 - 1e-12) / (1 - np.clip(edges, 1e-12, 1 - 1e-12))) - S.A1_MU
    ) / S.A1_S
    return np.diff(norm.cdf(z))


# --- logistic surrogate (predeclared) ----------------------------------------------------

SURROGATE_FEATURES = ("intercept", "logit_profile", "h1", "h2", "h3", "h4", "need_low", "need_high")


def surrogate_design(cell: str, k: int):
    hv, nv = S.CELL_VISIBILITY[cell]
    rows, index = [], []
    g = L.grid(k)
    for i, f in enumerate(g):
        for hi, h in enumerate(S.HISTORY_PATTERNS if hv else [None]):
            for ni, n in enumerate(S.NEED_LEVELS if nv else [None]):
                x = [1.0, math.log(f / (1 - f))]
                x += list(h) if hv else [0, 0, 0, 0]
                x += [1.0 if n == "low" else 0.0, 1.0 if n == "high" else 0.0] if nv else [0.0, 0.0]
                rows.append(x)
                index.append((i, hi if hv else None, ni if nv else None))
    x = np.array(rows, float)
    used = [j for j in range(x.shape[1]) if np.any(x[:, j] != 0)]
    return x[:, used], index, [SURROGATE_FEATURES[j] for j in used]


def fit_surrogate(table: np.ndarray, cell: str, k: int) -> dict:
    """Fractional-logit fit (equal weight per lattice state; cross-entropy; Newton/IRLS)
    of the cell's policy surface. Returns the surrogate table and its coefficients."""
    x, index, names = surrogate_design(cell, k)
    yv = np.array([table[i, 0 if h is None else h, 0 if n is None else n] for i, h, n in index])
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(x @ beta)))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        grad = x.T @ (yv - mu)
        hess = (x * w[:, None]).T @ x + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    fitted = 1 / (1 + np.exp(-(x @ beta)))
    sur = np.full_like(table, np.nan)
    hv, nv = S.CELL_VISIBILITY[cell]
    for (i, h, n), v in zip(index, fitted, strict=True):
        hs = [h] if hv else range(16)
        ns = [n] if nv else range(3)
        for a in hs:
            for b in ns:
                sur[i, a, b] = v
    return {
        "table": sur,
        "coef": dict(zip(names, beta.tolist(), strict=True)),
        "max_abs_error": float(np.max(np.abs(fitted - yv))),
        "rmse": float(np.sqrt(np.mean((fitted - yv) ** 2))),
    }


# --- secondary calibration: one global logit-intercept shift per DGP ---------------------


def calibrate_shift(
    policy_table_fn,
    dgp: DGP,
    k: int,
    target_mean: float,
    coords: tuple[int, ...],
    n: int = S.N_HOUSEHOLDS,
    r: int = 4,
    lo: float = -6.0,
    hi: float = 6.0,
    iters: int = 30,
) -> float:
    """Bisection on delta such that the mean CI-1 'mean' component over r common-random-
    number panels equals the real CI-1 mean. ``policy_table_fn(delta)`` returns the table."""
    from complete_journey.abm.moments import ci1_fast

    def gap(delta: float) -> float:
        tab = policy_table_fn(delta)
        vals = [ci1_fast(simulate(tab, dgp, k, (*coords, j), n=n)["behavior"])[0] for j in range(r)]
        return float(np.mean(vals)) - target_mean

    glo, ghi = gap(lo), gap(hi)
    require(glo < 0 < ghi, "calibration target outside the attainable range")
    for _ in range(iters):
        mid = (lo + hi) / 2
        if gap(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
