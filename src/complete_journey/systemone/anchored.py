"""P9-007 anchored out-of-family known truths (local only; frozen design).

A truth combines Phase 8's common heterogeneity layer, with (mu, s) or the mixture
nuisance parameters fitted to the real CI-1 only, with the frozen learned-policy dynamic
response (P9-007 §1):

    logit p_it = alpha_i + theta * g^H(h_it, n_it | f_i) + g^N(n_it | f_i)

From the frozen table pi_C(f, h, n) of cell C, clipped to [0.005, 0.995]:
- L = logit(pi_C);
- Lbar(f, n) = sum_h w(h | f) L(f, h, n), with w(h | f) = Binomial(4, f) pattern weights;
- c(f) = mean over the cell's needs of Lbar(f, n), uniform (the stationary need law);
- g^H(h, n | f) = L(f, h, n) - Lbar(f, n);
- g^N(n | f) = Lbar(f, n) - c(f).

f_i = clamp(expit(alpha_i), 0.05, 0.95), and g is linearly interpolated in f between
adjacent grid levels. Irrelevant-history inputs (§2):
- **serial donor**: the donor's realized purchases in t-1 ... t-4;
- **time-scrambled donor**: each position drawn uniformly, with replacement, from the
  donor's realized purchases over weeks 1 ... t-1.

As in the Phase 9 shuffled design, the donor is an independent shadow household paired
with the household. It shares alpha_i (the anchored analogue of Phase 9's pairing by
profile level) and evolves on its own history and need under the same dynamics. In
week 0 the scrambled input is the donor's initial 4-week history, as in Phase 9.

Calibration reproduces the frozen Phase 8 calibrators exactly:
- the CI-1 objective with diagonal inverse archived bootstrap variances;
- CALIBRATION-role common random numbers, R_cal = 20;
- Nelder-Mead with a wide simplex and a narrow restart;
- for the mixture arm: the P8-005 transforms, the three moment-matched starts from the
  primary solution at the same dynamics, and the start-selection rule.

Only the simulator they call differs. No P1-P5 moment enters construction or
calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from complete_journey.systemone import lattice as L
from complete_journey.systemone import settings as S
from complete_journey.systemone.closedloop import H_WEIGHTS, need_matrix
from complete_journey.systemone.seeds import rng
from complete_journey.systemone.settings import require

CLIP = (0.005, 0.995)
LAM = 0.8
DEGENERATE_SHARE = 0.05
NO_HISTORY, HISTORY_INSTALLED = "NO-HISTORY", "HISTORY-INSTALLED"
ARMS = ("logit_normal", "mixture")
_BITS = np.array(S.HISTORY_PATTERNS, dtype=np.int8)  # (16, 4); row index = history index


@dataclass(frozen=True)
class Truth:
    id: str
    history: str  # none | own | serial | scrambled
    need: bool
    cell: str | None  # table cell supplying g (None: g == 0)
    theta: float

    @property
    def label(self) -> str:
        return HISTORY_INSTALLED if self.history == "own" and self.theta > 0 else NO_HISTORY

    @property
    def has_history_input(self) -> bool:
        return self.history != "none"


#: The frozen P9-007 §3 grid (15 truths per policy x arm).
TRUTHS: tuple[Truth, ...] = (
    Truth("A1", "none", False, None, 0.0),
    Truth("A2", "none", True, S.LATENT, 0.0),
    Truth("A3", "scrambled", False, S.HISTORY, 1.0),
    Truth("A4", "serial", False, S.HISTORY, 1.0),
    Truth("A5", "scrambled", True, S.BOTH, 1.0),
    Truth("A6", "serial", True, S.BOTH, 1.0),
    Truth("A7", "none", True, S.BOTH, 0.0),
    *(
        Truth(f"B{i + 1}", "own", False, S.HISTORY, th)
        for i, th in enumerate((0.25, 0.5, 0.75, 1.0))
    ),
    *(Truth(f"B{i + 5}", "own", True, S.BOTH, th) for i, th in enumerate((0.25, 0.5, 0.75, 1.0))),
)
TRUTH_BY_ID = {t.id: t for t in TRUTHS}


# --- learned dynamic components --------------------------------------------------------


def components(table: np.ndarray, k: int = 10) -> dict:
    """g^H (k, 16, 3) and g^N (k, 3) from a dense primary table pi_C[f, h, n]."""
    require(table.shape == (len(L.grid(k)), 16, 3), "table shape")
    require(np.isfinite(table).all(), "the table must be complete for a truth cell")
    lg = logit(np.clip(table, *CLIP))
    f = np.array(L.grid(k))
    ones = _BITS.sum(axis=1)  # purchases in each pattern
    w = f[:, None] ** ones[None, :] * (1 - f[:, None]) ** (4 - ones[None, :])  # (k, 16)
    require(np.allclose(w.sum(axis=1), 1.0), "binomial weights")
    lbar = np.einsum("fh,fhn->fn", w, lg)  # (k, 3)
    c = lbar.mean(axis=1)  # uniform over needs; non-visible needs are broadcast copies
    return {"gH": lg - lbar[:, None, :], "gN": lbar - c[:, None]}


def interp_weights(alpha: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    g = np.array(L.grid(k))
    f = np.clip(expit(alpha), g[0], g[-1])
    x = (f - g[0]) / (g[1] - g[0])
    lo = np.minimum(np.floor(x).astype(int), len(g) - 2)
    return lo, x - lo


# --- common random numbers -------------------------------------------------------------


def draws(g: np.random.Generator, n: int, total: int) -> dict:
    """All random inputs of one panel. They do not depend on (mu, s), so calibration
    reuses them across evaluations (common random numbers)."""
    return {
        "xi": g.standard_normal(n),
        "mix_u": g.random(n),
        "init_u": g.random((n, 4)),
        "need0": g.integers(0, 3, n),
        "u": g.random((n, total)),
        "un": g.random((n, total)),
        "d_init_u": g.random((n, 4)),
        "d_need0": g.integers(0, 3, n),
        "d_u": g.random((n, total)),
        "d_un": g.random((n, total)),
        "scr": g.random((n, total, 4)),
    }


def household_alpha(het, xi: np.ndarray, mix_u: np.ndarray) -> np.ndarray:
    from complete_journey.abm.architectures import household_alpha as p8_alpha
    from complete_journey.abm.seeds import Family

    return p8_alpha(het, xi, mix_u, Family.A1)


# --- simulator -------------------------------------------------------------------------


def simulate(
    comp: dict | None,
    truth: Truth,
    het,
    dr: dict,
    t: int = S.T_WEEKS,
    burn: int = S.BURN_IN,
    k: int = 10,
) -> dict:
    """One anchored panel. ``comp`` holds the truth cell's components (None for A1)."""
    n = dr["xi"].shape[0]
    total = burn + t
    alpha = household_alpha(het, dr["xi"], dr["mix_u"])
    lo, w = interp_weights(alpha, k)
    if comp is not None:
        gH = (1 - w)[:, None, None] * comp["gH"][lo] + w[:, None, None] * comp["gH"][lo + 1]
        gN = (1 - w)[:, None] * comp["gN"][lo] + w[:, None] * comp["gN"][lo + 1]
    else:
        gH = np.zeros((n, 16, 3))
        gN = np.zeros((n, 3))
    rows = np.arange(n)
    tm = need_matrix(LAM)
    cum = np.cumsum(tm, axis=1)
    p0 = expit(alpha)
    hist = (dr["init_u"] < p0[:, None]).astype(np.int8)
    need = dr["need0"].copy() if truth.need else np.zeros(n, dtype=int)
    donor = truth.history in ("serial", "scrambled")
    if donor:
        d_hist = (dr["d_init_u"] < p0[:, None]).astype(np.int8)
        d_need = dr["d_need0"].copy() if truth.need else np.zeros(n, dtype=int)
        d_y = np.zeros((n, total), dtype=np.int8)
    y = np.zeros((n, total), dtype=np.int8)
    outside = 0
    count = 0
    d1_sum = 0.0

    def index(hidx, nd):
        return alpha + truth.theta * gH[rows, hidx, nd] + gN[rows, nd]

    for wk in range(total):
        if truth.history == "own":
            seen = hist
        elif truth.history == "serial":
            seen = d_hist
        elif truth.history == "scrambled":
            if wk == 0:
                seen = d_hist
            else:
                pick = np.floor(dr["scr"][:, wk, :] * wk).astype(int)  # weeks 0 .. wk-1
                seen = d_y[rows[:, None], pick]
        else:
            seen = None
        hidx = seen @ H_WEIGHTS if seen is not None else np.zeros(n, dtype=int)
        eta = index(hidx, need)
        p = expit(eta)
        if wk >= burn:
            outside += int(((p < CLIP[0]) | (p > CLIP[1])).sum())
            count += n
            if truth.has_history_input and comp is not None and truth.theta > 0:
                h1 = (hidx & 8) > 0
                p1 = expit(index(np.where(h1, hidx, hidx + 8), need))
                p0_ = expit(index(np.where(h1, hidx - 8, hidx), need))
                d1_sum += float((p1 - p0_).sum())
        yw = (dr["u"][:, wk] < p).astype(np.int8)
        y[:, wk] = yw
        hist = np.column_stack([yw, hist[:, :3]])
        if truth.need:
            need = (dr["un"][:, wk][:, None] > cum[need]).sum(axis=1)
        if donor:
            pd_ = expit(index(d_hist @ H_WEIGHTS, d_need))
            yd = (dr["d_u"][:, wk] < pd_).astype(np.int8)
            d_y[:, wk] = yd
            d_hist = np.column_stack([yd, d_hist[:, :3]])
            if truth.need:
                d_need = (dr["d_un"][:, wk][:, None] > cum[d_need]).sum(axis=1)
    beh = y[:, burn:]
    rates = beh.mean(axis=1)
    return {
        "behavior": beh,
        "share_outside_clip": outside / max(count, 1),
        "mean_rate": float(beh.mean()),
        "share_households_rate_gt_095": float((rates > 0.95).mean()),
        "delta1": d1_sum / max(count, 1) if count else 0.0,
    }


# --- calibration (frozen Phase 8 methods; only the simulator differs) ------------------


def _real_target():
    from complete_journey.abm.calibration import target_arrays
    from complete_journey.abm.stage1 import real_ci1_target

    return target_arrays(real_ci1_target())


def calibration_draws(n: int, r_cal: int) -> list[dict]:
    total = S.BURN_IN + S.T_WEEKS
    return [draws(rng("p9007_calibration", r), n, total) for r in range(r_cal)]


def simulated_ci1(comp, truth, het, crn: list[dict]) -> np.ndarray:
    from complete_journey.abm.moments import ci1_fast

    return np.mean([ci1_fast(simulate(comp, truth, het, d)["behavior"]) for d in crn], axis=0)


def calibrate_logit_normal(comp, truth, crn) -> dict:
    """The frozen primary calibrator (Phase 8 §F/§P): (mu, log s), default start."""
    from complete_journey.abm.architectures import Heterogeneity
    from complete_journey.abm.calibration import INITIAL_SIMPLEX_SCALES, NM_OPTIONS
    from complete_journey.abm.mixture import _gate

    values, se = _real_target()
    weights = 1.0 / se**2
    count = {"n": 0}

    def unpack(x):
        return Heterogeneity(float(x[0]), float(np.exp(x[1])))

    def objective(x):
        count["n"] += 1
        sim = simulated_ci1(comp, truth, unpack(x), crn)
        return float(np.sum(weights * (sim - values) ** 2))

    x0 = np.array([float(logit(np.clip(values[0], 1e-3, 1 - 1e-3))), np.log(1.0)])
    result = None
    for scale in INITIAL_SIMPLEX_SCALES:
        simplex = np.vstack([x0, x0 + scale * np.eye(2)])
        result = minimize(
            objective, x0, method="Nelder-Mead", options={**NM_OPTIONS, "initial_simplex": simplex}
        )
        x0 = result.x
    het = unpack(result.x)
    sim = simulated_ci1(comp, truth, het, crn)
    _, ok = _gate(sim, values, se)
    return {
        "arm": "logit_normal",
        "het": {"mu": het.mu, "s": het.s},
        "ci1_sim": sim.tolist(),
        "ci1_target": values.tolist(),
        "distance": float(np.sum(weights * (sim - values) ** 2)),
        "feasible": ok,
        "evaluations": count["n"],
        "converged": bool(result.success),
    }


def calibrate_mixture(comp, truth, crn, primary: dict) -> dict:
    """The frozen P8-005 mixture calibrator: three moment-matched starts from the primary
    solution at the same dynamics, transformed Nelder-Mead, frozen start selection."""
    from complete_journey.abm import mixture as M
    from complete_journey.abm.architectures import Heterogeneity
    from complete_journey.abm.calibration import INITIAL_SIMPLEX_SCALES, NM_OPTIONS

    values, se = _real_target()
    weights = 1.0 / se**2
    fits = []
    for st in M.starts(Heterogeneity(primary["het"]["mu"], primary["het"]["s"])):
        count = {"n": 0}

        def objective(u, count=count):
            count["n"] += 1
            sim = simulated_ci1(comp, truth, M.from_unconstrained(u), crn)
            return float(np.sum(weights * (sim - values) ** 2))

        x0 = M.to_unconstrained(st)
        result = None
        for scale in INITIAL_SIMPLEX_SCALES:
            simplex = np.vstack([x0, x0 + scale * np.eye(4)])
            result = minimize(
                objective,
                x0,
                method="Nelder-Mead",
                options={**NM_OPTIONS, "initial_simplex": simplex},
            )
            x0 = result.x
        het = M.from_unconstrained(result.x)
        sim = simulated_ci1(comp, truth, het, crn)
        _, ok = M._gate(sim, values, se)
        fits.append(
            {
                "het": het.as_dict(),
                "distance": float(np.sum(weights * (sim - values) ** 2)),
                "ci1_sim": sim.tolist(),
                "feasible": ok,
                "evaluations": count["n"],
                "converged": bool(result.success),
            }
        )
    order = sorted(range(len(fits)), key=lambda j: (fits[j]["distance"], j))
    best, rule = order[0], "lowest objective"
    if not fits[best]["feasible"]:
        passing = [j for j in order if fits[j]["feasible"]]
        if passing:
            best, rule = passing[0], "lowest objective among passing starts"
        else:
            rule = "no start passes the gate"
    chosen = fits[best]
    return {
        "arm": "mixture",
        "het": chosen["het"],
        "ci1_sim": chosen["ci1_sim"],
        "ci1_target": values.tolist(),
        "distance": chosen["distance"],
        "feasible": chosen["feasible"],
        "selected_start": best,
        "selection_rule": rule,
        "fits": fits,
    }


def het_from(cal: dict):
    from complete_journey.abm.architectures import Heterogeneity, MixtureHeterogeneity

    h = cal["het"]
    if cal["arm"] == "logit_normal":
        return Heterogeneity(h["mu"], h["s"])
    return MixtureHeterogeneity(h["pi"], h["mu1"], h["d"], h["s"])
