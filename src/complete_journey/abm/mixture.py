"""P8-005 heterogeneity robustness bracket: the frozen mixture calibrator (§P8-005.8).

Layer: alpha_i ~ pi·N(mu_1, s²) + (1 − pi)·N(mu_2, s²), mu_1 < mu_2, simulated as
alpha_i = mu_1 + d·B_i + s·xi_i (``architectures.MixtureHeterogeneity``).

**Frozen specification** (fixed in P8-005 before any implementation; none of it
comes from Stage 1 outcomes):
- bounds: pi ∈ [0.10, 0.90]; d = mu_2 − mu_1 ∈ [0.25, 6]; mu_1, mu_2 ∈ [−5, 5],
  so mu_1 ∈ [−5, 4.75]; s ∈ [0.05, 3];
- transforms onto the open interior:
  pi = 0.10 + 0.80·expit(u1); mu_1 = −5 + 9.75·expit(u2);
  d = 0.25 + (min(6, 5 − mu_1) − 0.25)·expit(u3); s = 0.05 + 2.95·expit(u4);
- Nelder–Mead with the primary calibrator's options (wide simplex 0.5, then a narrow
  restart of 0.1 in the transformed coordinates; xatol 1e-4, fatol 1e-10, maxiter
  400);
- three deterministic moment-matched starts from the primary solution (mu_hat, s_hat)
  at the same theta, projected into the interior with margin 0.01;
- the frozen CI-1 objective (diagonal inverse archived bootstrap variances,
  CALIBRATION-role common random numbers, R_cal = 20) and the identical 0.02 gate.

**Failure rules.** No start passes the gate → INFEASIBLE (recorded, never replaced).
A passing fit within 1e-3 of a bound is flagged MIXTURE-BOUNDARY (disclosure only).

**Start selection** (mechanical reconciliation of the two frozen sentences "the
lowest objective wins" and "no start passes the gate → INFEASIBLE"): the
lowest-objective start is selected if it passes the gate; otherwise the
lowest-objective passing start is selected, and the record says so. With no passing
start the lowest-objective fit is reported as INFEASIBLE.

The mixture parameters are nuisance parameters. They are never interpreted
behaviourally. The identification diagnostics here are descriptive and never change
CI-1 feasibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from complete_journey.abm import config
from complete_journey.abm.architectures import Heterogeneity, MixtureHeterogeneity, Spec
from complete_journey.abm.calibration import (
    INITIAL_SIMPLEX_SCALES,
    NM_OPTIONS,
    simulated_ci1,
    target_arrays,
)
from complete_journey.abm.config import require
from complete_journey.abm.guards import TargetVector, require_run_stage, require_target_allowed
from complete_journey.abm.seeds import Family

PI_LO, PI_HI = MixtureHeterogeneity.BOUNDS["pi"]
MU_LO, MU_HI = MixtureHeterogeneity.BOUNDS["mu"]
D_LO, D_HI = MixtureHeterogeneity.BOUNDS["d"]
S_LO, S_HI = MixtureHeterogeneity.BOUNDS["s"]
MU1_HI = MU_HI - D_LO  # 4.75: mu_2 <= 5 with d >= 0.25
START_PIS = (0.25, 0.50, 0.75)
START_MARGIN = 0.01  # natural units
BOUNDARY_FLAG_TOL = 1e-3
INFEASIBLE = "INFEASIBLE"
MIXTURE_BOUNDARY = "MIXTURE-BOUNDARY"
UNIMODAL = "UNIMODAL (d <= 2s)"  # mathematical property of the equal-spread mixture
BRACKET_UNAVAILABLE = "HETEROGENEITY BRACKET UNAVAILABLE"
#: Descriptive identification diagnostics (never gates): central-difference steps in
#: natural units and the pi grid of the profile.
JACOBIAN_STEPS = {"pi": 0.02, "mu1": 0.05, "d": 0.05, "s": 0.05}
PI_PROFILE_GRID = tuple(round(0.10 + 0.10 * k, 2) for k in range(9))  # 0.10 ... 0.90


# --- transforms -------------------------------------------------------------------------


def d_max(mu1: float) -> float:
    return min(D_HI, MU_HI - mu1)


def from_unconstrained(u: np.ndarray) -> MixtureHeterogeneity:
    u = np.asarray(u, float)
    require(u.shape == (4,) and np.isfinite(u).all(), "mixture coordinates must be 4 finite")
    pi = PI_LO + (PI_HI - PI_LO) * float(expit(u[0]))
    mu1 = MU_LO + (MU1_HI - MU_LO) * float(expit(u[1]))
    d = D_LO + (d_max(mu1) - D_LO) * float(expit(u[2]))
    s = S_LO + (S_HI - S_LO) * float(expit(u[3]))
    return MixtureHeterogeneity(pi, mu1, d, s)


def to_unconstrained(het: MixtureHeterogeneity, clip: bool = False) -> np.ndarray:
    """Inverse transform. Starts must be strictly interior; ``clip`` (diagnostics only)
    nudges a fit lying exactly on a bound inside by 1e-12 of the range."""

    def inner(x, lo, hi):
        frac = (x - lo) / (hi - lo)
        if clip:
            frac = min(max(frac, 1e-12), 1.0 - 1e-12)
        require(0.0 < frac < 1.0, "mixture start must lie strictly inside the bounds")
        return float(logit(frac))

    return np.array(
        [
            inner(het.pi, PI_LO, PI_HI),
            inner(het.mu1, MU_LO, MU1_HI),
            inner(het.d, D_LO, d_max(het.mu1)),
            inner(het.s, S_LO, S_HI),
        ]
    )


def starts(primary: Heterogeneity) -> tuple[MixtureHeterogeneity, ...]:
    """Three moment-matched starts from the primary (mu_hat, s_hat), in frozen order.

    For pi ∈ {0.25, 0.50, 0.75}: s = s_hat/2 and d = s_hat·sqrt(0.75/(pi(1 − pi))), so
    s² + pi(1 − pi)d² = s_hat²; mu_1 = mu_hat − (1 − pi)·d, so the mixture mean is mu_hat.
    Projection into the interior, in the frozen order, with margin 0.01: s, d, mu_1,
    then d <= 5 − mu_1 − 0.01. (Mechanical corner: if mu_1 sits so high that d's upper
    limit falls below its floor, mu_1 is lowered just enough to leave d an interior
    point.)
    """
    require(math.isfinite(primary.mu) and primary.s >= 0, "primary solution must be finite")
    e = START_MARGIN
    out = []
    for pi in START_PIS:
        s = float(np.clip(primary.s / 2.0, S_LO + e, S_HI - e))
        d = primary.s * math.sqrt(0.75 / (pi * (1.0 - pi)))
        d = float(np.clip(d, D_LO + e, D_HI - e))
        mu1 = float(np.clip(primary.mu - (1.0 - pi) * d, MU_LO + e, MU1_HI - e))
        if MU_HI - mu1 - e < D_LO + e:
            mu1 = MU_HI - D_LO - 2.0 * e - 1e-9
        d = min(d, MU_HI - mu1 - e)
        out.append(MixtureHeterogeneity(pi, mu1, d, s))
    return tuple(out)


def boundary_parameters(het: MixtureHeterogeneity, tol: float = BOUNDARY_FLAG_TOL) -> tuple:
    """Names of the bounds the fit lies within ``tol`` of (natural units)."""
    near = []
    checks = (
        ("pi_lo", het.pi - PI_LO),
        ("pi_hi", PI_HI - het.pi),
        ("mu1_lo", het.mu1 - MU_LO),
        ("mu1_hi", MU1_HI - het.mu1),
        ("d_lo", het.d - D_LO),
        ("d_hi", d_max(het.mu1) - het.d),
        ("mu2_hi", MU_HI - het.mu2),
        ("s_lo", het.s - S_LO),
        ("s_hi", S_HI - het.s),
    )
    for name, gap in checks:
        if gap <= tol:
            near.append(name)
    return tuple(near)


# --- calibration ---------------------------------------------------------------------


@dataclass(frozen=True)
class StartFit:
    start: MixtureHeterogeneity
    heterogeneity: MixtureHeterogeneity
    distance: float
    residuals: np.ndarray
    feasible: bool
    evaluations: int
    converged: bool


@dataclass(frozen=True)
class MixtureCalibration:
    spec: Spec
    heterogeneity: MixtureHeterogeneity
    ci1_sim: np.ndarray
    ci1_target: np.ndarray
    residuals: np.ndarray
    gate_bounds: np.ndarray
    distance: float
    feasible: bool
    selected_start: int
    selection_rule: str
    fits: tuple[StartFit, ...]
    flags: tuple[str, ...] = field(default=())

    @property
    def evaluations(self) -> int:
        return sum(f.evaluations for f in self.fits)

    @property
    def max_abs_residual(self) -> float:
        return float(np.max(np.abs(self.residuals)))


def _gate(sim: np.ndarray, values: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, bool]:
    bounds = np.maximum(config.CI1_GATE_ABS, 2.0 * se)
    return bounds, bool(np.all(np.abs(sim - values) <= bounds))


def calibrate_mixture(
    spec: Spec,
    target: TargetVector,
    stage: int,
    primary: Heterogeneity,
    n: int = config.N_HOUSEHOLDS,
    r_cal: int = config.R_CAL,
    availability: np.ndarray | None = None,
    periodic_amplitude: float = 0.0,
) -> MixtureCalibration:
    """Fit (pi, mu_1, d, s) to the target CI-1 with the dynamics held fixed."""
    require_run_stage(stage)
    require_target_allowed(target, "ci1_calibration")
    require(spec.family != Family.A0, "A0 is homogeneous; the mixture layer does not apply")
    values, se = target_arrays(target)
    weights = 1.0 / se**2

    fits = []
    for st in starts(primary):
        count = {"n": 0}

        def objective(u, count=count):
            count["n"] += 1
            sim = simulated_ci1(
                spec, from_unconstrained(u), stage, n, r_cal, availability, periodic_amplitude
            )
            return float(np.sum(weights * (sim - values) ** 2))

        x0 = to_unconstrained(st)
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
        het = from_unconstrained(result.x)
        sim = simulated_ci1(spec, het, stage, n, r_cal, availability, periodic_amplitude)
        _, ok = _gate(sim, values, se)
        fits.append(
            StartFit(
                st,
                het,
                float(np.sum(weights * (sim - values) ** 2)),
                sim - values,
                ok,
                count["n"],
                bool(result.success),
            )
        )

    order = sorted(range(len(fits)), key=lambda k: (fits[k].distance, k))
    best = order[0]
    rule = "lowest objective"
    if not fits[best].feasible:
        passing = [k for k in order if fits[k].feasible]
        if passing:
            best, rule = passing[0], "lowest objective among passing starts"
        else:
            rule = "no start passes the gate"
    chosen = fits[best]
    sim = chosen.residuals + values
    bounds, feasible = _gate(sim, values, se)
    flags = []
    if not feasible:
        flags.append(INFEASIBLE)
    elif boundary_parameters(chosen.heterogeneity):
        flags.append(MIXTURE_BOUNDARY)
    if chosen.heterogeneity.d <= 2.0 * chosen.heterogeneity.s:
        flags.append(UNIMODAL)
    return MixtureCalibration(
        spec=spec,
        heterogeneity=chosen.heterogeneity,
        ci1_sim=sim,
        ci1_target=values,
        residuals=chosen.residuals,
        gate_bounds=bounds,
        distance=chosen.distance,
        feasible=feasible,
        selected_start=best,
        selection_rule=rule,
        fits=tuple(fits),
        flags=tuple(flags),
    )


# --- descriptive identification diagnostics ------------------------------------------


def _shifted(het: MixtureHeterogeneity, name: str, delta: float) -> MixtureHeterogeneity:
    vals = {"pi": het.pi, "mu1": het.mu1, "d": het.d, "s": het.s}
    vals[name] += delta
    return MixtureHeterogeneity(vals["pi"], vals["mu1"], vals["d"], vals["s"])


def _feasible_step(het: MixtureHeterogeneity, name: str, step: float) -> tuple[float, float]:
    """(down, up) step sizes that keep both shifted points inside the bounds."""
    lo_hi = {
        "pi": (het.pi - PI_LO, PI_HI - het.pi),
        "mu1": (het.mu1 - MU_LO, min(MU1_HI - het.mu1, MU_HI - het.mu2)),
        "d": (het.d - D_LO, d_max(het.mu1) - het.d),
        "s": (het.s - S_LO, S_HI - het.s),
    }[name]
    return min(step, lo_hi[0]), min(step, lo_hi[1])


def ci1_jacobian(
    spec: Spec,
    het: MixtureHeterogeneity,
    stage: int,
    n: int = config.N_HOUSEHOLDS,
    r_cal: int = config.R_CAL,
) -> np.ndarray:
    """(6, 4) SE-scaled finite-difference Jacobian of simulated CI-1 in (pi, mu1, d, s).

    Central differences where both sides fit inside the bounds, one-sided otherwise,
    with the CALIBRATION-role common random numbers of the objective.
    """
    se = np.array([config.CI1_BOOT_SE[c] for c in config.CI1_COMPONENTS])
    base = simulated_ci1(spec, het, stage, n, r_cal)
    cols = []
    for name in ("pi", "mu1", "d", "s"):
        down, up = _feasible_step(het, name, JACOBIAN_STEPS[name])
        hi = simulated_ci1(spec, _shifted(het, name, up), stage, n, r_cal) if up > 0 else base
        lo = simulated_ci1(spec, _shifted(het, name, -down), stage, n, r_cal) if down > 0 else base
        width = up + down
        cols.append((hi - lo) / width / se if width > 0 else np.zeros(6))
    return np.column_stack(cols)


def condition_number(jacobian: np.ndarray) -> float:
    sv = np.linalg.svd(np.asarray(jacobian, float), compute_uv=False)
    return float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf


def separation(het: MixtureHeterogeneity) -> dict[str, float]:
    """Descriptive component separation; never interpreted behaviourally."""
    between = het.pi * (1.0 - het.pi) * het.d**2
    return {
        "d": het.d,
        "d_over_s": het.d / het.s,
        "between_share_of_variance": between / het.variance,
        "unimodal": bool(het.d <= 2.0 * het.s),
    }


def start_agreement(cal: MixtureCalibration) -> dict:
    """Spread of the three start solutions (objective and natural parameters)."""
    params = np.array(
        [
            [f.heterogeneity.pi, f.heterogeneity.mu1, f.heterogeneity.d, f.heterogeneity.s]
            for f in cal.fits
        ]
    )
    dist = np.array([f.distance for f in cal.fits])
    return {
        "objectives": dist.tolist(),
        "objective_range": float(dist.max() - dist.min()),
        "param_range": dict(
            zip(("pi", "mu1", "d", "s"), np.ptp(params, axis=0).tolist(), strict=True)
        ),
        "passing_starts": int(sum(f.feasible for f in cal.fits)),
        "converged_starts": int(sum(f.converged for f in cal.fits)),
    }


def diagnostics(
    cal: MixtureCalibration, stage: int, n: int = config.N_HOUSEHOLDS, r_cal: int = config.R_CAL
) -> dict:
    """Per-point descriptive diagnostics: Jacobian condition, separation, bounds, starts."""
    jac = ci1_jacobian(cal.spec, cal.heterogeneity, stage, n, r_cal)
    return {
        "jacobian_condition": condition_number(jac),
        "jacobian_singular_values": np.linalg.svd(jac, compute_uv=False).tolist(),
        "separation": separation(cal.heterogeneity),
        "boundary_parameters": list(boundary_parameters(cal.heterogeneity)),
        "start_agreement": start_agreement(cal),
        "selection_rule": cal.selection_rule,
        "flags": list(cal.flags),
    }


def pi_profile(
    spec: Spec,
    target: TargetVector,
    stage: int,
    best: MixtureHeterogeneity,
    n: int = config.N_HOUSEHOLDS,
    r_cal: int = config.R_CAL,
    grid=PI_PROFILE_GRID,
) -> list[dict]:
    """Profile of the CI-1 objective over pi, re-optimizing (mu_1, d, s) at each pi.

    Descriptive only: discloses the pi–s ridge. Each fixed-pi fit uses the same
    Nelder–Mead options, started from ``best`` with pi replaced.
    """
    require_run_stage(stage)
    require_target_allowed(target, "ci1_calibration")
    values, se = target_arrays(target)
    weights = 1.0 / se**2
    out = []
    for pi in grid:
        pi_c = float(np.clip(pi, PI_LO + 1e-6, PI_HI - 1e-6))
        st = MixtureHeterogeneity(pi_c, best.mu1, best.d, best.s)
        u_full = to_unconstrained(st, clip=True)

        def objective(v, u_full=u_full):
            u = np.array([u_full[0], *v])
            sim = simulated_ci1(spec, from_unconstrained(u), stage, n, r_cal)
            return float(np.sum(weights * (sim - values) ** 2))

        x0 = u_full[1:]
        result = None
        for scale in INITIAL_SIMPLEX_SCALES:
            simplex = np.vstack([x0, x0 + scale * np.eye(3)])
            result = minimize(
                objective,
                x0,
                method="Nelder-Mead",
                options={**NM_OPTIONS, "initial_simplex": simplex},
            )
            x0 = result.x
        het = from_unconstrained(np.array([u_full[0], *result.x]))
        sim = simulated_ci1(spec, het, stage, n, r_cal)
        _, ok = _gate(sim, values, se)
        out.append(
            {
                "pi": pi_c,
                "objective": float(result.fun),
                "gate_pass": ok,
                "mu1": het.mu1,
                "d": het.d,
                "s": het.s,
                "max_abs_residual": float(np.max(np.abs(sim - values))),
            }
        )
    return out
