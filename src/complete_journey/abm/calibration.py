"""Common heterogeneity calibrator: fit (mu, s) to CI-1 with dynamics held fixed (§F, §P).

The method is simulated minimum distance over the six CI-1 components:
- diagonal weights equal to the inverse archived bootstrap variances;
- CALIBRATION-role common random numbers, R_cal replicates;
- a deterministic Nelder–Mead search over (mu, log s): a wide simplex, then one
  narrow restart. A0 searches mu only.

The architecture's dynamic parameters are never touched.

**Feasibility gate.** Every component satisfies
|CI1_sim − CI1_target| ≤ max(0.02, 2·SE_j). Infeasible points are reported and later
excluded from any acceptance search.

Real CI-1 targets are refused until Stage 1 is authorized (``guards``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import logit

from complete_journey.abm import config
from complete_journey.abm.architectures import Heterogeneity, MixtureHeterogeneity, Spec, simulate
from complete_journey.abm.config import require
from complete_journey.abm.guards import TargetVector, require_run_stage, require_target_allowed
from complete_journey.abm.moments import ci1_fast
from complete_journey.abm.seeds import Draws, Family, Role, SeedKey, draws

NM_OPTIONS = {"xatol": 1e-4, "fatol": 1e-10, "maxiter": 400, "maxfev": 800}
INITIAL_SIMPLEX_SCALES = (0.5, 0.1)


@dataclass(frozen=True)
class CalibrationResult:
    spec: Spec
    heterogeneity: Heterogeneity
    ci1_sim: np.ndarray
    ci1_target: np.ndarray
    residuals: np.ndarray
    gate_bounds: np.ndarray
    distance: float
    feasible: bool
    evaluations: int
    converged: bool


def target_arrays(target: TargetVector) -> tuple[np.ndarray, np.ndarray]:
    require(tuple(target.values) == config.CI1_COMPONENTS, "CI-1 target needs the six components")
    values = np.array([target.values[c] for c in config.CI1_COMPONENTS])
    se_source = target.se if target.se is not None else config.CI1_BOOT_SE
    se = np.array([se_source[c] for c in config.CI1_COMPONENTS])
    require((se > 0).all(), "CI-1 SEs must be positive")
    return values, se


@lru_cache(maxsize=24)
def _calibration_draws(
    stage: int, replicate: int, n: int, family: Family, periodic: bool, mixture: bool
) -> Draws:
    """CALIBRATION-role draws of one replicate. They do not depend on θ or η, so the
    optimizer reuses them across evaluations. ``simulate`` would build exactly these
    draws from the same key, so results are bit-identical (pinned by a test)."""
    key = SeedKey(stage=stage, role=Role.CALIBRATION, replicate=replicate, family=family)
    return draws(key, n, family, periodic=periodic, mixture=mixture)


def simulated_ci1(
    spec: Spec,
    het: Heterogeneity,
    stage: int,
    n: int,
    r_cal: int,
    availability: np.ndarray | None = None,
    periodic_amplitude: float = 0.0,
) -> np.ndarray:
    reps = []
    mixture = isinstance(het, MixtureHeterogeneity)
    for r in range(r_cal):
        key = SeedKey(stage=stage, role=Role.CALIBRATION, replicate=r)
        fixed = _calibration_draws(stage, r, n, spec.family, periodic_amplitude > 0, mixture)
        world = simulate(spec, het, key, n, availability, periodic_amplitude, fixed_draws=fixed)
        reps.append(ci1_fast(world.behavior))
    return np.mean(reps, axis=0)


def calibrate(
    spec: Spec,
    target: TargetVector,
    stage: int,
    n: int = config.N_HOUSEHOLDS,
    r_cal: int = config.R_CAL,
    start: Heterogeneity | None = None,
    availability: np.ndarray | None = None,
    periodic_amplitude: float = 0.0,
) -> CalibrationResult:
    """Fit only (mu, s) of the common layer to the target CI-1; dynamics stay as given."""
    require_run_stage(stage)
    require_target_allowed(target, "ci1_calibration")
    values, se = target_arrays(target)
    weights = 1.0 / se**2
    homogeneous = spec.family == Family.A0
    if start is None:
        start = Heterogeneity(float(logit(np.clip(values[0], 1e-3, 1 - 1e-3))), 1.0)
    count = {"n": 0}

    def unpack(x):
        return Heterogeneity(float(x[0]), 0.0 if homogeneous else float(np.exp(x[1])))

    def objective(x):
        count["n"] += 1
        sim = simulated_ci1(spec, unpack(x), stage, n, r_cal, availability, periodic_amplitude)
        return float(np.sum(weights * (sim - values) ** 2))

    x0 = np.array([start.mu]) if homogeneous else np.array([start.mu, np.log(max(start.s, 1e-3))])
    # Deterministic search: a wide initial simplex, then one restart from the best point.
    # A wide simplex is needed because CRN Bernoulli draws make the objective stepwise.
    result = None
    for scale in INITIAL_SIMPLEX_SCALES:
        simplex = np.vstack([x0, x0 + scale * np.eye(len(x0))])
        result = minimize(
            objective, x0, method="Nelder-Mead", options={**NM_OPTIONS, "initial_simplex": simplex}
        )
        x0 = result.x
    het = unpack(result.x)
    sim = simulated_ci1(spec, het, stage, n, r_cal, availability, periodic_amplitude)
    residuals = sim - values
    bounds = np.maximum(config.CI1_GATE_ABS, 2.0 * se)
    return CalibrationResult(
        spec=spec,
        heterogeneity=het,
        ci1_sim=sim,
        ci1_target=values,
        residuals=residuals,
        gate_bounds=bounds,
        distance=float(np.sum(weights * residuals**2)),
        feasible=bool(np.all(np.abs(residuals) <= bounds)),
        evaluations=count["n"],
        converged=bool(result.success),
    )
