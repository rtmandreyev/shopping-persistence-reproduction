r"""Model-adequacy rule, tolerances, κ* and verdicts (design §S–§U, §AE; P8-003).

This is a project-specific, predeclared rule, not a standard econometric test.

**Population object (P8-003).** Δ_j(θ) = E_θ[m_j] − E_P[m_j], where both
expectations refer to the same estimator, N/T design, purchase-envelope operator and
measurement pipeline.
- A point is adequate iff |Δ_j(θ)| < τ_j for every primary moment j.
- A family is adequate iff some admissible interior point is.

**Tolerance (P8-003).** τ_j(κ) = max(κ·|E_j|, ρ_j), where:
- E_j = m_j^real − μ_j^A1;
- ρ_j = c·SE_j is the **design-resolution floor**;
- c = (z_{1−α} + Φ⁻¹((1 + P^{1/J})/2))·sqrt(1 + (g⁺·(1+ε_s))²/R_em)/(1 − f_e).

The floor is not a substantive indifference threshold. It is the tightest margin at
which the procedure can certify equivalence while meeting the perfect-model power P
under the stated approximations. A moment is RESOLUTION-LIMITED at κ when
ρ_j > κ·|E_j|.

**Uncertainty (P8-003).** The empirical and model terms are kept separate:
- σ±_j = sqrt(SE_j² + (g±·s_j(θ))²/R).
- SE_j is the empirical sampling SE: the archived bootstrap SE for the real panel, or
  the truth-point across-panel SD for RQ2 panels.
- s_j(θ) is the model-panel sampling SD.
- g± = (1.25, 1/1.2) under the emulator, and 1 under direct simulation. They act on
  the Monte Carlo term only.
- e_j is the validated emulator mean-error bound (0 under direct simulation).

**Rules:**
- **EQ_j:** |d_j| + e_j + z_{1−α}·σ⁺_j ≤ τ_j(κ). ADEQ = EQ_j for every j
  (intersection–union).
- **OUT_j:** |d_j| − e_j − z*·σ⁻_j > τ_j(κ), with z* = Φ⁻¹(1 − α/(2J)). INADEQ = OUT_j
  for some j (two-tailed Bonferroni).
- **Otherwise** NEITHER. The two conditions cannot overlap.

**κ\* (P8-003).** With U_j = |d_j| + e_j + z_{1−α}·σ⁺_j:
- κ*_j = 0 (RESOLUTION-LIMITED) if U_j ≤ ρ_j;
- otherwise κ*_j = U_j/|E_j| (+∞ if E_j = 0).

Then κ*(θ) = max_j κ*_j and κ*_F = min over θ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from complete_journey.abm import config
from complete_journey.abm.config import require
from complete_journey.abm.guards import TargetVector, require_target_allowed

ADEQ, INADEQ, NEITHER = "ADEQ", "INADEQ", "NEITHER"
ADEQUATE, INADEQUATE, INCONCLUSIVE = "ADEQUATE", "INADEQUATE", "INCONCLUSIVE"
HISTORY_FAVOURED = "HISTORY_FAVOURED"
LATENT_FAVOURED = "LATENT_FAVOURED"
EQUIVALENT = "EQUIVALENT"
NEITHER_FAMILY = "NEITHER"
UNRESOLVED = "UNRESOLVED"
RESOLUTION_QUALIFIER = "WITHIN DESIGN RESOLUTION"


def z_equivalence(alpha: float = config.ALPHA) -> float:
    return float(norm.ppf(1.0 - alpha))


def z_inadequacy(j: int = config.J_PRIMARY, alpha: float = config.ALPHA) -> float:
    """Two-tailed Bonferroni critical value Φ⁻¹(1 − α/(2J)); J = 5 gives 2.5758."""
    require(j >= 1, "J must be positive")
    return float(norm.ppf(1.0 - alpha / (2.0 * j)))


def per_moment_power_quantile(j: int = config.J_PRIMARY) -> float:
    """q_J = Φ⁻¹((1 + P^{1/J})/2): per-moment margin giving joint power ≥ P (Šidák)."""
    p = config.GATE_TRUTH_INCLUSION_MIN ** (1.0 / j)
    return float(norm.ppf((1.0 + p) / 2.0))


def resolution_constant(j: int = config.J_PRIMARY, alpha: float = config.ALPHA) -> float:
    """P8-003 design-resolution multiplier c (≈ 4.16 for J = 5)."""
    inflation = math.sqrt(
        1.0 + (config.G_PLUS * (1.0 + config.EMULATOR_SD_TOLERANCE)) ** 2 / config.R_EMULATOR
    )
    return (
        (z_equivalence(alpha) + per_moment_power_quantile(j))
        * inflation
        / (1.0 - config.EMULATOR_MEAN_ERROR_FRACTION)
    )


def truth_pass_probability(ratio: float, sd_inflation: float = 1.0) -> float:
    """P(|N(0, σ²)| ≤ τ − z_eq·σ·k) at Δ = 0 with τ = ratio·σ: 2Φ(ratio − z_eq·k) − 1."""
    margin = ratio - z_equivalence() * sd_inflation
    return max(0.0, 2.0 * float(norm.cdf(margin)) - 1.0)


@dataclass(frozen=True)
class Tolerances:
    """Per-moment explanandum E_j and design-resolution floor ρ_j (P8-003)."""

    explanandum: dict[str, float]
    rho: dict[str, float]

    def __post_init__(self):
        require(tuple(self.explanandum) == config.PRIMARY_IDS, "tolerances need P1..P5")
        require(tuple(self.rho) == config.PRIMARY_IDS, "resolution floors need P1..P5")
        require(all(v >= 0 for v in self.rho.values()), "floors must be non-negative")

    def tau(self, kappa: float) -> np.ndarray:
        require(kappa > 0, "κ must be positive")
        e = np.array([abs(self.explanandum[j]) for j in config.PRIMARY_IDS])
        r = np.array([self.rho[j] for j in config.PRIMARY_IDS])
        return np.maximum(kappa * e, r)

    def resolution_limited(self, kappa: float) -> tuple[str, ...]:
        """Moments with ρ_j > κ·|E_j| (the floor binds at κ)."""
        return tuple(
            j for j in config.PRIMARY_IDS if self.rho[j] > kappa * abs(self.explanandum[j])
        )

    def kappa_res(self) -> dict[str, float]:
        """κ_res,j = ρ_j/|E_j| (+∞ if E_j = 0): the smallest κ the design resolves."""
        return {
            j: (self.rho[j] / abs(self.explanandum[j]) if self.explanandum[j] != 0 else math.inf)
            for j in config.PRIMARY_IDS
        }


def resolution_floor(se: float, j: int = config.J_PRIMARY) -> float:
    require(se >= 0, "SE must be non-negative")
    return resolution_constant(j) * se


def tolerance(kappa: float, m_ref: float, mu_a1: float, se: float) -> float:
    """General τ(κ) = max(κ·|m_ref − μ_A1|, c·SE) (P8-003)."""
    require(kappa > 0, "κ must be positive")
    return max(kappa * abs(m_ref - mu_a1), resolution_floor(se))


def build_tolerances(reference: TargetVector, a1_means: dict[str, float]) -> Tolerances:
    """Tolerances from a reference target vector (P1..P5 with SEs) and A1 means.

    In Run B the reference must be synthetic; real references need Stage 1 (guard).
    """
    require_target_allowed(reference, "tolerance")
    require(reference.se is not None, "tolerance construction needs reference SEs")
    require(tuple(reference.values) == config.PRIMARY_IDS, "reference must be P1..P5")
    explanandum = {j: reference.values[j] - a1_means[j] for j in config.PRIMARY_IDS}
    rho = {j: resolution_floor(reference.se[j]) for j in config.PRIMARY_IDS}
    return Tolerances(explanandum, rho)


def _primary(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    require(arr.shape == (config.J_PRIMARY,), f"{name} must hold P1..P5")
    require(np.isfinite(arr).all(), f"{name} must be finite")
    return arr


@dataclass(frozen=True)
class Evidence:
    """Discrepancy evidence at one point for P1..P5 (P8-003 variance decomposition).

    - d: data-minus-simulation discrepancy;
    - se: empirical / truth-panel SE;
    - s: model-panel SD;
    - replicates: R;
    - e: emulator mean-error bound (0 for direct simulation).
    """

    d: np.ndarray
    se: np.ndarray
    s: np.ndarray
    replicates: int
    e: np.ndarray = field(default_factory=lambda: np.zeros(config.J_PRIMARY))
    emulator: bool = False

    def __post_init__(self):
        for name in ("d", "se", "s", "e"):
            object.__setattr__(self, name, _primary(name, getattr(self, name)))
        require(self.replicates >= 1, "R must be positive")
        require((self.se >= 0).all() and (self.s >= 0).all(), "SDs must be non-negative")
        require((self.e >= 0).all(), "emulator error bound must be non-negative")
        if not self.emulator:
            require((self.e == 0).all(), "direct simulation carries no emulator error")

    @property
    def g_plus(self) -> float:
        return config.G_PLUS if self.emulator else 1.0

    @property
    def g_minus(self) -> float:
        return config.G_MINUS if self.emulator else 1.0

    def sigma(self, g: float) -> np.ndarray:
        """sqrt(SE² + (g·s)²/R): g scales the Monte Carlo term only."""
        return np.sqrt(self.se**2 + (g * self.s) ** 2 / self.replicates)

    @property
    def sigma_plus(self) -> np.ndarray:
        return self.sigma(self.g_plus)

    @property
    def sigma_minus(self) -> np.ndarray:
        return self.sigma(self.g_minus)

    def upper(self) -> np.ndarray:
        """U_j = |d_j| + e_j + z_eq·σ⁺_j (95% upper confidence bound on |Δ_j|)."""
        return np.abs(self.d) + self.e + z_equivalence() * self.sigma_plus

    def lower(self, j: int = config.J_PRIMARY) -> np.ndarray:
        return np.abs(self.d) - self.e - z_inadequacy(j) * self.sigma_minus


def evidence_from_simulation(
    data: dict[str, float],
    data_se: dict[str, float],
    sim_means: np.ndarray,
    sim_sds: np.ndarray,
    replicates: int,
    emulator_error: np.ndarray | None = None,
) -> Evidence:
    """Evidence from the panel's moments and empirical SEs plus R model replicates."""
    require(replicates >= 2, "need at least two replicates for a simulated SD")
    require(set(data) == set(config.PRIMARY_IDS), "only the primary moments P1..P5 have veto")
    require(set(data_se) == set(config.PRIMARY_IDS), "empirical SEs needed for P1..P5")
    m = np.array([data[j] for j in config.PRIMARY_IDS])
    se = np.array([data_se[j] for j in config.PRIMARY_IDS])
    d = m - np.asarray(sim_means, float)
    if emulator_error is None:
        return Evidence(d, se, np.asarray(sim_sds, float), replicates)
    return Evidence(d, se, np.asarray(sim_sds, float), replicates, np.asarray(emulator_error), True)


def point_status(ev: Evidence, tau: np.ndarray) -> str:
    tau = np.asarray(tau, float)
    require(tau.shape == (config.J_PRIMARY,) and (tau >= 0).all(), "τ must hold P1..P5")
    if (ev.upper() <= tau).all():
        return ADEQ
    if (ev.lower() > tau).any():
        return INADEQ
    return NEITHER


def kappa_star_detail(ev: Evidence, tol: Tolerances) -> tuple[float, tuple[str, ...]]:
    """(κ*(θ), moments whose κ*_j = 0 because U_j ≤ ρ_j: resolution-limited)."""
    upper = ev.upper()
    values, limited = [], []
    for k, j in enumerate(config.PRIMARY_IDS):
        if upper[k] <= tol.rho[j]:
            values.append(0.0)
            limited.append(j)
        elif tol.explanandum[j] == 0:
            values.append(math.inf)
        else:
            values.append(upper[k] / abs(tol.explanandum[j]))
    return max(values), tuple(limited)


def kappa_star_point(ev: Evidence, tol: Tolerances) -> float:
    return kappa_star_detail(ev, tol)[0]


def point_discrepancy_ratio(ev: Evidence, tol: Tolerances) -> float:
    """Descriptive max_j |d_j|/|E_j| over moments with E_j ≠ 0 (nan if none)."""
    ratios = [
        abs(ev.d[k]) / abs(tol.explanandum[j])
        for k, j in enumerate(config.PRIMARY_IDS)
        if tol.explanandum[j] != 0
    ]
    return max(ratios) if ratios else math.nan


@dataclass(frozen=True)
class PointOutcome:
    coord: tuple[float, float]
    feasible: bool
    boundary: bool
    status: dict[float, str]  # κ -> ADEQ / INADEQ / NEITHER (empty if infeasible)
    kappa_star: float
    point_ratio: float = math.nan


@dataclass(frozen=True)
class FamilyVerdict:
    family: str
    role: str  # "primary" or "sensitivity"
    kappa: float
    verdict: str
    boundary_dependent: bool
    kappa_star: float
    argmin: tuple[float, float] | None
    resolution_limited: tuple[str, ...] = ()
    point_summary: float = math.nan

    @property
    def within_design_resolution(self) -> bool:
        """An ADEQUATE verdict relying on a binding resolution floor (P8-003)."""
        return self.verdict == ADEQUATE and bool(self.resolution_limited)

    @property
    def label(self) -> str:
        if self.within_design_resolution:
            return f"{ADEQUATE} {RESOLUTION_QUALIFIER}"
        return self.verdict


def family_verdict(
    family: str,
    role: str,
    outcomes: list[PointOutcome],
    kappa: float,
    tolerances: Tolerances | None = None,
) -> FamilyVerdict:
    """ADEQUATE (possibly BOUNDARY-DEPENDENT and/or WITHIN DESIGN RESOLUTION),
    INADEQUATE or INCONCLUSIVE (§S.4, §U, P8-003)."""
    require(role in ("primary", "sensitivity"), "family role must be primary or sensitivity")
    limited = tolerances.resolution_limited(kappa) if tolerances is not None else ()
    feasible = [o for o in outcomes if o.feasible]
    finite = [o for o in feasible if math.isfinite(o.kappa_star)]
    best = min(finite, key=lambda o: (o.kappa_star, o.coord)) if finite else None
    ks = best.kappa_star if best else math.inf
    argmin = best.coord if best else None
    ratios = [o.point_ratio for o in feasible if not math.isnan(o.point_ratio)]
    summary = min(ratios) if ratios else math.nan
    adequate = [o for o in feasible if o.status[kappa] == ADEQ]
    if adequate:
        boundary_only = all(o.boundary for o in adequate)
        return FamilyVerdict(
            family, role, kappa, ADEQUATE, boundary_only, ks, argmin, limited, summary
        )
    if not feasible or all(o.status[kappa] == INADEQ for o in feasible):
        return FamilyVerdict(family, role, kappa, INADEQUATE, False, ks, argmin, limited, summary)
    return FamilyVerdict(family, role, kappa, INCONCLUSIVE, False, ks, argmin, limited, summary)


@dataclass(frozen=True)
class Comparative:
    verdict: str
    a2_boundary_dependent: bool
    a1_prime_boundary_dependent: bool
    within_design_resolution: bool = False

    @property
    def label(self) -> str:
        """Never unqualified equivalence when a resolution floor binds (P8-003)."""
        if self.within_design_resolution and self.verdict in (
            EQUIVALENT,
            HISTORY_FAVOURED,
            LATENT_FAVOURED,
        ):
            return f"{self.verdict} {RESOLUTION_QUALIFIER}"
        return self.verdict


def comparative_verdict(a2: FamilyVerdict, a1_prime: FamilyVerdict) -> Comparative:
    """Exact §AE table. ADEQUATE (BOUNDARY-DEPENDENT) maps as ADEQUATE; the qualifiers
    carry through, including P8-003 WITHIN DESIGN RESOLUTION."""
    require(a2.role == "primary" and a1_prime.role == "primary", "primary families only")
    require(a2.family == "A2" and a1_prime.family == "A1_PRIME", "comparison is A2 vs A1'")
    require(a2.kappa == a1_prime.kappa, "verdicts must share κ")
    pair = (a2.verdict, a1_prime.verdict)
    table = {
        (ADEQUATE, INADEQUATE): HISTORY_FAVOURED,
        (INADEQUATE, ADEQUATE): LATENT_FAVOURED,
        (ADEQUATE, ADEQUATE): EQUIVALENT,
        (INADEQUATE, INADEQUATE): NEITHER_FAMILY,
    }
    verdict = table.get(pair, UNRESOLVED)
    resolution = a2.within_design_resolution or a1_prime.within_design_resolution
    return Comparative(verdict, a2.boundary_dependent, a1_prime.boundary_dependent, resolution)


def kappa_robustness(verdicts: dict[float, Comparative]) -> str:
    require(set(verdicts) == set(config.KAPPAS), "robustness needs all three κ")
    return "KAPPA_STABLE" if len({v.verdict for v in verdicts.values()}) == 1 else "KAPPA_DEPENDENT"


def operator_robustness(primary: Comparative, s_a: Comparative) -> str:
    return "OPERATOR_ROBUST" if primary.verdict == s_a.verdict else "OPERATOR_DEPENDENT"
