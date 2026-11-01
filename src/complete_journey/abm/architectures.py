"""Phase 8 household architectures and the behaviour simulator (design §E–§M, §Y).

All rungs share one index form, logit p_it = eta_it, with p_it = Λ(eta_it) and
y_it = 1{U_it < p_it}:

| Rung | eta_it |
|---|---|
| A0 | mu |
| A1 | alpha_i = mu + s·xi_i |
| A1′ | alpha_i + sigma_z·z_it, where z_it = rho·z_i,t−1 + sqrt(1 − rho²)·eps_it |
| A1′-HMM | alpha_i + sigma·s_it, where s_it ∈ {−1, +1} switches with probability q |
| A2 | alpha_i + delta·H_it, where H_i,t+1 = lambda·H_it + (1 − lambda)·y_it |
| A2-recency | alpha_i + delta·lambda^(d_it − 1), with d_it the weeks since the last purchase |

Optional additive layers:
- the shared periodic layer (§M): a·cos(pi·(t − phi_i));
- the calendar component (§Z): tau_t, which is zero in the primary design.

Timing is past-only by construction:
- **A1′ / HMM.** The latent state evolves from its own innovations only. No y
  enters its transition.
- **A2.** p_it uses H_it, which contains purchases up to t−1. y_it is drawn and only
  then updates H_i,t+1. y_it never affects its own p_it.
- **A2-recency.** d_it is computed from purchases strictly before t.

Horizons (§Y):
- **Primary.** Weeks −155 … 102 (156 burn-in weeks), keeping weeks 1–102. A1′ starts
  stationary, A2 starts at H = 0, and A0/A1 use the same horizon.
- **S-a** (§X). Behaviour exists only inside an imposed availability window [a_i, b_i]
  within weeks 1–102. The state is (re)initialized at entry (H = 0, z stationary, HMM
  equiprobable, no prior purchase). There is no burn-in, and y = 0 outside the window.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np
from scipy.special import expit

from complete_journey.abm import config
from complete_journey.abm.config import require
from complete_journey.abm.guards import CalendarPath
from complete_journey.abm.seeds import Draws, Family, SeedKey, Stream, draws


@dataclass(frozen=True)
class Heterogeneity:
    """Common logit-normal heterogeneity layer alpha_i = mu + s·xi_i (design §F)."""

    mu: float
    s: float

    def __post_init__(self):
        require(np.isfinite(self.mu) and np.isfinite(self.s), "heterogeneity must be finite")
        require(self.s >= 0, "heterogeneity SD must be non-negative")


@dataclass(frozen=True)
class MixtureHeterogeneity:
    """P8-005 robustness layer: alpha_i ~ pi·N(mu_1, s²) + (1 − pi)·N(mu_2, s²).

    Simulated as alpha_i = mu_1 + d·B_i + s·xi_i, with d = mu_2 − mu_1 and
    B_i = 1{u_i < 1 − pi} from the shared MIXTURE stream, so B_i ~ Bernoulli(1 − pi)
    marks component 2. The frozen bounds (§P8-005.8) are checked here.
    """

    pi: float
    mu1: float
    d: float
    s: float

    BOUNDS = MappingProxyType(
        {"pi": (0.10, 0.90), "mu": (-5.0, 5.0), "d": (0.25, 6.0), "s": (0.05, 3.0)}
    )

    def __post_init__(self):
        vals = (self.pi, self.mu1, self.d, self.s)
        require(all(np.isfinite(v) for v in vals), "mixture heterogeneity must be finite")
        b, tol = self.BOUNDS, 1e-12
        require(b["pi"][0] - tol <= self.pi <= b["pi"][1] + tol, "pi outside [0.10, 0.90]")
        require(b["d"][0] - tol <= self.d <= b["d"][1] + tol, "d outside [0.25, 6]")
        require(b["s"][0] - tol <= self.s <= b["s"][1] + tol, "s outside [0.05, 3]")
        require(b["mu"][0] - tol <= self.mu1 and self.mu2 <= b["mu"][1] + tol, "mu outside [-5, 5]")

    @property
    def mu2(self) -> float:
        return self.mu1 + self.d

    @property
    def mean(self) -> float:
        return self.mu1 + (1.0 - self.pi) * self.d

    @property
    def variance(self) -> float:
        return self.s**2 + self.pi * (1.0 - self.pi) * self.d**2

    def as_dict(self) -> dict[str, float]:
        return {"pi": self.pi, "mu1": self.mu1, "mu2": self.mu2, "d": self.d, "s": self.s}


def household_alpha(het, xi: np.ndarray, components: np.ndarray | None, family) -> np.ndarray:
    """alpha_i under the primary logit-normal layer or the P8-005 mixture layer."""
    if isinstance(het, MixtureHeterogeneity):
        require(family != Family.A0, "A0 is homogeneous; the mixture layer does not apply")
        require(components is not None, "the mixture layer needs MIXTURE stream uniforms")
        b = (components < 1.0 - het.pi).astype(float)
        return het.mu1 + het.d * b + het.s * xi
    require(isinstance(het, Heterogeneity), "unknown heterogeneity layer")
    s = 0.0 if family == Family.A0 else het.s
    return het.mu + s * xi


@dataclass(frozen=True)
class Spec:
    """Architecture specification. ``role`` is "primary", "sensitivity" or "truth"."""

    family: Family
    role: str
    params: tuple[tuple[str, float], ...] = ()

    def param(self, name: str) -> float:
        return dict(self.params)[name]


def a0() -> Spec:
    return Spec(Family.A0, "primary")


def a1() -> Spec:
    return Spec(Family.A1, "primary")


def a1_prime(rho: float, sigma_z: float) -> Spec:
    require(0.0 <= rho < 1.0, "A1' persistence must lie in [0, 1)")
    require(sigma_z >= 0.0, "A1' amplitude must be non-negative")
    return Spec(Family.A1_PRIME, "primary", (("rho", rho), ("sigma_z", sigma_z)))


def a1_prime_hmm(q: float, sigma: float) -> Spec:
    require(0.0 < q <= 0.5, "HMM switching probability must lie in (0, 0.5]")
    require(sigma >= 0.0, "HMM amplitude must be non-negative")
    return Spec(Family.A1_PRIME_HMM, "sensitivity", (("q", q), ("sigma", sigma)))


def a2(delta: float, lam: float) -> Spec:
    require(delta >= 0.0, "A2 amplitude must be non-negative")
    require(0.0 <= lam < 1.0, "A2 decay must lie in [0, 1)")
    return Spec(Family.A2, "primary", (("delta", delta), ("lam", lam)))


def a2_recency(delta: float, lam: float) -> Spec:
    require(delta >= 0.0, "recency amplitude must be non-negative")
    require(0.0 <= lam < 1.0, "recency decay must lie in [0, 1)")
    return Spec(Family.A2_RECENCY, "sensitivity", (("delta", delta), ("lam", lam)))


def a1_prime_heterogeneous_rho(mean: float, sd: float, sigma_z: float) -> Spec:
    """T2(b) truth only: household-specific persistence rho_i ~ Beta(mean, sd)."""
    require(0 < mean < 1 and 0 < sd and sd**2 < mean * (1 - mean), "invalid Beta mean/sd")
    return Spec(
        Family.A1_PRIME, "truth", (("rho_mean", mean), ("rho_sd", sd), ("sigma_z", sigma_z))
    )


def mixed(rho: float, sigma_z: float, delta: float, lam: float) -> Spec:
    """T4 masking truth only: latent AR state plus genuine geometric history."""
    return Spec(
        Family.A2,
        "truth",
        (("rho", rho), ("sigma_z", sigma_z), ("delta", delta), ("lam", lam)),
    )


def beta_parameters(mean: float, sd: float) -> tuple[float, float]:
    common = mean * (1 - mean) / sd**2 - 1
    return mean * common, (1 - mean) * common


@dataclass
class World:
    """One simulated world (weeks 1–102 retained). Never carries real data."""

    spec: Spec
    heterogeneity: Heterogeneity | MixtureHeterogeneity
    operator: str
    behavior: np.ndarray  # (n, 102) int8, y_it
    alpha: np.ndarray  # (n,)
    p: np.ndarray  # (n, 102) purchase probability used at week t
    history: np.ndarray | None  # (n, 102) A2 stock H_it used at week t
    last_purchase: np.ndarray | None  # (n, 102) recency: last purchase week before t
    latent: np.ndarray | None  # (n, 102) z_it or s_it at week t
    availability: np.ndarray | None  # (n, 2) S-a windows [a_i, b_i]
    seed_key: SeedKey
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.behavior.shape[0]


def _periodic(amplitude: float, phases: np.ndarray, week: int) -> np.ndarray:
    """a·cos(pi·(t − phi_i)) for integer t and phi_i ∈ {0, 1}: exactly ±a."""
    sign = np.where(np.mod(week - phases, 2) == 0, 1.0, -1.0)
    return amplitude * sign


def simulate(
    spec: Spec,
    het: Heterogeneity,
    key: SeedKey,
    n: int,
    availability: np.ndarray | None = None,
    periodic_amplitude: float = 0.0,
    calendar: CalendarPath | None = None,
    fixed_draws: Draws | None = None,
) -> World:
    """Generate behaviour for one world under the primary horizon or the S-a bracket.

    ``fixed_draws`` replaces the keyed draws. It exists for counterfactual tests and
    interventions that hold every random stream fixed while forcing a realized purchase.
    """
    require(spec.role in ("primary", "sensitivity", "truth"), "unknown spec role")
    require(periodic_amplitude >= 0.0, "periodic amplitude must be non-negative")
    if periodic_amplitude > 0:
        require(
            spec.family in (Family.A1_PRIME, Family.A2),
            "the shared periodic layer attaches to A1' and A2 only",
        )
    if calendar is not None:
        require(isinstance(calendar, CalendarPath), "calendar must be a guarded CalendarPath")
        calendar = np.asarray(calendar.values, dtype=float)
    params = dict(spec.params)
    family = spec.family
    key = replace(key, family=family)
    mixture = isinstance(het, MixtureHeterogeneity)
    d: Draws = (
        fixed_draws
        if fixed_draws is not None
        else draws(key, n, family, periodic=periodic_amplitude > 0, mixture=mixture)
    )
    require(d.uniforms.shape == (n, config.TOTAL_SIM_WEEKS), "draws must cover the horizon")
    components = d.mixture
    if mixture and components is None:
        components = key.generator(Stream.MIXTURE).random(n)
    alpha = household_alpha(het, d.xi, components, family)
    het_rho = None
    if "rho_mean" in params:
        a, b = beta_parameters(params["rho_mean"], params["rho_sd"])
        het_rho = key.generator(Stream.HET_PERSISTENCE).beta(a, b, n)
        het_rho = np.minimum(het_rho, config.PERSISTENCE_CAP)
    sa = availability is not None
    if sa:
        availability = np.asarray(availability, dtype=int)
        require(availability.shape == (n, 2), "availability must be (n, 2)")
        require(
            (availability[:, 0] >= config.RESPONSE_WEEKS[0] - 1).all()
            and (availability[:, 1] <= config.T_WEEKS).all()
            and (availability[:, 0] <= availability[:, 1]).all(),
            "availability windows must lie inside weeks 1-102",
        )
    first_index = config.BURN_IN_WEEKS if sa else 0  # S-a: no burn-in (entry is the start)
    keep = config.BURN_IN_WEEKS  # index of week 1

    rho = params.get("rho", 0.0) if het_rho is None else het_rho
    sigma_z = params.get("sigma_z", 0.0)
    has_latent = family == Family.A1_PRIME or (spec.role == "truth" and "sigma_z" in params)
    has_hmm = family == Family.A1_PRIME_HMM
    has_stock = family == Family.A2
    has_recency = family == Family.A2_RECENCY
    delta = params.get("delta", 0.0)
    lam = params.get("lam", 0.0)
    if has_latent and d.innovations is None:
        d = Draws(
            d.xi,
            d.uniforms,
            key.generator(Stream.AR_INNOVATION).standard_normal(d.uniforms.shape),
            d.hmm,
            d.phases,
            d.mixture,
        )

    T = config.T_WEEKS
    y_out = np.zeros((n, T), dtype=np.int8)
    p_out = np.zeros((n, T))
    h_out = np.zeros((n, T)) if has_stock else None
    last_out = np.full((n, T), np.nan) if has_recency else None
    z_out = np.zeros((n, T)) if (has_latent or has_hmm) else None

    z = np.zeros(n)
    state = np.ones(n)
    stock = np.zeros(n)  # A2: neutral H = 0 before burn-in (or at S-a entry)
    last = np.full(n, np.nan)  # recency: no prior purchase
    for idx in range(first_index, config.TOTAL_SIM_WEEKS):
        week = config.FIRST_SIM_WEEK + idx
        if has_latent:
            eps = d.innovations[:, idx]
            if idx == first_index:
                z = eps.copy()  # stationary N(0, 1) start
            else:
                z = rho * z + np.sqrt(1.0 - rho**2) * eps
        if has_hmm:
            u = d.hmm[:, idx]
            if idx == first_index:
                state = np.where(u < 0.5, 1.0, -1.0)  # stationary: equiprobable
            else:
                state = np.where(u < params["q"], -state, state)
        if sa:
            entering = availability[:, 0] == week
            active = (availability[:, 0] <= week) & (week <= availability[:, 1])
            if has_latent:
                z = np.where(entering, d.innovations[:, idx], z)
            if has_hmm:
                state = np.where(entering, np.where(d.hmm[:, idx] < 0.5, 1.0, -1.0), state)
            stock = np.where(entering, 0.0, stock)
            last = np.where(entering, np.nan, last)
        else:
            active = np.ones(n, dtype=bool)

        eta = alpha.copy()
        if has_latent:
            eta = eta + sigma_z * z
        if has_hmm:
            eta = eta + params["sigma"] * state
        if has_stock:
            eta = eta + delta * stock  # H_it: purchases up to t-1 only
        if has_recency:
            gap = week - last  # NaN where no prior purchase
            effect = np.where(np.isnan(last), 0.0, delta * np.power(lam, np.nan_to_num(gap) - 1))
            eta = eta + effect
        if periodic_amplitude > 0:
            eta = eta + _periodic(periodic_amplitude, d.phases, week)
        if calendar is not None and week >= config.RESPONSE_WEEKS[0] - 1:
            eta = eta + calendar[week - 1]
        p = expit(eta)
        y = (d.uniforms[:, idx] < p) & active

        if idx >= keep:
            j = idx - keep
            y_out[:, j] = y
            p_out[:, j] = np.where(active, p, 0.0)
            if h_out is not None:
                h_out[:, j] = stock
            if last_out is not None:
                last_out[:, j] = last
            if z_out is not None:
                z_out[:, j] = z if has_latent else state
        # Updates for t+1 happen only after y_it is drawn.
        if has_stock:
            stock = lam * stock + (1.0 - lam) * y
        if has_recency:
            last = np.where(y, float(week), last)

    return World(
        spec=spec,
        heterogeneity=het,
        operator=config.OPERATOR_SA if sa else config.OPERATOR_PRIMARY,
        behavior=y_out,
        alpha=alpha,
        p=p_out,
        history=h_out,
        last_purchase=last_out,
        latent=z_out,
        availability=availability if sa else None,
        seed_key=key,
        meta={"periodic_amplitude": periodic_amplitude, "calendar": calendar is not None},
    )
