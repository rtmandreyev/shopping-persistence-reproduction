"""Rule-based parameter regions and deterministic refinement (design §V).

Regions are built from P8-001 constants only. This module takes no empirical target,
reads no archive and imports no guard or tolerance code. A test pins that boundary.

- **Persistence axis** (12 points): {0} plus 11 log-spaced half-lives
  h_k = 0.5·50^(k/10) weeks (0.5 … 25). For A1′ the values are rho = 2^(−1/h); for A2
  they are lambda. Index 0 is rho = 0, or lambda = 0 (the one-lag slice).
- **Amplitude axis** (12 points, zero excluded):
  - A1′ and HMM: sigma = 2.5·k/12 (logit scale);
  - A2 and recency: raw delta = 5·k/12 (logit scale; P8-002). Δ_1 and Δ_12 are
    per-point labels computed in Stage 1, never grid coordinates.
- **Boundary flag.** True on the design-cap faces: persistence index 11 (the 25-week
  half-life) or amplitude index 11 (sigma = 2.5 or delta = 5).
- **Refinement.** A coarse cell is refined (edge midpoints plus centre) if its corners
  have mixed ADEQ status at any κ, or if the κ* argmin is one of its corners. At most
  two levels and 144 added points per family per panel. Order: argmin cells first,
  then mixed cells by ascending minimum corner κ*, ties by grid index.
- **Fractional indices** interpolate persistence in log half-life, or linearly in
  rho for the cell next to 0. Amplitude interpolates linearly on its own axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from complete_journey.abm import config
from complete_journey.abm.config import require
from complete_journey.abm.seeds import Family

PRIMARY_FAMILIES = (Family.A1_PRIME, Family.A2)
SENSITIVITY_FAMILIES = (Family.A1_PRIME_HMM, Family.A2_RECENCY)
LAST = config.GRID_SIZE - 1


def persistence_axis() -> tuple[float, ...]:
    return (0.0, *(config.half_life_to_persistence(h) for h in config.HALF_LIFE_AXIS))


def half_life_axis() -> tuple[float, ...]:
    return (0.0, *config.HALF_LIFE_AXIS)


def amplitude_axis(family: Family) -> tuple[float, ...]:
    """A1′/HMM: logit-scale SD. A2/recency: raw delta on the logit scale (P8-002)."""
    if family in (Family.A1_PRIME, Family.A1_PRIME_HMM):
        return tuple(config.SIGMA_Z_CAP * k / config.GRID_SIZE for k in range(1, 13))
    require(family in (Family.A2, Family.A2_RECENCY), f"no amplitude axis for {family}")
    return tuple(config.A2_DELTA_CAP * k / config.GRID_SIZE for k in range(1, 13))


def _interp_persistence(ip: float) -> float:
    axis, hl = persistence_axis(), half_life_axis()
    lo = math.floor(ip)
    if ip == lo:
        return axis[lo]
    frac = ip - lo
    if lo == 0:  # between rho = 0 and the first half-life: linear in rho
        return frac * axis[1]
    log_h = (1 - frac) * math.log(hl[lo]) + frac * math.log(hl[lo + 1])
    return config.half_life_to_persistence(math.exp(log_h))


def _interp_amplitude(family: Family, ia: float) -> float:
    axis = amplitude_axis(family)
    lo = math.floor(ia)
    if ia == lo:
        return axis[lo]
    frac = ia - lo
    return (1 - frac) * axis[lo] + frac * axis[lo + 1]


@dataclass(frozen=True)
class GridPoint:
    family: Family
    ip: float  # persistence index (fractional after refinement)
    ia: float  # amplitude index
    level: int
    params: tuple[tuple[str, float], ...]
    half_life: float
    amplitude_label: float  # sigma_z (A1′/HMM) or raw delta (A2/recency; P8-002)
    degenerate: bool = False  # §V.3: numerically degenerate; recorded, excluded

    @property
    def boundary(self) -> bool:
        """On a design-cap face (§U): the half-life cap or the amplitude cap."""
        return self.ip == LAST or self.ia == LAST

    @property
    def coord(self) -> tuple[float, float]:
        return (self.ip, self.ia)


def make_point(family: Family, ip: float, ia: float, level: int):
    require(0 <= ip <= LAST and 0 <= ia <= LAST, "grid index outside the region")
    rho = _interp_persistence(ip)
    amp = _interp_amplitude(family, ia)
    require(rho <= config.PERSISTENCE_CAP + 1e-15, "persistence exceeds the cap")
    half_life = config.persistence_to_half_life(rho)
    if family == Family.A1_PRIME:
        require(amp <= config.SIGMA_Z_CAP, "sigma_z exceeds the cap")
        params = (("rho", rho), ("sigma_z", amp))
    elif family == Family.A1_PRIME_HMM:
        require(amp <= config.SIGMA_Z_CAP, "HMM amplitude exceeds the cap")
        params = (("q", max((1.0 - rho) / 2.0, 1e-12)), ("sigma", amp))
    else:
        require(amp <= config.A2_DELTA_CAP, "delta exceeds the P8-002 cap")
        params = (("delta", amp), ("lam", rho))
    return GridPoint(family, ip, ia, level, params, half_life, amp)


def build_grid(
    family: Family,
    persistence_index=range(config.GRID_SIZE),
    amplitude_index=range(config.GRID_SIZE),
) -> list[GridPoint]:
    """Coarse grid (production: 12 × 12; Run B instantiates tiny subsets only)."""
    require(family in PRIMARY_FAMILIES + SENSITIVITY_FAMILIES, "unknown family")
    return [make_point(family, ip, ia, 0) for ip in persistence_index for ia in amplitude_index]


def sensitivity_grid(family: Family) -> list[GridPoint]:
    idx = config.SENSITIVITY_SUBGRID_INDEX
    return build_grid(family, idx, idx)


def refine(
    family: Family,
    evaluated: dict[tuple[float, float], dict[float, bool]],
    kappa_star: dict[tuple[float, float], float],
    step: float,
    level: int,
    budget: int = config.REFINEMENT_MAX_POINTS,
) -> list[GridPoint]:
    """One deterministic refinement level over cells of side ``step``.

    ``evaluated`` maps a coordinate to {κ: ADEQ}. ``kappa_star`` maps a coordinate to
    κ*(θ). It returns new points (edge midpoints and centres not yet evaluated) in the
    frozen order, capped by ``budget``.
    """
    require(1 <= level <= config.REFINEMENT_MAX_LEVELS, "refinement level outside 1..2")
    finite = {c: v for c, v in kappa_star.items() if math.isfinite(v)}
    argmin = min(finite, key=lambda c: (finite[c], c)) if finite else None
    cells = []
    for ip, ia in sorted(evaluated):
        corners = [(ip, ia), (ip + step, ia), (ip, ia + step), (ip + step, ia + step)]
        if not all(c in evaluated for c in corners):
            continue
        statuses = [evaluated[c] for c in corners]
        mixed = any(len({s[k] for s in statuses}) > 1 for k in statuses[0])
        has_argmin = argmin in corners
        if mixed or has_argmin:
            min_ks = min(kappa_star.get(c, math.inf) for c in corners)
            cells.append((0 if has_argmin else 1, min_ks, (ip, ia)))
    cells.sort()
    new, seen = [], set(evaluated)
    half = step / 2
    for _, _, (ip, ia) in cells:
        for c in (
            (ip + half, ia),
            (ip, ia + half),
            (ip + step, ia + half),
            (ip + half, ia + step),
            (ip + half, ia + half),
        ):
            if c in seen:
                continue
            if len(new) >= budget:
                return new
            seen.add(c)
            new.append(make_point(family, c[0], c[1], level))
    return new


def coordinate_of(family: Family, persistence: float, amplitude: float) -> tuple[float, float]:
    """Fractional grid coordinate (ip, ia) of a dynamic point; inverse of ``make_point``.

    Used to insert θ_true as an evaluation point (§AB). It returns coordinates outside
    [0, 11] unchanged, so the caller can see that a truth lies outside the region.
    """
    axis, hl = persistence_axis(), half_life_axis()
    require(0.0 <= persistence < 1.0, "persistence must lie in [0, 1)")
    if persistence == 0.0:
        ip = 0.0
    elif persistence <= axis[1]:
        ip = persistence / axis[1]
    else:
        h = config.persistence_to_half_life(persistence)
        k = max(i for i in range(1, len(hl)) if hl[i] <= h + 1e-12) if h >= hl[1] else 1
        if k >= LAST:
            ip = LAST + (math.log(h) - math.log(hl[LAST])) / (
                math.log(hl[LAST]) - math.log(hl[LAST - 1])
            )
        else:
            ip = k + (math.log(h) - math.log(hl[k])) / (math.log(hl[k + 1]) - math.log(hl[k]))
    amp_axis = amplitude_axis(family)
    step = amp_axis[1] - amp_axis[0]
    ia = (amplitude - amp_axis[0]) / step
    return float(ip), float(ia)


def inside_region(coord: tuple[float, float], tol: float = 1e-9) -> bool:
    return all(-tol <= c <= LAST + tol for c in coord)
