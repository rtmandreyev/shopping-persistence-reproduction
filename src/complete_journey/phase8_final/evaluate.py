"""Pure evaluation functions of the P8-006 direct plug-in procedure.

Model table: for every family point θ (coarse grid, first-level refinement points, the
A1 point, perfect-model θ_true) a DIRECT image at the family's real-CI-1 calibration:
P1..P5 means and SDs over R MODEL replicates, the replicate count R and the CI-1
feasibility flag. The table is the same for every panel (plug-in, P8-006).

Panel evaluation applies the unchanged P8-003 rule with direct-simulation uncertainty
(e = 0, g± = 1): σ²_j = SE_j² + s_j(θ)²/R(θ).

Refinement: the frozen first level of ``regions.refine`` over the coarse statuses. A
cell whose four corners are all CI-infeasible is never refined (no ADEQ corner, no
finite κ*), so the pruned precomputed set always contains every requested point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

from complete_journey.abm import config, regions
from complete_journey.abm.adequacy import (
    ADEQ,
    INADEQ,
    Evidence,
    PointOutcome,
    Tolerances,
    comparative_verdict,
    family_verdict,
    kappa_star_detail,
    point_discrepancy_ratio,
    point_status,
    z_equivalence,
    z_inadequacy,
)
from complete_journey.abm.config import require
from complete_journey.abm.seeds import Family

LAST = regions.LAST


@dataclass(frozen=True)
class Entry:
    """One direct model image at a family point (plug-in table)."""

    mean: np.ndarray  # (5,)
    sd: np.ndarray  # (5,)
    replicates: int
    feasible: bool

    def __post_init__(self):
        object.__setattr__(self, "mean", np.asarray(self.mean, float))
        object.__setattr__(self, "sd", np.asarray(self.sd, float))
        require(self.mean.shape == self.sd.shape == (config.J_PRIMARY,), "entry needs P1..P5")
        require(self.replicates >= 2, "an image needs at least two replicates")


Table = dict[tuple[float, float], Entry]


def boundary(coord: tuple[float, float]) -> bool:
    return coord[0] == LAST or coord[1] == LAST


def outcome(coord, entry: Entry, data: np.ndarray, se: np.ndarray, tol: Tolerances):
    """Pointwise status at every κ and κ*, direct simulation (e = 0, g = 1)."""
    if not entry.feasible:
        return PointOutcome(coord, False, boundary(coord), {}, math.inf), ()
    ev = Evidence(np.asarray(data, float) - entry.mean, se, entry.sd, entry.replicates)
    status = {k: point_status(ev, tol.tau(k)) for k in config.KAPPAS}
    ks, limited = kappa_star_detail(ev, tol)
    return (
        PointOutcome(coord, True, boundary(coord), status, ks, point_discrepancy_ratio(ev, tol)),
        limited,
    )


def binding_moment(entry: Entry, data, se, tol: Tolerances) -> str:
    """argmax_j κ*_j at a point (the binding moment, §U)."""
    ev = Evidence(np.asarray(data, float) - entry.mean, se, entry.sd, entry.replicates)
    upper = ev.upper()
    vals = []
    for k, j in enumerate(config.PRIMARY_IDS):
        e = abs(tol.explanandum[j])
        vals.append(0.0 if upper[k] <= tol.rho[j] else (upper[k] / e if e else math.inf))
    return config.PRIMARY_IDS[int(np.argmax(vals))]


@dataclass(frozen=True)
class FamilyResult:
    family: str
    outcomes: tuple[PointOutcome, ...]
    refinement_points: int

    def verdict(self, kappa: float, tol: Tolerances):
        return family_verdict(self.family, "primary", list(self.outcomes), kappa, tol)


def evaluate_family(
    family: str,
    coarse: Table,
    refined: Table,
    data,
    se,
    tol: Tolerances,
    levels: int = 1,
) -> FamilyResult:
    """Coarse lattice plus the frozen refinement (``levels`` = 1 confirmatory)."""
    outs: dict[tuple[float, float], PointOutcome] = {}
    for c, e in coarse.items():
        outs[c], _ = outcome(c, e, data, se, tol)
    added, step = 0, 1.0
    for level in range(1, levels + 1):
        evaluated = {
            c: {k: (o.feasible and o.status.get(k) == ADEQ) for k in config.KAPPAS}
            for c, o in outs.items()
        }
        kstar = {c: o.kappa_star for c, o in outs.items()}
        budget = config.REFINEMENT_MAX_POINTS - added
        if budget <= 0:
            break
        new = regions.refine(Family[family], evaluated, kstar, step, level, budget)
        for pt in new:
            require(pt.coord in refined, f"refinement point {pt.coord} missing from the table")
            outs[pt.coord], _ = outcome(pt.coord, refined[pt.coord], data, se, tol)
        added += len(new)
        step /= 2.0
    return FamilyResult(family, tuple(outs[c] for c in sorted(outs)), added)


def requested_refinement(family: str, coarse: Table, data, se, tol, level_tables=None):
    """Coordinates the frozen rule requests at level 1 (and level 2 if a level-1 table is
    supplied): used for the real-panel audit and the secondary level-2 calculation."""
    outs = {c: outcome(c, e, data, se, tol)[0] for c, e in coarse.items()}
    evaluated = {
        c: {k: (o.feasible and o.status.get(k) == ADEQ) for k in config.KAPPAS}
        for c, o in outs.items()
    }
    kstar = {c: o.kappa_star for c, o in outs.items()}
    lvl1 = regions.refine(Family[family], evaluated, kstar, 1.0, 1, config.REFINEMENT_MAX_POINTS)
    if level_tables is None:
        return [p.coord for p in lvl1], []
    for p in lvl1:
        outs[p.coord] = outcome(p.coord, level_tables[p.coord], data, se, tol)[0]
    evaluated = {
        c: {k: (o.feasible and o.status.get(k) == ADEQ) for k in config.KAPPAS}
        for c, o in outs.items()
    }
    kstar = {c: o.kappa_star for c, o in outs.items()}
    lvl2 = regions.refine(
        Family[family], evaluated, kstar, 0.5, 2, config.REFINEMENT_MAX_POINTS - len(lvl1)
    )
    return [p.coord for p in lvl1], [p.coord for p in lvl2]


def gate_record_inputs(
    true_family: FamilyResult,
    truth_out: PointOutcome,
    a1p: FamilyResult,
    a2: FamilyResult,
    tol: Tolerances,
) -> dict:
    ref = config.KAPPA_REFERENCE
    in_acc = {
        k: bool(truth_out.feasible and truth_out.status.get(k) == ADEQ) for k in config.KAPPAS
    }
    in_nr = bool(truth_out.feasible and truth_out.status.get(ref) != INADEQ)
    fv = true_family.verdict(ref, tol)
    comp = comparative_verdict(a2.verdict(ref, tol), a1p.verdict(ref, tol))
    recovery = (
        math.dist(fv.argmin, truth_out.coord)
        if (fv.argmin is not None and true_family.family != "A1")
        else None
    )
    return {
        "truth_in_acc": in_acc,
        "truth_in_nr": in_nr,
        "family_verdict": fv.verdict,
        "comparative": comp.verdict,
        "recovery_distance": recovery,
    }


# --- representability (direct: e = 0, g = 1) ------------------------------------------


def classify_direct(
    truth_mean, truth_sd, q: int, image: list[Entry], tol: Tolerances, kappa: float
) -> str:
    """P8-005.4 three-state classification with direct images (e = 0, g± = 1)."""
    from complete_journey.abm.representability import EXCLUDED, INDETERMINATE, REPRESENTABLE

    feas = [e for e in image if e.feasible]
    if not feas:
        return EXCLUDED
    m = np.asarray(truth_mean, float)
    st2 = np.asarray(truth_sd, float) ** 2 / q
    tau = tol.tau(kappa)
    inside_any, outside_all = False, True
    for e in feas:
        d = np.abs(e.mean - m)
        sig = np.sqrt(st2 + e.sd**2 / e.replicates)
        if (d + z_equivalence() * sig <= tau).all():
            inside_any = True
        if not (d - z_inadequacy() * sig > tau).any():
            outside_all = False
    if inside_any:
        return REPRESENTABLE
    return EXCLUDED if outside_all else INDETERMINATE


# --- exact intervals and aggregation ------------------------------------------------


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    require(0 <= k <= n and n > 0, "binomial counts out of range")
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def lower_bound_one_sided(k: int, n: int, level: float = 0.95) -> float:
    return 0.0 if k == 0 else float(beta.ppf(1 - level, k, n - k + 1))


def rate(flags) -> dict:
    flags = [bool(f) for f in flags]
    k, n = sum(flags), len(flags)
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci95": [lo, hi]}


def mde_bracket(det_counts: dict[float, tuple[int, int]], reachable: list[float]):
    """MDE_80 confidence-aware bracket end: the smallest reachable Δ_1 whose one-sided
    95% lower Clopper-Pearson bound reaches 0.80 for it and every larger reachable Δ_1;
    '> X pp' if none (X = largest reachable)."""
    if not reachable:
        return "NO REACHABLE HISTORY TRUTH"
    lbs = [lower_bound_one_sided(*det_counts[d]) for d in reachable]
    for i, d in enumerate(reachable):
        if all(lb >= config.DETECTION_THRESHOLD for lb in lbs[i:]):
            return float(d)
    return f"> {reachable[-1]:g} pp"


def pooled_stats(n1: int, mean1, sd1, rows) -> tuple[np.ndarray, np.ndarray, int]:
    """Exact pooled mean/SD (ddof=1) of an imported (n1, mean1, sd1) block and new rows."""
    rows = np.asarray(rows, float).reshape(-1, config.J_PRIMARY)
    n2 = rows.shape[0]
    mean1, sd1 = np.asarray(mean1, float), np.asarray(sd1, float)
    s1, ss1 = n1 * mean1, (n1 - 1) * sd1**2 + n1 * mean1**2
    s2, ss2 = rows.sum(0), (rows**2).sum(0)
    n = n1 + n2
    mean = (s1 + s2) / n
    var = (ss1 + ss2 - n * mean**2) / (n - 1)
    return mean, np.sqrt(np.maximum(var, 0.0)), n
