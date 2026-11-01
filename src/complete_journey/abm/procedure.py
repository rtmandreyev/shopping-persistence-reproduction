"""Emulator-based acceptance-set procedure for one truth panel, and the H2 direct check.

This is the frozen P8-001 procedure (§O, §S–§U, §AB, §AC) as amended by P8-003 and
P8-004, executed per heterogeneity arm (P8-005):

1. For each lattice point θ of a family, recalibrate the heterogeneity η to the panel's
   own CI-1 by local-linear inversion of the emulated CI-1 map (within the arm's bounds).
2. CI-1 gate on the emulated CI-1 at that η: |CI1_em − CI1_panel| ≤ max(0.02, 2·SE_j).
3. Emulated μ_j(θ, η), s_j(θ, η) → evidence with the full budgets e_j = 0.1·τ_j(0.15)
   and g± (P8-004), R = 50, against the panel's P1..P5 and the truth point's
   across-panel SE (P8-003) → ADEQ / INADEQ / NEITHER at each κ, and κ*.
4. The frozen deterministic refinement (§V): at most two levels and 144 added points
   per family and panel.
5. θ_true (perfect-model gate) is inserted as an extra evaluation point by emulator
   interpolation; it never alters the family verdict, which uses the production grid
   plus the frozen refinement only (P8-003 G2).

**H2 direct check (§AC).** The same classification by direct recalibration and direct
simulation (R = 25, MODEL streams 0…24) on the 6 × 6 subgrid. Agreement ≥ 95% of
points at every κ, with no ADEQ/INADEQ contradiction (an infeasible-versus-feasible
mismatch counts as a disagreement).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from complete_journey.abm import config, regions
from complete_journey.abm.adequacy import (
    ADEQ,
    INADEQ,
    PointOutcome,
    Tolerances,
    comparative_verdict,
    evidence_from_simulation,
    family_verdict,
    kappa_star_point,
    point_discrepancy_ratio,
    point_status,
)
from complete_journey.abm.config import require
from complete_journey.abm.emulator import ReferenceTable, predict, predict_ci1, recalibrate

H2_R = 25
H2_AGREEMENT_MIN = 0.95
INFEASIBLE_POINT = "INFEASIBLE"


@dataclass(frozen=True)
class Panel:
    """One truth panel: its CI-1, its P1..P5 and the truth point's across-panel SE."""

    ci1: np.ndarray  # (6,)
    primary: dict[str, float]
    primary_se: dict[str, float]

    def __post_init__(self):
        require(np.asarray(self.ci1).shape == (6,), "panel CI-1 must hold six components")
        require(tuple(self.primary) == config.PRIMARY_IDS, "panel moments must be P1..P5")
        require(set(self.primary_se) == set(config.PRIMARY_IDS), "panel SEs must be P1..P5")


def ci1_se() -> np.ndarray:
    return np.array([config.CI1_BOOT_SE[c] for c in config.CI1_COMPONENTS])


def emulator_budget(tolerances: Tolerances) -> np.ndarray:
    """P8-004: e_j = 0.1·τ_j(0.15), the full validated mean-error budget."""
    return config.EMULATOR_MEAN_ERROR_FRACTION * tolerances.tau(config.EMULATOR_GATE_KAPPA)


def evaluate_emulated(
    table: ReferenceTable,
    coord: tuple[float, float],
    panel: Panel,
    tolerances: Tolerances,
    boundary: bool = False,
) -> tuple[PointOutcome, dict]:
    """Pointwise status at a (fractional) coordinate via the emulator."""
    se6 = ci1_se()
    eta = recalibrate(table, coord[0], coord[1], panel.ci1, se6)
    ci1_em = predict_ci1(table, coord[0], coord[1], eta)
    resid = ci1_em - np.asarray(panel.ci1, float)
    feasible = bool(np.all(np.abs(resid) <= np.maximum(config.CI1_GATE_ABS, 2.0 * se6)))
    info = {"eta": eta.tolist(), "ci1_max_abs_residual": float(np.max(np.abs(resid)))}
    if not feasible:
        return PointOutcome(coord, False, boundary, {}, math.inf), info
    mean, sd = predict(table, coord[0], coord[1], eta)
    ev = evidence_from_simulation(
        panel.primary,
        panel.primary_se,
        mean,
        np.maximum(sd, 0.0),
        config.R_EMULATOR,
        emulator_error=emulator_budget(tolerances),
    )
    status = {k: point_status(ev, tolerances.tau(k)) for k in config.KAPPAS}
    return (
        PointOutcome(
            coord,
            True,
            boundary,
            status,
            kappa_star_point(ev, tolerances),
            point_discrepancy_ratio(ev, tolerances),
        ),
        info,
    )


def _boundary(coord) -> bool:
    return coord[0] == regions.LAST or coord[1] == regions.LAST


@dataclass(frozen=True)
class FamilyEvaluation:
    family: str
    outcomes: tuple[PointOutcome, ...]  # production grid plus frozen refinement
    refinement_points: int
    truth: PointOutcome | None = None

    def verdict(self, kappa: float, tolerances: Tolerances):
        return family_verdict(self.family, "primary", list(self.outcomes), kappa, tolerances)


def evaluate_family(
    table: ReferenceTable,
    panel: Panel,
    tolerances: Tolerances,
    theta_true: tuple[float, float] | None = None,
) -> FamilyEvaluation:
    """Production lattice + the frozen two-level refinement, all via the emulator."""
    ips, ias = table.lattice
    outcomes: dict[tuple[float, float], PointOutcome] = {}
    for ip in ips:
        for ia in ias:
            c = (float(ip), float(ia))
            outcomes[c], _ = evaluate_emulated(table, c, panel, tolerances, _boundary(c))
    family = regions.Family[table.family]
    added = 0
    step = float(ips[1] - ips[0]) if len(ips) > 1 else 1.0
    for level in (1, 2):
        evaluated = {
            c: {k: (o.feasible and o.status.get(k) == ADEQ) for k in config.KAPPAS}
            for c, o in outcomes.items()
        }
        kstar = {c: o.kappa_star for c, o in outcomes.items()}
        budget = config.REFINEMENT_MAX_POINTS - added
        if budget <= 0:
            break
        new = regions.refine(family, evaluated, kstar, step, level, budget)
        for pt in new:
            outcomes[pt.coord], _ = evaluate_emulated(
                table, pt.coord, panel, tolerances, pt.boundary
            )
        added += len(new)
        step /= 2.0
    truth = None
    if theta_true is not None:
        truth, _ = evaluate_emulated(table, theta_true, panel, tolerances, _boundary(theta_true))
    ordered = tuple(outcomes[c] for c in sorted(outcomes))
    return FamilyEvaluation(table.family, ordered, added, truth)


def evaluate_single(table: ReferenceTable, panel: Panel, tolerances: Tolerances):
    """A1 (no dynamic parameters): its single calibrated point via its own table."""
    outcome, _ = evaluate_emulated(table, (0.0, 0.0), panel, tolerances, False)
    return FamilyEvaluation(table.family, (outcome,), 0, outcome)


def gate_inputs(
    true_family: FamilyEvaluation,
    a1_prime: FamilyEvaluation,
    a2: FamilyEvaluation,
    tolerances: Tolerances,
) -> dict:
    """G1–G5 inputs for one panel (P8-003): pointwise truth inclusion, verdicts, NR."""
    ref = config.KAPPA_REFERENCE
    t = true_family.truth
    require(t is not None, "the true family needs θ_true inserted")
    in_acc = {k: bool(t.feasible and t.status.get(k) == ADEQ) for k in config.KAPPAS}
    in_nr = bool(t.feasible and t.status.get(ref) != INADEQ)
    fam = true_family.verdict(ref, tolerances).verdict
    comp = comparative_verdict(a2.verdict(ref, tolerances), a1_prime.verdict(ref, tolerances))
    argmin = true_family.verdict(ref, tolerances).argmin
    recovery = (
        math.dist(argmin, t.coord) if argmin is not None and true_family.family != "A1" else None
    )
    return {
        "truth_in_acc": in_acc,
        "truth_in_nr": in_nr,
        "family_verdict": fam,
        "comparative": comp.verdict,
        "recovery_distance": recovery,
    }


# --- H2 direct-procedure check --------------------------------------------------------


def classification(outcome: PointOutcome, kappa: float) -> str:
    return outcome.status[kappa] if outcome.feasible else INFEASIBLE_POINT


def h2_agreement(pairs: list[tuple[PointOutcome, PointOutcome]]) -> dict:
    """(direct, emulated) outcome pairs on the subgrid → agreement per κ and contradictions."""
    require(pairs, "H2 needs evaluated pairs")
    out, passed = {}, True
    for k in config.KAPPAS:
        agree = [classification(d, k) == classification(e, k) for d, e in pairs]
        contra = sum(
            {classification(d, k), classification(e, k)} == {ADEQ, INADEQ} for d, e in pairs
        )
        share = float(np.mean(agree))
        ok = share >= H2_AGREEMENT_MIN and contra == 0
        out[str(k)] = {"agreement": share, "adeq_inadeq_contradictions": int(contra), "pass": ok}
        passed = passed and ok
    return {"by_kappa": out, "pairs": len(pairs), "pass": passed}
