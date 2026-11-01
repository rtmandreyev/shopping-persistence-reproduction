"""Direct (non-emulator) acceptance-set procedure for one panel (design §O, §S–§U).

For each grid point θ of a family:
1. calibrate (mu, s) to the panel's CI-1, applying the gate;
2. simulate R MODEL-role worlds (common random numbers across θ) under the same
   operator;
3. compute P1..P5 for each world;
4. form the evidence and the pointwise status at every κ, plus κ*(θ).

Run B uses this only with tiny N, R and grids on synthetic panels. Production uses the
validated emulator plus direct audits (§AC). Synthetic and real targets go through
``guards`` before any use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from complete_journey.abm import config
from complete_journey.abm.adequacy import (
    PointOutcome,
    Tolerances,
    evidence_from_simulation,
    kappa_star_point,
    point_discrepancy_ratio,
    point_status,
)
from complete_journey.abm.architectures import Spec, World, simulate
from complete_journey.abm.calibration import calibrate
from complete_journey.abm.config import require
from complete_journey.abm.guards import TargetVector, require_run_stage, require_target_allowed
from complete_journey.abm.moments import ci1_fast, primary_moments
from complete_journey.abm.operators import world_panel
from complete_journey.abm.regions import GridPoint
from complete_journey.abm.seeds import Role, SeedKey


@dataclass(frozen=True)
class PanelTargets:
    """CI-1 and P1..P5 of the panel under analysis (synthetic in Run B)."""

    ci1: TargetVector
    primary: TargetVector


def world_targets(world: World, primary_se: dict[str, float] | None = None) -> PanelTargets:
    """Targets of a synthetic DATA-role world (truth or pseudo-reference panel).

    ``primary_se`` is the empirical sampling SE of P1..P5 for this panel. For RQ2 and
    perfect-model panels (P8-003), it is the truth point's across-panel SD.
    """
    require(world.seed_key.role == Role.DATA, "targets come from DATA-role worlds only")
    ci1 = TargetVector(
        "synthetic", dict(zip(config.CI1_COMPONENTS, ci1_fast(world.behavior), strict=True))
    )
    primary = primary_moments(world_panel(world)).values
    return PanelTargets(ci1, TargetVector("synthetic", dict(primary), primary_se))


def truth_point_se(panels: list[PanelTargets]) -> dict[str, float]:
    """P8-003: empirical SE for truth panels = across-panel SD of P1..P5 at the point."""
    require(len(panels) >= 2, "an across-panel SD needs at least two truth panels")
    arr = np.array([[p.primary.values[j] for j in config.PRIMARY_IDS] for p in panels])
    return dict(zip(config.PRIMARY_IDS, arr.std(axis=0, ddof=1), strict=True))


def simulate_reference(
    spec: Spec,
    het,
    stage: int,
    n: int,
    replicates: int,
    availability: np.ndarray | None = None,
    periodic_amplitude: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Means and SDs of P1..P5 over MODEL-role replicates (CRN across grid points)."""
    require_run_stage(stage)
    rows = []
    for r in range(replicates):
        key = SeedKey(stage=stage, role=Role.MODEL, replicate=r)
        world = simulate(spec, het, key, n, availability, periodic_amplitude)
        rows.append(list(primary_moments(world_panel(world)).values.values()))
    arr = np.array(rows)
    return arr.mean(axis=0), arr.std(axis=0, ddof=1)


def evaluate_point(
    point: GridPoint | None,
    spec: Spec,
    targets: PanelTargets,
    tolerances: Tolerances,
    stage: int,
    n: int,
    replicates: int,
    r_cal: int,
    availability: np.ndarray | None = None,
) -> PointOutcome:
    require_target_allowed(targets.ci1, "ci1_calibration")
    require_target_allowed(targets.primary, "comparison")
    if point is not None and point.degenerate:
        return PointOutcome(point.coord, False, bool(point.boundary), {}, float("inf"))
    cal = calibrate(spec, targets.ci1, stage=stage, n=n, r_cal=r_cal, availability=availability)
    coord = point.coord if point is not None else (-1.0, -1.0)
    boundary = bool(point.boundary) if point is not None else False
    if not cal.feasible:
        return PointOutcome(coord, False, boundary, {}, float("inf"))
    require(targets.primary.se is not None, "P8-003: the panel's empirical SEs are required")
    means, sds = simulate_reference(spec, cal.heterogeneity, stage, n, replicates, availability)
    ev = evidence_from_simulation(
        targets.primary.values, targets.primary.se, means, sds, replicates
    )
    status = {k: point_status(ev, tolerances.tau(k)) for k in config.KAPPAS}
    return PointOutcome(
        coord,
        True,
        boundary,
        status,
        kappa_star_point(ev, tolerances),
        point_discrepancy_ratio(ev, tolerances),
    )
