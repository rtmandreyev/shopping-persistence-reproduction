"""P8-005 calibration frontier: SECONDARY / DESCRIPTIVE / POST-STAGE-1 PREDECLARED (§P8-005.9).

For each primary family and heterogeneity arm, over the full production grid
(including CI-1-infeasible points):
- x: the continuous CI-1 discrepancy D_CI = max_j |CI1_sim,j − CI1_target,j| under the
  frozen metric (the gate is D_CI ≤ max(0.02, 2·SE_j) per component);
- y: the direct R = 50 images of P1, P2, P3 at the calibrated heterogeneity.

The frontier is the maximum (and range) of each of P1, P2, P3 among points with
D_CI ≤ t, for the frozen thresholds t, plus the full scatter.

It is never a gate: it reads point records and returns a new summary. It does not
alter CI-1 feasibility, any verdict or any region, and does not justify loosening
0.02. Thresholds above 0.02 are descriptive only. It never uses real dynamic targets.
"""

from __future__ import annotations

import copy

import numpy as np

from complete_journey.abm.config import require

THRESHOLDS = (0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100)
COORDS = ("P1", "P2", "P3")
STATUS = "SECONDARY / DESCRIPTIVE / POST-STAGE-1 PREDECLARED"


def ci1_discrepancy(residuals) -> float:
    r = np.asarray(residuals, float)
    require(r.shape == (6,) and np.isfinite(r).all(), "CI-1 residuals must hold six values")
    return float(np.max(np.abs(r)))


def frontier(points: list[dict], thresholds=THRESHOLDS) -> dict:
    """Frontier summary from point records with ``ci1_residuals`` and ``images``.

    Each record needs ``ci1_residuals`` (6) and ``images`` {P1, P2, P3}. Records are
    deep-copied, never modified.
    """
    rows = copy.deepcopy(points)
    require(rows, "the frontier needs at least one grid point")
    scatter = []
    for r in rows:
        require(all(c in r["images"] for c in COORDS), "frontier records need P1..P3 images")
        require(isinstance(r.get("ci1_feasible"), bool), "frontier records need a CI-1 flag")
        scatter.append(
            {
                "ip": r.get("ip"),
                "ia": r.get("ia"),
                "d_ci": ci1_discrepancy(r["ci1_residuals"]),
                "ci1_feasible": r.get("ci1_feasible"),
                **{c: float(r["images"][c]) for c in COORDS},
            }
        )
    table = {}
    for t in thresholds:
        sel = [s for s in scatter if s["d_ci"] <= t]
        entry = {"points": len(sel)}
        for c in COORDS:
            vals = [s[c] for s in sel]
            entry[c] = (
                {"max": max(vals), "min": min(vals), "range": max(vals) - min(vals)}
                if vals
                else None
            )
        table[f"{t:.3f}"] = entry
    return {
        "status": STATUS,
        "is_gate": False,
        "thresholds": list(thresholds),
        "frontier": table,
        "scatter": scatter,
        "points_total": len(scatter),
        "points_ci1_infeasible": int(sum(s["ci1_feasible"] is False for s in scatter)),
    }
