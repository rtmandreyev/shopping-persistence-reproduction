"""Reference-table emulator (design §AC) for both heterogeneity arms (P8-005), plus the
P8-004 CRN-paired validation.

A ``ReferenceTable`` holds, for every lattice point (ip, ia) of one family, K + 1
heterogeneity design nodes η, where K is the number of heterogeneity parameters:
- primary logit-normal arm: η = (μ, s), K = 2, nodes η̂ + {0, 0.10·e_μ, 0.10·e_s}
  (the frozen Stage 1 rule);
- P8-005 mixture arm: η = (π, μ_1, d, s), K = 4, nodes η̂ + {0, 0.10·e_k} for each k
  (the same rule per coordinate; a node that would leave the frozen bounds steps
  −0.10 instead).

At each node it stores the simulated means μ_j and SDs s_j of P1..P5 (R = 50 MODEL
replicates, the same streams at every node and lattice point) and the simulated CI-1
vector. Values are local-affine in η at each lattice point (K + 1 nodes determine the
map exactly) and bilinear in θ across the lattice. A single-point table (A1) has no θ
dimension.

- ``predict`` / ``predict_ci1`` return emulated values at a fractional (ip, ia) and η.
- ``recalibrate`` inverts the emulated CI-1 map by weighted linear least squares in η,
  within the arm's parameter bounds.

**Predeclared engineering refinement** (§AC "If the gate fails", step 1): a failing
validation point's neighbouring cells get their centre added as an extra node set at
R = 100. Inside a refined cell, θ-interpolation is piecewise linear on the four
triangles formed by the centre and each cell edge.

**P8-004.** The validation reference at each held-out validation point is direct
simulation with the **same 50 MODEL-replicate streams** as the table, so the
comparison isolates interpolation error. The thresholds are unchanged
(``rq2.emulator_gate``). An independent R = 200 direct comparison (MODEL replicates
50…249, never used by the table) is a reported, non-binding diagnostic. Validation
points and their outputs never enter table construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear

from complete_journey.abm import config
from complete_journey.abm.config import require

LOGIT_NORMAL, MIXTURE = "logit_normal", "mixture"
ETA_NAMES = {LOGIT_NORMAL: ("mu", "s"), MIXTURE: ("pi", "mu1", "d", "s")}
NODE_STEP = 0.10
VALIDATION_OFFSET = 0.03
#: Recalibration bounds in η (the frozen parameter bounds of each arm).
ETA_BOUNDS = {
    LOGIT_NORMAL: (np.array([-np.inf, 0.0]), np.array([np.inf, np.inf])),
    MIXTURE: (np.array([0.10, -5.0, 0.25, 0.05]), np.array([0.90, 4.75, 6.0, 3.0])),
}
TABLE_R = config.R_EMULATOR
REFINED_R = 100
INDEPENDENT_R = config.R_DIRECT
INDEPENDENT_FIRST_REPLICATE = TABLE_R  # replicates 50 … 249: disjoint from the table


def eta_dimension(arm: str) -> int:
    require(arm in ETA_NAMES, f"unknown heterogeneity arm {arm}")
    return len(ETA_NAMES[arm])


def clamp_eta(arm: str, eta) -> np.ndarray:
    """Project η into the arm's bounds (mixture: also μ_2 = μ_1 + d ≤ 5)."""
    lo, hi = ETA_BOUNDS[arm]
    x = np.minimum(np.maximum(np.asarray(eta, float), lo), hi)
    if arm == MIXTURE:
        x[2] = max(min(x[2], 5.0 - x[1]), 0.25)
    return x


def design_nodes(arm: str, base) -> np.ndarray:
    """(K + 1, K) design nodes: η̂ plus a +0.10 step in each coordinate (−0.10 if out)."""
    base = np.asarray(base, float)
    k = eta_dimension(arm)
    require(base.shape == (k,), "η̂ has the wrong dimension")
    nodes = [base.copy()]
    for i in range(k):
        up = base.copy()
        up[i] += NODE_STEP
        if not np.allclose(clamp_eta(arm, up), up):
            up[i] = base[i] - NODE_STEP
        nodes.append(clamp_eta(arm, up))
    out = np.array(nodes)
    require(
        np.linalg.matrix_rank(np.column_stack([np.ones(k + 1), out])) == k + 1,
        "design nodes must be affinely independent",
    )
    return out


def validation_offsets(arm: str) -> tuple[np.ndarray, np.ndarray]:
    """The two frozen offsets (±0.03, ∓0.03, …) in η, alternating sign by coordinate."""
    k = eta_dimension(arm)
    plus = np.array([VALIDATION_OFFSET * (1 if i % 2 == 0 else -1) for i in range(k)])
    return plus, -plus


@dataclass(frozen=True)
class ReferenceTable:
    family: str
    arm: str
    lattice: tuple[tuple[float, ...], tuple[float, ...]]  # persistence, amplitude indices
    eta_design: np.ndarray  # (P_ip, P_ia, K + 1, K)
    mean: np.ndarray  # (P_ip, P_ia, K + 1, J)
    sd: np.ndarray  # (P_ip, P_ia, K + 1, J)
    ci1: np.ndarray  # (P_ip, P_ia, K + 1, 6)
    #: Predeclared refinement: cell (i, j) -> (eta_design, mean, sd, ci1) at its centre.
    centres: dict = field(default_factory=dict)
    origin: str = "simulated"

    def __post_init__(self):
        require(self.origin == "simulated", "reference tables hold simulated values only")
        k = eta_dimension(self.arm)
        n_ip, n_ia = len(self.lattice[0]), len(self.lattice[1])
        require(self.eta_design.shape == (n_ip, n_ia, k + 1, k), "η design shape mismatch")
        for arr in (self.mean, self.sd, self.ci1):
            require(arr.shape[:3] == (n_ip, n_ia, k + 1), "table value shape mismatch")
        require(
            np.isfinite(self.mean).all() and np.isfinite(self.ci1).all(),
            "table values must be finite",
        )

    @property
    def k(self) -> int:
        return eta_dimension(self.arm)


def _affine(eta_design: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Coefficients C (K + 1, M) with values ≈ [1, η] @ C, exact from K + 1 nodes."""
    x = np.column_stack([np.ones(eta_design.shape[0]), eta_design])
    require(abs(np.linalg.det(x)) > 1e-12, "η design nodes must not be affinely dependent")
    return np.linalg.solve(x, values)


def _axis_cell(axis: tuple[float, ...], x: float) -> tuple[int, float]:
    if len(axis) == 1:
        require(abs(x - axis[0]) < 1e-12, "point outside a single-point table")
        return 0, 0.0
    require(axis[0] - 1e-12 <= x <= axis[-1] + 1e-12, "point outside the table")
    i = min(max(int(np.searchsorted(axis, x, side="right")) - 1, 0), len(axis) - 2)
    return i, (x - axis[i]) / (axis[i + 1] - axis[i])


def _node_coef(table: ReferenceTable, field_name: str, i: int, j: int) -> np.ndarray:
    return _affine(table.eta_design[i, j], getattr(table, field_name)[i, j])


def _centre_coef(table: ReferenceTable, field_name: str, cell) -> np.ndarray:
    eta, mean, sd, ci1 = table.centres[cell]
    return _affine(eta, {"mean": mean, "sd": sd, "ci1": ci1}[field_name])


def _blend(table: ReferenceTable, field_name: str, ip: float, ia: float) -> np.ndarray:
    """θ-interpolation of the per-node affine coefficient matrices (K + 1, M)."""
    ips, ias = table.lattice
    i, fi = _axis_cell(ips, ip)
    j, fj = _axis_cell(ias, ia)
    single_i, single_j = len(ips) == 1, len(ias) == 1
    if single_i and single_j:
        return _node_coef(table, field_name, 0, 0)
    require(not (single_i or single_j), "tables are single-point or full 2-D lattices")
    corner = {
        (0, 0): (i, j),
        (1, 0): (i + 1, j),
        (0, 1): (i, j + 1),
        (1, 1): (i + 1, j + 1),
    }
    if (i, j) in table.centres:
        # Refined cell: linear on the triangle (centre, two adjacent corners).
        u, v = fi - 0.5, fj - 0.5
        if abs(u) >= abs(v):
            side = 1 if u >= 0 else 0
            c1, c2 = (side, 0), (side, 1)
            t = 2 * abs(u)  # 0 at centre, 1 on the edge
            s = (v + 0.5 * t) / t if t > 0 else 0.5
        else:
            side = 1 if v >= 0 else 0
            c1, c2 = (0, side), (1, side)
            t = 2 * abs(v)
            s = (u + 0.5 * t) / t if t > 0 else 0.5
        out = (1 - t) * _centre_coef(table, field_name, (i, j))
        out = out + t * (1 - s) * _node_coef(table, field_name, *corner[c1])
        return out + t * s * _node_coef(table, field_name, *corner[c2])
    weights = {
        (0, 0): (1 - fi) * (1 - fj),
        (1, 0): fi * (1 - fj),
        (0, 1): (1 - fi) * fj,
        (1, 1): fi * fj,
    }
    out = 0.0
    for c, w in weights.items():
        if w:
            out = out + w * _node_coef(table, field_name, *corner[c])
    return out


def predict(table: ReferenceTable, ip: float, ia: float, eta) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([1.0, *np.asarray(eta, float)])
    require(x.size == table.k + 1, "η has the wrong dimension")
    return x @ _blend(table, "mean", ip, ia), x @ _blend(table, "sd", ip, ia)


def predict_ci1(table: ReferenceTable, ip: float, ia: float, eta) -> np.ndarray:
    x = np.array([1.0, *np.asarray(eta, float)])
    require(x.size == table.k + 1, "η has the wrong dimension")
    return x @ _blend(table, "ci1", ip, ia)


def recalibrate(table: ReferenceTable, ip: float, ia: float, target, se) -> np.ndarray:
    """η minimizing the inverse-variance-weighted emulated CI-1 distance, within bounds."""
    coef = _blend(table, "ci1", ip, ia)  # (K + 1, 6)
    w = np.sqrt(1.0 / np.asarray(se, float) ** 2)
    a = coef[1:].T * w[:, None]
    b = (np.asarray(target, float) - coef[0]) * w
    lo, hi = ETA_BOUNDS[table.arm]
    res = lsq_linear(a, b, bounds=(lo, hi), method="bvls")
    eta = clamp_eta(table.arm, res.x)
    require(all(math.isfinite(v) for v in eta), "emulated recalibration failed")
    return eta


def with_centres(table: ReferenceTable, centres: dict) -> ReferenceTable:
    """A copy of the table with predeclared refinement centres added."""
    merged = {**table.centres, **centres}
    return ReferenceTable(
        table.family,
        table.arm,
        table.lattice,
        table.eta_design,
        table.mean,
        table.sd,
        table.ci1,
        merged,
    )


def neighbouring_cells(table: ReferenceTable, ip: float, ia: float) -> list[tuple[int, int]]:
    """Lattice cells whose closure contains (ip, ia) (predeclared refinement targets)."""
    ips, ias = table.lattice
    cells_i = [k for k in range(len(ips) - 1) if ips[k] - 1e-12 <= ip <= ips[k + 1] + 1e-12]
    cells_j = [k for k in range(len(ias) - 1) if ias[k] - 1e-12 <= ia <= ias[k + 1] + 1e-12]
    return sorted((i, j) for i in cells_i for j in cells_j)


# --- P8-004 CRN-paired validation ----------------------------------------------------


@dataclass(frozen=True)
class ValidationRecord:
    family: str
    point: tuple[float, float]
    offset: int
    eta: np.ndarray
    emulated_mean: np.ndarray
    emulated_sd: np.ndarray
    crn_mean: np.ndarray  # direct, same 50 MODEL streams as the table
    crn_sd: np.ndarray
    independent_mean: np.ndarray | None = None  # diagnostic, replicates 50 … 249
    independent_sd: np.ndarray | None = None


def crn_gate(records: list[ValidationRecord], tolerances) -> dict:
    """Primary P8-004 gate on the CRN-paired references (thresholds unchanged)."""
    from complete_journey.abm.rq2 import emulator_gate

    require(records, "the emulator gate needs validation records")
    return emulator_gate(
        np.array([r.emulated_mean for r in records]),
        np.array([r.crn_mean for r in records]),
        np.array([r.emulated_sd for r in records]),
        np.array([r.crn_sd for r in records]),
        tolerances,
    )


def independent_diagnostic(records: list[ValidationRecord], tolerances) -> dict:
    """Non-binding diagnostic: the same thresholds against independent R = 200."""
    from complete_journey.abm.rq2 import emulator_gate

    rows = [r for r in records if r.independent_mean is not None]
    require(rows, "no independent diagnostic references")
    out = emulator_gate(
        np.array([r.emulated_mean for r in rows]),
        np.array([r.independent_mean for r in rows]),
        np.array([r.emulated_sd for r in rows]),
        np.array([r.independent_sd for r in rows]),
        tolerances,
    )
    out["binding"] = False
    return out


def failing_points(records: list[ValidationRecord], tolerances) -> list[tuple[float, float]]:
    """Validation points failing either the mean or the SD criterion (CRN reference)."""
    tau = tolerances.tau(config.EMULATOR_GATE_KAPPA)
    bad = set()
    lo, hi = 1.0 - config.EMULATOR_SD_TOLERANCE, 1.0 + config.EMULATOR_SD_TOLERANCE
    for r in records:
        mean_bad = (
            np.abs(r.emulated_mean - r.crn_mean) > config.EMULATOR_MEAN_ERROR_FRACTION * tau
        ).any()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                r.crn_sd > 0, r.emulated_sd / r.crn_sd, np.where(r.emulated_sd == 0, 1.0, np.inf)
            )
        if mean_bad or ((ratio < lo) | (ratio > hi)).any():
            bad.add(tuple(r.point))
    return sorted(bad)
