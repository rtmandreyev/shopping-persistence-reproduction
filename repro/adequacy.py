"""Independent re-implementation of the paper's adequacy rule (Appendix D), numpy and scipy only.

It deliberately imports nothing from ``complete_journey``: it is a second implementation,
written from the paper's description and used to recompute every stored known-truth and
real-panel verdict.

Rule. For moment j at a family point with simulated mean m_j, SD v_j and R replicates, given
an observed moment vector x with standard errors SE:
    d_j = x_j - m_j,  sigma_j = sqrt(SE_j^2 + v_j^2 / R)
    tolerance omega_j(kappa) = max(kappa |E_j|, floor_j)
    EQ_j  iff |d_j| + 1.645 sigma_j <= omega_j            (one-sided 95% upper bound)
    OUT_j iff |d_j| - 2.5758 sigma_j > omega_j            (Bonferroni over five moments)
A point is ADEQ if every moment is EQ, INADEQ if any moment is OUT, else NEITHER.
A family is ADEQUATE if some feasible point is ADEQ, INADEQUATE if every feasible point is
INADEQ (or none is feasible), else INCONCLUSIVE. The family's points are the 12 x 12 coarse
lattice plus one level of refinement chosen by the recorded rule (cells whose corners
disagree on adequacy at some kappa, or that contain the kappa* minimiser; edge midpoints and
centre; at most 144 points).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

KAPPAS = (0.15, 0.25, 0.35)
Z_EQ = float(norm.ppf(0.95))
Z_OUT = float(norm.ppf(1 - 0.05 / 10))
TABLE = {
    ("ADEQUATE", "INADEQUATE"): "HISTORY_FAVOURED",
    ("INADEQUATE", "ADEQUATE"): "LATENT_FAVOURED",
    ("ADEQUATE", "ADEQUATE"): "EQUIVALENT",
    ("INADEQUATE", "INADEQUATE"): "NEITHER",
}


def comparative(a2: str, a1p: str) -> str:
    return TABLE.get((a2, a1p), "UNRESOLVED")


def parse(coord: str) -> tuple[float, float]:
    a, b = coord.split(",")
    return float(a), float(b)


class Rule:
    def __init__(self, E, floor):
        self.E = np.abs(np.asarray(E, float))
        self.floor = np.asarray(floor, float)
        self.tau = {k: np.maximum(k * self.E, self.floor) for k in KAPPAS}

    def points(self, x, se, M, S, R, F):
        """Status ('A','I','N','X' for infeasible) per kappa and kappa* for arrays of points."""
        d = np.abs(np.asarray(x)[None, :] - M)
        sig = np.sqrt(np.asarray(se)[None, :] ** 2 + S**2 / R[:, None])
        U, L = d + Z_EQ * sig, d - Z_OUT * sig
        st = {}
        for k in KAPPAS:
            s = np.where((U <= self.tau[k]).all(1), "A",
                         np.where((L > self.tau[k]).any(1), "I", "N"))
            st[k] = np.where(F, s, "X")
        kj = np.where(U <= self.floor[None, :], 0.0, U / self.E[None, :])
        ks = np.where(F, kj.max(1), np.inf)
        return st, ks, kj


def arrays(entries: dict):
    coords = [parse(c) for c in entries]
    vals = list(entries.values())
    M = np.array([e["mean"] for e in vals])
    S = np.array([e["sd"] for e in vals])
    R = np.array([e["n"] for e in vals], float)
    F = np.array([e["feasible"] for e in vals])
    return coords, M, S, R, F


def refine(evaluated: dict, kstar: dict, step: float, budget: int = 144):
    """One refinement level: the recorded rule (edge midpoints and centre of selected cells)."""
    fin = {c: v for c, v in kstar.items() if math.isfinite(v)}
    argmin = min(fin, key=lambda c: (fin[c], c)) if fin else None
    cells = []
    for ip, ia in sorted(evaluated):
        corners = [(ip, ia), (ip + step, ia), (ip, ia + step), (ip + step, ia + step)]
        if not all(c in evaluated for c in corners):
            continue
        mixed = any(len({evaluated[c][k] for c in corners}) > 1 for k in KAPPAS)
        has = argmin in corners
        if mixed or has:
            cells.append((0 if has else 1, min(kstar.get(c, math.inf) for c in corners), (ip, ia)))
    cells.sort()
    new, seen, h = [], set(evaluated), step / 2
    for _, _, (ip, ia) in cells:
        for c in ((ip + h, ia), (ip, ia + h), (ip + step, ia + h), (ip + h, ia + step), (ip + h, ia + h)):
            if c in seen:
                continue
            if len(new) >= budget:
                return new
            seen.add(c)
            new.append(c)
    return new


class Family:
    """A family's coarse lattice plus its precomputed refinement points."""

    def __init__(self, table: dict, rule: Rule):
        self.rule = rule
        self.cc, self.cM, self.cS, self.cR, self.cF = arrays(table["coarse"])
        self.ref = {parse(c): e for c, e in table["refined"].items()}

    def _verdicts(self, statuses):
        out = {}
        for k in KAPPAS:
            s = [x for x in statuses[k] if x != "X"]
            if any(x == "A" for x in s):
                out[k] = "ADEQUATE"
            elif all(x == "I" for x in s):
                out[k] = "INADEQUATE"
            else:
                out[k] = "INCONCLUSIVE"
        return out

    def evaluate(self, x, se, all_points: bool = False):
        st, ks, _ = self.rule.points(x, se, self.cM, self.cS, self.cR, self.cF)
        statuses = {k: list(st[k]) for k in KAPPAS}
        if all_points:
            new = list(self.ref)
        else:
            ev = {c: {k: st[k][i] == "A" for k in KAPPAS} for i, c in enumerate(self.cc)}
            new = refine(ev, {c: ks[i] for i, c in enumerate(self.cc)}, 1.0)
            missing = [c for c in new if c not in self.ref]
            if missing:
                raise KeyError(f"refinement point(s) not in the stored tables: {missing[:3]}")
        if new:
            ent = {f"{c[0]:g},{c[1]:g}": self.ref[c] for c in new}
            _, M, S, R, F = arrays(ent)
            st2, _, _ = self.rule.points(x, se, M, S, R, F)
            for k in KAPPAS:
                statuses[k] += list(st2[k])
        return self._verdicts(statuses)
