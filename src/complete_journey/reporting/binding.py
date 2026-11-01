"""Binding-moment labelling for κ* tables (a correction made after an independent audit).

The frozen Phase 8 helper ``phase8_final.evaluate.binding_moment`` returns
``PRIMARY_IDS[argmax_j κ*_j]``. When every per-moment κ*_j is 0 (each moment's upper
bound already lies inside its resolution floor) the argmax of an all-zero vector is index
0, so it reports "P1" although no moment binds. The frozen module is hash-pinned by the
Phase 8 design snapshot and is left byte-for-byte unchanged; reporting code uses this
helper instead, and the frozen ``real_rq1.json`` / ``t10`` cell is superseded by the
post-audit errata.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

PRIMARY_IDS = ("P1", "P2", "P3", "P4", "P5")


def per_moment_kappa_star(
    upper: Sequence[float], explanandum: Mapping[str, float], floor: Mapping[str, float]
) -> dict[str, float]:
    """κ*_j = 0 if U_j ≤ ρ_j, else U_j/|E_j| (+∞ if E_j = 0), as in ``abm.adequacy``."""
    out = {}
    for u, j in zip(upper, PRIMARY_IDS, strict=True):
        e = abs(explanandum[j])
        out[j] = 0.0 if u <= floor[j] else (u / e if e else math.inf)
    return out


def binding_moment(kappa_star_by_moment: Mapping[str, float]) -> str | None:
    """The moment with the largest κ*_j, or ``None`` when no moment binds (all κ*_j = 0).

    Ties among positive maxima keep the frozen order P1..P5.
    """
    values = [kappa_star_by_moment[j] for j in PRIMARY_IDS]
    top = max(values)
    if top <= 0.0:
        return None
    return PRIMARY_IDS[values.index(top)]
