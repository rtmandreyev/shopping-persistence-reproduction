"""Nonlinear span-parity engine (design §N, §AB.1).

The inputs are **simulated images only**: for every feasible grid point of a family,
the simulated mean of the image moments (production: P1, P2, P3) and their simulated
sampling SDs. The input type is ``SimulatedImage``, so no ``TargetVector`` (real or
synthetic) can enter.

- **Image region.** The image region of a family is the union of boxes of half-width
  2·s_j around its image points (a simulation-only scale).
- **Coverage.** C(F→G) is the share of F's image points inside G's image region.
- **Parity holds** if C(A1′→A2) ≥ 0.90 and C(A2→A1′) ≥ 0.90.
- **Trimming** (applied once). Otherwise each family keeps the points whose image lies
  in the other family's (untrimmed) image region.
- **STOP** if either family keeps fewer than 50% of its feasible points.

**P8-005.** ``span_check`` (the gate, with trimming and STOP) is kept only to reproduce
the archived Stage 1 (snapshot ``e59d101b…``). The replacement Stage 1 (S1-P8005) calls
``descriptive_map`` only: the same images and boxes, reported as a DESCRIPTIVE
REPRESENTATIONAL MAP, with no coverage threshold, no retention floor, no trimming and
no status that can stop anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from complete_journey.abm import config
from complete_journey.abm.config import require

PARITY, TRIMMED, STOP = "PARITY", "TRIMMED", "STOP"


@dataclass(frozen=True)
class SimulatedImage:
    family: str
    means: np.ndarray  # (points, K)
    sds: np.ndarray  # (points, K)
    origin: str = "simulated"

    def __post_init__(self):
        require(self.origin == "simulated", "span images must be simulated, never targets")
        m, s = np.asarray(self.means, float), np.asarray(self.sds, float)
        require(m.ndim == 2 and m.shape == s.shape and m.shape[0] > 0, "image shape mismatch")
        require(np.isfinite(m).all() and (s >= 0).all(), "image values must be finite")
        object.__setattr__(self, "means", m)
        object.__setattr__(self, "sds", s)


@dataclass(frozen=True)
class SpanResult:
    coverage_a_in_b: float
    coverage_b_in_a: float
    keep_a: np.ndarray
    keep_b: np.ndarray
    status: str


def inside_region(points: np.ndarray, region: SimulatedImage) -> np.ndarray:
    """Boolean per point: inside at least one box of half-width 2·s_j of the region."""
    half = config.SPAN_BOX_SD * region.sds  # (G, K)
    diff = np.abs(points[:, None, :] - region.means[None, :, :])  # (P, G, K)
    return (diff <= half[None, :, :]).all(axis=2).any(axis=1)


def span_check(a: SimulatedImage, b: SimulatedImage) -> SpanResult:
    require(
        isinstance(a, SimulatedImage) and isinstance(b, SimulatedImage), "simulated images only"
    )
    require(a.means.shape[1] == b.means.shape[1], "images must share coordinates")
    a_in_b = inside_region(a.means, b)
    b_in_a = inside_region(b.means, a)
    ca, cb = float(a_in_b.mean()), float(b_in_a.mean())
    if ca >= config.SPAN_COVERAGE_MIN and cb >= config.SPAN_COVERAGE_MIN:
        return SpanResult(ca, cb, np.ones(len(a_in_b), bool), np.ones(len(b_in_a), bool), PARITY)
    retained = min(a_in_b.mean(), b_in_a.mean())
    status = STOP if retained < config.SPAN_RETAIN_MIN else TRIMMED
    return SpanResult(ca, cb, a_in_b, b_in_a, status)


DESCRIPTIVE_MAP = "DESCRIPTIVE REPRESENTATIONAL MAP"


def descriptive_map(a: SimulatedImage, b: SimulatedImage) -> dict:
    """P8-005 span map: coverage and unique/common counts, descriptive only.

    Returns no gate status, applies no threshold and trims nothing.
    """
    require(
        isinstance(a, SimulatedImage) and isinstance(b, SimulatedImage), "simulated images only"
    )
    require(a.means.shape[1] == b.means.shape[1], "images must share coordinates")
    a_in_b = inside_region(a.means, b)
    b_in_a = inside_region(b.means, a)
    return {
        "status": DESCRIPTIVE_MAP,
        "coverage_a_in_b": float(a_in_b.mean()),
        "coverage_b_in_a": float(b_in_a.mean()),
        "a_points": int(a_in_b.size),
        "b_points": int(b_in_a.size),
        "a_common": int(a_in_b.sum()),
        "a_unique": int((~a_in_b).sum()),
        "b_common": int(b_in_a.sum()),
        "b_unique": int((~b_in_a).sum()),
        "a_in_b": a_in_b.tolist(),
        "b_in_a": b_in_a.tolist(),
    }
