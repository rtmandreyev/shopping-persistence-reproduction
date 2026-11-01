"""Finite state lattice, placebo sampling frame (P9-002 §2), drift sentinels and smoke states.

Every set here is built algorithmically from the lattice before any provider output
exists, and each is hashed so it can be frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict

from complete_journey.systemone import settings as S
from complete_journey.systemone.seeds import rng
from complete_journey.systemone.settings import require
from complete_journey.systemone.state import SemanticState


def grid(k: int) -> tuple[float, ...]:
    require(k in S.GRIDS, f"no frozen grid with K={k}")
    return S.GRIDS[k]


def band(profile: float) -> str:
    for name, lo, hi in S.PROFILE_BANDS:
        if name == "low" and profile <= hi + 1e-12:
            return name
        if name == "high" and profile >= lo - 1e-12:
            return name
    return "mid"


def cell_states(cell: str, k: int, profile_stated: bool = True) -> list[SemanticState]:
    """All states of one information cell (default placebo value)."""
    hv, nv = S.CELL_VISIBILITY[cell]
    profiles = list(grid(k)) if profile_stated else [None]
    hists = list(S.HISTORY_PATTERNS) if hv else [None]
    needs = list(S.NEED_LEVELS) if nv else [None]
    return [SemanticState(cell, f, h, n) for f in profiles for h in hists for n in needs]


def lattice(k: int, profile_stated: bool = True) -> dict[str, list[SemanticState]]:
    return {c: cell_states(c, k, profile_stated) for c in S.CELLS}


def expected_counts(k: int) -> dict[str, int]:
    return {
        S.NEITHER: k,
        S.LATENT: 3 * k,
        S.HISTORY: 16 * k,
        S.BOTH: 48 * k,
        "profile_unavailable": 1 + 3 + 16 + 48,
    }


def all_states(k: int) -> list[SemanticState]:
    """Production template-A lattice: profile-stated cells plus the profile-unavailable
    secondary surface (static only)."""
    out = [s for c in S.CELLS for s in cell_states(c, k)]
    out += [s for c in S.CELLS for s in cell_states(c, k, profile_stated=False)]
    require(len(out) == len(set(out)), "duplicate lattice states")
    return out


def stratum(s: SemanticState) -> tuple:
    """P9-002 §2 strata (only HISTORY and BOTH are sampled; the others are fully covered)."""
    require(s.profile is not None, "placebo strata exist for profile-stated states only")
    if s.cell == S.HISTORY:
        return (s.cell, band(s.profile), s.count)
    if s.cell == S.BOTH:
        return (s.cell, band(s.profile), s.count, s.need)
    return (s.cell,)


def placebo_frame(k: int) -> list[SemanticState]:
    """The predeclared stratified placebo subset (P9-002 §2), seeded, sorted by key."""
    chosen: list[SemanticState] = []
    chosen += cell_states(S.NEITHER, k) + cell_states(S.LATENT, k)
    for cell in (S.HISTORY, S.BOTH):
        strata: dict[tuple, list[SemanticState]] = defaultdict(list)
        for s in cell_states(cell, k):
            strata[stratum(s)].append(s)
        for idx, key in enumerate(sorted(strata, key=repr)):
            members = sorted(strata[key], key=lambda x: x.key())
            m = min(
                len(members),
                max(S.PLACEBO_MIN_PER_STRATUM, math.ceil(S.PLACEBO_FRACTION * len(members))),
            )
            g = rng("placebo_frame", k, S.CELLS.index(cell), idx)
            pick = sorted(g.choice(len(members), size=m, replace=False).tolist())
            chosen += [members[i] for i in pick]
    return sorted(chosen, key=lambda x: x.key())


def stratum_weights(frame: list[SemanticState], k: int) -> dict[str, float]:
    """Inverse-inclusion weights: each covered state stands for N_stratum / n_stratum states."""
    pop = defaultdict(int)
    for c in S.CELLS:
        for s in cell_states(c, k):
            pop[stratum(s)] += 1
    covered = defaultdict(int)
    for s in frame:
        covered[stratum(s)] += 1
    return {s.key(): pop[stratum(s)] / covered[stratum(s)] for s in frame}


def keys_hash(states: list[SemanticState]) -> str:
    data = json.dumps(sorted(s.key() for s in states), separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def sentinels(k: int) -> list[SemanticState]:
    """24 drift sentinels covering profile extremes and middle, history extremes, every
    need level and every information cell (deterministic rule, no randomness)."""
    g = grid(k)
    lo, mid, hi = g[0], g[(len(g) - 1) // 2], g[-1]
    zero, one, h1000, h0111 = (0, 0, 0, 0), (1, 1, 1, 1), (1, 0, 0, 0), (0, 1, 1, 1)
    out = [SemanticState(S.NEITHER, f, None, None) for f in (lo, mid, hi)]
    out += [
        SemanticState(S.LATENT, f, None, n) for f, n in ((lo, "low"), (mid, "usual"), (hi, "high"))
    ]
    out += [
        SemanticState(S.HISTORY, f, h, None)
        for f, h in ((lo, zero), (lo, one), (mid, h1000), (mid, h0111), (hi, zero), (hi, one))
    ]
    out += [
        SemanticState(S.BOTH, f, h, n)
        for f in (lo, mid, hi)
        for h, n in ((zero, "low"), (one, "high"), (one, "low"), (zero, "high"))
    ]
    require(
        len(out) == S.SENTINELS and len(set(out)) == S.SENTINELS,
        "sentinel rule must give 24 distinct states",
    )
    return out


def smoke_states(k: int) -> list[SemanticState]:
    """Designated technical smoke states: one seeded random state per information cell.
    The selection uses no scientific expectation."""
    out = []
    for i, cell in enumerate(S.CELLS):
        members = sorted(cell_states(cell, k), key=lambda x: x.key())
        out.append(members[int(rng("smoke_states", k, i).integers(len(members)))])
    return out
