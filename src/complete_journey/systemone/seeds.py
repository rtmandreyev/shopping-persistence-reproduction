"""Phase 9 seed tree: one new SeedSequence root, spawned by purpose and coordinates."""

from __future__ import annotations

import numpy as np

from complete_journey.systemone import settings as S
from complete_journey.systemone.settings import require


def rng(purpose: str, *coords: int) -> np.random.Generator:
    require(purpose in S.SEED_PURPOSE, f"unknown seed purpose {purpose}")
    require(all(int(c) >= 0 for c in coords), "seed coordinates must be non-negative")
    key = (9, S.SEED_PURPOSE[purpose], *(int(c) for c in coords))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(S.SEED_ROOT, spawn_key=key)))
