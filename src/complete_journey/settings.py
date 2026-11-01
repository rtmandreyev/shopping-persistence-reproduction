"""Small, dependency-light project settings.

Kept intentionally minimal. Analysis choices belong in gated phases, not here.
"""

from __future__ import annotations

import random

import numpy as np

RANDOM_SEED = 20260101


def seed_everything(seed: int = RANDOM_SEED) -> int:
    """Seed Python and NumPy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    return seed
