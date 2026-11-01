"""Deterministic seed tree and common-random-number streams (design §AA).

Every draw comes from ``SeedSequence(entropy=20261015, spawn_key=(8, stage,
truth_class, truth_point, q, role, family, grid_point, replicate, stream))``.
Unused key positions are 0, so a stream depends only on its key, never on execution
order.

Roles:
- DATA streams generate truth and data panels;
- MODEL streams generate model-evaluation panels;
- CALIBRATION streams drive the CI-1 heterogeneity fit.

The three roles are always independent of one another.

Common random numbers:
- Within a family search, a MODEL replicate uses the same streams at every grid
  point: ``grid_point`` is not part of MODEL keys.
- Across A1′ and A2, the heterogeneity base draws (ALPHA), the Bernoulli uniforms
  (BERNOULLI) and the P8-005 mixture component uniforms (MIXTURE) are shared: they are
  keyed with ``family = SHARED``. MIXTURE is drawn only under the mixture arm, and a
  new stream code never perturbs any existing stream.
- Architecture-specific streams (AR_INNOVATION, HMM, PERIODIC phases) carry the
  family code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

from complete_journey.abm import config
from complete_journey.abm.config import require


class Role(IntEnum):
    DATA = 0
    MODEL = 1
    CALIBRATION = 2


class Stream(IntEnum):
    ALPHA = 1  # household heterogeneity base normals xi_i (shared across families)
    BERNOULLI = 2  # U_it uniforms (shared across families)
    AR_INNOVATION = 3  # A1' epsilon_it
    HMM = 4  # A1' sensitivity hidden-state uniforms
    PERIODIC = 5  # shared periodic layer phases phi_i
    HET_PERSISTENCE = 6  # T2(b) household-specific persistence draws
    AVAILABILITY = 7  # S-a assignment of availability windows
    MIXTURE = 8  # P8-005 mixture component uniforms (shared across families)


class Family(IntEnum):
    SHARED = 0
    A0 = 1
    A1 = 2
    A1_PRIME = 3
    A2 = 4
    A1_PRIME_HMM = 5
    A2_RECENCY = 6


SHARED_STREAMS = frozenset({Stream.ALPHA, Stream.BERNOULLI, Stream.MIXTURE})
#: Which family-specific stream each architecture may draw.
FAMILY_STREAMS = {
    Family.A0: frozenset(),
    Family.A1: frozenset(),
    Family.A1_PRIME: frozenset({Stream.AR_INNOVATION}),
    Family.A2: frozenset(),
    Family.A1_PRIME_HMM: frozenset({Stream.HMM}),
    Family.A2_RECENCY: frozenset(),
}


@dataclass(frozen=True)
class SeedKey:
    """Context of a draw; the stream is chosen at draw time."""

    stage: int
    role: Role
    replicate: int = 0
    truth_class: int = 0
    truth_point: int = 0
    q: int = 0
    family: Family = Family.SHARED
    grid_point: int = 0

    def spawn_key(self, stream: Stream) -> tuple[int, ...]:
        require(self.stage >= 0 and self.replicate >= 0, "seed key fields must be non-negative")
        family = Family.SHARED if stream in SHARED_STREAMS else self.family
        # CRN across grid points: model and calibration streams ignore the grid point.
        grid = 0 if self.role in (Role.MODEL, Role.CALIBRATION) else self.grid_point
        return (
            config.SEED_NAMESPACE,
            int(self.stage),
            int(self.truth_class),
            int(self.truth_point),
            int(self.q),
            int(self.role),
            int(family),
            int(grid),
            int(self.replicate),
            int(stream),
        )

    def generator(self, stream: Stream) -> np.random.Generator:
        seq = np.random.SeedSequence(entropy=config.SEED_ENTROPY, spawn_key=self.spawn_key(stream))
        return np.random.default_rng(seq)

    def with_replicate(self, replicate: int) -> SeedKey:
        return replace(self, replicate=replicate)


@dataclass(frozen=True)
class Draws:
    """Pre-drawn random inputs of one world over the full simulated horizon."""

    xi: np.ndarray  # (n,) heterogeneity base normals
    uniforms: np.ndarray  # (n, TOTAL_SIM_WEEKS) Bernoulli uniforms
    innovations: np.ndarray | None  # (n, TOTAL_SIM_WEEKS) A1' N(0,1) innovations
    hmm: np.ndarray | None  # (n, TOTAL_SIM_WEEKS) HMM uniforms
    phases: np.ndarray | None  # (n,) periodic phases in {0, 1}
    mixture: np.ndarray | None = None  # (n,) P8-005 component uniforms (mixture arm only)


def draws(
    key: SeedKey,
    n: int,
    family: Family,
    periodic: bool = False,
    weeks: int = config.TOTAL_SIM_WEEKS,
    mixture: bool = False,
) -> Draws:
    """All random inputs for one world; shared streams are identical across families."""
    require(n > 0, "a world needs at least one household")
    key = replace(key, family=family)
    allowed = FAMILY_STREAMS[family]
    xi = key.generator(Stream.ALPHA).standard_normal(n)
    uniforms = key.generator(Stream.BERNOULLI).random((n, weeks))
    innovations = (
        key.generator(Stream.AR_INNOVATION).standard_normal((n, weeks))
        if Stream.AR_INNOVATION in allowed
        else None
    )
    hmm = key.generator(Stream.HMM).random((n, weeks)) if Stream.HMM in allowed else None
    # Design §AA lists the periodic phases among the architecture-specific streams;
    # the layer's definition and phase distribution are identical across families (§M).
    phases = key.generator(Stream.PERIODIC).integers(0, 2, n) if periodic else None
    components = key.generator(Stream.MIXTURE).random(n) if mixture else None
    return Draws(xi, uniforms, innovations, hmm, phases, components)


#: Truth-class codes for DATA keys (§AA). Stage separates gate panels (Stage 1) from
#: RQ2 panels (Stage 3), so the two never share streams.
TRUTH_CLASS_CODE = {"T0": 1, "T1": 2, "T2": 3, "T3": 4, "T4": 5, "T5": 6, "PSEUDO": 9}


def truth_key(stage: int, truth_class: str, truth_point: int, q: int) -> SeedKey:
    """DATA key of truth panel q at a truth point (distinct per point, q and stage)."""
    require(truth_class in TRUTH_CLASS_CODE, f"unknown truth class {truth_class}")
    require(truth_point >= 0 and q >= 0, "truth point and panel index must be non-negative")
    return SeedKey(
        stage=stage,
        role=Role.DATA,
        truth_class=TRUTH_CLASS_CODE[truth_class],
        truth_point=truth_point + 1,
        q=q,
    )
