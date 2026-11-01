"""Phase 8 real-data and heavy-run guards (design §AP, §AI, §AK).

**Real-data guard.** Every target vector carries its origin ("synthetic" or "real").
Architecture search, CI-1 calibration, tolerance construction and adequacy evaluation
refuse real targets unless the committed ``config.AUTHORIZED_STAGE`` reaches the stage
that may use them. In Run B that stage is 0, so no real target is usable.
- Stage 1 may calibrate A1 to the real CI-1 and build tolerances.
- Stage 4 may run the one-time real comparison.

The Run C archive may be read only for **contract verification**: hash checks and
constant pins. Raising the authorized stage is a later, explicit researcher decision
recorded in a commit. No flag, environment variable or keyword can bypass the guard.

**Heavy-run guard.** The production modes (emulator build, RQ2, real comparison,
N-scaling) refuse until their stage is authorized. There is no generic force switch.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from complete_journey.abm import config
from complete_journey.abm.config import require
from complete_journey.paths import PROJECT_ROOT
from complete_journey.reporting.phase7_publication import ACCEPTED_INPUTS, RUN_C

REAL_ARCHIVE = f"{RUN_C}/moment_register_real.csv"
#: Stage required for each use of real data (design §AI).
REAL_USE_STAGE = {
    "ci1_calibration": 1,
    "tolerance": 1,
    "comparison": 4,
    "availability": 4,  # real E-2 pairs as S-a availability windows
    "calendar": 5,  # real E-1-fitted calendar path (Stage 5 sensitivity only)
}
#: Stage required for each heavy production mode (design §AK).
HEAVY_MODE_STAGE = {
    "build-emulator": 1,
    "perfect-model-production": 1,
    "run-rq2": 3,
    "run-real": 4,
    "run-n-scaling": 5,
}


@dataclass(frozen=True)
class TargetVector:
    """Named target values (CI-1 components or P1..P5) with their origin."""

    origin: str
    values: dict[str, float]
    se: dict[str, float] | None = None

    def __post_init__(self):
        require(self.origin in ("synthetic", "real"), "target origin must be synthetic or real")
        require(all(np.isfinite(v) for v in self.values.values()), "targets must be finite")
        if self.origin == "synthetic":
            require(
                not reproduces_archive(self.values),
                "target labelled synthetic reproduces the real Run C archive",
            )


#: Phase 8 target names -> archived (moment, component) under operator O.
ARCHIVE_NAME = {
    **{p8: (mid, comp) for p8, mid, comp in config.PRIMARY_MOMENTS},
    **{c: ("CI-1", c) for c in config.CI1_COMPONENTS},
}


@lru_cache(maxsize=1)
def _archive_estimates() -> dict:
    return {k: v["estimate"] for k, v in archive_contract_values().items()}


def reproduces_archive(values: dict[str, float]) -> bool:
    """True if every named value equals its archived real estimate (to 1e-12).

    It mirrors the Phase 7 fingerprint guard: real values cannot pose as synthetic.
    """
    names = [k for k in values if k in ARCHIVE_NAME]
    if not names:
        return False
    archive = _archive_estimates()
    return all(abs(values[k] - archive[ARCHIVE_NAME[k]]) <= 1e-12 for k in names)


@dataclass(frozen=True)
class CalendarPath:
    """A common calendar path tau_t for weeks 1–102, with its origin (§Z, §AP)."""

    origin: str
    values: tuple[float, ...]

    def __post_init__(self):
        require(self.origin in ("synthetic", "real"), "calendar origin must be synthetic or real")
        require(len(self.values) == config.T_WEEKS, "calendar path must cover weeks 1-102")
        if self.origin == "real":
            require_stage("calendar")


def require_stage(use: str) -> None:
    require(use in REAL_USE_STAGE, f"unknown real-data use: {use}")
    require(
        config.AUTHORIZED_STAGE >= REAL_USE_STAGE[use],
        f"Phase 8: real-data use '{use}' needs Stage {REAL_USE_STAGE[use]} authorization; "
        f"the committed authorized stage is {config.AUTHORIZED_STAGE} (Run B)",
    )


def require_run_stage(stage: int) -> None:
    """Computations stamped with a stage may not exceed the committed authorized stage.

    Seed-stage key 11 is the P8-005 replacement Stage 1 (S1-P8005): protocol Stage 1,
    refused until ``config.P8005_STAGE1_AUTHORIZED`` is set by its authorization commit.
    """
    if stage == config.P8005_STAGE_KEY:
        require_p8005_authorized()
        return
    require(
        0 <= stage <= config.AUTHORIZED_STAGE,
        f"Phase 8: stage {stage} computations need authorization "
        f"(committed authorized stage: {config.AUTHORIZED_STAGE})",
    )


def require_p8005_authorized() -> None:
    require(
        config.P8005_STAGE1_AUTHORIZED and config.AUTHORIZED_STAGE >= 1,
        "Phase 8: the P8-005 replacement Stage 1 (S1-P8005, seed stage key 11) needs its "
        "researcher authorization commit (config.P8005_STAGE1_AUTHORIZED); there is no "
        "bypass.",
    )


def require_target_allowed(target: TargetVector, use: str) -> None:
    """Refuse real targets for any use whose stage is not yet authorized."""
    require(use in REAL_USE_STAGE, f"unknown real-data use: {use}")
    if target.origin == "real":
        require_stage(use)


def require_heavy_authorized(mode: str) -> None:
    require(mode in HEAVY_MODE_STAGE, f"unknown production mode: {mode}")
    require(
        config.AUTHORIZED_STAGE >= HEAVY_MODE_STAGE[mode],
        f"Phase 8: '{mode}' is a production mode that needs Stage {HEAVY_MODE_STAGE[mode]} "
        f"authorization (committed authorized stage: {config.AUTHORIZED_STAGE}). Production "
        "runs require an explicit researcher authorization commit; there is no bypass.",
    )


def verify_real_archive(root=PROJECT_ROOT) -> str:
    """Contract verification only: the Run C register hash equals the accepted hash."""
    digest = hashlib.sha256((root / REAL_ARCHIVE).read_bytes()).hexdigest()
    require(digest == ACCEPTED_INPUTS[REAL_ARCHIVE], "Run C moment archive drifted")
    return digest


def archive_contract_values(root=PROJECT_ROOT) -> dict[tuple[str, str], dict]:
    """Contract verification only: archived (estimate, SE) per (moment, component), O.

    It returns a plain mapping for pinning tests. It is never a ``TargetVector``, so it
    cannot enter calibration, tolerance or adequacy code.
    """
    verify_real_archive(root)
    out = {}
    with (root / REAL_ARCHIVE).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["operator"] == config.OPERATOR_O and row["source"] == "real":
                out[(row["moment_id"], row["component"])] = {
                    "estimate": float(row["estimate"]),
                    "se": float(row["bootstrap_se"]) if row["bootstrap_se"] else None,
                }
    return out
