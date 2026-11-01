"""RQ2 freeze and the guarded one-time real reveal (P8-006).

Order (mandatory): Stage 1 hard gates pass → full T0–T4 RQ2 completes → RQ2 artifacts
written (write-once) → RQ2 freeze record (hash of every RQ2 artifact) → only then the
reveal ledger may be created, and only through it can the real P1..P5 targets or the
real E-2 availability windows be read.

- After the freeze, no RQ2 artifact can be rewritten (``write_once`` refuses different
  bytes), and the freeze can never be replaced.
- ``config.AUTHORIZED_STAGE`` stays at 1, so no legacy code path can compare A1′/A2
  with real dynamic targets. P8-006 authorizes the Stage 4 uses (comparison, S-a
  availability) exclusively through this module, after the freeze. (The archived real
  P1..P5 are read earlier only for the P8-003 A1 tolerance construction, E_j and ρ_j,
  an authorized Stage 1 use.)
- The Stage 1 gate record is hashed into the freeze, and the pipeline recomputes the
  gates from the checkpoints and requires equality before unlocking.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from complete_journey.abm.config import require
from complete_journey.abm.guards import TargetVector
from complete_journey.abm.operators import household_keys
from complete_journey.abm.seeds import Role, SeedKey, Stream
from complete_journey.paths import PROJECT_ROOT
from complete_journey.phase8_final import settings as S

RQ2_ARTIFACTS = (
    "rq2_truths.json",
    "rq2_per_truth.json",
    "rq2_results.json",
    "calibration_noise.json",
)
STAGE1_RECORD = "stage1_gates.json"
FREEZE = "rq2_freeze.json"
LEDGER = "real_reveal_ledger.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dump(payload) -> bytes:
    def default(v):
        if isinstance(v, float):
            return str(v)
        return v.tolist() if hasattr(v, "tolist") else str(v)

    def keys(v):
        if isinstance(v, dict):
            return {
                ("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): keys(x)
                for k, x in v.items()
            }
        if isinstance(v, (list, tuple)):
            return [keys(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v):
            return str(v)
        return v

    return (json.dumps(keys(payload), indent=2, sort_keys=True, default=default) + "\n").encode()


def archive_path(name: str, archive: str | None = None) -> Path:
    return PROJECT_ROOT / (archive or S.ARCHIVE) / name


def write_once(name: str, payload, archive: str | None = None) -> str:
    """Write an archive artifact; an existing file must be byte-identical (resume)."""
    path = archive_path(name, archive)
    data = _dump(payload)
    if path.exists():
        require(path.read_bytes() == data, f"refusing to rewrite archived artifact {name}")
        return _sha(data)
    if name in RQ2_ARTIFACTS:
        require(not archive_path(FREEZE, archive).exists(), "RQ2 is frozen: no RQ2 rewrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return _sha(data)


def _exclusive(path: Path, payload: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(_dump(payload))
        handle.flush()
        os.fsync(handle.fileno())


def freeze_rq2(archive: str | None = None) -> dict:
    """Create the RQ2 freeze record once; afterwards it must match the artifacts."""
    hashes = {
        n: _sha(archive_path(n, archive).read_bytes()) for n in (*RQ2_ARTIFACTS, STAGE1_RECORD)
    }
    path = archive_path(FREEZE, archive)
    if not path.exists():
        _exclusive(
            path,
            {
                "rq2_artifact_sha256": hashes,
                "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "run_label": S.RUN_LABEL,
            },
        )
    return verify_freeze(archive)


def verify_freeze(archive: str | None = None) -> dict:
    rec = json.loads(archive_path(FREEZE, archive).read_text())
    for n, h in rec["rq2_artifact_sha256"].items():
        require(
            _sha(archive_path(n, archive).read_bytes()) == h, f"frozen RQ2 artifact {n} changed"
        )
    rec["freeze_sha256"] = _sha(archive_path(FREEZE, archive).read_bytes())
    return rec


def unlock(archive: str | None = None) -> dict:
    """Create (or, on resume, re-verify) the one-time reveal ledger."""
    require(S.FINAL_RUN_AUTHORIZED, "the final run is not authorized")
    gates = json.loads(archive_path(STAGE1_RECORD, archive).read_text())
    require(gates.get("pass") is True, "Stage 1 hard gates did not pass: no reveal")
    freeze = verify_freeze(archive)
    path = archive_path(LEDGER, archive)
    if not path.exists():
        _exclusive(
            path,
            {
                "freeze_sha256": freeze["freeze_sha256"],
                "stage1_gates_sha256": _sha(archive_path(STAGE1_RECORD, archive).read_bytes()),
                "unlocked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "uses": ["comparison (real P1..P5)", "S-a availability (real E-2)"],
            },
        )
    ledger = json.loads(path.read_text())
    require(
        ledger["freeze_sha256"] == freeze["freeze_sha256"],
        "reveal ledger does not match the RQ2 freeze",
    )
    return ledger


def real_targets(archive: str | None = None) -> TargetVector:
    unlock(archive)
    from complete_journey.abm import stage1  # the archived real P1..P5 accessor

    return stage1.real_primary_reference()


def real_windows(n: int, archive: str | None = None) -> dict:
    """Real E-2 (f, l) pairs drawn as S-a availability windows (frozen Phase 7 rule)."""
    unlock(archive)
    from complete_journey.prediction.operators import draw_empirical_windows, empirical_window_pairs
    from complete_journey.prediction.panel import load_real_panel

    pairs = empirical_window_pairs(load_real_panel())
    require(len(pairs) >= n, f"only {len(pairs)} empirical windows for {n} households")
    rng = SeedKey(S.MODEL_STAGE, Role.MODEL, replicate=0).generator(Stream.AVAILABILITY)
    drawn = draw_empirical_windows(pairs, household_keys(n), rng)
    windows = drawn[["f", "l"]].to_numpy(dtype=int).tolist()
    data = json.dumps(windows).encode()
    return {"windows": windows, "sha256": _sha(data), "pairs": int(len(pairs))}
