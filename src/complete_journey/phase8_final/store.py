"""Checkpoint storage for the final run, and the integrity-checked S1-P8005 import.

- ``ShardedStore``: the frozen ``CheckpointStore`` (atomic writes, record and output
  hashes, config-hash refusal, duplicate refusal) with files sharded two levels deep by
  job id, so no directory holds more than a few hundred files.
- ``legacy``: read-only access to the S1-P8005 checkpoints. It re-verifies the S1-P8005
  snapshot (``08a6a672…``): its recorded source hashes must equal the current
  ``abm/*.py``, ``prediction/*.py`` and ``scripts/run_abm.py`` byte for byte, which
  proves that architectures, seeds, calibration, mixture, moments and operators are
  unchanged. Every imported record is then verified by the frozen store (record hash,
  output hash, config hash). Imported checkpoints are never copied or modified.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from complete_journey.abm import stage1_p8005 as s1p
from complete_journey.abm.checkpoint import CheckpointStore
from complete_journey.abm.config import require
from complete_journey.paths import PROJECT_ROOT

S1P8005_SNAPSHOT_SHA256 = "08a6a67279cfc7d9351d3fc62c7dc5a2da7eae6d4fb40679fd23d910ffd1ef94"


class ShardedStore(CheckpointStore):
    """CheckpointStore whose files live at root/ab/cd/<job_id>.json."""

    def _path(self, job_id: str) -> Path:
        return self.root / job_id[:2] / job_id[2:4] / f"{job_id}.json"

    def _atomic_write(self, path: Path, data: bytes) -> None:
        # The frozen writer puts its temp file beside the target; only the shard
        # directory has to exist first.
        path.parent.mkdir(parents=True, exist_ok=True)
        super()._atomic_write(path, data)

    def files(self):
        return self.root.glob("*/*/*.json")


@lru_cache(maxsize=1)
def legacy() -> CheckpointStore:
    """The S1-P8005 store, after re-verifying its snapshot against the current source."""
    digest = s1p.verify_snapshot()
    require(digest == S1P8005_SNAPSHOT_SHA256, "unexpected S1-P8005 snapshot")
    return s1p.store()


def legacy_load(job: s1p.Job) -> dict:
    rec = legacy().load(job.key())
    require(rec is not None, f"missing S1-P8005 checkpoint {job}")
    return rec["payload"]


def legacy_record_sha(job: s1p.Job) -> str:
    rec = legacy().load(job.key())
    require(rec is not None, f"missing S1-P8005 checkpoint {job}")
    return rec["output_sha256"]


def file_sha(relative: str, root: Path = PROJECT_ROOT) -> str:
    return hashlib.sha256((root / relative).read_bytes()).hexdigest()


def import_verification() -> dict:
    """Snapshot, source, config and seed-semantics evidence for the S1-P8005 import."""
    digest = s1p.verify_snapshot()
    snap = json.loads((PROJECT_ROOT / s1p.SNAPSHOT).read_text())
    semantic = [
        "src/complete_journey/abm/architectures.py",
        "src/complete_journey/abm/seeds.py",
        "src/complete_journey/abm/calibration.py",
        "src/complete_journey/abm/mixture.py",
        "src/complete_journey/abm/moments.py",
        "src/complete_journey/abm/operators.py",
        "src/complete_journey/abm/config.py",
        "src/complete_journey/abm/regions.py",
        "src/complete_journey/abm/mapping.py",
        "src/complete_journey/abm/adequacy.py",
        "src/complete_journey/abm/stage1_p8005.py",
    ]
    hashes = {f: file_sha(f) for f in semantic}
    require(
        all(snap["source_sha256"][f] == h for f, h in hashes.items()),
        "S1-P8005 import refused: a simulation-semantic source file changed",
    )
    moment_engine = {
        f: h
        for f, h in snap["source_sha256"].items()
        if f.startswith("src/complete_journey/prediction/")
    }
    return {
        "s1p8005_snapshot_sha256": digest,
        "config_hash": legacy().config_hash,
        "semantic_source_sha256": hashes,
        "moment_engine_sha256": moment_engine,
        "seed_semantics": snap["seed_tree"],
        "archived_stage1": s1p.verify_archived_stage1(),
    }
