"""Phase 8 resumable checkpoint store: atomic, content-addressed, identifier-free.

One JSON file per job, written to a temporary sibling and moved into place with
``os.replace``, so an interrupted job can never be observed as complete. Payload bytes
are hashed into the record, so tampering is detected on load instead of silently resumed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from complete_journey.abm.config import Phase8ContractError, require

SCHEMA = "phase8-checkpoint-v1"
NONFINITE = {"inf": "__inf__", "-inf": "__-inf__", "nan": "__nan__"}
#: Household-/product-/store-level identifiers may never enter a checkpoint payload.
FORBIDDEN_IDENTIFIER_KEYS = frozenset({"household_key", "BASKET_ID", "PRODUCT_ID", "STORE_ID"})
JOB_ID_LENGTH = 24


def _sha256_hex(data: bytes) -> str:
    """Hex sha256 digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    """Compact, key-sorted, NaN-free JSON bytes; the only serialisation used here."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return text.encode("utf-8")


def _to_builtin(value: Any) -> Any:
    """Recursively replace numpy scalars/arrays (and containers) with plain Python."""
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return _to_builtin(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        # Strict JSON has no inf/nan; kappa* = inf is a legitimate result, so encode it.
        return NONFINITE[("nan" if math.isnan(value) else ("inf" if value > 0 else "-inf"))]
    if isinstance(value, dict):
        return {k if isinstance(k, str) else str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _forbidden_identifier_keys(value: Any) -> list[str]:
    """Sorted forbidden identifier keys found anywhere in value, recursively."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key in FORBIDDEN_IDENTIFIER_KEYS:
                found.add(key)
            found |= set(_forbidden_identifier_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found |= set(_forbidden_identifier_keys(item))
    return sorted(found)


@dataclass(frozen=True)
class JobKey:
    """Identity of one resumable simulation job: stage, family, point, reps and seeds."""

    stage: int
    family: str
    point: str
    rep_start: int
    rep_end: int
    seed_keys: tuple[tuple[int, ...], ...]

    def as_json_fields(self) -> dict[str, Any]:
        """Key fields as canonical-JSON-ready plain Python (seed tuples become lists)."""
        return {
            "stage": int(self.stage),
            "family": str(self.family),
            "point": str(self.point),
            "rep_start": int(self.rep_start),
            "rep_end": int(self.rep_end),
            "seed_keys": [list(seed) for seed in self.seed_keys],
        }

    @property
    def job_id(self) -> str:
        """Content address: first 24 hex chars of the sha256 of the key fields."""
        return _sha256_hex(_canonical_json(self.as_json_fields()))[:JOB_ID_LENGTH]


class CheckpointStore:
    """Directory of per-job checkpoint records, addressed by JobKey.job_id."""

    def __init__(self, root: Path, config_hash: str, source_commit: str) -> None:
        self.root = Path(root)
        self.config_hash = str(config_hash)
        self.source_commit = str(source_commit)

    # --- public API -------------------------------------------------------------------

    def save(self, key: JobKey, payload: dict) -> dict:
        """Atomically persist a completed job payload; idempotent for identical output.

        Raises Phase8ContractError if the payload carries a forbidden identifier key,
        if an existing checkpoint for this job is unreadable or belongs to another
        config, or if it holds a different output for the same job.
        """
        require(isinstance(key, JobKey), "key must be a JobKey")
        require(isinstance(payload, dict), "payload must be a dict")
        clean = _to_builtin(payload)
        offenders = _forbidden_identifier_keys(clean)
        require(
            not offenders,
            f"payload contains forbidden identifier keys: {offenders}",
        )
        output_sha256 = _sha256_hex(_canonical_json(clean))
        existing = self.load(key)
        if existing is not None:
            require(
                existing["output_sha256"] == output_sha256, "duplicate job with different output"
            )
            return existing
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "job_id": key.job_id,
            "config_hash": self.config_hash,
            "source_commit": self.source_commit,
            "stage": key.stage,
            "family": key.family,
            "point": key.point,
            "rep_start": key.rep_start,
            "rep_end": key.rep_end,
            "seed_keys": [list(seed) for seed in key.seed_keys],
            "status": "complete",
            "output_sha256": output_sha256,
            "payload": clean,
        }
        record["record_sha256"] = _sha256_hex(_canonical_json(record))
        self._atomic_write(self._path(key.job_id), _canonical_json(record))
        return record

    def load(self, key: JobKey) -> dict | None:
        """Return the validated complete record for key, or None if none is stored.

        Raises Phase8ContractError for an unreadable, tampered, mis-keyed record or one
        written under a different config_hash.
        """
        path = self._path(key.job_id)
        if not path.exists():
            return None
        record = self._read_record(path)
        require(record.get("schema") == SCHEMA, f"corrupt checkpoint {path.name}: bad schema")
        body = {k: v for k, v in record.items() if k != "record_sha256"}
        require(
            record.get("record_sha256") == _sha256_hex(_canonical_json(body)),
            f"corrupt checkpoint {path.name}: record_sha256 mismatch",
        )
        require(
            record.get("output_sha256") == _sha256_hex(_canonical_json(record.get("payload"))),
            f"corrupt checkpoint {path.name}: output_sha256 mismatch",
        )
        require(record.get("config_hash") == self.config_hash, "config mismatch")
        require(self._fields_match(record, key), f"checkpoint {path.name} does not match job key")
        return record

    def completed(self, keys: Iterable[JobKey]) -> list[JobKey]:
        """Keys with a valid complete checkpoint, in input order.

        Absent checkpoints are skipped; a present but unreadable, tampered, foreign-config
        or mis-keyed checkpoint raises instead of being treated as pending.
        """
        return [key for key in keys if self._is_complete(key)]

    def pending(self, keys: Iterable[JobKey]) -> list[JobKey]:
        """Keys still needing a run, in input order (the resume list)."""
        items = list(keys)
        done = {key.job_id for key in items if self._is_complete(key)}
        return [key for key in items if key.job_id not in done]

    # --- internals --------------------------------------------------------------------

    def _path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def _is_complete(self, key: JobKey) -> bool:
        record = self.load(key)
        return record is not None and record.get("status") == "complete"

    @staticmethod
    def _fields_match(record: dict, key: JobKey) -> bool:
        expected = key.as_json_fields()
        return all(record.get(field) == value for field, value in expected.items())

    @staticmethod
    def _read_record(path: Path) -> dict:
        try:
            raw = path.read_bytes()
            record = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Phase8ContractError(f"corrupt checkpoint {path.name}: unreadable JSON") from exc
        require(isinstance(record, dict), f"corrupt checkpoint {path.name}: not a JSON object")
        return record

    def _atomic_write(self, path: Path, data: bytes) -> None:
        """Write data to a same-directory temp file, fsync, then os.replace into place."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
