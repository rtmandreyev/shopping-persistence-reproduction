"""Content-addressed, append-only raw archive with a run lock (B-009 decision logs).

- ``responses/ab/cd/<request_hash>.json`` holds the one successful response. It is
  written atomically and never overwritten.
- ``attempts/<request_hash>/<n>.json`` holds every failed or malformed attempt,
  separately.
- Every record carries its own SHA-256. Corruption is refused on read.
- No credential ever enters a record. ``redact`` scrubs anything that looks like one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
from pathlib import Path

from complete_journey.systemone.settings import Phase9ContractError, require

SCHEMA = "phase9-archive-v1"
_SECRET = re.compile(
    r"(Bearer\s+)[A-Za-z0-9._\-]+|sk-[A-Za-z0-9._\-]{8,}|(api[_-]?key\"?\s*[:=]\s*\"?)[^\s\",]+",
    re.I,
)


def redact(text: str) -> str:
    return _SECRET.sub(lambda m: (m.group(1) or m.group(2) or "") + "[REDACTED]", str(text))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dump(rec: dict) -> bytes:
    return (json.dumps(rec, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode()


class Archive:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _resp(self, h: str) -> Path:
        return self.root / "responses" / h[:2] / h[2:4] / f"{h}.json"

    def _atomic(self, path: Path, data: bytes, exclusive: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if exclusive:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            return
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def store_success(self, request_hash: str, record: dict) -> dict:
        """Persist a successful response exactly once. The record is written to a temp file,
        fsynced, then hard-linked into place (atomic and exclusive). A second write is refused."""
        rec = {"schema": SCHEMA, "request_hash": request_hash, "status": "success", **record}
        _no_secrets(rec)
        body = {k: v for k, v in rec.items() if k != "record_sha256"}
        rec["record_sha256"] = _sha(_dump(body))
        path = self._resp(request_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            f.write(_dump(rec))
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise Phase9ContractError(
                f"refusing to overwrite successful response {request_hash}"
            ) from None
        finally:
            tmp.unlink(missing_ok=True)
        return rec

    def store_attempt(self, request_hash: str, record: dict) -> Path:
        """Failed or malformed attempts, kept apart from successes. Each is written to a
        temp file, then hard-linked to the next free index (atomic and exclusive)."""
        d = self.root / "attempts" / request_hash
        d.mkdir(parents=True, exist_ok=True)
        n = len(list(d.glob("*.json")))
        while True:
            rec = {"schema": SCHEMA, "request_hash": request_hash, "attempt": n, **record}
            rec = json.loads(redact(json.dumps(rec)))
            _no_secrets(rec)
            path = d / f"{n:03d}.json"
            tmp = d / f".{n:03d}.{os.getpid()}.tmp"
            tmp.write_bytes(_dump(rec))
            try:
                os.link(tmp, path)
                return path
            except FileExistsError:
                n += 1
            finally:
                tmp.unlink(missing_ok=True)

    def load(self, request_hash: str) -> dict | None:
        path = self._resp(request_hash)
        if not path.exists():
            return None
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise Phase9ContractError(f"corrupt archive record {request_hash}") from exc
        body = {k: v for k, v in rec.items() if k != "record_sha256"}
        require(rec.get("schema") == SCHEMA, f"corrupt archive record {request_hash}: schema")
        require(
            rec.get("record_sha256") == _sha(_dump(body)),
            f"corrupt archive record {request_hash}: hash",
        )
        require(rec.get("request_hash") == request_hash, f"archive record {request_hash} mis-keyed")
        return rec

    def done(self, request_hash: str) -> bool:
        return self.load(request_hash) is not None

    def successes(self):
        for p in sorted((self.root / "responses").glob("*/*/*.json")):
            yield self.load(p.stem)

    def attempts(self, request_hash: str) -> list[dict]:
        d = self.root / "attempts" / request_hash
        return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def _no_secrets(rec: dict) -> None:
    text = json.dumps(rec)
    require(redact(text) == text, "a record would contain a credential-like string")


class RunLock:
    """One live run per archive root. A lock whose process is gone on this host is stale."""

    def __init__(self, root: Path):
        self.path = Path(root) / "run.lock"

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            info = json.loads(self.path.read_text()) if self.path.exists() else None
        except (json.JSONDecodeError, OSError):
            info = {}
        if info is not None:
            if info.get("host") == socket.gethostname() and _alive(info.get("pid")):
                raise Phase9ContractError(f"run lock held by live process {info.get('pid')}")
            self.path.unlink(missing_ok=True)  # stale
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            raise Phase9ContractError("run lock acquired concurrently by another process") from None
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "utc": time.time()}, f)

    def release(self) -> None:
        if self.path.exists():
            info = json.loads(self.path.read_text())
            if info.get("pid") == os.getpid():
                self.path.unlink()


def _alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
