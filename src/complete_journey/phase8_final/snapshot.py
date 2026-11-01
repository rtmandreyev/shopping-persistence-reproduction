"""Immutable design snapshot of the Phase 8 final run (P8-006)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path

from complete_journey.abm import config as abm_config
from complete_journey.abm.config import require
from complete_journey.paths import PROJECT_ROOT
from complete_journey.phase8_final import settings as S

SOURCE_GLOBS = (
    "src/complete_journey/abm/*.py",
    "src/complete_journey/prediction/*.py",
    "src/complete_journey/phase8_final/*.py",
)
SOURCE_FILES = ("scripts/run_abm.py", "scripts/final_run.py", "scripts/launch_abm.py")
DECISIONS = ("P8-001", "P8-002", "P8-003", "P8-004", "P8-005", "P8-006")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_hashes(root: Path = PROJECT_ROOT) -> dict[str, str]:
    paths = sorted(
        {p for g in SOURCE_GLOBS for p in root.glob(g)} | {root / f for f in SOURCE_FILES}
    )
    return {str(p.relative_to(root)): _sha(p.read_bytes()) for p in paths}


JOB_SEMANTIC_GLOBS = ("src/complete_journey/abm/*.py", "src/complete_journey/prediction/*.py")
JOB_SEMANTIC_FILES = (
    "src/complete_journey/phase8_final/settings.py",
    "src/complete_journey/phase8_final/jobs.py",
    "src/complete_journey/phase8_final/store.py",
)


def job_semantics_hash(root: Path = PROJECT_ROOT) -> str:
    """Hash of every file that determines a job payload: the checkpoint config hash.
    Analysis, reporting and runner fixes therefore never invalidate completed jobs."""
    paths = sorted(
        {p for g in JOB_SEMANTIC_GLOBS for p in root.glob(g)}
        | {root / f for f in JOB_SEMANTIC_FILES}
    )
    blob = "".join(f"{p.relative_to(root)}:{_sha(p.read_bytes())}\n" for p in paths)
    return _sha(blob.encode())


def decision_hashes(root: Path = PROJECT_ROOT) -> dict[str, str]:
    text = (root / "docs/DECISIONS.md").read_text()
    out = {}
    for d in DECISIONS:
        start = text.index(f"### {d} ")
        nxt = text.find("\n### ", start + 5)
        out[d] = _sha(text[start : nxt if nxt != -1 else len(text)].encode())
    return out


def settings_dump() -> dict:
    return {
        k: (list(v) if isinstance(v, tuple) else (dict(v) if hasattr(v, "keys") else v))
        for k, v in vars(S).items()
        if k.isupper()
    }


def build_snapshot(root: Path = PROJECT_ROOT) -> dict:
    from complete_journey.abm import stage1_p8005 as s1p

    versions = {p: metadata.version(p) for p in ("numpy", "scipy", "pandas", "scikit-learn")}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {
        "schema": S.SNAPSHOT_SCHEMA,
        "run_label": S.RUN_LABEL,
        "source_commit_at_snapshot": head,
        "source_sha256": source_hashes(root),
        "job_semantics_sha256": job_semantics_hash(root),
        "decision_sha256": decision_hashes(root),
        "settings": settings_dump(),
        "imports": {
            "s1p8005_snapshot_sha256": s1p.verify_snapshot(),
            "archived_stage1_sha256": s1p.verify_archived_stage1(),
        },
        "procedure": {
            "rq2": "direct plug-in: every family evaluated at its real-CI-1 calibration "
            "(S1A semantics, stage key 11); synthetic panels never recalibrated; "
            "P8-003 rule with e = 0, g = 1",
            "refinement": "coarse 12x12 + frozen first level (exact pruning of all-infeasible "
            "cells); level 2 = real-panel secondary only",
            "representability": "separate Q_pop = 200 population block; direct images",
            "real_reveal": "after Stage 1 pass + RQ2 freeze (hash) + one-time ledger",
            "s_a": "logit-normal only (secondary); mixture S-a deferred",
            "deferred": [
                "T5 N=10,000",
                "mixture S-a",
                "full second-level refinement",
                "exhaustive per-panel recalibration RQ2",
            ],
            "emulators": "retired from confirmatory computation; P8-004/H2 historical",
        },
        "abm_frozen_constants": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in vars(abm_config).items()
            if k.isupper() and isinstance(v, (int, float, str, tuple))
        },
        "software": {"python": platform.python_version(), **versions},
        "machine": {"platform": platform.platform(), "logical_cpus": os.cpu_count()},
        "checkpoint_schema": "phase8-checkpoint-v1 (sharded ab/cd/<job_id>.json)",
        "seed_tree": "SeedSequence(20261015, (8, stage, truth_class, truth_point, q, role, "
        "family, grid_point, replicate, stream)); model stage 11; RQ2 DATA stage 3; "
        f"population block q + {S.POP_Q_OFFSET}",
    }


def write_snapshot(root: Path = PROJECT_ROOT) -> str:
    require(S.FINAL_RUN_AUTHORIZED, "the final run is not authorized")
    path = root / S.SNAPSHOT
    require(not path.exists(), "the final-run snapshot exists and is never rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(build_snapshot(root), indent=2, sort_keys=True, default=str) + "\n").encode()
    path.write_bytes(data)
    digest = _sha(data)
    (root / S.SNAPSHOT_SHA).write_text(digest + "\n")
    return digest


def amendment_paths(root: Path = PROJECT_ROOT):
    base = root / S.SNAPSHOT
    return [base.with_name(f"design_snapshot.amend-{i}.json") for i in range(1, 100)]


def verify_snapshot(root: Path = PROJECT_ROOT) -> str:
    """Verify the immutable base snapshot and its immutable amendment chain.

    An amendment (``design_snapshot.amend-N.json`` + ``.sha256``) records a mechanical,
    non-scientific source fix: its parent hash, the reason, the changed files, the full new
    source map and the unchanged job-semantics hash. Checkpoints therefore stay valid.
    Returns the hash of the newest link.
    """
    require(S.FINAL_RUN_AUTHORIZED, "the final run is not authorized")
    data = (root / S.SNAPSHOT).read_bytes()
    digest = _sha(data)
    require(digest == (root / S.SNAPSHOT_SHA).read_text().strip(), "final snapshot tampered")
    snap = json.loads(data)
    require(snap["schema"] == S.SNAPSHOT_SCHEMA, "not a final-run snapshot")
    sources, decisions = snap["source_sha256"], snap["decision_sha256"]
    for path in amendment_paths(root):
        if not path.exists():
            break
        adata = path.read_bytes()
        sha_file = path.with_suffix(".sha256")
        require(_sha(adata) == sha_file.read_text().strip(), f"{path.name} tampered")
        am = json.loads(adata)
        require(am["parent_sha256"] == digest, f"{path.name} does not chain to its parent")
        require(
            am["job_semantics_sha256"] == snap["job_semantics_sha256"],
            "an amendment may not change job semantics",
        )
        require(
            am["decision_sha256"] == snap["decision_sha256"],
            "an amendment may not change P8 decisions",
        )
        sources = am["source_sha256"]
        digest = _sha(adata)
    require(sources == source_hashes(root), "source drifted from the snapshot chain")
    require(decisions == decision_hashes(root), "P8 decisions drifted")
    require(snap["job_semantics_sha256"] == job_semantics_hash(root), "job semantics drifted")
    return digest


def write_amendment(reason: str, root: Path = PROJECT_ROOT) -> str:
    """Append one immutable amendment for a mechanical, non-scientific source fix."""
    require(S.FINAL_RUN_AUTHORIZED, "the final run is not authorized")
    base = json.loads((root / S.SNAPSHOT).read_bytes())
    parent = _sha((root / S.SNAPSHOT).read_bytes())
    prev_sources = base["source_sha256"]
    path = None
    for path in amendment_paths(root):
        if not path.exists():
            break
        am = json.loads(path.read_bytes())
        parent, prev_sources = _sha(path.read_bytes()), am["source_sha256"]
    require(
        job_semantics_hash(root) == base["job_semantics_sha256"],
        "a job-semantics change cannot be an amendment (it needs a new run)",
    )
    require(decision_hashes(root) == base["decision_sha256"], "P8 decisions drifted")
    current = source_hashes(root)
    changed = {
        f: {"old": prev_sources.get(f), "new": h}
        for f, h in current.items()
        if prev_sources.get(f) != h
    }
    changed.update(
        {f: {"old": h, "new": None} for f, h in prev_sources.items() if f not in current}
    )
    rec = {
        "schema": S.SNAPSHOT_SCHEMA + "-amendment",
        "parent_sha256": parent,
        "reason": reason,
        "changed_files": changed,
        "source_sha256": current,
        "decision_sha256": decision_hashes(root),
        "job_semantics_sha256": base["job_semantics_sha256"],
        "scientific_change": False,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    data = (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    path.with_suffix(".sha256").write_text(_sha(data) + "\n")
    return _sha(data)


def checkpoint_config_hash(root: Path = PROJECT_ROOT) -> str:
    verify_snapshot(root)
    return job_semantics_hash(root)
