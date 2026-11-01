"""Phase 2 current-source validation; the historical generation provenance stays immutable.

``outputs/tables/panels/provenance.json`` records the source state that generated the
accepted Phase 2 tables (refreshed once under P3-004). Later phases legitimately changed
some recorded files. That record is historical evidence and is never rewritten.
This module answers a different question: is the *current* source still compatible
with the accepted generation?

* The historical record itself is pinned byte-for-byte.
* Every recorded tracked artifact, and every local Parquet artifact that is present,
  must still match its recorded hash.
* Each recorded source file is either unchanged, outside the Phase 2 build import
  scope, or an exact pinned post-generation revision with a stated reason.
* Dependency files may change only if the Phase 2 runtime packages keep their
  generation-time locked versions.

Any other drift fails loudly. Passing validation never claims that current source
generated the accepted tables.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import MappingProxyType

from complete_journey.paths import PROJECT_ROOT

HISTORICAL_PROVENANCE = "outputs/tables/panels/provenance.json"
HISTORICAL_PROVENANCE_SHA256 = "fa486c16f81ebe6d284ccdfe60c09c5ecd54f2f7b978ee8fcda39af6b1ddb1fa"

#: Files the Phase 2 build actually executes: the import closure of
#: ``complete_journey.data.canonical``, its entry point, and its SQL.
BUILD_SCOPE_FILES = frozenset(
    {
        "src/complete_journey/__init__.py",
        "src/complete_journey/paths.py",
        "src/complete_journey/data/__init__.py",
        "src/complete_journey/data/audit.py",
        "src/complete_journey/data/canonical.py",
        "src/complete_journey/data/contracts.py",
        "src/complete_journey/data/invariants.py",
        "src/complete_journey/data/provenance.py",
        "src/complete_journey/data/raw_tables.py",
        "scripts/build_canonical_layer.py",
    }
)
BUILD_SCOPE_PREFIXES = ("sql/panels/", "sql/qa/")
DEPENDENCY_FILES = frozenset({"pyproject.toml", "uv.lock"})
#: Generation-time locked versions (uv.lock at 7a647c5) of the Phase 2 runtime packages.
PHASE2_RUNTIME_PINS = MappingProxyType(
    {"duckdb": "1.5.6", "numpy": "2.5.3", "pandas": "3.0.6", "pyarrow": "25.0.1"}
)

#: Exact post-generation revisions of build-scope files, each with a stated reason.
POST_GENERATION_REVISIONS = MappingProxyType(
    {
        "src/complete_journey/data/contracts.py": MappingProxyType(
            {
                "historical_sha256": "8ad8c17629c820db629b46b21561f041937b80c0b66e59c45c7a16473b56f8c4",
                "current_sha256": "00d195fcf13ffc4c81e0cc32d9c26e976640315dd994df682f14905a1602a3ff",
                "reason": (
                    "P4-001 contract limitation/missing-semantics prose only; re-emitted to "
                    "contracts.json without a data build; no SQL, schema or row logic changed"
                ),
            }
        ),
    }
)


class SourceValidationError(ValueError):
    """Current source is not demonstrably compatible with the accepted generation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceValidationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def in_build_scope(relative: str) -> bool:
    return (
        relative in BUILD_SCOPE_FILES
        or relative in DEPENDENCY_FILES
        or relative.startswith(BUILD_SCOPE_PREFIXES)
    )


def locked_versions(lock_text: str, packages) -> dict[str, str]:
    """Return ``{package: version}`` from uv.lock ``[[package]]`` blocks."""
    found = {}
    for block in lock_text.split("[[package]]"):
        name = re.search(r'^name = "([^"]+)"', block, re.MULTILINE)
        version = re.search(r'^version = "([^"]+)"', block, re.MULTILINE)
        if name and version and name.group(1) in packages:
            found[name.group(1)] = version.group(1)
    return found


def validate_current_source(root: Path = PROJECT_ROOT) -> dict:
    """Return a classification report or raise ``SourceValidationError``."""
    record_path = root / HISTORICAL_PROVENANCE
    _require(record_path.is_file(), "historical Phase 2 provenance missing")
    _require(
        sha256(record_path) == HISTORICAL_PROVENANCE_SHA256,
        "historical Phase 2 provenance changed; it must never be rewritten",
    )
    record = json.loads(record_path.read_bytes())

    for artifact in record["tracked_artifacts"]:
        path = root / artifact["path"]
        _require(
            path.is_file() and sha256(path) == artifact["sha256"],
            f"accepted Phase 2 tracked artifact drift: {artifact['path']}",
        )
    local_checked = 0
    for artifact in record["local_artifacts"]:
        path = root / artifact["path"]
        if path.is_file():
            _require(
                sha256(path) == artifact["sha256"],
                f"accepted Phase 2 local artifact drift: {artifact['path']}",
            )
            local_checked += 1

    classes: dict[str, list[str]] = {
        "unchanged": [],
        "outside_build_scope": [],
        "authorized_revision": [],
        "dependency_runtime_pinned": [],
    }
    for entry in record["source"]["files"]:
        relative, recorded = entry["path"], entry["sha256"]
        path = root / relative
        _require(path.is_file(), f"recorded Phase 2 source file missing: {relative}")
        current = sha256(path)
        if current == recorded:
            classes["unchanged"].append(relative)
        elif not in_build_scope(relative):
            classes["outside_build_scope"].append(relative)
        elif relative in DEPENDENCY_FILES:
            classes["dependency_runtime_pinned"].append(relative)
        else:
            revision = POST_GENERATION_REVISIONS.get(relative)
            _require(
                revision is not None
                and revision["historical_sha256"] == recorded
                and revision["current_sha256"] == current,
                f"unauthorized drift in Phase 2 build-scope source: {relative}",
            )
            classes["authorized_revision"].append(relative)
    if classes["dependency_runtime_pinned"]:
        versions = locked_versions((root / "uv.lock").read_text(), PHASE2_RUNTIME_PINS)
        _require(
            versions == dict(PHASE2_RUNTIME_PINS),
            f"Phase 2 runtime package versions changed: {versions}",
        )
    return {
        "historical_provenance_sha256": HISTORICAL_PROVENANCE_SHA256,
        "historical_source_state_sha256": record["source"]["state_sha256"],
        "tracked_artifacts_verified": len(record["tracked_artifacts"]),
        "local_artifacts_verified": local_checked,
        "local_artifacts_recorded": len(record["local_artifacts"]),
        "source_classes": {k: sorted(v) for k, v in classes.items()},
        "claim": "current source compatible with the accepted generation; "
        "historical provenance remains the generating-state record",
    }
