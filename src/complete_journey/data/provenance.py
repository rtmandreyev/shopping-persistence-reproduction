"""Stable and volatile provenance for generated Phase 1 audit artifacts.

The tracked record identifies exact source and artifact bytes without embedding
absolute paths or volatile run details. Timestamp, platform, and timings are
written separately to an ignored run-metadata file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from complete_journey import __version__
from complete_journey.paths import PROJECT_ROOT, RAW_MANIFEST


def file_sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    """Return a streamed SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


SOURCE_SCOPE_PATHS: tuple[str, ...] = (
    "sql/audit",
    "src",
    "scripts",
    "pyproject.toml",
    "uv.lock",
)


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"source path must be inside project root: {path}") from exc


def capture_source_records(
    source_paths: list[Path],
    project_root: Path = PROJECT_ROOT,
    *,
    content_overrides: dict[Path, bytes] | None = None,
) -> list[dict[str, Any]]:
    """Capture canonical path, byte count, and SHA-256 for exact source bytes."""
    overrides = {path.resolve(): content for path, content in (content_overrides or {}).items()}
    records: list[dict[str, Any]] = []
    for path in sorted(path.resolve() for path in source_paths):
        content = overrides.get(path)
        if content is None:
            content = path.read_bytes()
        records.append(
            {
                "path": _relative_path(path, project_root),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return records


def source_state_sha256(source_records: list[dict[str, Any]]) -> str:
    """Return the canonical identity for an ordered source-record set."""
    canonical = "".join(
        f"{record['path']}\0{record['bytes']}\0{record['sha256']}\n"
        for record in sorted(source_records, key=lambda item: str(item["path"]))
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _file_record(path: Path, project_root: Path) -> dict[str, Any]:
    return capture_source_records([path], project_root)[0]


def _source_pathspecs(source_paths: list[Path], project_root: Path) -> list[str] | None:
    pathspecs = set(SOURCE_SCOPE_PATHS)
    for path in source_paths:
        try:
            relative = _relative_path(path, project_root)
        except ValueError:
            return None
        pathspecs.add(relative)
        parent = Path(relative).parent.as_posix()
        if parent != ".":
            pathspecs.add(parent)
    return sorted(pathspecs)


def source_worktree_status(
    source_paths: list[Path],
    project_root: Path = PROJECT_ROOT,
) -> str:
    """Return canonical Git porcelain status for the closed generator scope."""
    pathspecs = _source_pathspecs(source_paths, project_root)
    if pathspecs is None:
        return "!source-outside-project-root"
    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *pathspecs,
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return f"!git-error:{type(exc).__name__}:{exc}"
    if result.returncode != 0:
        return f"!git-exit:{result.returncode}:{result.stderr.strip()}"
    return result.stdout.strip()


def source_worktree_dirty(
    source_paths: list[Path],
    project_root: Path = PROJECT_ROOT,
) -> bool:
    """Return whether the closed generator scope differs from Git HEAD/index."""
    return bool(source_worktree_status(source_paths, project_root))


def relevant_source_paths(
    sql_files: list[Path],
    project_root: Path = PROJECT_ROOT,
) -> list[Path]:
    """Return the closed source scope whose bytes define an audit generation."""
    paths = set(sql_files)
    paths.update((project_root / "src").rglob("*.py"))
    paths.update((project_root / "scripts").glob("*.py"))
    paths.update(
        path
        for path in (project_root / "pyproject.toml", project_root / "uv.lock")
        if path.is_file()
    )
    return sorted(path.resolve() for path in paths if path.is_file())


def manifest_reference(
    raw_manifest: Path = RAW_MANIFEST,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Identify the committed raw-data manifest and its declared inventory."""
    if not raw_manifest.is_file():
        return {"present": False}
    with raw_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "present": True,
        "path": raw_manifest.relative_to(project_root).as_posix(),
        "sha256": file_sha256(raw_manifest),
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
    }


def build_stable_provenance(
    checks: list[dict[str, Any]],
    *,
    artifact_dir: Path,
    source_worktree_dirty: bool,
    source_paths: list[Path] | None = None,
    source_records: list[dict[str, Any]] | None = None,
    project_root: Path = PROJECT_ROOT,
    raw_manifest: Path = RAW_MANIFEST,
) -> dict[str, Any]:
    """Build deterministic provenance from captured source and artifact bytes."""
    if source_records is None:
        if source_paths is None:
            raise ValueError("source_paths or source_records is required")
        source_records = capture_source_records(source_paths, project_root)
    state_sha256 = source_state_sha256(source_records)

    artifacts: list[dict[str, Any]] = []
    for check in sorted(checks, key=lambda item: str(item["artifact"])):
        path = artifact_dir / str(check["artifact"])
        artifacts.append(
            {
                "path": f"outputs/tables/audit/{check['artifact']}",
                "generator": check["sql_file"],
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "row_count": int(check["rows"]),
                "schema": check["schema"],
            }
        )

    return {
        "format_version": 1,
        "source": {
            "state_sha256": state_sha256,
            "worktree_dirty": source_worktree_dirty,
            "files": source_records,
        },
        "raw_manifest": manifest_reference(raw_manifest, project_root),
        "artifacts": artifacts,
    }


def build_run_metadata(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build ignored, intentionally volatile execution metadata."""
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "project_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "duckdb": duckdb.__version__,
        "query_seconds": {str(check["artifact"]): check.get("seconds") for check in checks},
    }


def write_provenance(payload: dict[str, Any], path: Path) -> None:
    """Write JSON deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
