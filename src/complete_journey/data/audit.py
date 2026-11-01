"""Phase 1 data audit runner.

The runner verifies raw inputs, reads them through a fresh per-run DuckDB scratch
catalog, validates every generated artifact, and only then promotes a complete
staged generation into the authoritative public directory.
"""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Any

import duckdb
import pandas as pd

from complete_journey.data import invariants
from complete_journey.data.provenance import (
    build_run_metadata,
    build_stable_provenance,
    capture_source_records,
    relevant_source_paths,
    source_state_sha256,
    source_worktree_status,
    write_provenance,
)
from complete_journey.data.raw_tables import (
    TABLE_SCHEMAS,
    RawDataError,
    discover_raw_files,
    register_views,
    snapshot_raw_files,
)
from complete_journey.paths import PROJECT_ROOT, RAW_MANIFEST

SQL_DIR = PROJECT_ROOT / "sql" / "audit"
ARTIFACT_DIR = PROJECT_ROOT / "outputs" / "tables" / "audit"
SCRATCH_DIR = PROJECT_ROOT / "data" / "warehouse"
PRIVATE_ARTIFACT_DIR = SCRATCH_DIR / "audit_private"

DUCKDB_SETTINGS: tuple[str, ...] = (
    "SET threads = 4",
    "SET memory_limit = '6GB'",
    "SET preserve_insertion_order = false",
)


@contextmanager
def fresh_scratch_connection(base_dir: Path = SCRATCH_DIR):
    """Yield a fresh per-run DuckDB connection and remove it afterward."""
    base_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="audit-", dir=base_dir) as temporary:
        database = Path(temporary) / "audit.duckdb"
        con = duckdb.connect(str(database))
        try:
            yield con
        finally:
            con.close()


@contextmanager
def staging_directory(
    *,
    prefix: str,
    directory: Path,
    cleanup: Callable[[Path], None] = shutil.rmtree,
) -> Iterator[Path]:
    """Yield a staging directory whose cleanup is always best-effort."""
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(mkdtemp(prefix=prefix, dir=directory))
    try:
        yield path
    finally:
        if path.exists():
            try:
                cleanup(path)
            except Exception as cleanup_exc:  # noqa: BLE001 - cleanup is post-transactional
                print(
                    "WARNING: audit staging cleanup failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )


def atomic_exchange(source: Path, target: Path) -> None:
    """Atomically exchange two existing filesystem entries on macOS or Linux."""
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    libc = ctypes.CDLL(None, use_errno=True)

    if sys.platform == "darwin":
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, target_bytes, 0x00000002)  # RENAME_SWAP
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, source_bytes, -100, target_bytes, 0x00000002)  # RENAME_EXCHANGE
    else:
        raise OSError(errno.ENOTSUP, "atomic directory exchange is unsupported on this platform")

    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), f"{source} <-> {target}")


def publish_generations(
    generations: Sequence[tuple[Path, Path]],
    *,
    exchange: Callable[[Path, Path], None] = atomic_exchange,
    replace: Callable[[Path, Path], None] = os.replace,
    cleanup: Callable[[Path], None] = shutil.rmtree,
) -> None:
    """Publish several generations as one rollback-safe batch."""
    installed: list[tuple[Path, Path, bool]] = []
    try:
        for staging, target in generations:
            if not staging.is_dir():
                raise FileNotFoundError(f"staged generation missing: {staging}")
            target.parent.mkdir(parents=True, exist_ok=True)
            had_target = target.exists()
            if had_target:
                exchange(staging, target)
            else:
                replace(staging, target)
            installed.append((staging, target, had_target))
    except Exception as exc:
        rollback_errors: list[str] = []
        for staging, target, had_target in reversed(installed):
            try:
                if had_target:
                    exchange(staging, target)
                elif target.exists():
                    replace(target, staging)
            except Exception as rollback_exc:  # noqa: BLE001 - preserve every rollback failure
                rollback_errors.append(f"{target}: {type(rollback_exc).__name__}: {rollback_exc}")
        if rollback_errors:
            exc.add_note("publication rollback failure(s): " + "; ".join(rollback_errors))
        raise

    for staging, _, had_target in installed:
        if had_target and staging.exists():
            try:
                cleanup(staging)
            except Exception as cleanup_exc:  # noqa: BLE001 - publication already committed
                print(
                    "WARNING: obsolete audit generation cleanup failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )


def publish_generation(
    staging: Path,
    target: Path,
    *,
    exchange: Callable[[Path, Path], None] = atomic_exchange,
    replace: Callable[[Path, Path], None] = os.replace,
    cleanup: Callable[[Path], None] = shutil.rmtree,
) -> None:
    """Publish one directory through the rollback-safe batch primitive."""
    publish_generations(
        [(staging, target)],
        exchange=exchange,
        replace=replace,
        cleanup=cleanup,
    )


def _materialize(con: duckdb.DuckDBPyConnection) -> None:
    """Copy raw views into the fresh scratch catalog exactly once per run."""
    for table in TABLE_SCHEMAS:
        con.execute(f"CREATE OR REPLACE TABLE t_{table} AS SELECT * FROM {table}")
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM t_{table}")


def manifest_verify() -> tuple[bool, str]:
    """Verify raw hashes and exact inventory against the committed manifest."""
    sys_path = PROJECT_ROOT / "scripts" / "build_raw_manifest.py"
    if not sys_path.is_file():
        return False, "scripts/build_raw_manifest.py not found"
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_raw_manifest", sys_path)
    if spec is None or spec.loader is None:
        return False, "could not load scripts/build_raw_manifest.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return_code = module.verify()
    except Exception as exc:  # noqa: BLE001 - any failure means raw data is suspect
        return False, f"{type(exc).__name__}: {exc}"
    if return_code != 0:
        return False, f"raw manifest verifier returned {return_code}"
    return True, "all raw files and inventory match manifest"


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def _run_audit(
    sql_dir: Path = SQL_DIR,
    artifact_dir: Path = ARTIFACT_DIR,
    strict: bool = True,
    *,
    scratch_dir: Path = SCRATCH_DIR,
    private_artifact_dir: Path | None = None,
    public_artifacts: set[str] | None = None,
    private_artifacts: set[str] | None = None,
    project_root: Path = PROJECT_ROOT,
    raw_manifest: Path = RAW_MANIFEST,
) -> dict[str, Any]:
    """Run, validate, and atomically publish the full Phase 1 audit."""
    public = set(invariants.PUBLIC_ARTIFACTS if public_artifacts is None else public_artifacts)
    private = set(invariants.PRIVATE_ARTIFACTS if private_artifacts is None else private_artifacts)
    canonical_public = set(invariants.PUBLIC_ARTIFACTS)
    canonical_private = set(invariants.PRIVATE_ARTIFACTS)
    if artifact_dir.resolve() == ARTIFACT_DIR.resolve() and (
        public != canonical_public or private != canonical_private
    ):
        raise SystemExit(
            "production artifact directory requires canonical public/private classification"
        )
    if public & private:
        raise SystemExit(f"artifact declared public and private: {sorted(public & private)}")

    sql_files = sorted(sql_dir.glob("*.sql"))
    discovered_stems = {path.stem for path in sql_files}
    expected_stems = public | private
    missing_sql = expected_stems - discovered_stems
    unexpected_sql = discovered_stems - expected_stems
    if missing_sql or unexpected_sql:
        raise SystemExit(
            "audit SQL declaration mismatch: "
            f"missing={sorted(missing_sql)} unexpected={sorted(unexpected_sql)}"
        )

    try:
        sql_dir.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise SystemExit(f"sql_dir must be inside project_root: {sql_dir}") from exc

    sql_payloads = {path.resolve(): path.read_bytes() for path in sql_files}
    source_paths = relevant_source_paths(sql_files, project_root=project_root)
    source_records_before = capture_source_records(
        source_paths,
        project_root=project_root,
        content_overrides=sql_payloads,
    )
    source_state_before = source_state_sha256(source_records_before)
    source_status_before = source_worktree_status(source_paths, project_root=project_root)
    source_dirty_before = bool(source_status_before)

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    private_target = private_artifact_dir or (scratch_dir / "audit_private")

    print("== raw integrity gate ==")
    manifest_pre_ok, manifest_pre_detail = manifest_verify()
    print(f"  manifest pre-read: {manifest_pre_detail}")
    if not manifest_pre_ok:
        raise SystemExit("STOP: raw manifest verification failed; refusing to audit modified data.")

    raw_before = snapshot_raw_files()
    discovered = sorted(discover_raw_files())
    print(f"  discovered {len(discovered)} raw tables")

    with staging_directory(prefix=".audit-public-", directory=artifact_dir.parent) as public_stage:
        with staging_directory(prefix=".audit-private-", directory=scratch_dir) as private_stage:
            artifacts: dict[str, pd.DataFrame] = {}
            checks: list[dict[str, Any]] = []
            public_checks: list[dict[str, Any]] = []
            sql_failures: list[str] = []

            with fresh_scratch_connection(scratch_dir) as con:
                for setting in DUCKDB_SETTINGS:
                    con.execute(setting)
                register_views(con)
                print("  materializing fresh scratch tables (raw files are read-only)...")
                materialize_start = time.time()
                _materialize(con)
                print(f"  scratch tables ready in {time.time() - materialize_start:.1f}s")

                print(f"\n== executing {len(sql_files)} declared audit queries ==")
                for sql_file in sql_files:
                    name = sql_file.stem
                    start = time.time()
                    try:
                        result = con.execute(sql_payloads[sql_file.resolve()].decode("utf-8"))
                        description = result.description
                        columns = [str(item[0]) for item in description]
                        schema = [
                            {"name": str(item[0]), "type": str(item[1])} for item in description
                        ]
                        rows = result.fetchall()
                    except Exception as exc:  # noqa: BLE001
                        detail = f"SQL FAILED {name}: {type(exc).__name__}: {exc}"
                        sql_failures.append(detail)
                        if strict:
                            raise SystemExit(detail) from exc
                        print(f"  {detail}")
                        continue

                    elapsed = time.time() - start
                    frame = pd.DataFrame(rows, columns=columns)
                    destination = public_stage if name in public else private_stage
                    frame.to_csv(destination / f"{name}.csv", index=False)
                    artifacts[name] = frame
                    check = {
                        "artifact": f"{name}.csv",
                        "sql_file": sql_file.resolve()
                        .relative_to(project_root.resolve())
                        .as_posix(),
                        "rows": len(frame),
                        "schema": schema,
                        "seconds": round(elapsed, 3),
                    }
                    checks.append(check)
                    if name in public:
                        public_checks.append(check)
                    print(f"  {name:<34} {len(frame):>8,} rows  {elapsed:>6.1f}s")

            if sql_failures:
                raise SystemExit(
                    f"{len(sql_failures)} SQL query failure(s); staged artifacts not published"
                )

            manifest_post_ok, manifest_post_detail = manifest_verify()
            print(f"  manifest post-read: {manifest_post_detail}")
            if not manifest_post_ok:
                raise SystemExit(
                    "STOP: raw manifest verification failed after data read; "
                    "staged artifacts not published."
                )

            print("\n== hard invariants ==")
            manifest_result = (
                manifest_pre_ok and manifest_post_ok,
                f"pre-read: {manifest_pre_detail}; post-read: {manifest_post_detail}",
            )
            results: list[invariants.InvariantResult] = [
                invariants.check_manifest_valid(manifest_result),
                invariants.check_expected_tables_discovered(discovered),
            ]
            results += invariants.check_artifact_schemas(artifacts)
            results += invariants.check_row_counts_consistent(artifacts)
            results += invariants.check_no_null_join_keys(artifacts)
            results += invariants.check_semantic_consistency(artifacts)
            results += invariants.check_raw_unchanged(raw_before, snapshot_raw_files())

            failures = [result for result in results if not result.passed]
            for result in failures:
                print(f"  {result}")
            print(f"  {len(results) - len(failures)}/{len(results)} invariants passed")
            if failures:
                raise SystemExit(
                    f"{len(failures)} HARD INVARIANT(S) FAILED - staged artifacts not published."
                )

            source_paths_after = relevant_source_paths(sql_files, project_root=project_root)
            source_records_after = capture_source_records(
                source_paths_after,
                project_root=project_root,
            )
            source_state_after = source_state_sha256(source_records_after)
            source_status_after = source_worktree_status(
                source_paths_after,
                project_root=project_root,
            )
            if (
                source_state_after != source_state_before
                or source_status_after != source_status_before
            ):
                raise SystemExit(
                    "STOP: generating source changed during audit; staged artifacts not published."
                )

            stable = build_stable_provenance(
                public_checks,
                artifact_dir=public_stage,
                source_records=source_records_before,
                source_worktree_dirty=source_dirty_before,
                project_root=project_root,
                raw_manifest=raw_manifest,
            )
            write_provenance(stable, public_stage / "provenance.json")
            write_provenance(build_run_metadata(checks), public_stage / "run_metadata.json")
            (public_stage / ".gitkeep").touch()

            expected_public_files = {
                *(f"{name}.csv" for name in public),
                "provenance.json",
                "run_metadata.json",
                ".gitkeep",
            }
            actual_public_files = {path.name for path in public_stage.iterdir()}
            if actual_public_files != expected_public_files:
                raise SystemExit(
                    "public staging inventory mismatch: "
                    f"expected={sorted(expected_public_files)} actual={sorted(actual_public_files)}"
                )

            generations: list[tuple[Path, Path]] = []
            if private:
                generations.append((private_stage, private_target))
            generations.append((public_stage, artifact_dir))
            publish_generations(generations)

    print(f"\nartifacts + stable provenance written to {_display_path(artifact_dir)}")
    return {
        "artifacts": {name: len(frame) for name, frame in artifacts.items()},
        "checks": checks,
        "invariants_passed": len(results),
        "raw_files": len(raw_before),
    }


def run_audit(
    sql_dir: Path = SQL_DIR,
    artifact_dir: Path = ARTIFACT_DIR,
    strict: bool = True,
    *,
    scratch_dir: Path = SCRATCH_DIR,
    private_artifact_dir: Path | None = None,
    project_root: Path = PROJECT_ROOT,
    raw_manifest: Path = RAW_MANIFEST,
) -> dict[str, Any]:
    """Run the production audit with its canonical public/private classification."""
    return _run_audit(
        sql_dir=sql_dir,
        artifact_dir=artifact_dir,
        strict=strict,
        scratch_dir=scratch_dir,
        private_artifact_dir=private_artifact_dir,
        public_artifacts=set(invariants.PUBLIC_ARTIFACTS),
        private_artifacts=set(invariants.PRIVATE_ARTIFACTS),
        project_root=project_root,
        raw_manifest=raw_manifest,
    )


def main() -> int:
    try:
        summary = run_audit()
    except RawDataError as exc:
        print(f"RAW DATA ERROR: {exc}")
        return 2
    except SystemExit as exc:
        print(f"\n{exc}")
        return 1
    print(f"\naudit complete: {len(summary['artifacts'])} artifacts")
    return 0
