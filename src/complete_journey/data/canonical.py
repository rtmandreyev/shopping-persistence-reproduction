"""Phase 2A canonical analytical data layer builder.

Builds the durable canonical tables for Phase 2A from the verified Phase 1 raw
interface (:func:`complete_journey.data.raw_tables.register_views`), materialises
them as Parquet under ``data/processed/``, runs the QA check suite, writes the
tracked non-identifying aggregate summaries, and records contracts + provenance.

Design constraints (from the Phase 2A brief):

* Raw data is immutable. This module only reads raw views.
* Parsing is reused from Phase 1; no new raw ingestion is introduced.
* Canonical Parquet is local and Git-ignored; the customer-level diagnostic
  (``panel_eligibility``) is written under ``data/processed/private/`` and must
  never become Git-trackable.
* Only aggregate, non-identifying summaries are tracked.
* A HARD_INVARIANT failure aborts before anything is written.

This is deliberately a lightweight runner: it does not reproduce Phase 1's full
atomic multi-generation publication framework.
"""

from __future__ import annotations

import csv
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from complete_journey.data import contracts
from complete_journey.data.audit import manifest_verify
from complete_journey.data.provenance import (
    capture_source_records,
    file_sha256,
    manifest_reference,
    relevant_source_paths,
    source_state_sha256,
    source_worktree_status,
    write_provenance,
)
from complete_journey.data.raw_tables import (
    TABLE_SCHEMAS,
    RawDataError,
    register_views,
    snapshot_raw_files,
)
from complete_journey.paths import PROJECT_ROOT

PANELS_SQL_DIR = PROJECT_ROOT / "sql" / "panels"
QA_SQL_DIR = PROJECT_ROOT / "sql" / "qa"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PRIVATE_DIR = PROCESSED_DIR / "private"
PANEL_ARTIFACT_DIR = PROJECT_ROOT / "outputs" / "tables" / "panels"
SCRATCH_DIR = PROJECT_ROOT / "data" / "warehouse"
DOCS_LINEAGE = PROJECT_ROOT / "docs" / "data_lineage.csv"

#: Phase label written into the generated metadata documents (contracts.json and
#: provenance.json). One build command produces the COMPLETE Phase 2 analytical
#: data model -- the 2A canonical tables, the 2B econometric mart / campaign
#: infrastructure / choice ingredients, and the 2C predictive feature store -- so
#: the label names the phase itself rather than one of its sub-stages. It is
#: defined once so the two generated documents cannot disagree.
PHASE_METADATA_LABEL = "2"

#: Kept identical to the Phase 1 audit settings for the (dominant) raw
#: materialisation step.
DUCKDB_SETTINGS: tuple[str, ...] = (
    "SET threads = 4",
    "SET memory_limit = '8GB'",
)

#: Applied AFTER raw materialisation, before any canonical aggregation.
#: Floating-point sums are order-dependent, and DuckDB's parallel aggregation
#: order is not stable run-to-run, so SUM(DOUBLE) columns differ in their last
#: bits between otherwise identical builds. Pinning the aggregation step to a
#: single thread makes the summation order deterministic, so the canonical
#: Parquet is byte-reproducible. The expensive CSV read stays parallel above.
DETERMINISTIC_SETTINGS: tuple[str, ...] = (
    "SET threads = 1",
    "SET preserve_insertion_order = true",
)


def _materialize_raw(con: duckdb.DuckDBPyConnection) -> None:
    """Copy raw views into the scratch catalog once (raw files stay read-only)."""
    for table in TABLE_SCHEMAS:
        con.execute(f"CREATE OR REPLACE TABLE t_{table} AS SELECT * FROM {table}")
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM t_{table}")


def _atomic_write(path: Path, writer: Any) -> None:
    """Write through a sibling temp file and os.replace for per-file atomicity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    writer(tmp)
    os.replace(tmp, path)


def _write_parquet(
    con: duckdb.DuckDBPyConnection, table: str, path: Path, key: tuple[str, ...]
) -> None:
    """Write one table to Parquet, ordered by its declared key for determinism."""
    order_by = ", ".join(f'"{column}"' for column in key)

    def _writer(tmp: Path) -> None:
        con.execute(
            f'COPY (SELECT * FROM "{table}" ORDER BY {order_by}) '
            f"TO '{tmp.as_posix()}' (FORMAT PARQUET)"
        )

    _atomic_write(path, _writer)


def _write_csv(frame: Any, path: Path) -> None:
    _atomic_write(path, lambda tmp: frame.to_csv(tmp, index=False))


def _describe(con: duckdb.DuckDBPyConnection, table: str) -> list[tuple[str, str]]:
    rows = con.execute(f'DESCRIBE "{table}"').fetchall()
    return [(str(r[0]), str(r[1])) for r in rows]


def _validate_columns(
    actual: list[tuple[str, str]], expected: tuple[tuple[str, str], ...]
) -> list[str]:
    problems: list[str] = []
    actual_names = [name for name, _ in actual]
    expected_names = [name for name, _ in expected]
    if actual_names != expected_names:
        problems.append(
            f"column set/order mismatch: actual={actual_names} expected={expected_names}"
        )
        return problems
    for (name, family), (_, actual_type) in zip(expected, actual, strict=True):
        try:
            got = contracts.column_family(actual_type)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if got != family:
            problems.append(
                f"column {name}: family {got} != declared {family} (type {actual_type})"
            )
    return problems


def _key_tuple_sql(key: tuple[str, ...]) -> str:
    if len(key) == 1:
        return f'"{key[0]}"'
    return "(" + ", ".join(f'"{c}"' for c in key) + ")"


def _validate_table(
    con: duckdb.DuckDBPyConnection, name: str, spec: dict[str, object]
) -> list[str]:
    """Return a list of contract violations (empty = passes)."""
    problems: list[str] = []
    actual = _describe(con, name)
    problems += _validate_columns(actual, spec["columns"])  # type: ignore[arg-type]

    row_count = int(con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0])
    expected_rows = spec.get("expected_row_count")
    if expected_rows is not None and row_count != expected_rows:
        problems.append(f"row count {row_count} != structurally required {expected_rows}")

    key = spec["key"]  # type: ignore[assignment]
    dup = int(
        con.execute(
            f'SELECT count(*) - count(DISTINCT {_key_tuple_sql(key)}) FROM "{name}"'  # type: ignore[arg-type]
        ).fetchone()[0]
    )
    if dup != 0:
        problems.append(f"declared key not unique: {dup} duplicate key rows")

    nulls = int(
        con.execute(
            f'SELECT count(*) FROM "{name}" WHERE {_key_tuple_sql(key)} IS NULL'  # type: ignore[arg-type]
        ).fetchone()[0]
    )
    if nulls != 0:
        problems.append(f"declared key contains {nulls} null rows")
    return problems


def build(
    *,
    panels_sql_dir: Path = PANELS_SQL_DIR,
    qa_sql_dir: Path = QA_SQL_DIR,
    processed_dir: Path = PROCESSED_DIR,
    private_dir: Path | None = None,
    panel_artifact_dir: Path = PANEL_ARTIFACT_DIR,
    scratch_dir: Path = SCRATCH_DIR,
    docs_lineage: Path = DOCS_LINEAGE,
    project_root: Path = PROJECT_ROOT,
    strict: bool = True,
) -> dict[str, Any]:
    """Build, validate, and publish the Phase 2A canonical layer."""
    private_dir = private_dir if private_dir is not None else processed_dir / "private"
    # A clean rebuild may start from a deleted data/processed/, so re-create the
    # tracked placeholder that keeps the directory in Git.
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / ".gitkeep").touch()

    table_specs: dict[str, dict[str, object]] = {
        **contracts.CANONICAL_TABLES,
        **contracts.PRIVATE_TABLES,
        **contracts.PHASE2B_TABLES,
        **contracts.PHASE2C_TABLES,
    }

    # --- declared-vs-present SQL check -------------------------------------
    panel_files = {p.stem: p for p in sorted(panels_sql_dir.glob("*.sql"))}
    declared_panels = {str(spec["sql"]).removesuffix(".sql") for spec in table_specs.values()}
    qa_files = {p.stem: p for p in sorted(qa_sql_dir.glob("*.sql"))}
    declared_qa = set(contracts.CHECK_SQL) | set(contracts.SUMMARY_SQL)
    problems: list[str] = []
    if set(panel_files) != declared_panels:
        problems.append(
            f"panels SQL mismatch: missing={sorted(declared_panels - set(panel_files))} "
            f"unexpected={sorted(set(panel_files) - declared_panels)}"
        )
    if set(qa_files) != declared_qa:
        problems.append(
            f"qa SQL mismatch: missing={sorted(declared_qa - set(qa_files))} "
            f"unexpected={sorted(set(qa_files) - declared_qa)}"
        )
    if problems:
        raise SystemExit("STOP: " + "; ".join(problems))

    sql_paths = [*panel_files.values(), *qa_files.values()]
    source_paths = relevant_source_paths(sql_paths, project_root=project_root)
    source_records = capture_source_records(source_paths, project_root=project_root)
    source_state = source_state_sha256(source_records)
    source_dirty = bool(source_worktree_status(source_paths, project_root=project_root))

    print("== raw integrity gate ==")
    manifest_pre_ok, manifest_pre_detail = manifest_verify()
    print(f"  manifest pre-read: {manifest_pre_detail}")
    if not manifest_pre_ok:
        raise SystemExit(
            "STOP: raw manifest verification failed; refusing to build canonical layer."
        )
    raw_before = snapshot_raw_files()

    con = duckdb.connect()
    try:
        for setting in DUCKDB_SETTINGS:
            con.execute(setting)
        register_views(con)
        print("  materializing strict raw views (raw files are read-only)...")
        _materialize_raw(con)
        for setting in DETERMINISTIC_SETTINGS:
            con.execute(setting)

        print("\n== building canonical tables ==")
        built: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        for name, spec in table_specs.items():
            sql_text = (panels_sql_dir / str(spec["sql"])).read_text(encoding="utf-8")
            con.execute(f'CREATE OR REPLACE TABLE "{name}" AS {sql_text}')
            table_problems = _validate_table(con, name, spec)
            row_count = int(con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0])
            built[name] = {"rows": row_count, "problems": table_problems}
            status = "ok" if not table_problems else f"{len(table_problems)} problem(s)"
            print(f"  {name:<26} {row_count:>10,} rows  {status}")
            if table_problems:
                failures += [f"{name}: {p}" for p in table_problems]

        if failures:
            raise SystemExit("STOP: contract violations:\n  " + "\n  ".join(failures))

        print(f"\n== running {len(contracts.CHECK_SQL)} QA check files ==")
        qa_frames = []
        for stem in contracts.CHECK_SQL:
            frame = con.execute((qa_sql_dir / f"{stem}.sql").read_text(encoding="utf-8")).fetchdf()
            if tuple(frame.columns) != contracts.CHECK_COLUMNS:
                raise SystemExit(f"STOP: {stem} did not emit the uniform check schema")
            qa_frames.append(frame)
        qa = pd.concat(qa_frames, ignore_index=True)

        print("\n== running eligibility aggregate summaries ==")
        summaries: dict[str, Any] = {}
        for stem, filename in contracts.SUMMARY_SQL.items():
            summaries[filename] = con.execute(
                (qa_sql_dir / f"{stem}.sql").read_text(encoding="utf-8")
            ).fetchdf()
            print(f"  {filename:<56} {len(summaries[filename]):>4} rows")

        # --- Python-side reconciliation of the tracked summaries -----------
        universe = built["panel_eligibility"]["rows"]
        summary_checks = _reconcile_summaries(summaries, universe)
        qa = pd.concat([qa, pd.DataFrame(summary_checks)], ignore_index=True)

        hard_failures = qa[(qa["check_class"] == "HARD_INVARIANT") & (qa["status"] == "FAIL")]
        n_hard = int((qa["check_class"] == "HARD_INVARIANT").sum())
        n_emp = int((qa["check_class"] == "EMPIRICAL_FINDING").sum())
        passed = int(((qa["check_class"] == "HARD_INVARIANT") & (qa["status"] == "PASS")).sum())
        print(
            f"\n== QA: {passed}/{n_hard} hard invariants passed, "
            f"{n_hard - passed} failed, {n_emp} empirical findings reported =="
        )
        for _, row in hard_failures.iterrows():
            print(
                f"  FAIL {row['table_name']}.{row['check_name']}: "
                f"observed={row['observed']} expected={row['expected']}"
            )
        if not hard_failures.empty:
            raise SystemExit("STOP: HARD INVARIANT(S) FAILED - nothing written.")

        print("\n== writing artifacts ==")
        local_artifacts: list[dict[str, Any]] = []
        for name, spec in table_specs.items():
            target_dir = private_dir if spec["scope"] == "private" else processed_dir
            path = target_dir / f"{name}.parquet"
            _write_parquet(con, name, path, spec["key"])  # type: ignore[arg-type]
            size = path.stat().st_size
            digest = file_sha256(path)
            local_artifacts.append(
                {
                    "name": name,
                    "path": path.relative_to(project_root).as_posix(),
                    "scope": spec["scope"],
                    "rows": built[name]["rows"],
                    "bytes": size,
                    "sha256": digest,
                }
            )
            print(
                f"  {path.relative_to(project_root).as_posix():<52} {built[name]['rows']:>10,} rows  {size / 1024:>8.1f} KiB"
            )
        local_artifacts.sort(key=lambda a: str(a["path"]))

        tracked: list[dict[str, Any]] = []
        for filename, frame in summaries.items():
            path = panel_artifact_dir / filename
            _write_csv(frame, path)
            tracked.append(
                {
                    "path": path.relative_to(project_root).as_posix(),
                    "rows": int(len(frame)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        tracked.sort(key=lambda a: str(a["path"]))

        qa_path = panel_artifact_dir / "qa_summary.csv"
        _write_csv(qa, qa_path)
        tracked.append(
            {
                "path": qa_path.relative_to(project_root).as_posix(),
                "rows": int(len(qa)),
                "bytes": qa_path.stat().st_size,
                "sha256": file_sha256(qa_path),
            }
        )
        inventory_path = panel_artifact_dir / "predictive_feature_inventory.csv"
        _write_feature_inventory(inventory_path)
        tracked.append(
            {
                "path": inventory_path.relative_to(project_root).as_posix(),
                "rows": len(contracts.predictive_feature_inventory()),
                "bytes": inventory_path.stat().st_size,
                "sha256": file_sha256(inventory_path),
            }
        )
        (panel_artifact_dir / ".gitkeep").touch()
    finally:
        con.close()

    print("\n== contract + provenance documents ==")
    generated_contracts = _build_contracts_document(table_specs, built, local_artifacts, qa)
    contracts_path = panel_artifact_dir / "contracts.json"
    write_provenance(generated_contracts, contracts_path)
    print(f"  {contracts_path.relative_to(project_root).as_posix()}")

    lineage_path = _write_lineage(docs_lineage)
    print(f"  {lineage_path.relative_to(project_root).as_posix()}")

    manifest_post_ok, manifest_post_detail = manifest_verify()
    print(f"  manifest post-read: {manifest_post_detail}")
    raw_after = snapshot_raw_files()
    if not manifest_post_ok or raw_before != raw_after:
        raise SystemExit("STOP: raw data changed during the build.")
    print("  raw files unchanged (size + mtime identical, manifest valid)")

    provenance = {
        "format_version": 1,
        "phase": PHASE_METADATA_LABEL,
        "source": {
            "state_sha256": source_state,
            "worktree_dirty": source_dirty,
            "files": source_records,
        },
        "raw_manifest": manifest_reference(),
        "local_artifacts": local_artifacts,
        "tracked_artifacts": tracked,
        "qa": {
            "total_checks": int(len(qa)),
            "hard_invariants": n_hard,
            "hard_passed": passed,
            "empirical_findings": n_emp,
        },
    }
    provenance_path = panel_artifact_dir / "provenance.json"
    write_provenance(provenance, provenance_path)
    print(f"  {provenance_path.relative_to(project_root).as_posix()}")

    run_metadata = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "duckdb": duckdb.__version__,
        "python": sys.version.split()[0],
        "system": platform.platform(),
        "artifacts": {a["name"]: a["rows"] for a in local_artifacts},
    }
    write_provenance(run_metadata, panel_artifact_dir / "run_metadata.json")

    return {
        "tables": {name: built[name]["rows"] for name in built},
        "local_artifacts": local_artifacts,
        "tracked_artifacts": tracked,
        "qa": {"total": int(len(qa)), "hard": n_hard, "passed": passed, "empirical": n_emp},
        "source_state": source_state,
    }


def _reconcile_summaries(summaries: dict[str, Any], universe: int) -> list[dict[str, str]]:
    """HARD checks that tracked aggregate summaries reconcile to the local table."""
    checks: list[dict[str, str]] = []

    def add(name: str, ok: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "table_name": "panel_eligibility_summaries",
                "check_name": name,
                "check_class": "HARD_INVARIANT",
                "status": "PASS" if ok else "FAIL",
                "observed": str(observed),
                "expected": str(expected),
            }
        )

    dist = summaries["panel_eligibility_active_weeks_distribution.csv"]
    dist_households = int(dist.loc[dist["statistic"] == "households", "value"].iloc[0])
    add(
        "summary_distribution_households_reconcile",
        dist_households == universe,
        dist_households,
        universe,
    )

    thr = summaries["panel_eligibility_active_weeks_thresholds.csv"]
    thr_at_one = int(thr.loc[thr["active_weeks_threshold"] == 1, "households_at_or_above"].iloc[0])
    add("summary_threshold_counts_reconcile", thr_at_one == universe, thr_at_one, universe)
    shares_ok = bool(((thr["share_at_or_above"] >= 0) & (thr["share_at_or_above"] <= 1)).all())
    add(
        "summary_threshold_shares_in_range",
        shares_ok,
        "in [0,1]" if shares_ok else "OUT OF RANGE",
        "[0,1]",
    )

    bucket = summaries["panel_eligibility_support_bucket_by_early_late.csv"]
    add(
        "summary_bucket_counts_reconcile",
        int(bucket["households"].sum()) == universe,
        int(bucket["households"].sum()),
        universe,
    )

    halves = summaries["panel_eligibility_support_halves_crosstab.csv"]
    add(
        "summary_halves_counts_reconcile",
        int(halves["households"].sum()) == universe,
        int(halves["households"].sum()),
        universe,
    )

    # --- longitudinal-support diagnostics (Phase 2A extension) -------------
    required_measures = {
        "observed_span_weeks",
        "active_weeks",
        "leading_no_purchase_weeks",
        "trailing_no_purchase_weeks",
        "longest_internal_no_purchase_gap",
        "active_weeks_first_half",
        "active_weeks_second_half",
        "active_weeks_early_third",
        "active_weeks_middle_third",
        "active_weeks_late_third",
        "active_weeks_block_1",
        "active_weeks_block_2",
        "active_weeks_block_3",
        "active_weeks_block_4",
    }
    dists = summaries["panel_eligibility_support_distributions.csv"]
    present = set(dists["measure"])
    add(
        "summary_support_distributions_complete",
        required_measures <= present,
        f"{len(required_measures & present)}/{len(required_measures)} measures",
        sorted(required_measures),
    )
    ordered = True
    for _, row in dists.iterrows():
        seq = [
            row["min"],
            row["p05"],
            row["p10"],
            row["p25"],
            row["median"],
            row["p75"],
            row["p90"],
            row["p95"],
            row["max"],
        ]
        if any(a > b for a, b in zip(seq, seq[1:], strict=False)):
            ordered = False
    add(
        "summary_support_distributions_ordered",
        ordered,
        "monotone" if ordered else "OUT OF ORDER",
        "min<=p05<=...<=max",
    )

    thresholds_lookup = dict(
        zip(thr["active_weeks_threshold"], thr["households_at_or_above"], strict=True)
    )
    for filename, column in (
        ("panel_eligibility_halves_by_active_weeks_thresholds.csv", "active_in_both_halves"),
        ("panel_eligibility_thirds_by_active_weeks_thresholds.csv", "active_in_all_three_thirds"),
    ):
        frame = summaries[filename]
        grouped = frame.groupby("active_weeks_threshold")["households"].sum()
        # The two flag rows partition the households with active_weeks >= threshold,
        # so their sum must equal that threshold's count in the thresholds table.
        observed = [int(grouped[t]) for t in grouped.index]
        expected = [int(thresholds_lookup[t]) for t in grouped.index]
        add(
            f"summary_{column}_partitions_reconcile",
            len(grouped) == 6 and observed == expected,
            observed,
            expected,
        )
        add(
            f"summary_{column}_shares_in_range",
            bool(((frame["share_of_households"] >= 0) & (frame["share_of_households"] <= 1)).all()),
            "in [0,1]",
            "[0,1]",
        )

    halves_thr = summaries["panel_eligibility_halves_by_active_weeks_thresholds.csv"]
    both_at_one = int(
        halves_thr.loc[
            (halves_thr["active_weeks_threshold"] == 1)
            & (halves_thr["active_in_both_halves"] == 1),
            "households",
        ].iloc[0]
    )
    crosstab_both = int(
        halves.loc[
            (halves["first_half_active"] == 1) & (halves["second_half_active"] == 1), "households"
        ].iloc[0]
    )
    add(
        "summary_halves_both_active_consistent",
        both_at_one == crosstab_both,
        both_at_one,
        crosstab_both,
    )

    span = summaries["panel_eligibility_observed_span_by_active_weeks.csv"]
    span_ok = (
        len(span) == 25
        and bool((span["households"] <= universe).all())
        and bool(((span["share_of_households"] >= 0) & (span["share_of_households"] <= 1)).all())
    )
    for _, group in span.groupby("observed_span_weeks_threshold"):
        if not group["households"].is_monotonic_decreasing:
            span_ok = False
    for _, group in span.groupby("active_weeks_threshold"):
        if not group["households"].is_monotonic_decreasing:
            span_ok = False
    add(
        "summary_span_by_activity_consistent",
        span_ok,
        "25 rows, bounded, monotone" if span_ok else "VIOLATION",
        "bounded+monotone",
    )

    gap = summaries["panel_eligibility_gap_thresholds.csv"]
    gap_ok = (
        len(gap) == 12
        and bool((gap["households"] <= universe).all())
        and bool(((gap["share_of_households"] >= 0) & (gap["share_of_households"] <= 1)).all())
    )
    for _, group in gap.groupby("active_weeks_threshold"):
        if not group["households"].is_monotonic_increasing:
            gap_ok = False
    # gaps<=20 & active>=20 cannot exceed the active>=20 count alone
    if 20 in set(gap["active_weeks_threshold"]):
        cap = int(thr.loc[thr["active_weeks_threshold"] == 20, "households_at_or_above"].iloc[0])
        combined = gap.loc[
            (gap["gap_threshold_weeks"] == 20) & (gap["active_weeks_threshold"] == 20),
            "households",
        ]
        if combined.empty or int(combined.iloc[0]) > cap:
            gap_ok = False
    add(
        "summary_gap_thresholds_consistent",
        gap_ok,
        "12 rows, bounded, monotone" if gap_ok else "VIOLATION",
        "bounded+monotone",
    )

    # --- Phase 2B: sample flag counts (S0/S1/S2) ---------------------------
    samples = summaries["sample_flag_counts.csv"]
    by_sample = dict(zip(samples["sample"], samples["households"], strict=True))
    expected_samples = {
        "sample_full": 2500,
        "sample_sustained_support": 2115,
        "sample_higher_support": 1807,
    }
    add(
        "summary_sample_counts_reconcile",
        all(int(by_sample.get(k, -1)) == v for k, v in expected_samples.items()),
        {k: int(v) for k, v in by_sample.items()},
        expected_samples,
    )
    weeks = dict(zip(samples["sample"], samples["household_weeks"], strict=True))
    add(
        "summary_sample_household_weeks_consistent",
        int(weeks["sample_full"]) == universe * 102
        and int(weeks["sample_sustained_support"])
        == int(by_sample["sample_sustained_support"]) * 102
        and int(weeks["sample_higher_support"]) == int(by_sample["sample_higher_support"]) * 102,
        {k: int(v) for k, v in weeks.items()},
        "households * 102",
    )

    return checks


def _build_contracts_document(
    table_specs: dict[str, dict[str, object]],
    built: dict[str, dict[str, Any]],
    local_artifacts: list[dict[str, Any]],
    qa: Any,
) -> dict[str, Any]:
    """Deterministic machine-readable contract document for the canonical layer."""
    artifact_by_name = {str(a["name"]): a for a in local_artifacts}
    tables: list[dict[str, Any]] = []
    for name in sorted(table_specs):
        spec = table_specs[name]
        artifact = artifact_by_name[name]
        tables.append(
            {
                "table": name,
                "title": spec["title"],
                "grain": spec["grain"],
                "key": list(spec["key"]),  # type: ignore[arg-type]
                "required_columns": [list(c) for c in spec["columns"]],  # type: ignore[arg-type]
                "population": spec["population"],
                "time_support": spec["time_support"],
                "zero_semantics": spec["zero_semantics"],
                "missing_semantics": spec["missing_semantics"],
                "expected_row_count": spec["expected_row_count"],
                "actual_row_count": built[name]["rows"],
                "scope": spec["scope"],
                "git_tracked": False,
                "parquet_path": artifact["path"],
                "parquet_sha256": artifact["sha256"],
                "generating_sql": f"sql/panels/{spec['sql']}",
                "source_tables": list(spec["sources"]),  # type: ignore[arg-type]
                "limitations": spec["limitations"],
            }
        )
    return {
        "format_version": 1,
        "phase": PHASE_METADATA_LABEL,
        "note": (
            "Canonical Parquet artifacts are local and Git-ignored. This document "
            "and the aggregate summaries under outputs/tables/panels/ contain no "
            "household identifiers."
        ),
        "tables": tables,
        "qa": {
            "hard_invariants": int((qa["check_class"] == "HARD_INVARIANT").sum()),
            "hard_failed": int(
                ((qa["check_class"] == "HARD_INVARIANT") & (qa["status"] == "FAIL")).sum()
            ),
            "empirical_findings": int((qa["check_class"] == "EMPIRICAL_FINDING").sum()),
        },
    }


def _write_lineage(path: Path) -> Path:
    """Write the deterministic machine-readable lineage CSV from the contracts."""

    def _writer(tmp: Path) -> None:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["output_table", "source_table", "relationship", "transformation_stage"]
            )
            for output_table, source_table, relationship, stage in contracts.LINEAGE:
                writer.writerow([output_table, source_table, relationship, stage])

    _atomic_write(path, _writer)
    return path


def _write_feature_inventory(path: Path) -> Path:
    """Write the deterministic, non-identifying Phase 2C feature inventory."""

    def _writer(tmp: Path) -> None:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(contracts.PREDICTIVE_INVENTORY_COLUMNS)
            for row in contracts.predictive_feature_inventory():
                writer.writerow(row)

    _atomic_write(path, _writer)
    return path


def main() -> int:
    try:
        summary = build()
    except RawDataError as exc:
        print(f"RAW DATA ERROR: {exc}")
        return 2
    except SystemExit as exc:
        print(f"\n{exc}")
        return 1
    print("\ncanonical layer complete:")
    for name, rows in summary["tables"].items():
        print(f"  {name:<26} {rows:>10,} rows")
    qa = summary["qa"]
    print(
        f"  QA: {qa['passed']}/{qa['hard']} hard invariants, {qa['empirical']} empirical findings"
    )
    return 0
