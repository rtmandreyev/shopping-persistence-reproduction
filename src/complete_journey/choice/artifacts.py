"""Closed preparation artifact policy and immutable upstream verification."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from complete_journey.choice.contracts import (
    ACCEPTED_RUN_C_SHA256,
    ACCEPTED_RUN_D_SHA256,
    CORE_OUTPUTS,
    GROUP,
    IGNORE_APPEND,
    LOCAL_PATHS,
    OUTPUTS,
    PREPARATION_EVIDENCE_SHA256,
    PREPARATION_VALIDATION_PATH,
    PROTECTED_INPUTS,
    REGISTER_FIELDS,
    REGISTER_IDS,
    REGISTER_PATH,
    REGISTER_STATUSES,
    RUN_B_PROVENANCE_SHA256,
    RUN_D_OUTPUTS,
    SCOPE,
    SUMMARY_FIELDS,
    UPSTREAM_IGNORE_SHA256,
    ChoiceContractError,
    require,
    source_paths,
)
from complete_journey.choice.final_contracts import (
    APPROVED_STATEMENTS,
    D2_PATH,
    FINAL_IGNORE_APPEND,
    FINAL_OUTPUTS,
    PROMOTED_IDS,
    PUBLICATION_PROVENANCE,
    phase5_ledger_state,
)
from complete_journey.paths import PROJECT_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def source_identity(root: Path = PROJECT_ROOT) -> dict:
    files = {str(p.relative_to(root)): sha256(p) for p in source_paths(root)}
    return {"files": files, "state_sha256": hashlib.sha256(canonical_json(files)).hexdigest()}


def verify_inputs(root: Path = PROJECT_ROOT) -> dict:
    """Return historical input hashes, including the accepted ledger prefix.

    The ledger may extend that prefix only by the exact authorized Phase 5 history
    (P5-002, then the exact P5-003 closure, then later phases from the Phase 6
    heading; P7-002). The returned hash does not identify the current ledger or
    claim it generated old evidence; publication provenance records it separately.
    """
    for relative, digest in PROTECTED_INPUTS.items():
        path = root / relative
        matches = path.is_file() and sha256(path) == digest
        if not matches and relative == "docs/DECISIONS.md" and path.is_file():
            try:
                matches = phase5_ledger_state(path.read_bytes(), digest) != "preparation"
            except ChoiceContractError:
                matches = False
        require(matches, f"accepted input/evidence drift: {relative}")
    return dict(PROTECTED_INPUTS)


def ledger_state(root: Path = PROJECT_ROOT) -> str:
    """Return preparation, run_e or closed for the current ledger, or refuse it."""
    ledger = root / "docs/DECISIONS.md"
    require(ledger.is_file(), "exact P5-002 ledger entry required")
    return phase5_ledger_state(ledger.read_bytes(), PROTECTED_INPUTS["docs/DECISIONS.md"])


def ignored(root: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", relative], cwd=root, capture_output=True
    )
    require(result.returncode in (0, 1), "ignore-policy probe failed")
    return result.returncode == 0


def publication_stage(root: Path = PROJECT_ROOT) -> bool:
    """Validate exact policy bytes and identify the narrowly authorized final stage."""
    policy = (root / ".gitignore").read_bytes()
    final_suffix = FINAL_IGNORE_APPEND.encode()
    final = policy.endswith(final_suffix)
    if final:
        policy = policy[: -len(final_suffix)]
    suffix = IGNORE_APPEND.encode()
    require(
        policy.endswith(suffix)
        and hashlib.sha256(policy[: -len(suffix)]).hexdigest() == UPSTREAM_IGNORE_SHA256,
        "ignore policy differs from exact Run B/Run E authorization",
    )
    return final


def _register_records(root: Path) -> list[dict]:
    with (root / REGISTER_PATH).open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == REGISTER_FIELDS, "result register closed schema")
        records = list(reader)
    require(
        len(records) == len(REGISTER_IDS)
        and {r["result_id"] for r in records} == set(REGISTER_IDS)
        and all(set(r) == set(REGISTER_FIELDS) and None not in r.values() for r in records),
        "result register universe",
    )
    return records


def validate_researcher_decision(root: Path = PROJECT_ROOT) -> list[dict]:
    """Require the exact historical ledger plus P5-002 and exactly four promotions.

    After the exact P5-003 closure, later authorized phases may follow (P7-002).
    """
    require(ledger_state(root) != "preparation", "exact P5-002 ledger entry required")
    require((root / REGISTER_PATH).is_file(), "promoted result register required")
    records = _register_records(root)
    require(
        {r["result_id"] for r in records if r["status"] == "PROMOTED"} == set(PROMOTED_IDS),
        "exact researcher-approved promoted set required",
    )
    for row in records:
        require(
            row["statement"] == APPROVED_STATEMENTS[row["result_id"]],
            "researcher-approved statement mismatch",
        )
        require(
            all(
                row[k].strip() for k in ("evidence", "interpretation_boundary", "researcher_notes")
            ),
            "promotion evidence/boundary/notes required",
        )
    return records


def assert_policy(root: Path = PROJECT_ROOT) -> None:
    final = publication_stage(root)
    if final:
        validate_researcher_decision(root)
    for relative in FINAL_OUTPUTS.values():
        require(
            ignored(root, relative) != final,
            f"final artifact allowlist differs from publication stage: {relative}",
        )
    for relative in OUTPUTS.values():
        require(not ignored(root, relative), f"preparation artifact not allowlisted: {relative}")
    for relative in (
        *LOCAL_PATHS.values(),
        "outputs/tables/choice/final/results.csv",
        "outputs/models/choice/model.pkl",
        "outputs/figures/choice/result.png",
        "outputs/tables/choice/customer_rows.csv",
    ):
        require(ignored(root, relative), f"unauthorized result/customer path trackable: {relative}")
    allowed = (
        set(OUTPUTS.values())
        | set(CORE_OUTPUTS.values())
        | set(RUN_D_OUTPUTS.values())
        | {PREPARATION_VALIDATION_PATH, D2_PATH, PUBLICATION_PROVENANCE}
        | (set(FINAL_OUTPUTS.values()) if final else set())
    )
    for relative in (
        *CORE_OUTPUTS.values(),
        *RUN_D_OUTPUTS.values(),
        PREPARATION_VALIDATION_PATH,
        D2_PATH,
        PUBLICATION_PROVENANCE,
    ):
        require(ignored(root, relative), "Run C/D exploratory path must remain ignored")
    register = root / REGISTER_PATH
    if register.exists() and not final:
        records = _register_records(root)
        require(
            all(r["status"] in REGISTER_STATUSES for r in records),
            "Phase 5 result promotion prohibited",
        )
    for tree in (
        "model",
        "models",
        "figure",
        "figures",
        "table",
        "tables",
        "diagnostic",
        "diagnostics",
    ):
        for scope in ("choice", "choices", "purchased_composition"):
            directory = root / "outputs" / tree / scope
            for p in directory.rglob("*"):
                if p.is_file() and p.name != ".gitkeep":
                    require(
                        str(p.relative_to(root)) in allowed,
                        "unapproved Phase 5 result/customer artifact",
                    )
    for p in (root / "outputs").rglob("phase5_*"):
        if p.is_file():
            require(str(p.relative_to(root)) in allowed, "unapproved Phase 5 namespace artifact")


def summary_bytes(rows: list[dict], expected_keys: set[tuple[str, str, str]]) -> bytes:
    require(
        type(rows) is list and len(rows) == len(expected_keys), "preflight row universe mismatch"
    )
    keys = []
    for row in rows:
        require(
            type(row) is dict and set(row) == set(SUMMARY_FIELDS),
            "preflight closed schema mismatch",
        )
        require(
            all(type(row[k]) is str for k in ("sample", "domain", "metric", "scope")),
            "preflight label type",
        )
        require(row["scope"] == SCOPE, "preflight scope mismatch")
        require(
            type(row["value"]) in (int, float) and math.isfinite(row["value"]),
            "preflight scalar value type",
        )
        keys.append((row["sample"], row["domain"], row["metric"]))
    require(
        len(set(keys)) == len(keys) and set(keys) == expected_keys,
        "preflight unknown/duplicate metric",
    )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: (r["sample"], r["domain"], r["metric"])))
    return handle.getvalue().encode()


def qa_bytes(qa: dict[str, int], expected: dict[str, int]) -> bytes:
    require(set(qa) == set(expected), "QA metric universe mismatch")
    require(all(type(v) is int for v in qa.values()), "QA values must be integer counts")
    require(qa == expected, "accepted construction accounting mismatch")
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(("check_name", "observed", "expected", "status", "scope"))
    for k in sorted(expected):
        writer.writerow((k, qa[k], expected[k], "PASS", SCOPE))
    return handle.getvalue().encode()


def validate_contract(document: dict, expected_columns: dict[str, tuple[str, ...]]) -> None:
    require(
        type(document) is dict and set(document) == {"format_version", "scope", "group", "tables"},
        "contract keys",
    )
    require(
        type(document["format_version"]) is int and document["format_version"] == 1,
        "contract version",
    )
    require(document["scope"] == SCOPE and document["group"] == GROUP, "contract scope/group")
    require(
        type(document["tables"]) is dict and set(document["tables"]) == set(LOCAL_PATHS),
        "contract table universe",
    )
    for name, table in document["tables"].items():
        require(
            type(table) is dict and set(table) == {"path", "rows", "columns", "sha256"},
            "contract table schema",
        )
        require(table["path"] == LOCAL_PATHS[name], "contract unsafe/unapproved data path")
        require(
            type(table["rows"]) is int and table["rows"] == {"ledger": 6571, "pure": 6544}[name],
            "contract rows",
        )
        require(
            type(table["columns"]) is list and tuple(table["columns"]) == expected_columns[name],
            "contract column universe",
        )
        require(
            type(table["sha256"]) is str
            and len(table["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in table["sha256"]),
            "contract hash",
        )


def validate_parquet_schema(table: pq.ParquetFile, columns: tuple[str, ...]) -> None:
    from complete_journey.choice.events import LEDGER_DTYPES, PURE_COLUMNS, PURE_DTYPES

    types = dict(PURE_DTYPES if columns == PURE_COLUMNS else LEDGER_DTYPES)
    arrow_names = {"int64": "int64", "float64": "double", "string": "string", "bool": "bool"}
    require(tuple(table.schema_arrow.names) == columns, "Parquet closed column schema")
    require(
        all(str(field.type) == arrow_names[types[field.name]] for field in table.schema_arrow),
        "Parquet dtype schema drift",
    )


def write_preparation(payloads: dict[str, bytes], root: Path = PROJECT_ROOT) -> None:
    """Validate the complete intended payload before replacing authorized files."""
    require(
        set(payloads)
        <= set(OUTPUTS.values()) | set(LOCAL_PATHS.values()) | {PREPARATION_VALIDATION_PATH},
        "unapproved preparation writer path",
    )
    require(
        all(type(p) is str and type(b) is bytes for p, b in payloads.items()),
        "preparation payload types",
    )
    from complete_journey.choice.events import LEDGER_COLUMNS, PURE_COLUMNS
    from complete_journey.choice.preflight import QA_EXPECTED
    from complete_journey.choice.schema import validate_summary_csv

    assert_policy(root)
    verify_inputs(root)
    for relative, payload in payloads.items():
        if relative in PREPARATION_EVIDENCE_SHA256:
            require(
                hashlib.sha256(payload).hexdigest() == PREPARATION_EVIDENCE_SHA256[relative],
                "accepted Run B evidence replacement refused",
            )
        if relative == PREPARATION_VALIDATION_PATH:
            require(
                json.loads(payload) == preparation_validation_record(root),
                "closed preparation revalidation record",
            )
        elif relative == OUTPUTS["contract"]:
            validate_contract(json.loads(payload), {"ledger": LEDGER_COLUMNS, "pure": PURE_COLUMNS})
        elif relative == OUTPUTS["qa"]:
            require(payload == qa_bytes(QA_EXPECTED, QA_EXPECTED), "writer QA schema/value refusal")
        elif relative == OUTPUTS["summary"]:
            validate_summary_csv(payload)
        elif relative == OUTPUTS["provenance"]:
            require(
                hashlib.sha256(payload).hexdigest() == RUN_B_PROVENANCE_SHA256,
                "historical Run B provenance replacement refused",
            )
        else:
            name = next(k for k, v in LOCAL_PATHS.items() if v == relative)
            table = pq.ParquetFile(io.BytesIO(payload))
            columns = {"ledger": LEDGER_COLUMNS, "pure": PURE_COLUMNS}[name]
            validate_parquet_schema(table, columns)
            require(
                tuple(table.schema_arrow.names) == columns
                and table.metadata.num_rows == {"ledger": 6571, "pure": 6544}[name],
                "writer local table schema/row refusal",
            )
    # All semantic/schema guards run before staging. Never stage in raw/accepted namespaces.
    scratch = tempfile.gettempdir()  # honours TMPDIR; portable fallback (post-audit revision)
    with tempfile.TemporaryDirectory(prefix="phase5-stage-", dir=scratch) as temp:
        staged = {}
        for i, (relative, payload) in enumerate(sorted(payloads.items())):
            p = Path(temp) / str(i)
            p.write_bytes(payload)
            require(sha256(p) == hashlib.sha256(payload).hexdigest(), "staged bytes mismatch")
            staged[relative] = p
        for relative, p in staged.items():
            target = root / relative
            if target.is_file() and sha256(target) == sha256(p):
                continue  # Preserve accepted evidence bytes AND mtime on verified reuse.
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(p, target)


def provenance(root: Path = PROJECT_ROOT, replacement: dict[str, bytes] | None = None) -> dict:
    replacement = replacement or {}
    hashes = {}
    for relative in (*LOCAL_PATHS.values(), OUTPUTS["contract"], OUTPUTS["qa"], OUTPUTS["summary"]):
        if relative in replacement:
            hashes[relative] = hashlib.sha256(replacement[relative]).hexdigest()
        elif (root / relative).exists():
            hashes[relative] = sha256(root / relative)
    return {
        "format_version": 1,
        "scope": SCOPE,
        "source": source_identity(root),
        "inputs": verify_inputs(root),
        "artifacts": hashes,
    }


def verify_preparation_evidence(root: Path = PROJECT_ROOT) -> None:
    for relative, digest in PREPARATION_EVIDENCE_SHA256.items():
        require(
            (root / relative).is_file() and sha256(root / relative) == digest,
            f"accepted Run B evidence drift: {relative}",
        )


def validate_source_manifest(source: dict) -> None:
    """Verify an internally canonical identity, not agreement with current source."""
    require(
        type(source) is dict and set(source) == {"files", "state_sha256"},
        "historical source manifest schema",
    )
    files = source["files"]
    require(type(files) is dict and bool(files), "historical source file universe")
    require(
        all(
            type(p) is str
            and p
            and not Path(p).is_absolute()
            and ".." not in Path(p).parts
            and type(d) is str
            and len(d) == 64
            and all(c in "0123456789abcdef" for c in d)
            for p, d in files.items()
        ),
        "historical source file/hash schema",
    )
    require(
        source["state_sha256"] == hashlib.sha256(canonical_json(files)).hexdigest(),
        "historical source canonical SHA mismatch",
    )


def verify_accepted_core(root: Path = PROJECT_ROOT) -> dict:
    """Return pinned historical evidence, never a current-generation attribution.

    The return keys are provenance, report and preparation_validation. All seven
    accepted Run C bytes, Run B evidence/provenance and protected inputs must match.
    This is independent of current source and never refreshes any evidence record.
    """
    require(
        set(ACCEPTED_RUN_C_SHA256) == set(CORE_OUTPUTS.values()) | {PREPARATION_VALIDATION_PATH},
        "accepted Run C pin universe",
    )
    for relative, digest in ACCEPTED_RUN_C_SHA256.items():
        require(
            (root / relative).is_file() and sha256(root / relative) == digest,
            f"accepted Run C archive drift: {relative}; no stale-source acceptance",
        )
    assert_policy(root)
    verify_preparation_evidence(root)
    inputs = verify_inputs(root)
    historical_b = root / OUTPUTS["provenance"]
    require(
        historical_b.is_file() and sha256(historical_b) == RUN_B_PROVENANCE_SHA256,
        "historical Run B provenance drift",
    )
    manifest = json.loads((root / CORE_OUTPUTS["provenance"]).read_bytes())
    require(
        type(manifest) is dict
        and set(manifest)
        == {
            "scope",
            "source",
            "inputs",
            "preparation_evidence",
            "preparation_provenance_sha256",
            "preparation_validation_sha256",
            "artifacts",
        },
        "closed accepted core provenance",
    )
    validate_source_manifest(manifest["source"])
    require(
        manifest["scope"] == "RUN_C_UNPROMOTED_CORE"
        and manifest["inputs"] == inputs
        and manifest["preparation_evidence"] == PREPARATION_EVIDENCE_SHA256
        and manifest["preparation_provenance_sha256"] == RUN_B_PROVENANCE_SHA256
        and manifest["preparation_validation_sha256"]
        == ACCEPTED_RUN_C_SHA256[PREPARATION_VALIDATION_PATH]
        and manifest["artifacts"]
        == {p: ACCEPTED_RUN_C_SHA256[p] for k, p in CORE_OUTPUTS.items() if k != "provenance"},
        "accepted core historical provenance mismatch",
    )
    validation = json.loads((root / PREPARATION_VALIDATION_PATH).read_bytes())
    require(
        validation
        == {
            "scope": "RUN_C_PREPARATION_REVALIDATION",
            "historical_run_b_provenance_sha256": RUN_B_PROVENANCE_SHA256,
            "validated_by_source": manifest["source"],
            "inputs": inputs,
            "preparation_evidence": dict(PREPARATION_EVIDENCE_SHA256),
        },
        "accepted historical Run C preparation revalidation mismatch",
    )
    report = json.loads((root / CORE_OUTPUTS["report"]).read_bytes())
    from complete_journey.choice.run_c import validate_report

    validate_report(report)
    return {"provenance": manifest, "report": report, "preparation_validation": validation}


def verify_accepted_robustness(root: Path = PROJECT_ROOT) -> dict:
    """Return exact historical Run D provenance/report without fitting or rewriting.

    All seven immutable Run D pins and the accepted Run C/B/input chain must
    match. The canonical generating source remains historical under final policy.
    """
    require(
        set(ACCEPTED_RUN_D_SHA256) == set(RUN_D_OUTPUTS.values()), "accepted Run D pin universe"
    )
    for relative, digest in ACCEPTED_RUN_D_SHA256.items():
        require(
            (root / relative).is_file() and sha256(root / relative) == digest,
            f"accepted Run D archive drift: {relative}; no stale-source acceptance",
        )
    verify_accepted_core(root)
    manifest = json.loads((root / RUN_D_OUTPUTS["provenance"]).read_bytes())
    require(
        type(manifest) is dict
        and set(manifest) == {"scope", "status", "source", "inputs", "accepted_run_c", "artifacts"},
        "closed accepted Run D provenance",
    )
    validate_source_manifest(manifest["source"])
    require(
        manifest["scope"] == "RUN_D_UNPROMOTED_ROBUSTNESS"
        and manifest["status"] == "UNREVIEWED"
        and manifest["inputs"] == verify_inputs(root)
        and manifest["accepted_run_c"] == ACCEPTED_RUN_C_SHA256
        and manifest["artifacts"]
        == {p: ACCEPTED_RUN_D_SHA256[p] for k, p in RUN_D_OUTPUTS.items() if k != "provenance"},
        "accepted Run D historical provenance mismatch",
    )
    report = json.loads((root / RUN_D_OUTPUTS["report"]).read_bytes())
    from complete_journey.choice.robustness import validate_report

    validate_report(report)
    return {"provenance": manifest, "report": report}


def preparation_validation_record(root: Path = PROJECT_ROOT) -> dict:
    """Current verification source is not the historical Run B generating source."""
    require(
        sha256(root / OUTPUTS["provenance"]) == RUN_B_PROVENANCE_SHA256,
        "historical Run B provenance drift",
    )
    return {
        "scope": "RUN_C_PREPARATION_REVALIDATION",
        "historical_run_b_provenance_sha256": RUN_B_PROVENANCE_SHA256,
        "validated_by_source": source_identity(root),
        "inputs": verify_inputs(root),
        "preparation_evidence": dict(PREPARATION_EVIDENCE_SHA256),
    }


def refresh_preparation_provenance(root: Path = PROJECT_ROOT) -> None:
    """Record current revalidation separately; NEVER rewrite historical provenance."""
    assert_policy(root)
    verify_preparation_evidence(root)
    record = preparation_validation_record(root)
    write_preparation({PREPARATION_VALIDATION_PATH: canonical_json(record)}, root)
    validate_saved(root)


def validate_saved(root: Path = PROJECT_ROOT) -> dict | None:
    assert_policy(root)
    inputs = verify_inputs(root)
    paths = [
        root / LOCAL_PATHS["ledger"],
        root / LOCAL_PATHS["pure"],
        root / OUTPUTS["contract"],
        root / OUTPUTS["qa"],
        root / OUTPUTS["provenance"],
    ]
    if not any(p.exists() for p in paths):
        require(not (root / OUTPUTS["summary"]).exists(), "partial orphan preparation summary")
        return None
    verify_preparation_evidence(root)
    require(
        all(p.is_file() for p in paths), "partial preparation artifacts; explicit rebuild required"
    )
    from complete_journey.choice.events import LEDGER_COLUMNS, PURE_COLUMNS

    contract = json.loads((root / OUTPUTS["contract"]).read_text())
    validate_contract(contract, {"ledger": LEDGER_COLUMNS, "pure": PURE_COLUMNS})
    document = json.loads((root / OUTPUTS["provenance"]).read_text())
    require(
        type(document) is dict
        and set(document) == {"format_version", "scope", "source", "inputs", "artifacts"},
        "provenance schema",
    )
    require(
        type(document["format_version"]) is int
        and document["format_version"] == 1
        and document["scope"] == SCOPE,
        "provenance header",
    )
    require(
        sha256(root / OUTPUTS["provenance"]) == RUN_B_PROVENANCE_SHA256,
        "historical Run B provenance drift",
    )
    require(document["inputs"] == inputs, "accepted input provenance drift")
    if document["source"] != source_identity(root):
        validation = root / PREPARATION_VALIDATION_PATH
        require(
            validation.is_file(),
            "current preparation revalidation stale; explicit refresh required",
        )
        if json.loads(validation.read_bytes()) != preparation_validation_record(root):
            # Only the exact accepted seven-file archive admits historical validation.
            # It remains Run C validation, not a claim that Run D generated Run B/C.
            verify_accepted_core(root)
    expected_paths = set(LOCAL_PATHS.values()) | {OUTPUTS["contract"], OUTPUTS["qa"]}
    if (root / OUTPUTS["summary"]).exists():
        expected_paths.add(OUTPUTS["summary"])
    require(
        type(document["artifacts"]) is dict and set(document["artifacts"]) == expected_paths,
        "provenance artifact universe",
    )
    for relative, digest in document["artifacts"].items():
        require(sha256(root / relative) == digest, f"preparation artifact drift: {relative}")
    for table in contract["tables"].values():
        parquet = pq.ParquetFile(root / table["path"])
        validate_parquet_schema(parquet, tuple(table["columns"]))
        require(
            parquet.metadata.num_rows == table["rows"]
            and tuple(parquet.schema_arrow.names) == tuple(table["columns"]),
            "Parquet schema/row contract mismatch",
        )
        require(
            table["sha256"] == document["artifacts"][table["path"]],
            "contract/provenance hash disagreement",
        )
    return document
