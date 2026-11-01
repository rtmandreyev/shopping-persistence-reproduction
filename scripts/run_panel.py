#!/usr/bin/env python
"""Phase 4 accepted-evidence verification/publication; no empirical estimation."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from complete_journey.econometrics.archives import verify_history  # noqa: E402
from complete_journey.econometrics.artifacts import (  # noqa: E402
    OUTPUTS,
    assert_policy,
    sha256,
    source_identity,
    validate_existing_summary,
    verify_inputs,
)
from complete_journey.econometrics.final_contract import FINAL_PATHS  # noqa: E402
from complete_journey.econometrics.samples import require  # noqa: E402
from complete_journey.paths import PROJECT_ROOT  # noqa: E402


def run_check(*, check_existing_source: bool = True) -> tuple:
    """Hash-check unchanged inputs and immutable design/results; no mart or fits."""
    assert_policy()
    history = verify_history()
    inputs = verify_inputs()
    validate_existing_summary(require_current_source=check_existing_source, inputs=inputs)
    return None, inputs, history


def run_preflight() -> dict:
    """Read back accepted Run B evidence; never refresh historical provenance."""
    run_check()
    with (PROJECT_ROOT / OUTPUTS["summary"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {"rows": len(rows), "hashes": {k: sha256(PROJECT_ROOT / p) for k, p in OUTPUTS.items()}}


def run_core() -> dict:
    """Verify immutable Run C creation-time records; never regenerate the ladder."""
    from complete_journey.econometrics.core_artifacts import validate_saved

    run_check()
    saved = validate_saved()
    require(saved is not None, "accepted Run C artifacts missing; STOP rather than regenerate")
    return {"models": len(saved), "reused": True}


def run_sensitivity() -> dict:
    """Verify immutable reviewed Run D evidence; no refitting or audit rewrite."""
    from complete_journey.econometrics.core_artifacts import validate_saved
    from complete_journey.econometrics.sensitivity_artifacts import validate_existing

    run_check()
    require(validate_saved() is not None, "accepted Run C baseline missing")
    rows = validate_existing()
    require(rows is not None, "accepted Run D artifacts missing; STOP rather than refit")
    return {"models": len(rows), "reused": True}


def run_final() -> dict:
    """Select/render the three approved paper artifacts from accepted aggregates."""
    from complete_journey.econometrics.publication import publish_final, validate_final

    _, inputs, history = run_check()
    source = source_identity(PROJECT_ROOT)
    hashes = publish_final()
    validate_final()
    require(inputs == verify_inputs() and history == verify_history(), "accepted evidence changed")
    require(source == source_identity(PROJECT_ROOT), "publication source changed during render")
    return {"hashes": hashes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--check", action="store_true", help="Read-only accepted input/history checks"
    )
    modes.add_argument("--preflight", action="store_true", help="Verify immutable Run B preflight")
    modes.add_argument(
        "--core", action="store_true", help="Verify immutable accepted Run C records"
    )
    modes.add_argument(
        "--sensitivities", action="store_true", help="Verify immutable reviewed Run D"
    )
    modes.add_argument(
        "--publish-final", action="store_true", help="Publish only reviewed F10/T4/T5"
    )
    args = parser.parse_args()
    if args.check:
        _, inputs, _ = run_check()
        if any((PROJECT_ROOT / p).exists() for p in FINAL_PATHS):
            from complete_journey.econometrics.publication import validate_final

            validate_final()
        print(
            f"PHASE 4 CHECK PASS: {inputs['parquet_artifacts']} accepted Parquet, "
            f"{inputs['phase3_final_artifacts']} Phase 3 artifacts; immutable B/C/D verified"
        )
    elif args.preflight:
        report = run_preflight()
        print(f"RUN B ARCHIVE PASS: {report['rows']} design rows; provenance unchanged")
        for kind, digest in report["hashes"].items():
            print(f"{kind} SHA256 {digest}")
    elif args.core:
        report = run_core()
        print(f"RUN C ARCHIVE PASS: {report['models']} creation-time records; no ladder refit")
    elif args.sensitivities:
        report = run_sensitivity()
        print(f"RUN D ARCHIVE PASS: {report['models']} reviewed rows; no sensitivity refit")
    else:
        report = run_final()
        print("RUN E PUBLICATION PASS: exactly F10/T4/T5; primary M3 paper uncertainty DK8")
        for path, digest in report["hashes"].items():
            print(f"{path} SHA256 {digest}")
    print("Phase 4 ACCEPTED AND CLOSED; later phases governed by docs/DECISIONS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
