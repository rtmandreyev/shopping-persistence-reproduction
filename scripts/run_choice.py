#!/usr/bin/env python
"""Phase 5 (accepted and closed, P5-003) preparation checks and historical Run C/D archives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from complete_journey.choice.contracts import ChoiceContractError  # noqa: E402
from complete_journey.choice.preflight import build, check, preflight  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--check", action="store_true", help="read-only preparation/source/privacy check"
    )
    modes.add_argument(
        "--build", action="store_true", help="canonical factual event/history tables only"
    )
    modes.add_argument(
        "--preflight",
        action="store_true",
        help="outcome-free design and synthetic implementation checks",
    )
    modes.add_argument(
        "--refresh-preparation-provenance",
        action="store_true",
        help="separate current-source revalidation; historical Run B provenance unchanged",
    )
    modes.add_argument(
        "--core",
        action="store_true",
        help="frozen baseline M0/M1, household × week only; ignored UNPROMOTED results",
    )
    modes.add_argument(
        "--robustness",
        action="store_true",
        help="only predeclared R1–R4; ignored UNPROMOTED results; inference review pending",
    )
    args = parser.parse_args(argv)
    try:
        if args.refresh_preparation_provenance:
            from complete_journey.choice.artifacts import refresh_preparation_provenance

            refresh_preparation_provenance()
        elif args.robustness:
            from complete_journey.choice.run_d import run_robustness

            report, reused = run_robustness()
            print(
                f"Phase 5 Run D {'REUSED VERIFIED ARCHIVE' if reused else 'ESTIMATED'}: bounded R1–R4; UNPROMOTED; researcher inference review required"
            )
            for variant in ("R1", "R2", "R3", "R4"):
                for row in report["variants"][variant]["results"]:
                    print(
                        f"{variant} {row['term']}: {row['coefficient']:.9f} probability points; {row['inference_status']}"
                    )
            return 0
        elif args.core:
            from complete_journey.choice.run_c import run_core

            report, reused = run_core()
            print(
                f"Phase 5 Run C core {'REUSED VERIFIED ARCHIVE' if reused else 'ESTIMATED'}: only M0/M1; UNPROMOTED; researcher review required"
            )
            for row in report["results"]:
                print(
                    f"{row['model']} {row['term']}: {row['coefficient']:.9f} probability points; {row['percentage_points']:.6f} percentage points; {row['inference_status']}"
                )
            return 0
        elif args.check:
            check()
        elif args.build:
            build()
        else:
            preflight()
    except (ChoiceContractError, ValueError, FileNotFoundError) as error:
        print(f"Phase 5 STOP: {error}", file=sys.stderr)
        return 1
    print(
        "Phase 5 preparation PASS; ACCEPTED AND CLOSED (P5-003); historical Run C core / Run D R1–R4 archives only; no refit authorized"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
