#!/usr/bin/env python
"""Phase 7 entry point: contracts, real-data preflight, synthetic preflight, Run C.

Real-data fitting exists only in the Run C modes. It needs the committed
authorization token (``config.REAL_FIT_AUTHORIZATION``), a design that matches the
committed snapshot, and the one-time test ledger state:

  --check      contracts, config pins, feature allowlist, Phase 2 current-source
               validation; no data rows, no fits
  --preflight  real-data counts, eligibility, split support, feature timing and the
               Tier 2 truncation tests, plus reconciliation to published Phase 4
               aggregates; no predictive fit
  --synthetic  S1-S7 synthetic fixtures through the full frozen pipeline (non-evidence)
  --run-real   Run C one-time reveal; refuses once the test ledger exists
  --audit-real reproduce the archived Run C numbers from frozen inputs and the
               ledger's recorded selections; never reselects, never reopens the test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from complete_journey.prediction import preflight  # noqa: E402
from complete_journey.prediction.panel import Phase7ContractError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true", help="contracts/config only")
    modes.add_argument(
        "--preflight", action="store_true", help="real-data counts and timing; no fits"
    )
    modes.add_argument("--synthetic", action="store_true", help="synthetic S1-S7 pipeline")
    modes.add_argument("--run-real", action="store_true", help="Run C one-time real reveal")
    modes.add_argument(
        "--audit-real",
        action="store_true",
        help="reproduce archived Run C numbers (no reselection)",
    )
    args = parser.parse_args(argv)
    try:
        if args.check:
            report = preflight.check()
        elif args.preflight:
            report = preflight.real_preflight()
        elif args.synthetic:
            report = preflight.synthetic_preflight()
        elif args.run_real:
            from complete_journey.prediction import run_c

            report = run_c.run_real()
        else:
            from complete_journey.prediction import run_c

            report = run_c.audit_real()
    except (Phase7ContractError, ValueError, FileNotFoundError) as error:
        print(f"Phase 7 STOP: {error}", file=sys.stderr)
        return 1
    print(preflight.render(report))
    if args.run_real or args.audit_real:
        print("Phase 7 Run C PASS: frozen pipeline only; no ABM, no Jev; Phase 8 not begun")
    else:
        print(
            "Phase 7 PASS: no predictive model fitted or scored on real outcomes in this "
            "mode; no ABM, no Jev; Phase 8 not begun"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
