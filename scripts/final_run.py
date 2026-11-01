#!/usr/bin/env python
"""Phase 8 final run (P8-006). Normally started through scripts/launch_abm.py.

--execute       run / resume the stage machine (used by the launcher)
--snapshot      write the immutable final-run snapshot (authorization commit only)
--verify        verify the snapshot and the S1-P8005 import
--preflight     synthetic and engineering preflight (no scientific computation)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from complete_journey.abm.config import Phase8ContractError  # noqa: E402

ENG_WORK = "outputs/diagnostics/phase8_final_eng"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--execute", action="store_true")
    g.add_argument("--snapshot", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--engineering", action="store_true", help="bounded runner test (no science)")
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args(argv)
    try:
        if args.execute:
            from complete_journey.phase8_final.pipeline import Pipeline

            print(json.dumps(Pipeline(workers=args.workers).execute(), default=str), flush=True)
        elif args.snapshot:
            from complete_journey.phase8_final.snapshot import write_snapshot

            print(json.dumps({"snapshot_sha256": write_snapshot()}))
        elif args.verify:
            from complete_journey.phase8_final.snapshot import verify_snapshot
            from complete_journey.phase8_final.store import import_verification

            print(
                json.dumps(
                    {"final_snapshot": verify_snapshot(), "import": import_verification()},
                    indent=1,
                    default=str,
                )
            )
        elif args.engineering:
            from complete_journey.phase8_final import jobs as J
            from complete_journey.phase8_final.runner import Runner

            r = Runner(
                "engineering", "engineering", work=ENG_WORK, workers=3, deadline=False, probe=False
            )
            r.acquire_lock()
            try:
                r.stage = "ENGINEERING"
                out = r.run_jobs(
                    [J.FJob("_eng", "logit_normal", index=i, sub=2000) for i in range(40)]
                )
                r.write_status(force=True, health="RUN_COMPLETE")
            finally:
                r.release_lock()
            print(json.dumps(out, default=str), flush=True)
        else:
            from complete_journey.phase8_final.preflight import run_all

            print(json.dumps(run_all(), indent=1, default=str))
    except Phase8ContractError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
