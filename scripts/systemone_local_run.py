#!/usr/bin/env python
"""Phase 9 local production run (executor). Run from the repository root; the original
production runs were started detached by a separate launcher.

  --execute       the frozen production design (N=2500, T=102, burn-in 26, R=200,
                  Q_pop=200, lambda in {0.5, 0.8, 0.95}, all three policies)
  --engineering   a bounded engineering run (separate work directory)
  --p9007         the P9-007 anchored out-of-family extension (with --execute or
                  --engineering)

No network, no credentials: see ``complete_journey.systemone.localrun``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    from complete_journey.systemone import localrun as LR

    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--execute", action="store_true")
    g.add_argument("--engineering", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--p9007", action="store_true", help="the P9-007 anchored extension")
    a = p.parse_args(argv)
    if a.p9007:
        from complete_journey.systemone import anchoredrun as AR

        cfg = AR.ENGINEERING if a.engineering else AR.PRODUCTION
        res = AR.run(cfg, a.workers)
    else:
        cfg = LR.ENGINEERING if a.engineering else LR.PRODUCTION
        res = LR.run(cfg, a.workers)
    print(json.dumps({"results_sha256": res["results_sha256"], "label": cfg.label}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
