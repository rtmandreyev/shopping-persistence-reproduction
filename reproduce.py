#!/usr/bin/env python
"""Reproduce Andreyev (2026), "Heterogeneity, Latent State or Purchase History?"

    ./reproduce.sh                                   # --quick: 0.5 to 2 minutes, no data needed
    ./reproduce.sh --check                           # verification only: no rendering, no compiling
    ./reproduce.sh --standard [--data-dir PATH]      # adds the raw-data reproduction
    ./reproduce.sh --learned-model-simulations --archive DIR [--checkpoint-stores DIR]
    ./reproduce.sh --known-truth-production          # why the in-family production run is not repeated
    ./reproduce.sh --model-queries                   # status of the original learned-model queries

Apart from uv's environment (.venv/), every mode writes only under the build directory
(default build/), reads no credentials and ends with a report,
build/reproduction_report_<mode>.json. No step needs the network, and
reproduce.py blocks network access in its own process.
An option that the selected mode does not use is a usage error.
Exit status: 0 if no check failed, 1 if any check failed, 2 for usage errors.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True


def _no_network(*_a, **_k):
    raise RuntimeError("network access is disabled in reproduce.py")


# Options and the modes that use them; any other combination is rejected.
USED_BY = {
    "data_dir": ("--data-dir", {"standard"}),
    "archive": ("--archive", {"learned-model-simulations"}),
    "checkpoint_stores": ("--checkpoint-stores", {"learned-model-simulations"}),
    "small_scale": ("--small-scale", {"learned-model-simulations"}),
    "workers": ("--workers", {"quick", "check", "standard", "learned-model-simulations"}),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="checks that need no data (default)")
    g.add_argument("--check", action="store_true", help="the verification checks of --quick only")
    g.add_argument("--standard", action="store_true", help="--quick plus the raw-data reproduction")
    g.add_argument("--learned-model-simulations", action="store_true",
                   help="rerun the Section 7 simulations from an authorized response archive")
    g.add_argument("--known-truth-production", action="store_true",
                   help="explain why the in-family production run is not repeated")
    g.add_argument("--model-queries", action="store_true",
                   help="explain the status of the original learned-model queries")
    p.add_argument("--build-dir", type=Path, default=ROOT / "build", metavar="DIR", help="output folder (default: build/)")
    p.add_argument("--data-dir", type=Path, default=None, metavar="PATH",
                   help="the release folder dunnhumby_The-Complete-Journey/ or the folder that contains it "
                        "(default: data/raw)")
    p.add_argument("--archive", type=Path, default=None, metavar="DIR",
                   help="with --learned-model-simulations: an authorized copy of the stored responses")
    p.add_argument("--checkpoint-stores", type=Path, default=None, metavar="DIR",
                   help="with --learned-model-simulations: the in-family checkpoint stores (phase8_final/ and "
                        "phase8_stage1_p8005/), needed by the anchored run")
    p.add_argument("--small-scale", action="store_true",
                   help="with --learned-model-simulations: run at small scale only (minutes)")
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="parallel processes (default: one fewer than the CPUs, at most 8)")
    a = p.parse_args(argv)

    mode = ("standard" if a.standard else "learned-model-simulations" if a.learned_model_simulations
            else "known-truth-production" if a.known_truth_production else "model-queries" if a.model_queries
            else "check" if a.check else "quick")
    for dest, (flag, modes) in USED_BY.items():
        if getattr(a, dest) not in (None, False) and mode not in modes:
            p.error(f"{flag} is used only with --{next(iter(modes))}" if len(modes) == 1
                    else f"{flag} is not used with --{mode}")
    if a.workers is not None and a.workers < 1:
        p.error("--workers must be at least 1")
    data_dir = a.data_dir if a.data_dir is not None else ROOT / "data" / "raw"
    workers = a.workers if a.workers is not None else max(1, min(8, (os.cpu_count() or 2) - 1))

    socket.socket.connect = _no_network  # no mode needs the network
    socket.create_connection = _no_network

    from repro import modes as M
    from repro.report import Report

    if mode == "known-truth-production":
        print(M.KNOWN_TRUTH_TEXT)
        rep = Report("known-truth-production", a.build_dir)
        rep.add("in-family known-truth production run", "SKIPPED-BY-DESIGN", "not repeated by this package")
        print(f"report: {rep.write()}")
        return 0
    if mode == "model-queries":
        print(M.MODEL_QUERIES_TEXT)
        rep = Report("model-queries", a.build_dir)
        rep.add("original learned-model queries", "SKIPPED-BY-DESIGN", "status only; no call is made")
        print(f"report: {rep.write()}")
        return 0

    estimates = {"quick": "about 0.5-2 minutes", "check": "under a minute",
                 "standard": "about 2-8 minutes (needs the raw files)",
                 "learned-model-simulations": "minutes" if a.small_scale else
                 "hours: the original runs took 2.27 h and 7.52 h of wall time on 8 workers"}
    print(f"mode: {mode} (expected runtime {estimates[mode]}); build directory: {a.build_dir}", flush=True)
    rep = Report(mode, a.build_dir)
    if mode == "quick":
        M.quick(rep, a.build_dir, workers)
    elif mode == "check":
        M.quick(rep, a.build_dir, workers, verify_only=True)
    elif mode == "standard":
        M.standard(rep, a.build_dir, data_dir, workers)
    else:
        M.learned_model_simulations(rep, a.build_dir, a.archive, workers, a.small_scale, a.checkpoint_stores)
    path = rep.write()
    s = {k: sum(r.status == k for r in rep.results) for k in ("PASS", "FAIL", "SKIPPED-BY-DESIGN", "WARN")}
    print(f"\nsummary: {s}\nreport: {path}")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
