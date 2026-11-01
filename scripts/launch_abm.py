#!/usr/bin/env python
"""Detached, resume-aware launcher of the Phase 8 final run (P8-006).

    cd ~/complete-journey && uv run python scripts/launch_abm.py --resume

It verifies the final snapshot and the S1-P8005 import, checks the run lock (a live run
is never duplicated; a stale lock is cleared), checks free disk (>= 10 GB) and power,
then starts ``caffeinate -dims python scripts/final_run.py --execute`` in its OWN
SESSION, so the run survives Claude usage limits, chat disconnects, Terminal closure
and Wi-Fi loss. Completed jobs are skipped; the run continues at the correct stage.

  --resume   start or continue the run (default)
  --status   print the run status and exit
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WORK = ROOT / "outputs/diagnostics/phase8_final"


def alive_run() -> int | None:
    lock = WORK / "run.lock"  # noqa: F821 - WORK rebound by --engineering
    if not lock.exists():
        return None
    try:
        pid = int(json.loads(lock.read_text())["pid"])
        os.kill(pid, 0)
    except (ValueError, KeyError, OSError):
        return None
    cmd = subprocess.run(
        ["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True
    ).stdout
    return pid if "final_run.py" in cmd else None


def status() -> dict:
    try:
        return json.loads((WORK / "status.json").read_text())
    except (OSError, ValueError):
        return {}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--status", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--engineering", action="store_true", help="bounded launcher/resume test")
    args = p.parse_args(argv)
    global WORK
    if args.engineering:
        WORK = ROOT / "outputs/diagnostics/phase8_final_eng"
    st = status()
    pid = alive_run()
    if args.status:
        print(
            json.dumps(
                {
                    "alive_pid": pid,
                    **{
                        k: st.get(k)
                        for k in (
                            "stage",
                            "health",
                            "updated_utc",
                            "workers",
                            "eta_seconds",
                            "stop_message",
                        )
                    },
                },
                indent=1,
            )
        )
        return 0
    if pid:
        print(f"The Phase 8 final run is already running (pid {pid}). Nothing to do.")
        print("Watch it with: cd ~/complete-journey && uv run python scripts/watch_abm.py")
        return 0
    if st.get("health") == "RUN_COMPLETE":
        print("The heavy run is complete (status RUN_COMPLETE). Nothing to resume.")
        return 0
    if args.engineering:
        digest = "engineering"
    else:
        from complete_journey.phase8_final.snapshot import verify_snapshot
        from complete_journey.phase8_final.store import import_verification

        digest = verify_snapshot()
        import_verification()
    WORK.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(WORK).free / 1e9
    if free < 10.0:
        print(f"REFUSED: only {free:.1f} GB free (need >= 10 GB).", file=sys.stderr)
        return 2
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    if "AC Power" not in power:
        print("WARNING: the Mac is on battery. Plug it in for a multi-day run.")
    mode = "--engineering" if args.engineering else "--execute"
    cmd = ["caffeinate", "-dims", sys.executable, str(ROOT / "scripts/final_run.py"), mode]
    if args.workers:
        cmd += ["--workers", str(args.workers)]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log = open(WORK / "run.log", "a")  # noqa: SIM115 - handed to the detached child
    log.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%S')} snapshot {digest[:16]} ===\n")
    log.flush()
    child = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (WORK / "launch.json").write_text(
        json.dumps(
            {
                "caffeinate_pid": child.pid,
                "started_epoch": time.time(),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "command": " ".join(cmd),
                "snapshot_sha256": digest,
                "free_gb": free,
            },
            indent=1,
        )
    )
    print(f"Launched (caffeinate pid {child.pid}, own session). Snapshot {digest[:16]}.")
    print("Watch: cd ~/complete-journey && uv run python scripts/watch_abm.py")
    print(
        "Resume after any interruption: cd ~/complete-journey && "
        "uv run python scripts/launch_abm.py --resume"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
