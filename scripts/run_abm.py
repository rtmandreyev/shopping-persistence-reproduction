#!/usr/bin/env python
"""Phase 8 entry point (P8-001). Run B allows synthetic correctness modes only.

  --check               config pins, archive contract (hash only), guard state
  --synthetic           B1–B12 fixtures, the A1′/A2 order comparison and the
                        purchase-envelope diagnostic (SYNTHETIC NON-EVIDENCE)
  --tiny-span           tiny span-parity engine run on synthetic images
  --tiny-perfect-model  tiny perfect-model exercise on synthetic truth panels
  --s1p-preflight       P8-005 synthetic fixtures C1-C23 (SYNTHETIC NON-EVIDENCE)
  --s1p-dry-run         tiny S1-P8005 plumbing run (throwaway store; never evidence)
  --s1p-cost-probe      engineering cost probe (timings only; outputs discarded)

S1-P8005 (P8-005 replacement Stage 1) modes refuse until its authorization commit
(``abm.config.P8005_STAGE1_AUTHORIZED``): --s1p-snapshot, --s1p-execute,
--s1p-verify-resume. The archived Stage 1 modes (--stage1-*) refuse once the source
drifts from snapshot e59d101b.

Production modes refuse until their stage is authorized by an explicit researcher
commit (``abm.config.AUTHORIZED_STAGE``). There is no force flag:

  --build-emulator  --run-rq2  --run-real  --run-n-scaling

Summaries go to stdout and to the ignored ``outputs/diagnostics/phase8_run_b/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from complete_journey.abm import config, guards, preflight  # noqa: E402
from complete_journey.abm.config import Phase8ContractError  # noqa: E402
from complete_journey.paths import PROJECT_ROOT  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs/diagnostics/phase8_run_b"
HEAVY = {
    "build_emulator": "build-emulator",
    "run_rq2": "run-rq2",
    "run_real": "run-real",
    "run_n_scaling": "run-n-scaling",
}


def _default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _keys(value):
    """JSON needs string keys; tuple keys become 'a|b'."""
    if isinstance(value, dict):
        return {
            ("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): _keys(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_keys(v) for v in value]
    return value


def _emit(name: str, payload: dict) -> None:
    text = json.dumps(_keys(payload), indent=2, sort_keys=True, default=_default)
    print(text)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(text + "\n")


def check() -> dict:
    archive = guards.archive_contract_values()
    pins = {
        "ci1_se_pinned": all(
            abs(archive[("CI-1", c)]["se"] - config.CI1_BOOT_SE[c]) < 1e-15
            for c in config.CI1_COMPONENTS
        ),
        "naive_m3_se_pinned": abs(archive[("VI-2", "O-M3")]["se"] - config.NAIVE_M3_SE) < 1e-15,
        "naive_rec_se_pinned": abs(
            archive[("VI-6", "delta_log_loss_recency")]["se"] - config.NAIVE_REC_SE
        )
        < 1e-15,
    }
    if not all(pins.values()):
        raise Phase8ContractError(f"configuration pins disagree with the archive: {pins}")
    return {
        "authorized_stage": config.AUTHORIZED_STAGE,
        "archive_sha256": guards.verify_real_archive(),
        "persistence_cap": config.PERSISTENCE_CAP,
        "burn_in_half_lives": config.BURN_IN_WEEKS / config.HALF_LIFE_CAP_WEEKS,
        "primary_moments": config.PRIMARY_MOMENTS,
        "kappas": config.KAPPAS,
        **pins,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--synthetic", action="store_true")
    modes.add_argument("--tiny-span", action="store_true")
    modes.add_argument("--tiny-perfect-model", action="store_true")
    for flag in HEAVY.values():
        modes.add_argument(f"--{flag}", action="store_true", help="production; guarded")
    modes.add_argument("--stage1-snapshot", action="store_true", help="write the snapshot")
    modes.add_argument("--stage1-resolution", action="store_true", help="A1 + tolerances")
    modes.add_argument("--stage1-span", action="store_true", help="S1-A span jobs")
    modes.add_argument("--stage1-span-report", action="store_true", help="S1-A analysis")
    modes.add_argument("--stage1-verify-resume", action="store_true", help="resume audit")
    modes.add_argument("--s1p-preflight", action="store_true", help="P8-005 C1-C23 fixtures")
    modes.add_argument("--s1p-dry-run", action="store_true", help="tiny S1-P8005 plumbing run")
    modes.add_argument("--s1p-cost-probe", action="store_true", help="engineering cost probe")
    modes.add_argument("--s1p-snapshot", action="store_true", help="write the S1-P8005 snapshot")
    modes.add_argument("--s1p-execute", action="store_true", help="S1-P8005 heavy run")
    modes.add_argument("--s1p-verify-resume", action="store_true", help="S1-P8005 resume audit")
    parser.add_argument("--through", choices=("TRUTH", "S1A", "ALL"), default="ALL")
    parser.add_argument("--projected-core-hours", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        for attr, mode in HEAVY.items():
            if getattr(args, attr):
                guards.require_heavy_authorized(mode)
                raise Phase8ContractError(f"'{mode}' is not implemented in Run B")
        if args.stage1_snapshot or args.stage1_resolution or args.stage1_span:
            guards.require_heavy_authorized("build-emulator")  # Stage 1 authorization
        if args.stage1_snapshot:
            from complete_journey.abm import stage1

            _emit("stage1_snapshot", {"snapshot_sha256": stage1.write_snapshot()})
        elif args.stage1_resolution:
            from complete_journey.abm import stage1

            run = stage1.run_jobs([stage1.Job("a1", "A1", -1.0, -1.0)], workers=1)
            report = stage1.resolution_report()
            report["run"] = run
            digest = stage1.write_json(f"{stage1.ARCHIVE}/s1_resolution.json", report)
            _emit("stage1_resolution", {**report, "artifact_sha256": digest})
        elif args.stage1_span:
            from complete_journey.abm import stage1

            workers = args.workers or stage1.WORKERS
            _emit("stage1_span_run", stage1.run_jobs(stage1.span_jobs(), workers, args.limit))
        elif args.stage1_span_report:
            from complete_journey.abm import stage1

            report = stage1.span_report()
            digest = stage1.write_json(f"{stage1.ARCHIVE}/s1_span.json", report)
            _emit("stage1_span_report", {**report, "artifact_sha256": digest})
        elif args.stage1_verify_resume:
            from complete_journey.abm import stage1

            _emit("stage1_verify_resume", stage1.verify_resume())
        elif args.s1p_preflight:
            from complete_journey.abm import preflight_p8005

            _emit("s1p_preflight", preflight_p8005.run_all())
        elif args.s1p_dry_run:
            from complete_journey.abm import preflight_p8005

            _emit("s1p_dry_run", preflight_p8005.pipeline_dry_run())
        elif args.s1p_cost_probe:
            from complete_journey.abm import preflight_p8005

            _emit("s1p_cost_probe", preflight_p8005.cost_probe())
        elif args.s1p_snapshot or args.s1p_execute or args.s1p_verify_resume:
            from complete_journey.abm import stage1_p8005

            guards.require_p8005_authorized()
            if args.s1p_snapshot:
                digest = stage1_p8005.write_snapshot(projected_core_hours=args.projected_core_hours)
                _emit("s1p_snapshot", {"snapshot_sha256": digest})
            elif args.s1p_execute:
                workers = args.workers or stage1_p8005.WORKERS
                _emit("s1p_execute", stage1_p8005.execute(workers, args.through))
            else:
                _emit("s1p_verify_resume", stage1_p8005.verify_resume())
        elif args.check:
            _emit("check", check())
        elif args.synthetic:
            _emit("synthetic", {k: fn() for k, fn in preflight.SYNTHETIC_FIXTURES.items()})
        elif args.tiny_span:
            _emit("tiny_span", preflight.tiny_span())
        elif args.tiny_perfect_model:
            _emit("tiny_perfect_model", preflight.tiny_perfect_model())
    except Phase8ContractError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
