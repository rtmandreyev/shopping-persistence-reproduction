"""Synthetic / engineering preflight of the final run (no scientific result).

- ``stream_continuation``: re-simulating MODEL replicates 0..49 at an imported S1A point
  reproduces the imported image bit for bit (same seeds, same η̂, same moment engine).
- ``pooled_extension``: the R = 50 → 200 pooling equals a clean deterministic run.
- ``dry_run``: the whole stage machine at toy size in a throwaway namespace, with
  SYNTHETIC stand-ins for the real targets and windows (no real dynamic data is read).
"""

from __future__ import annotations

import json
import shutil
from contextlib import ExitStack
from unittest import mock

import numpy as np

from complete_journey.abm import config as abm_config
from complete_journey.abm import stage1_p8005 as s1p
from complete_journey.abm.emulator import LOGIT_NORMAL, MIXTURE
from complete_journey.abm.guards import TargetVector
from complete_journey.paths import PROJECT_ROOT
from complete_journey.phase8_final import analysis as A
from complete_journey.phase8_final import jobs as J
from complete_journey.phase8_final import reveal as R
from complete_journey.phase8_final import settings as S
from complete_journey.phase8_final.evaluate import pooled_stats
from complete_journey.phase8_final.store import legacy_load

DRY_WORK = "outputs/diagnostics/phase8_final_dryrun"


def stream_continuation(points=((LOGIT_NORMAL, "A2", 3.0, 4.0), (MIXTURE, "A1_PRIME", 6.0, 2.0))):
    out = {}
    for arm, fam, a, b in points:
        m = legacy_load(s1p.Job("map", arm, fam, a, b))
        rows, failed = J.model_rows(
            s1p.spec_for(fam, a, b), s1p.het_from_payload(m["het"]), range(abm_config.R_EMULATOR)
        )
        summ = J.summarize(rows)
        same = all(
            summ["primary_mean"][j] == m["primary_mean"][j]
            and summ["primary_sd"][j] == m["primary_sd"][j]
            for j in abm_config.PRIMARY_IDS
        )
        out[f"{arm}:{fam}:{a:g},{b:g}"] = {"bit_identical": bool(same), "failed": failed}
    out["pass"] = all(v["bit_identical"] for k, v in out.items() if k != "pass")
    return out


def pooled_extension(n: int = 300):
    with mock.patch.object(J, "N", n):
        spec = s1p.spec_for("A2", 2.0, 5.0)
        het = s1p.het_from_payload(legacy_load(s1p.Job("map", LOGIT_NORMAL, "A2", 2.0, 5.0))["het"])
        a, _ = J.model_rows(spec, het, range(0, 6))
        b, _ = J.model_rows(spec, het, range(6, 20))
        full, _ = J.model_rows(spec, het, range(0, 20))
    sa = J.summarize(a)
    mean, sd, k = pooled_stats(len(a), A.vec(sa["primary_mean"]), A.vec(sa["primary_sd"]), b)
    sf = J.summarize(full)
    ok = (
        k == len(full)
        and np.allclose(mean, A.vec(sf["primary_mean"]), rtol=0, atol=1e-12)
        and np.allclose(sd, A.vec(sf["primary_sd"]), rtol=0, atol=1e-12)
        and np.array_equal(np.array(a + b), np.array(full))
    )
    return {
        "rows_concatenate_exactly": bool(np.array_equal(np.array(a + b), np.array(full))),
        "max_abs_mean_diff": float(np.max(np.abs(mean - A.vec(sf["primary_mean"])))),
        "max_abs_sd_diff": float(np.max(np.abs(sd - A.vec(sf["primary_sd"])))),
        "pass": bool(ok),
    }


def dry_run(keep: bool = False, fresh: bool = True) -> dict:
    """Toy-size plumbing run of every stage. ENGINEERING ONLY; never evidence."""
    from complete_journey.phase8_final import pipeline as P
    from complete_journey.phase8_final import snapshot as SN

    root = PROJECT_ROOT / DRY_WORK
    if fresh:
        shutil.rmtree(root, ignore_errors=True)
    synthetic = TargetVector(
        "synthetic",
        {"P1": 0.11, "P2": 0.07, "P3": 0.11, "P4": 0.02, "P5": 0.42},
        {"P1": 0.0045, "P2": 0.0034, "P3": 0.0036, "P4": 0.0015, "P5": 0.0045},
    )

    def fake_targets(archive=None):
        R.unlock(archive)
        return synthetic

    def fake_windows(n, archive=None):
        R.unlock(archive)
        rng = np.random.default_rng(5)
        f = rng.integers(1, 60, n)
        win = np.column_stack([f, np.minimum(f + rng.integers(20, 60, n), 102)]).tolist()
        return {"windows": win, "sha256": "synthetic", "pairs": n}

    coords = {LOGIT_NORMAL: [(0.5, 0.0)], MIXTURE: [(0.0, 0.5)]}
    patches = [
        mock.patch.object(S, "FINAL_RUN_AUTHORIZED", True),
        mock.patch.object(S, "WORK", DRY_WORK),
        mock.patch.object(S, "ARCHIVE", f"{DRY_WORK}/archive"),
        mock.patch.object(S, "R_COARSE", 52),
        mock.patch.object(S, "R_REFINE", 3),
        mock.patch.object(S, "R_THETA_TRUE", 3),
        mock.patch.object(S, "R_AUDIT", 5),
        mock.patch.object(S, "Q", 4),
        mock.patch.object(S, "Q_POP", 4),
        mock.patch.object(S, "PANEL_CHUNK", 2),
        mock.patch.object(
            S,
            "CALNOISE_PLAN",
            ((LOGIT_NORMAL, "A2", ("HMM q=0.05",), 1), (MIXTURE, "A1_PRIME", ("HMM q=0.05",), 1)),
        ),
        mock.patch.object(S, "CALNOISE_THETA", ((0.0, 0.0),)),
        mock.patch.object(J, "N", 300),
        mock.patch.object(abm_config, "N_HOUSEHOLDS", 300),
        mock.patch.object(abm_config, "R_CAL", 2),
        mock.patch.object(abm_config, "Q_GATE", 4),
        mock.patch.object(abm_config, "REFINEMENT_MAX_POINTS", 0),
        mock.patch.object(A, "LATTICE", (0.0, 11.0)),
        mock.patch.object(A, "pruned_refinement_points", lambda arm, f: coords[arm]),
        mock.patch.object(SN, "verify_snapshot", lambda root=None: "dry-run"),
        mock.patch.object(SN, "checkpoint_config_hash", lambda root=None: "dry-run"),
        mock.patch.object(R, "real_targets", fake_targets),
        mock.patch.object(R, "real_windows", fake_windows),
    ]
    out = {}
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        pipe = P.Pipeline(workers=2, deadline=False, probe=False)
        r = pipe.runner
        r.inprocess = True  # spawned workers would not see the toy-size patches
        r.acquire_lock()
        try:
            pipe.setup()
            try:
                pipe.stage1()
                out["stage1"] = "pass"
            except Exception as exc:  # toy sizes fail gates; plumbing continues
                out["stage1"] = f"stopped (expected at toy size): {exc}"
                rec = R.archive_path(R.STAGE1_RECORD)  # throwaway dry-run archive only
                data = json.loads(rec.read_text()) if rec.exists() else {}
                data["pass"] = True
                rec.write_text(json.dumps(data))
                stack.enter_context(
                    mock.patch.object(  # recompute returns the forced record
                        A, "stage1_report", lambda store, tol: json.loads(rec.read_text())
                    )
                )
                rec.write_bytes(R._dump(json.loads(rec.read_text())))
            pipe.rq2_pop()
            pipe.rq2()
            pipe.rq2_archive()
            try:
                R.write_once("rq2_results.json", {"tamper": 1})
                out["rq2_rewrite_refused"] = False
            except Exception:
                out["rq2_rewrite_refused"] = True
            pipe.real()
            pipe.s_a()
            out["artifacts"] = sorted(p.name for p in R.archive_path("").glob("*.json"))
            out["checkpoints"] = sum(1 for _ in r.store.files())
            out["status_stage"] = r.status.get("stage")
        finally:
            r.release_lock()
    out["pass"] = (
        bool(out.get("rq2_rewrite_refused"))
        and "real_rq1.json" in out.get("artifacts", [])
        and "s_a_bracket.json" in out.get("artifacts", [])
    )
    if not keep:
        shutil.rmtree(root, ignore_errors=True)
    return out


def run_all() -> dict:
    out = {"stream_continuation": stream_continuation(), "pooled_extension": pooled_extension()}
    out["pass"] = all(v["pass"] for v in out.values())
    return out
