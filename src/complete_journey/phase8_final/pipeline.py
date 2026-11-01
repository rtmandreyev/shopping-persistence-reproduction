"""Stage machine of the Phase 8 final run (P8-006): one resumable ``execute``.

SETUP → STAGE1 → RQ2_POP → RQ2 → RQ2_ARCHIVE → REAL_RQ1 → S_A, then the run process
exits with status RUN_COMPLETE; SYNTHESIS and VALIDATION follow outside the heavy run.
Stops: a failed G1–G5 gate (STOPPED_GATE), the disk guard, a deadline projection beyond
72 h (PAUSED_DEADLINE), an import-integrity failure or a job error (FAILED).
"""

from __future__ import annotations

import hashlib
import json
import subprocess

from complete_journey.abm import config as abm_config
from complete_journey.abm import rq2
from complete_journey.abm import stage1_p8005 as s1p
from complete_journey.abm.config import require
from complete_journey.abm.emulator import LOGIT_NORMAL, MIXTURE
from complete_journey.abm.representability import INFEASIBLE_TRUTH
from complete_journey.paths import PROJECT_ROOT
from complete_journey.phase8_final import analysis as A
from complete_journey.phase8_final import reveal as R
from complete_journey.phase8_final import settings as S
from complete_journey.phase8_final.evaluate import Entry, pooled_stats, requested_refinement
from complete_journey.phase8_final.jobs import ARMS, FAMILIES, FJob, truth_payload
from complete_journey.phase8_final.runner import Runner, RunStop, atomic_json, now_utc
from complete_journey.phase8_final.store import import_verification

PRIMARY = abm_config.PRIMARY_IDS


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _sha_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimated_jobs() -> dict[str, list[FJob]]:
    """Plan estimates for stages whose exact job lists depend on earlier stages."""
    est_truths = [i for i in A.rq2_indices()]
    rq2_panels = [
        FJob("panels", arm, index=i, sub=c, block="rq2")
        for arm in ARMS
        for i in est_truths
        for c in range(S.Q // S.PANEL_CHUNK)
    ]
    pop = [
        FJob("panels", arm, index=i, sub=c, block="pop")
        for arm in ARMS
        for i in est_truths
        for c in range(S.Q_POP // S.PANEL_CHUNK)
    ]
    cal = [
        FJob("calnoise", arm, fam, 0, 0, index=k, sub=q)
        for arm, fam, tr, n in S.CALNOISE_PLAN
        for k in range(len(tr))
        for q in range(n * len(S.CALNOISE_THETA))
    ]
    audits = [FJob("audit", arm, f, index=k) for arm in ARMS for f in FAMILIES for k in range(6)]
    ref2 = [FJob("ref2", arm, f, index=k) for arm in ARMS for f in FAMILIES for k in range(40)]
    sa = [FJob("sa", LOGIT_NORMAL, f, index=k) for f in FAMILIES for k in range(144)] + [
        FJob("saref", LOGIT_NORMAL, f, index=k) for f in FAMILIES for k in range(40)
    ]
    return {"RQ2_POP": pop, "RQ2": rq2_panels + cal, "REAL_RQ1": audits + ref2, "S_A": sa}


class Pipeline:
    def __init__(self, workers: int | None = None, deadline: bool = True, probe: bool = True):
        from complete_journey.phase8_final.snapshot import checkpoint_config_hash, verify_snapshot

        self.digest = verify_snapshot()
        self.runner = Runner(
            checkpoint_config_hash(), git_head(), workers=workers, deadline=deadline, probe=probe
        )
        self.runner.status["snapshot_sha256"] = self.digest
        self.tol = s1p.tolerances()

    def set_stage(self, stage: str) -> None:
        self.runner.stage = stage
        self.runner.write_status(force=True, health="RUNNING")

    def execute(self) -> dict:
        r = self.runner
        r.acquire_lock()
        try:
            r.status.setdefault("launched_utc", now_utc())
            self.setup()
            self.stage1()
            self.rq2_pop()
            self.rq2()
            self.rq2_archive()
            self.real()
            self.s_a()
            r.stage = "SYNTHESIS"
            r.write_status(force=True, health="RUN_COMPLETE", run_complete_utc=now_utc())
            return {"status": "RUN_COMPLETE"}
        except RunStop as stop:
            r.record_error(str(stop))
            r.write_status(force=True, health=stop.status, stop_message=str(stop))
            return {"status": stop.status, "message": str(stop)}
        except Exception as exc:  # unexpected (mechanical) error: record it, keep checkpoints
            r.record_error(f"{type(exc).__name__}: {exc}")
            r.write_status(force=True, health="FAILED", stop_message=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            r.release_lock()

    # ---------------------------------------------------------------------------
    def setup(self) -> None:
        self.set_stage("SETUP")
        try:
            ver = import_verification()
        except Exception as exc:
            raise RunStop("FAILED", f"import integrity failed: {exc}") from exc
        r = self.runner
        jobs1 = A.ext_jobs() + A.theta_jobs() + A.gpanel_jobs() + A.ref_jobs()
        r.update_plan("STAGE1", jobs1)
        for st, js in estimated_jobs().items():
            r.update_plan(st, js, estimated=True)
        legacy_counts = {
            "S1A coarse images (calibration + R=50)": 4 * 144 + 2,
            "A1 resolution (R=200, tolerances)": 1,
            "S1-P8005 gate truth configurations": 30,
            "S1-P8005 logit-normal gate panels (200 per truth)": len(A.gate_feasible(LOGIT_NORMAL)),
            "S1-P8005 A1' logit-normal H2 pairs (calibration-noise evidence)": 349,
        }
        manifest = {
            "verification": ver,
            "REUSED": legacy_counts,
            "EXTENDED": {"coarse images R=50 -> 200 (MODEL replicates 50..199)": len(A.ext_jobs())},
            "NEW": {
                "first-level refinement images": len(A.ref_jobs()),
                "theta_true images (R=200)": len(A.theta_jobs()),
                "mixture gate panels": len(A.gpanel_jobs()),
                "RQ2 truth configurations, panels, population block, calibration-noise "
                "checks, real audits, level-2 secondary, S-a": "see plan.json",
            },
        }
        atomic_json(r.work / "import_manifest.json", manifest)
        R.write_once("import_manifest.json", manifest)

    def stage1(self) -> None:
        self.set_stage("STAGE1")
        r = self.runner
        r.run_jobs(A.ext_jobs() + A.theta_jobs() + A.gpanel_jobs() + A.ref_jobs())
        report = A.stage1_report(r.store, self.tol)
        R.write_once(R.STAGE1_RECORD, report)
        if not report["pass"]:
            raise RunStop(
                "STOPPED_GATE",
                "a P8-003 G1-G5 hard gate failed: STOP (no RQ2 archive reveal, no real RQ1)",
            )

    def rq2_pop(self) -> None:
        self.set_stage("RQ2_POP")
        r = self.runner
        r.run_jobs(A.truth_jobs(LOGIT_NORMAL))
        r.run_jobs(A.truth_jobs(MIXTURE))
        pop = A.panel_jobs(LOGIT_NORMAL, r.store, "pop") + A.panel_jobs(MIXTURE, r.store, "pop")
        r.update_plan("RQ2_POP", A.truth_jobs(LOGIT_NORMAL) + A.truth_jobs(MIXTURE) + pop)
        r.run_jobs(pop)

    def rq2(self) -> None:
        self.set_stage("RQ2")
        r = self.runner
        panels = A.panel_jobs(LOGIT_NORMAL, r.store, "rq2") + A.panel_jobs(MIXTURE, r.store, "rq2")
        est = [j for j in estimated_jobs()["RQ2"] if j.kind == "calnoise"]
        r.update_plan("RQ2", panels + est, estimated=True)
        r.run_jobs(panels)
        checks = A.calnoise_jobs(r.store)  # needs the panels: first n successful per truth
        r.update_plan("RQ2", panels + checks)
        r.run_jobs(checks)

    def rq2_archive(self) -> None:
        self.set_stage("RQ2_ARCHIVE")
        r = self.runner
        truths, per_truth, se = {}, {}, {}
        for arm in ARMS:
            tables = A.build_tables(r.store, arm)
            truths[arm], per_truth[arm] = {}, {}
            for i in A.rq2_indices():
                t = truth_payload(arm, i, r.store)
                lab = t["label"]
                truths[arm][lab] = {
                    k: t.get(k)
                    for k in (
                        "status",
                        "kind",
                        "truth_class",
                        "delta",
                        "realized_pp",
                        "coord",
                        "inside_lattice",
                        "lattice_note",
                        "reason",
                        "het",
                        "periodic",
                    )
                }
                if t["status"] != "FEASIBLE":
                    per_truth[arm][lab] = {
                        "status": INFEASIBLE_TRUTH,
                        "truth_class": t["truth_class"],
                        "lam": dict(rq2.production_truth_points()[i].params).get("lam"),
                        "delta1_pp": dict(rq2.production_truth_points()[i].params).get("delta1_pp"),
                    }
                    continue
                ev = A.rq2_truth_eval(r.store, arm, i, tables, self.tol)
                per_truth[arm][lab] = ev
                rows, sd = A.panel_arrays(A.block_panels(r.store, arm, i, "rq2"))
                se[(arm, i)] = sd
                r.write_status(force=True, rq2_eval=f"{arm}:{lab}")
        results = A.rq2_report(r.store, self.tol, per_truth)
        noise = {
            "checks": A.calnoise_report(r.store, self.tol, se),
            "a1prime_logit_h2_evidence": A.h2_evidence(self.tol),
            "plan": [list(map(str, p)) for p in S.CALNOISE_PLAN],
            "theta": S.CALNOISE_THETA,
            "rule": f"{S.CALNOISE_LABEL} if p90 |Δμ|/τ(0.25) > {S.CALNOISE_SHIFT_P90_MAX} "
            f"for any moment or status disagreement > {S.CALNOISE_DISAGREE_MAX}",
        }
        R.write_once("rq2_truths.json", truths)
        R.write_once("rq2_per_truth.json", A.strip(per_truth))
        R.write_once("rq2_results.json", A.strip(results))
        R.write_once("calibration_noise.json", A.strip(noise))
        rec = R.freeze_rq2()
        r.write_status(
            force=True, rq2_freeze=rec["freeze_sha256"], rq2_frozen_utc=rec["frozen_utc"]
        )

    # ---------------------------------------------------------------------------
    def real(self) -> None:
        self.set_stage("REAL_RQ1")
        r = self.runner
        recomputed = R._dump(A.stage1_report(r.store, self.tol))
        require(
            recomputed == R.archive_path(R.STAGE1_RECORD).read_bytes(),
            "archived Stage 1 gate record differs from the checkpoint recomputation",
        )
        ledger = R.unlock()
        r.write_status(force=True, reveal="UNLOCKED", reveal_utc=ledger["unlocked_utc"])
        target = R.real_targets()
        data = A.vec(target.values)
        se = A.vec(target.se)
        out, audits_all, ref2_all, overrides = {}, [], [], {}
        tables = {arm: A.build_tables(r.store, arm) for arm in ARMS}
        prelim = {}
        for arm in ARMS:
            prelim[arm] = {f: A.real_family(f, tables[arm], data, se, self.tol) for f in FAMILIES}
            for f in FAMILIES:
                for c in A.audit_points(prelim[arm][f], str(abm_config.KAPPA_REFERENCE)):
                    if c in tables[arm][f]["refined"]:
                        audits_all.append(FJob("audit", arm, f, c[0], c[1], block="ref"))
        r.update_plan("REAL_RQ1", audits_all)
        r.run_jobs(audits_all)
        for arm in ARMS:
            fams = {}
            for f in FAMILIES:
                override = {}
                for job in [j for j in audits_all if j.arm == arm and j.family == f]:
                    base = r.store.load(FJob("ref", arm, f, job.a, job.b).key())["payload"]
                    ext = r.store.load(job.key())["payload"]
                    mean, sd, n = pooled_stats(
                        base["replicates_ok"],
                        A.vec(base["primary_mean"]),
                        A.vec(base["primary_sd"]),
                        ext["rows"],
                    )
                    override[(job.a, job.b)] = Entry(mean, sd, n, bool(base["ci1_feasible"]))
                overrides[(arm, f)] = override
                fams[f] = A.real_family(f, tables[arm], data, se, self.tol, override)
                fams[f]["audited_points"] = sorted(override)
                if S.REAL_SECONDARY_LEVEL2:
                    lvl1, lvl2 = requested_refinement(
                        f,
                        tables[arm][f]["coarse"],
                        data,
                        se,
                        self.tol,
                        {**tables[arm][f]["refined"], **override},
                    )
                    ref2_all += [FJob("ref2", arm, f, a, b) for a, b in lvl2]
            a1_out = A.outcome((0.0, 0.0), tables[arm]["A1"], data, se, self.tol)[0]
            out[arm] = {
                "families": fams,
                "comparative": A.comparative_block(fams, self.tol),
                "A1": {
                    "feasible": a1_out.feasible,
                    "status": a1_out.status,
                    "kappa_star": a1_out.kappa_star,
                },
            }
        ref = str(abm_config.KAPPA_REFERENCE)
        verdicts = {arm: out[arm]["comparative"]["verdict"][ref] for arm in ARMS}
        out["heterogeneity_label"] = (
            "ROBUST TO FROZEN HETEROGENEITY BRACKET"
            if len(set(verdicts.values())) == 1
            else "HETEROGENEITY-DEPENDENT"
        )
        r.run_jobs(ref2_all)
        secondary = {}
        for arm in ARMS:
            secondary[arm] = {}
            for f in FAMILIES:
                lvl2 = {
                    (j.a, j.b): A.ref_entry(r.store.load(j.key())["payload"])
                    for j in ref2_all
                    if j.arm == arm and j.family == f
                }
                refined = {**tables[arm][f]["refined"], **overrides[(arm, f)], **lvl2}
                t2 = {**tables[arm], f: {"coarse": tables[arm][f]["coarse"], "refined": refined}}
                secondary[arm][f] = A.real_family(f, t2, data, se, self.tol, levels=2)
        out["secondary_level2"] = {
            arm: {f: A.strip(v)["kappas"] for f, v in d.items()} for arm, d in secondary.items()
        }
        out["ledger"] = ledger
        R.write_once("real_rq1.json", A.strip(out))
        self._real_cache = (data, se, out)

    def s_a(self) -> None:
        self.set_stage("S_A")
        r = self.runner
        win_path = r.work / "sa_windows.json"
        if not win_path.exists():
            atomic_json(win_path, R.real_windows(abm_config.N_HOUSEHOLDS))
        target = R.real_targets()
        data, se = A.vec(target.values), A.vec(target.se)
        coarse_jobs = [FJob("sa", LOGIT_NORMAL, "A1")] + [
            FJob("sa", LOGIT_NORMAL, f, ip, ia)
            for f in FAMILIES
            for ip in A.LATTICE
            for ia in A.LATTICE
        ]
        r.update_plan("S_A", coarse_jobs)
        r.run_jobs(coarse_jobs)

        def entry(job):
            p = r.store.load(job.key())["payload"]
            return A.ref_entry(p)

        tables = {"A1": entry(coarse_jobs[0])}
        ref_jobs = []
        for f in FAMILIES:
            coarse = {
                (ip, ia): entry(FJob("sa", LOGIT_NORMAL, f, ip, ia))
                for ip in A.LATTICE
                for ia in A.LATTICE
            }
            lvl1, _ = requested_refinement(f, coarse, data, se, self.tol)
            ref_jobs += [FJob("saref", LOGIT_NORMAL, f, a, b) for a, b in lvl1]
            tables[f] = {"coarse": coarse, "refined": {}}
        r.run_jobs(ref_jobs)
        for f in FAMILIES:
            tables[f]["refined"] = {(j.a, j.b): entry(j) for j in ref_jobs if j.family == f}
        fams = {f: A.real_family(f, tables, data, se, self.tol) for f in FAMILIES}
        comp = A.comparative_block(fams, self.tol)
        real = json.loads(R.archive_path("real_rq1.json").read_text())
        ref = str(abm_config.KAPPA_REFERENCE)
        primary = real[LOGIT_NORMAL]["comparative"]["verdict"][ref]
        label = (
            "OPERATOR_ROBUST"
            if comp["verdict"][ref] == primary
            else "OPERATOR-DEPENDENT / UNRESOLVED"
        )
        R.write_once(
            "s_a_bracket.json",
            A.strip(
                {
                    "families": fams,
                    "comparative": comp,
                    "primary_comparative": primary,
                    "operator_label": label,
                    "windows_sha256": json.loads(win_path.read_text())["sha256"],
                }
            ),
        )
