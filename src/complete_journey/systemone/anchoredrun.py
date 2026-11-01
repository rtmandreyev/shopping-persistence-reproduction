"""P9-007 checkpointed, network-free runner for the anchored out-of-family extension.

The infrastructure is the same as ``systemone.localrun``: frozen-archive verification,
the snapshot and refusal on change, the frozen Phase 8 ``CheckpointStore`` (atomic,
hashed, duplicate-refusing), deterministic resume, the run lock, ``status.json`` /
``plan.json`` / ``progress.jsonl`` for the read-only monitor, sockets blocked and no
credentials in every process.

Jobs:
1. ``calibration`` (policy, truth): the logit-normal arm, then the mixture arm from the
   primary solution (frozen Phase 8 methods; CI-1 only).
2. ``panels`` (policy, truth, arm, block, replicate chunk): anchored panels plus the
   frozen P1-P5 / CI-1 moments and the degeneracy and Delta_1 diagnostics. Block 0 is
   the R recovery panels; block 1 is the Q_pop population panels.
3. ``recovery`` (policy, truth, arm): CI-1 admissibility, the DEGENERATE rule, Level 1,
   and, for admissible non-degenerate truths only, Level 2 (unchanged Phase 8 plug-in
   classification).
4. ``assemble`` (parent): ``results.json``.
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from complete_journey.systemone import anchored as A
from complete_journey.systemone import localrun as LR
from complete_journey.systemone import settings as S
from complete_journey.systemone.settings import Phase9ContractError, require

WORK_PRODUCTION = S.PROJECT_ROOT / "outputs" / "diagnostics" / "phase9_p9007"
WORK_ENGINEERING = S.PROJECT_ROOT / "outputs" / "diagnostics" / "phase9_p9007_eng"
STAGES = ("calibration", "panels", "recovery", "assemble")
SEED_TREE = {
    "root": S.SEED_ROOT,
    "purposes": {k: S.SEED_PURPOSE[k] for k in ("p9007_calibration", "p9007_panel")},
    "calibration": "p9007_calibration (replicate r < R_cal); common to every truth and arm",
    "panels": "p9007_panel (policy_index, truth_index, arm_index, block, replicate)",
}


@dataclass(frozen=True)
class AnchoredConfig:
    label: str = "production"
    k: int = 10
    n: int = S.N_HOUSEHOLDS
    t_weeks: int = S.T_WEEKS
    burn_in: int = S.BURN_IN
    panels: int = S.R_PANELS
    q_pop: int = S.Q_POP
    chunk: int = 10
    r_cal: int = 20
    lam: float = A.LAM
    degenerate_share: float = A.DEGENERATE_SHARE
    clip: tuple = A.CLIP
    policies: tuple = S.POLICIES
    truths: tuple = tuple(t.id for t in A.TRUTHS)

    def work(self) -> Path:
        return WORK_PRODUCTION if self.label == "production" else WORK_ENGINEERING


PRODUCTION = AnchoredConfig()
ENGINEERING = AnchoredConfig(label="engineering", n=300, panels=4, q_pop=4, chunk=2, r_cal=2)


@dataclass(frozen=True)
class Job:
    stage: str
    policy: str
    truth: str
    arm: str = ""
    block: int = -1
    rep_start: int = 0
    rep_end: int = 0

    def key(self):
        from complete_journey.abm.checkpoint import JobKey

        ti = [t.id for t in A.TRUTHS].index(self.truth)
        ai = A.ARMS.index(self.arm) if self.arm else -1
        pi = S.POLICIES.index(self.policy)
        seeds = tuple(
            (pi, ti, max(ai, 0), max(self.block, 0), r) for r in range(self.rep_start, self.rep_end)
        )
        return JobKey(
            STAGES.index(self.stage),
            self.policy,
            f"p9007|{self.stage}|{self.truth}|{self.arm}|{self.block}",
            self.rep_start,
            self.rep_end,
            seeds,
        )

    def label(self) -> dict:
        tr = A.TRUTH_BY_ID[self.truth]
        desc = {
            "none": "no history",
            "own": "own history",
            "serial": "serial donor history",
            "scrambled": "time-scrambled donor history",
        }[tr.history]
        return {
            "stage": self.stage,
            "policy": self.policy,
            "condition": f"{tr.id} ({desc}{' + latent need' if tr.need else ''}; θ={tr.theta:g})",
            "lam": A.LAM if tr.need else None,
            "variant": self.arm or "both arms",
            "block": {-1: "-", 0: "R panels", 1: "Q_pop panels"}[self.block],
            "reps": f"{self.rep_start}-{self.rep_end}",
        }


def plan(cfg: AnchoredConfig) -> list[Job]:
    jobs = [Job("calibration", p, t) for p in cfg.policies for t in cfg.truths]
    for p in cfg.policies:
        for t in cfg.truths:
            for arm in A.ARMS:
                for block, total in ((0, cfg.panels), (1, cfg.q_pop)):
                    for r0 in range(0, total, cfg.chunk):
                        jobs.append(Job("panels", p, t, arm, block, r0, min(total, r0 + cfg.chunk)))
    jobs += [Job("recovery", p, t, arm) for p in cfg.policies for t in cfg.truths for arm in A.ARMS]
    return jobs


def truth_components(cfg: AnchoredConfig, pds: dict) -> dict:
    """{(policy, cell): components} for every cell any truth uses."""
    cells = {A.TRUTH_BY_ID[t].cell for t in cfg.truths} - {None}
    return {
        (p, c): A.components(pds[p].table(c, "primary"), cfg.k) for p in cfg.policies for c in cells
    }


def components_hash(comps: dict) -> str:
    import hashlib

    h = hashlib.sha256()
    for key in sorted(comps):
        for name in ("gH", "gN"):
            h.update(f"{key}:{name}".encode())
            h.update(np.ascontiguousarray(comps[key][name], dtype=np.float64).tobytes())
    return h.hexdigest()


# --- job execution (worker processes) --------------------------------------------------


def execute(job: Job, cfg: AnchoredConfig, args: dict) -> dict:
    from complete_journey.systemone import recovery as R

    tr = A.TRUTH_BY_ID[job.truth]
    comp = args.get("comp")
    if job.stage == "calibration":
        crn = A.calibration_draws(cfg.n, cfg.r_cal)
        ln = A.calibrate_logit_normal(comp, tr, crn)
        mx = A.calibrate_mixture(comp, tr, crn, ln)
        return {"logit_normal": ln, "mixture": mx}
    if job.stage == "panels":
        from complete_journey.systemone.seeds import rng

        het = A.het_from(args["calibration"])
        pi, ti, ai = (
            S.POLICIES.index(job.policy),
            [t.id for t in A.TRUTHS].index(job.truth),
            A.ARMS.index(job.arm),
        )
        out, fails = [], 0
        for r in range(job.rep_start, job.rep_end):
            dr = A.draws(
                rng("p9007_panel", pi, ti, ai, job.block, r), cfg.n, cfg.burn_in + cfg.t_weeks
            )
            sim = A.simulate(comp, tr, het, dr, cfg.t_weeks, cfg.burn_in, cfg.k)
            mom = R.panel_moments(sim["behavior"])
            if mom is None:
                fails += 1
                continue
            mom.update(
                {
                    k: sim[k]
                    for k in (
                        "share_outside_clip",
                        "mean_rate",
                        "share_households_rate_gt_095",
                        "delta1",
                    )
                }
            )
            mom["replicate"] = r
            out.append(mom)
        return {"panels": out, "failures": fails}
    if job.stage == "recovery":
        rows, pop = args["rows"], args["pop"]
        adm = R.ci1_admissible(pop)
        diag = {
            k: float(np.mean([p[k] for p in pop["panels"]]))
            for k in ("share_outside_clip", "mean_rate", "share_households_rate_gt_095", "delta1")
        }
        degenerate = diag["share_outside_clip"] > cfg.degenerate_share
        truth_class = R.HISTORY_TRUTH if tr.label == A.HISTORY_INSTALLED else R.NO_HISTORY_TRUTH
        res = {
            "truth": tr.label,
            "calibration_feasible": bool(args["calibration"]["feasible"]),
            "admissibility": adm,
            "degenerate": degenerate,
            "diagnostics_pop": diag,
            "level1": R.level1(rows),
        }
        if adm["label"] == R.FEASIBLE_ARTIFICIAL_TRUTH and not degenerate:
            res["level2"] = R.level2(rows, pop, truth_class)
            res["level2_status"] = "EVALUATED"
        else:
            res["level2_status"] = (
                "INFEASIBLE ANCHORED TRUTH"
                if adm["label"] != R.FEASIBLE_ARTIFICIAL_TRUTH
                else "DEGENERATE"
            )
            if degenerate and adm["label"] != R.FEASIBLE_ARTIFICIAL_TRUTH:
                res["level2_status"] = "INFEASIBLE ANCHORED TRUTH; DEGENERATE"
        return res
    raise Phase9ContractError(f"unknown stage {job.stage}")


def _run_job(payload):
    job, cfg, args = payload
    t0 = time.time()
    return job, execute(job, cfg, args), time.time() - t0


def _worker_init():
    LR.isolate()


# --- runner ----------------------------------------------------------------------------


def run(cfg: AnchoredConfig, workers: int, max_job_retries: int = 2) -> dict:
    import signal

    signal.signal(signal.SIGTERM, LR._sigterm)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    LR.isolate()
    work = cfg.work()
    work.mkdir(parents=True, exist_ok=True)
    lock = LR.RunLock(work)
    lock.acquire()
    try:
        return _run(cfg, workers, max_job_retries, work)
    finally:
        lock.release()


def _run(cfg: AnchoredConfig, workers: int, max_job_retries: int, work: Path) -> dict:
    import multiprocessing as mp

    from complete_journey.phase8_final.store import ShardedStore

    archives = LR.verify_freeze(cfg.policies)
    pds = LR.policy_data(LR.PRODUCTION)
    comps = truth_components(cfg, pds)
    jobs = plan(cfg)
    snap = {
        "schema": "phase9-p9007-snapshot-v1",
        "decision": "P9-007",
        "config": asdict(cfg),
        "archive_digests": archives,
        "components_sha256": components_hash(comps),
        "seed_tree": SEED_TREE,
        "plan_sha256": LR.sha([j.key().job_id for j in jobs]),
        "jobs": len(jobs),
        "source_sha256": LR.source_hashes(),
    }
    snap = json.loads(json.dumps(snap, default=str))
    snap_path = work / "snapshot.json"
    if snap_path.exists():
        prev = json.loads(snap_path.read_text())
        diff = sorted(k for k in snap if prev.get(k) != snap[k])
        require(not diff, f"resume refused: the snapshot changed ({diff})")
    else:
        LR._atomic_json(snap_path, snap)
    snap_sha = LR.sha(snap)
    commit = LR.git_commit()
    store = ShardedStore(work / "checkpoints", snap_sha, commit)
    ids = {j: j.key().job_id for j in jobs}
    keys = {j: j.key() for j in jobs}
    LR._atomic_json(
        work / "plan.json",
        {"snapshot_sha256": snap_sha, "jobs": [{"id": ids[j], **j.label()} for j in jobs]},
    )

    class _Cfg:  # Progress expects .label
        label = cfg.label

    prog = LR.Progress(work, _Cfg, jobs, snap_sha, commit, workers, ids)
    for j in jobs:
        if store.load(keys[j]) is not None:
            prog.done.add(ids[j])
    prog.skipped = len(prog.done)
    cal_cache: dict = {}

    def calibration(policy, truth):
        if (policy, truth) not in cal_cache:
            rec = store.load(keys[Job("calibration", policy, truth)])
            require(rec is not None, "calibration incomplete before panels")
            cal_cache[(policy, truth)] = rec["payload"]
        return cal_cache[(policy, truth)]

    def comp_for(job):
        cell = A.TRUTH_BY_ID[job.truth].cell
        return comps[(job.policy, cell)] if cell else None

    def panels_of(job, block):
        ok, fails = [], 0
        for j in jobs:
            if j.stage == "panels" and (j.policy, j.truth, j.arm, j.block) == (
                job.policy,
                job.truth,
                job.arm,
                block,
            ):
                rec = store.load(keys[j])
                require(rec is not None, "a panel chunk is missing before recovery")
                ok += rec["payload"]["panels"]
                fails += rec["payload"]["failures"]
        return {"panels": ok, "failures": fails}

    def args_for(job):
        if job.stage == "calibration":
            return {"comp": comp_for(job)}
        if job.stage == "panels":
            return {
                "comp": comp_for(job),
                "calibration": calibration(job.policy, job.truth)[job.arm],
            }
        return {
            "calibration": calibration(job.policy, job.truth)[job.arm],
            "rows": panels_of(job, 0),
            "pop": panels_of(job, 1),
        }

    try:
        for stage in STAGES[:-1]:
            prog.stage = stage
            pending = [j for j in jobs if j.stage == stage and ids[j] not in prog.done]
            tries: dict = {}
            while pending:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=mp.get_context("spawn"),
                    initializer=_worker_init,
                ) as ex:
                    futs, queue, pending, broken = {}, list(pending), [], False
                    while queue or futs:
                        while queue and len(futs) < 2 * workers:
                            j = queue.pop(0)
                            futs[ex.submit(_run_job, (j, cfg, args_for(j)))] = j
                            prog.running[ids[j]] = j.label()
                        prog.write()
                        finished, _ = wait(futs, timeout=5, return_when=FIRST_COMPLETED)
                        for fu in finished:
                            j = futs.pop(fu)
                            prog.running.pop(ids[j], None)
                            try:
                                _, payload, secs = fu.result()
                            except BrokenProcessPool:
                                broken = True
                                pending.append(j)
                                continue
                            except Exception:  # noqa: BLE001 - mechanical job failure
                                tries[ids[j]] = tries.get(ids[j], 0) + 1
                                with open(work / "failures.log", "a") as fl:
                                    fl.write(
                                        f"{time.ctime()} {ids[j]} {j.label()}\n{traceback.format_exc()}\n"
                                    )
                                if tries[ids[j]] > max_job_retries:
                                    prog.failed.append({"id": ids[j], **j.label()})
                                else:
                                    queue.append(j)
                                continue
                            store.save(keys[j], payload)
                            prog.record(ids[j], secs)
                        if broken:
                            pending += queue + list(futs.values())
                            for j in list(futs.values()):
                                prog.running.pop(ids[j], None)
                            break
                prog.write(immediate=True)
                if prog.failed:
                    raise Phase9ContractError(
                        f"{len(prog.failed)} job(s) failed after {max_job_retries} retries"
                    )
        prog.stage = "assemble"
        prog.write(immediate=True)
        results = assemble(cfg, jobs, keys, store, snap, snap_sha)
        prog.health = "RUN_COMPLETE"
        prog.message = f"results sha256 {results['results_sha256']}"
        prog.write(immediate=True)
        return results
    except BaseException as exc:
        prog.health = "STOPPED"
        prog.message = f"{type(exc).__name__}: {exc}"
        prog.write(immediate=True)
        raise


def assemble(cfg, jobs, keys, store, snap, snap_sha) -> dict:
    out = {}
    for p in cfg.policies:
        out[p] = {}
        for t in cfg.truths:
            cal = store.load(keys[Job("calibration", p, t)])["payload"]
            out[p][t] = {
                "truth": A.TRUTH_BY_ID[t].label,
                "spec": asdict(A.TRUTH_BY_ID[t]),
                "calibration": cal,
                "arms": {
                    arm: store.load(keys[Job("recovery", p, t, arm)])["payload"] for arm in A.ARMS
                },
            }
    summary = {
        "schema": "phase9-p9007-results-v1",
        "status": "CANDIDATE RESULTS (researcher review required)",
        "decision": "P9-007",
        "label": cfg.label,
        "snapshot_sha256": snap_sha,
        "archive_digests": snap["archive_digests"],
        "components_sha256": snap["components_sha256"],
        "config": asdict(cfg),
        "policies": out,
    }
    summary["results_sha256"] = LR.sha(summary)
    LR._atomic_json(cfg.work() / "results.json", summary)
    return summary
