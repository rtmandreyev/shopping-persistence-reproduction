"""Phase 9 local production run: detached, checkpointed, resume-safe, network-free.

It executes the frozen Phase 9 closed-loop design (P9-001…P9-005) with the same
functions, seeds and coordinates as ``e2e.policy_pipeline``, split into independently
checkpointed atomic jobs:

1. ``calibration``: one global logit-intercept shift per (policy, DGP) (secondary);
2. ``panels``: closed-loop panels plus the frozen P1–P5 / CI-1 moments, in chunks of
   replicates, for every (policy, DGP, variant, block), where block 0 is the R recovery
   panels and block 1 the Q_pop population panels;
3. ``analytic``: the analytic-vs-Monte-Carlo check per (policy, non-shuffled DGP);
4. ``recovery``: Level 1 and Level 2 per (policy, DGP, variant), read from stage 2;
5. ``assemble`` (parent process): static estimands and validity, cross-policy labels,
   and the results file.

Inputs are the three frozen production archives (digests checked against
``docs/phases/PHASE_09/production_archive_freeze.json``) read through
``PolicyData.from_archive``. A state that is absent from an archive is a loud error
(``PolicyData._reps``). No job can create one, and nothing here imports the transport,
reads a credential or opens a socket: every process blocks sockets and drops credential
variables at start-up.

Every job is saved through the frozen Phase 8 ``CheckpointStore`` (atomic write,
record and output hashes, config-hash refusal, duplicate refusal). On resume, completed
jobs are skipped. The run refuses to resume if the snapshot (source hashes, config,
archive digests, policy-table hash, seed tree, job plan) changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from complete_journey.systemone import closedloop as C
from complete_journey.systemone import settings as S
from complete_journey.systemone.settings import Phase9ContractError, require

ROOT = S.PROJECT_ROOT
FREEZE_FILE = S.PREFLIGHT_DIR / "production_archive_freeze.json"
WORK_PRODUCTION = ROOT / "outputs" / "diagnostics" / "phase9_local"
WORK_ENGINEERING = ROOT / "outputs" / "diagnostics" / "phase9_local_eng"
STAGES = ("calibration", "panels", "analytic", "recovery", "assemble")


def is_credential_var(name: str) -> bool:
    """Every *_API_KEY variable and the external-run scope are removed from the local run."""
    return name.endswith("_API_KEY") or name == S.EXTERNAL_ENV


SOURCE_GLOBS = (
    "src/complete_journey/systemone/*.py",
    "src/complete_journey/abm/*.py",
    "src/complete_journey/phase8_final/*.py",
    "src/complete_journey/prediction/*.py",
)
#: Seed coordinates, identical to e2e.policy_pipeline (policy index = S.POLICIES order).
SEED_TREE = {
    "root": S.SEED_ROOT,
    "purposes": dict(S.SEED_PURPOSE),
    "panels": "(policy_index, dgp_index, variant_index, block, replicate); block 0 = R, 1 = Q_pop",
    "calibration": "(policy_index, dgp_index, 9, j), j < 2 common random numbers",
    "analytic": "(policy_index, dgp_index, 7, j), j < 4",
}


# --- isolation -----------------------------------------------------------------------


def _blocked(*_a, **_k):
    raise Phase9ContractError("network access is disabled in the Phase 9 local run")


def isolate() -> None:
    """Drop credential variables and disable sockets in this process."""
    for var in [k for k in os.environ if is_credential_var(k)]:
        os.environ.pop(var, None)
    socket.socket.connect = _blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = _blocked  # type: ignore[method-assign]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    require(
        "complete_journey.systemone.transport" not in sys.modules,
        "the transport module must not be loaded in the local run",
    )


# --- configuration -------------------------------------------------------------------


@dataclass(frozen=True)
class LocalConfig:
    label: str = "production"
    k: int = 10
    n: int = S.N_HOUSEHOLDS
    t_weeks: int = S.T_WEEKS
    burn_in: int = S.BURN_IN
    panels: int = S.R_PANELS
    q_pop: int = S.Q_POP
    lams: tuple = S.NEED_LAMBDAS
    chunk: int = 10
    calibrate: bool = True
    calibration_r: int = 2
    calibration_iters: int = 18
    analytic_sims: int = 4
    policies: tuple = S.POLICIES

    def work(self) -> Path:
        return WORK_PRODUCTION if self.label == "production" else WORK_ENGINEERING


PRODUCTION = LocalConfig()
ENGINEERING = LocalConfig(
    label="engineering", n=300, panels=4, q_pop=4, lams=(0.8,), chunk=2, analytic_sims=2
)


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha(obj) -> str:
    return hashlib.sha256(_canon(obj)).hexdigest()


# --- frozen inputs -------------------------------------------------------------------


def archive_root(policy: str) -> Path:
    return S.ARCHIVE_ROOT / "production" / policy


def archive_digest(policy: str) -> dict:
    """Digest over (request_hash, record_sha256) of every archived success, plus the
    file count; failed attempts are listed separately."""
    from complete_journey.systemone.archive import Archive

    recs = sorted(
        (r["request_hash"], r["record_sha256"]) for r in Archive(archive_root(policy)).successes()
    )
    return {"successes": len(recs), "sha256": sha(recs)}


def verify_freeze(policies) -> dict:
    require(FREEZE_FILE.exists(), f"no archive freeze record {FREEZE_FILE.name}")
    frozen = json.loads(FREEZE_FILE.read_text())["archives"]
    out = {}
    for p in policies:
        d = archive_digest(p)
        require(
            d["sha256"] == frozen[p]["success_digest_sha256"]
            and d["successes"] == frozen[p]["successes"],
            f"production archive {p} differs from its frozen digest",
        )
        out[p] = d["sha256"]
    return out


def dgp_list(cfg: LocalConfig) -> list[C.DGP]:
    from complete_journey.systemone.e2e import dgps

    return dgps(cfg.lams)


def variants_of(cfg: LocalConfig, dgp: C.DGP) -> list[str]:
    out = ["raw"]
    if cfg.calibrate:
        out.append("calibrated")
    if not dgp.shuffled and dgp.cell != S.NEITHER:
        out.append("surrogate")
    return out


def policy_data(cfg: LocalConfig) -> dict:
    from complete_journey.systemone.policy import PolicyData

    return {p: PolicyData.from_archive(archive_root(p), p, cfg.k) for p in cfg.policies}


def base_tables(cfg: LocalConfig, pds: dict) -> dict:
    """Raw primary tables and surrogate tables, keyed by (policy, dgp_index)."""
    out = {}
    for p in cfg.policies:
        for di, dgp in enumerate(dgp_list(cfg)):
            raw = pds[p].table(dgp.cell, "primary")
            entry = {"raw": raw}
            if not dgp.shuffled and dgp.cell != S.NEITHER:
                entry["surrogate"] = C.fit_surrogate(raw, dgp.cell, cfg.k)["table"]
            out[(p, di)] = entry
    return out


def tables_hash(tables: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(tables, key=str):
        for name in sorted(tables[key]):
            h.update(f"{key}:{name}".encode())
            h.update(np.ascontiguousarray(tables[key][name], dtype=np.float64).tobytes())
    return h.hexdigest()


def shifted_table(raw: np.ndarray, delta: float) -> np.ndarray:
    """Exactly ``PolicyData.table(cell, 'primary', shift=delta)`` on the raw table."""
    from complete_journey.systemone.policy import shifted

    if not delta:
        return raw
    out = raw.copy()
    m = np.isfinite(raw)
    out[m] = shifted(raw[m], delta)
    return out


# --- job plan ------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    stage: str
    policy: str
    dgp_index: int
    variant: str = ""
    block: int = -1
    rep_start: int = 0
    rep_end: int = 0

    def key(self, cfg: LocalConfig):
        from complete_journey.abm.checkpoint import JobKey

        pi = cfg.policies.index(self.policy)
        vi = self.variant_index(cfg)
        seeds = tuple(
            (S.POLICIES.index(self.policy), self.dgp_index, vi, self.block, r)
            for r in range(self.rep_start, self.rep_end)
        )
        return JobKey(
            STAGES.index(self.stage),
            self.policy,
            f"{self.stage}|{pi}|{self.dgp_index}|{self.variant}|{self.block}",
            self.rep_start,
            self.rep_end,
            seeds,
        )

    def variant_index(self, cfg: LocalConfig) -> int:
        if not self.variant:
            return -1
        return variants_of(cfg, dgp_list(cfg)[self.dgp_index]).index(self.variant)

    def label(self, cfg: LocalConfig) -> dict:
        dgp = dgp_list(cfg)[self.dgp_index]
        return {
            "stage": self.stage,
            "policy": self.policy,
            "condition": dgp.cell + (" (shuffled history)" if dgp.shuffled else ""),
            "lam": dgp.lam,
            "variant": self.variant or "-",
            "block": {-1: "-", 0: "R panels", 1: "Q_pop panels"}[self.block],
            "reps": f"{self.rep_start}-{self.rep_end}",
        }


def plan(cfg: LocalConfig) -> list[Job]:
    jobs: list[Job] = []
    ds = dgp_list(cfg)
    if cfg.calibrate:
        jobs += [Job("calibration", p, di) for p in cfg.policies for di in range(len(ds))]
    for p in cfg.policies:
        for di, dgp in enumerate(ds):
            for v in variants_of(cfg, dgp):
                for block, total in ((0, cfg.panels), (1, cfg.q_pop)):
                    for r0 in range(0, total, cfg.chunk):
                        jobs.append(Job("panels", p, di, v, block, r0, min(total, r0 + cfg.chunk)))
    jobs += [
        Job("analytic", p, di) for p in cfg.policies for di, d in enumerate(ds) if not d.shuffled
    ]
    jobs += [
        Job("recovery", p, di, v)
        for p in cfg.policies
        for di, d in enumerate(ds)
        for v in variants_of(cfg, d)
    ]
    return jobs


# --- job execution (worker processes) --------------------------------------------------


def _worker_init():
    isolate()


def execute(job: Job, cfg: LocalConfig, args: dict) -> dict:
    """Run one atomic job and return its payload. Pure function of (job, cfg, args)."""
    from complete_journey.systemone import recovery as R

    dgp = dgp_list(cfg)[job.dgp_index]
    pidx = S.POLICIES.index(job.policy)
    if job.stage == "calibration":
        raw = args["raw"]
        delta = C.calibrate_shift(
            lambda d: shifted_table(raw, d),
            dgp,
            cfg.k,
            args["real_mean"],
            (pidx, job.dgp_index, 9),
            n=cfg.n,
            r=cfg.calibration_r,
            iters=cfg.calibration_iters,
        )
        return {"delta": float(delta)}
    if job.stage == "panels":
        vi = job.variant_index(cfg)
        res = [
            R._panel_job(
                (args["table"], dgp, cfg.k, (pidx, job.dgp_index, vi, job.block, r), cfg.n)
            )
            for r in range(job.rep_start, job.rep_end)
        ]
        ok = [r for r in res if r is not None]
        return {"panels": ok, "failures": len(res) - len(ok)}
    if job.stage == "analytic":
        from complete_journey.systemone import lattice as L

        tab = args["raw"]
        lv = len(L.grid(cfg.k)) // 2
        an = C.analytic(tab, lv, L.grid(cfg.k)[lv], dgp)
        sims = [
            C.simulate(tab, dgp, cfg.k, (pidx, job.dgp_index, 7, j), n=cfg.n)
            for j in range(cfg.analytic_sims)
        ]
        mc = np.concatenate([s["behavior"][s["profile_index"] == lv] for s in sims])
        return {
            "level": lv,
            "mc_rate": float(mc.mean()),
            "analytic_rate": float(an["rate_by_week"].mean()),
            "mc_joint11": float((mc[:, :-1] * mc[:, 1:]).mean()),
            "analytic_joint11": float(an["joint11_by_week"].mean()),
            "households": int(mc.shape[0]),
            "tv_after_burn_in": an["tv_after_burn_in"],
        }
    if job.stage == "recovery":
        from complete_journey.systemone import estimands as E
        from complete_journey.systemone.e2e import truth_of

        truth = truth_of(dgp)
        rows, pop = args["rows"], args["pop"]
        vtab = args["table"]
        hv, _ = S.CELL_VISIBILITY[dgp.cell]
        installed = (
            0.0 if (dgp.shuffled or not hv) else E.ahe_from_table(vtab, cfg.k, dgp.cell, dgp.lam)
        )
        sens = E.ahe_from_table(vtab, cfg.k, dgp.cell, dgp.lam) if (dgp.shuffled and hv) else None
        return {
            "truth": truth,
            "installed_ahe": installed,
            "installed_ahe_pp": 100 * installed,
            "installed_basis": "structural zero"
            if (dgp.shuffled or not hv)
            else "measured on this variant's table",
            "policy_history_sensitivity_ahe": sens,
            "level1": R.level1(rows),
            "level2": R.level2(rows, pop, truth),
        }
    raise Phase9ContractError(f"unknown stage {job.stage}")


def _run_job(payload):
    job, cfg, args = payload
    t0 = time.time()
    return job, execute(job, cfg, args), time.time() - t0


# --- runner --------------------------------------------------------------------------


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def source_hashes() -> dict:
    out = {}
    for pattern in SOURCE_GLOBS:
        for f in sorted(ROOT.glob(pattern)):
            out[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def snapshot(cfg: LocalConfig, archives: dict, thash: str, jobs: list[Job]) -> dict:
    return {
        "schema": "phase9-local-snapshot-v1",
        "config": asdict(cfg),
        "archive_digests": archives,
        "policy_tables_sha256": thash,
        "seed_tree": SEED_TREE,
        "plan_sha256": sha([j.key(cfg).job_id for j in jobs]),
        "jobs": len(jobs),
        "source_sha256": source_hashes(),
    }


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


@dataclass
class Progress:
    work: Path
    cfg: LocalConfig
    jobs: list
    snapshot_sha: str
    commit: str
    workers: int
    ids: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    done: set = field(default_factory=set)
    skipped: int = 0
    failed: list = field(default_factory=list)
    running: dict = field(default_factory=dict)
    recent: list = field(default_factory=list)
    last_save: float | None = None
    health: str = "RUNNING"
    stage: str = "starting"
    message: str = ""
    _last_write: float = 0.0

    def record(self, job_id: str, seconds: float):
        self.done.add(job_id)
        now = time.time()
        self.last_save = now
        self.recent.append((now, seconds))
        self.recent = self.recent[-200:]
        with open(self.work / "progress.jsonl", "a") as f:
            f.write(json.dumps({"job_id": job_id, "utc": now, "seconds": round(seconds, 3)}) + "\n")

    def write(self, immediate: bool = False):
        now = time.time()
        if not immediate and now - self._last_write < 1.0:
            return
        self._last_write = now
        by_stage = {}
        for j in self.jobs:
            s = by_stage.setdefault(j.stage, {"total": 0, "done": 0})
            s["total"] += 1
            s["done"] += self.ids[j] in self.done
        window = [r for r in self.recent if now - r[0] <= 600]
        rate = len(window) / min(600.0, max(1.0, now - self.started)) if window else 0.0
        remaining = len(self.jobs) - len(self.done)
        current = next(iter(self.running.values()), None)
        _atomic_json(
            self.work / "status.json",
            {
                "label": self.cfg.label,
                "health": self.health,
                "stage": self.stage,
                "message": self.message,
                "jobs_total": len(self.jobs),
                "jobs_done": len(self.done),
                "jobs_skipped_on_resume": self.skipped,
                "jobs_failed": self.failed,
                "by_stage": by_stage,
                "running": list(self.running.values()),
                "current": current,
                "workers_configured": self.workers,
                "workers_busy": min(len(self.running), self.workers),
                "jobs_in_flight": len(self.running),
                "started_epoch": self.started,
                "updated_epoch": now,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                "throughput_jobs_per_min": 60 * rate,
                "eta_seconds": remaining / rate if rate > 0 else None,
                "last_checkpoint_epoch": self.last_save,
                "source_commit": self.commit,
                "snapshot_sha256": self.snapshot_sha,
                "api_calls": "DISABLED",
                "pid": os.getpid(),
            },
        )


class RunLock:
    def __init__(self, work: Path):
        self.path = work / "run.lock"

    def acquire(self):
        if self.path.exists():
            try:
                pid = int(json.loads(self.path.read_text())["pid"])
                os.kill(pid, 0)
                raise Phase9ContractError(f"another local run is alive (pid {pid})")
            except (ValueError, KeyError, ProcessLookupError, json.JSONDecodeError):
                pass  # stale lock
        _atomic_json(self.path, {"pid": os.getpid(), "utc": time.time()})

    def release(self):
        try:
            if int(json.loads(self.path.read_text())["pid"]) == os.getpid():
                self.path.unlink()
        except (OSError, ValueError, KeyError):
            pass


def _sigterm(signum, _frame):
    raise SystemExit(f"terminated by signal {signum}")


def run(cfg: LocalConfig, workers: int, max_job_retries: int = 2) -> dict:
    """Plan, verify, execute pending jobs, assemble. Safe to call again after any stop."""
    import signal

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # survives a closed Terminal
    isolate()
    work = cfg.work()
    work.mkdir(parents=True, exist_ok=True)
    lock = RunLock(work)
    lock.acquire()
    try:
        return _run(cfg, workers, max_job_retries, work)
    finally:
        lock.release()


def _run(cfg: LocalConfig, workers: int, max_job_retries: int, work: Path) -> dict:
    from complete_journey.abm.stage1 import real_ci1_target
    from complete_journey.phase8_final.store import ShardedStore

    archives = verify_freeze(cfg.policies)
    pds = policy_data(cfg)
    tables = base_tables(cfg, pds)
    thash = tables_hash(tables)
    jobs = plan(cfg)
    snap = json.loads(json.dumps(snapshot(cfg, archives, thash, jobs), default=str))
    snap_path = work / "snapshot.json"
    if snap_path.exists():
        prev = json.loads(snap_path.read_text())
        diff = sorted(k for k in snap if prev.get(k) != snap[k])
        require(not diff, f"resume refused: the snapshot changed ({diff})")
    else:
        _atomic_json(snap_path, snap)
    snap_sha = sha(snap)
    commit = git_commit()
    store = ShardedStore(work / "checkpoints", snap_sha, commit)
    _atomic_json(
        work / "plan.json",
        {
            "snapshot_sha256": snap_sha,
            "jobs": [{"id": j.key(cfg).job_id, **j.label(cfg)} for j in jobs],
        },
    )
    ids = {j: j.key(cfg).job_id for j in jobs}
    keys = {j: j.key(cfg) for j in jobs}
    prog = Progress(work, cfg, jobs, snap_sha, commit, workers, ids)
    for j in jobs:  # verified load of every existing checkpoint (corruption is a stop)
        if store.load(keys[j]) is not None:
            prog.done.add(ids[j])
    prog.skipped = len(prog.done)
    real_mean = real_ci1_target().values["mean"]
    deltas: dict = {}

    def table_for(job: Job) -> np.ndarray:
        t = tables[(job.policy, job.dgp_index)]
        if job.variant == "calibrated":
            return shifted_table(t["raw"], deltas[(job.policy, job.dgp_index)])
        return t[job.variant]

    def panels_of(job: Job, block: int) -> dict:
        ok, fails = [], 0
        for j in jobs:
            if j.stage == "panels" and (j.policy, j.dgp_index, j.variant, j.block) == (
                job.policy,
                job.dgp_index,
                job.variant,
                block,
            ):
                rec = store.load(keys[j])
                require(rec is not None, "a panel chunk is missing before recovery")
                ok += rec["payload"]["panels"]
                fails += rec["payload"]["failures"]
        return {"panels": ok, "failures": fails}

    def args_for(job: Job) -> dict:
        if job.stage == "calibration":
            return {"raw": tables[(job.policy, job.dgp_index)]["raw"], "real_mean": real_mean}
        if job.stage == "panels":
            return {"table": table_for(job)}
        if job.stage == "analytic":
            return {"raw": tables[(job.policy, job.dgp_index)]["raw"]}
        return {"table": table_for(job), "rows": panels_of(job, 0), "pop": panels_of(job, 1)}

    import multiprocessing as mp

    try:
        for stage in STAGES[:-1]:
            prog.stage = stage
            if stage in ("panels", "recovery") and cfg.calibrate:
                for p in cfg.policies:
                    for di in range(len(dgp_list(cfg))):
                        rec = store.load(Job("calibration", p, di).key(cfg))
                        require(rec is not None, "calibration incomplete before panels")
                        deltas[(p, di)] = rec["payload"]["delta"]
            pending = [j for j in jobs if j.stage == stage and ids[j] not in prog.done]
            tries: dict = {}
            while pending:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=mp.get_context("spawn"),
                    initializer=_worker_init,
                ) as ex:
                    futs = {}
                    queue = list(pending)
                    pending = []
                    broken = False
                    while queue or futs:
                        while queue and len(futs) < 2 * workers:
                            j = queue.pop(0)
                            futs[ex.submit(_run_job, (j, cfg, args_for(j)))] = j
                            prog.running[ids[j]] = j.label(cfg)
                        prog.write()
                        finished, _ = wait(futs, timeout=5, return_when=FIRST_COMPLETED)
                        for fu in finished:
                            j = futs.pop(fu)
                            jid = ids[j]
                            prog.running.pop(jid, None)
                            try:
                                _, payload, secs = fu.result()
                            except BrokenProcessPool:
                                broken = True
                                pending.append(j)
                                continue
                            except Exception:  # noqa: BLE001 - mechanical job failure
                                tries[jid] = tries.get(jid, 0) + 1
                                with open(work / "failures.log", "a") as fl:
                                    fl.write(
                                        f"{time.ctime()} {jid} {j.label(cfg)}\n"
                                        f"{traceback.format_exc()}\n"
                                    )
                                if tries[jid] > max_job_retries:
                                    prog.failed.append({"id": jid, **j.label(cfg)})
                                else:
                                    queue.append(j)
                                continue
                            store.save(keys[j], payload)
                            prog.record(jid, secs)
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
        results = assemble(cfg, pds, jobs, store, deltas, tables, snap, snap_sha)
        prog.health = "RUN_COMPLETE"
        prog.message = f"results sha256 {results['results_sha256']}"
        prog.write(immediate=True)
        return results
    except BaseException as exc:
        prog.health = "STOPPED"
        prog.message = f"{type(exc).__name__}: {exc}"
        prog.write(immediate=True)
        raise


# --- assembly ------------------------------------------------------------------------


def assemble(cfg, pds, jobs, store, deltas, tables, snap, snap_sha) -> dict:
    from complete_journey.systemone import estimands as E
    from complete_journey.systemone import reporting as Rep

    ds = dgp_list(cfg)
    policies = {}
    for p in cfg.policies:
        pd = pds[p]
        out = {"policy": p, "static": {}, "recovery": {}, "analytic_check": {}, "calibration": {}}
        for conv in ("primary", "normalized"):
            out["static"][conv] = E.static_report(pd, conv, lams=cfg.lams)
        out["convention_dependence"] = E.convention_dependence(
            out["static"]["primary"], out["static"]["normalized"]
        )
        for di, dgp in enumerate(ds):
            if cfg.calibrate:
                out["calibration"][dgp.name] = deltas[(p, di)]
            for v in variants_of(cfg, dgp):
                rec = store.load(Job("recovery", p, di, v).key(cfg))
                require(rec is not None, "missing recovery checkpoint")
                out["recovery"][f"{dgp.name}:{v}"] = rec["payload"]
            if not dgp.shuffled:
                rec = store.load(Job("analytic", p, di).key(cfg))
                require(rec is not None, "missing analytic checkpoint")
                out["analytic_check"][dgp.name] = rec["payload"]
        policies[p] = out
    summary = {
        "schema": "phase9-local-results-v1",
        "status": "CANDIDATE RESULTS (researcher review required)",
        "label": cfg.label,
        "snapshot_sha256": snap_sha,
        "archive_digests": snap["archive_digests"],
        "policy_tables_sha256": snap["policy_tables_sha256"],
        "config": asdict(cfg),
        "policies": policies,
    }
    labels = {p: Rep.flat_labels(r) for p, r in policies.items()}
    if S.P_J13 in labels and S.P_J14 in labels:
        jev = Rep.jev_generic(labels[S.P_J13], labels[S.P_J14])
        summary["jev_generic"] = jev
        if S.P_M in labels:
            summary["cross_vendor"] = Rep.cross_vendor(jev, labels[S.P_M])
    if S.P_J13 in pds and S.P_M in pds:
        summary["exploratory"] = Rep.exploratory_disagreement(pds[S.P_J13], pds[S.P_M])
    if S.P_J14 in pds and S.P_M in pds:
        summary["exploratory_j14_m"] = Rep.exploratory_disagreement(pds[S.P_J14], pds[S.P_M])
    summary["results_sha256"] = sha({k: v for k, v in summary.items()})
    _atomic_json(cfg.work() / "results.json", summary)
    return summary
