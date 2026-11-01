"""Resumable, failure-propagating executor and stage machine of the final run.

Reliability (P8-006):
- ``ProcessPoolExecutor`` (spawn): a dead worker raises ``BrokenProcessPool`` instead of
  hanging; the pool is rebuilt (at most ``EXECUTOR_MAX_RESTARTS`` times) and only jobs
  without a checkpoint are resubmitted. No completion for ``JOB_TIMEOUT_HOURS`` while
  jobs are in flight counts as a hang and also rebuilds the pool.
- Run lock with stale-lock detection; disk guard (stop below 10 GB free); last-error
  record; stage status file; append-only progress index; launch-time plan manifest.
- Every job is checkpointed atomically in a sharded store keyed by the final snapshot;
  completed jobs are skipped on resume. Arithmetic stages are recomputed (deterministic)
  and archive writes are write-once.
- Deadline governance: projected completion beyond the 72-hour ceiling pauses the run
  at the next safe point (no scope change).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path

from complete_journey.abm.config import Phase8ContractError, require
from complete_journey.paths import PROJECT_ROOT
from complete_journey.phase8_final import jobs as J
from complete_journey.phase8_final import settings as S
from complete_journey.phase8_final.store import ShardedStore


class RunStop(Phase8ContractError):
    """A mandatory or safe stop: the run ends cleanly, checkpoints intact."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def deadline_start_epoch() -> float:
    return (
        datetime.strptime(S.DEADLINE_START_UTC, "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=UTC)
        .timestamp()
    )


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


# --- worker side -------------------------------------------------------------------

_STORE: ShardedStore | None = None


def _init_worker(root: str, config_hash: str, commit: str) -> None:
    global _STORE
    for var in S.THREAD_ENV:
        os.environ[var] = "1"
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _STORE = ShardedStore(Path(root), config_hash, commit)


def _work(job: J.FJob) -> dict:
    t0, c0 = time.time(), time.process_time()
    rec = _STORE.load(job.key())
    if rec is not None:
        return {"job_id": job.key().job_id, "skipped": True, "cpu": 0.0, "wall": 0.0}
    payload = J.RUNNERS[job.kind](job, _STORE)
    payload["runtime_seconds"] = time.time() - t0
    payload["cpu_seconds"] = time.process_time() - c0
    record = _STORE.save(job.key(), payload)
    return {
        "job_id": job.key().job_id,
        "skipped": False,
        "cpu": payload["cpu_seconds"],
        "wall": payload["runtime_seconds"],
        "sha": record["output_sha256"],
    }


# --- parent side -------------------------------------------------------------------


class Runner:
    def __init__(
        self,
        config_hash: str,
        commit: str,
        work: str | None = None,
        workers: int | None = None,
        deadline: bool = True,
        probe: bool = True,
    ):
        self.work = PROJECT_ROOT / (work or S.WORK)
        self.work.mkdir(parents=True, exist_ok=True)
        self.store = ShardedStore(self.work / "checkpoints", config_hash, commit)
        self.config_hash, self.commit = config_hash, commit
        self.status_path = self.work / "status.json"
        self.progress_path = self.work / "progress.jsonl"
        self.error_path = self.work / "last_error.json"
        self.plan_path = self.work / "plan.json"
        self.lock_path = self.work / "run.lock"
        self.deadline, self.probe = deadline, probe
        # The researcher relaxed the 72 h ceiling to advisory (2026-10-20): projections are
        # still recorded at every timing checkpoint, but they no longer pause the run.
        self.pause_on_deadline = False
        self.status = self._read_status()
        self.workers = workers or self.status.get("workers_decided") or S.WORKERS_TRY
        self.stage = "SETUP"
        self.started = time.time()
        self._last_status = 0.0
        self._recent: list[tuple[float, float]] = []  # (time, expected cost) of completions
        self.stop_reason: tuple[str, str] | None = None
        self.inprocess = False

    # status / lock ----------------------------------------------------------------
    def _read_status(self) -> dict:
        try:
            return json.loads(self.status_path.read_text())
        except (OSError, ValueError):
            return {}

    def write_status(self, force: bool = False, **extra) -> None:
        if not force and time.time() - self._last_status < 5:
            return
        self._last_status = time.time()
        self.status.update(
            {
                "stage": self.stage,
                "updated_utc": now_utc(),
                "pid": os.getpid(),
                "workers": self.workers,
                "run_label": S.RUN_LABEL,
                "config_hash": self.config_hash,
                "commit": self.commit,
                **extra,
            }
        )
        atomic_json(self.status_path, self.status)

    def record_error(self, message: str) -> None:
        atomic_json(self.error_path, {"utc": now_utc(), "stage": self.stage, "error": message})

    def acquire_lock(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            live = None
            try:
                pid = int(json.loads(self.lock_path.read_text())["pid"])
                os.kill(pid, 0)
                cmd = subprocess.run(
                    ["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True
                ).stdout
                if "final_run.py" in cmd or "phase8_final" in cmd:
                    live = pid
            except (ProcessLookupError, PermissionError, ValueError, KeyError, OSError):
                live = None  # stale lock: the recorded process is gone
            require(
                live is None, f"a final run is already alive (pid {live}); refusing a second copy"
            )
            self.lock_path.unlink(missing_ok=True)
        fd = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as handle:
            json.dump({"pid": os.getpid(), "started_utc": now_utc()}, handle)

    def release_lock(self) -> None:
        try:
            if int(json.loads(self.lock_path.read_text())["pid"]) == os.getpid():
                self.lock_path.unlink()
        except (OSError, ValueError, KeyError):
            pass

    def free_gb(self) -> float:
        return shutil.disk_usage(self.work).free / 1e9

    # plan / progress / deadline ---------------------------------------------------
    def update_plan(self, stage: str, jobs: list[J.FJob], estimated: bool = False) -> None:
        plan = {}
        try:
            plan = json.loads(self.plan_path.read_text())
        except (OSError, ValueError):
            pass
        kinds: dict[str, dict] = {}
        for j in jobs:
            k = f"{j.kind}:{j.arm}"
            kinds.setdefault(k, {"total": 0, "cost_each": j.cost})["total"] += 1
        plan.setdefault("stages", {})[stage] = {"kinds": kinds, "estimated": estimated}
        plan["stage_order"] = list(S.STAGES)
        plan["deadline_start_utc"] = S.DEADLINE_START_UTC
        plan["ceiling_hours"] = S.DEADLINE_CEILING_HOURS
        atomic_json(self.plan_path, plan)

    def progress(self, job: J.FJob, info: dict) -> None:
        line = {
            "t": time.time(),
            "stage": self.stage,
            **job.label(),
            "job_id": info["job_id"],
            "skipped": info["skipped"],
            "cpu": round(info["cpu"], 2),
            "wall": round(info["wall"], 2),
            "cost": job.cost,
        }
        with self.progress_path.open("a") as handle:
            handle.write(json.dumps(line) + "\n")
        if not info["skipped"]:
            self._recent.append((time.time(), job.cost))

    def throughput(self) -> float | None:
        """Expected-cost seconds completed per wall second over the last hour."""
        cut = time.time() - 3600
        recent = [(t, c) for t, c in self._recent if t >= cut]
        if len(recent) < 10:
            return None
        span = max(time.time() - recent[0][0], 900.0)
        return sum(c for _, c in recent) / span

    def remaining_cost(self) -> float:
        try:
            plan = json.loads(self.plan_path.read_text())
        except (OSError, ValueError):
            return 0.0
        seen: dict[str, str] = {}
        if self.progress_path.exists():
            for line in self.progress_path.read_text().splitlines():
                rec = json.loads(line)
                seen[rec["job_id"]] = f"{rec['kind']}:{rec['arm']}"  # one entry per job
        done: dict[str, int] = {}
        for k in seen.values():
            done[k] = done.get(k, 0) + 1
        total = 0.0
        for st in plan.get("stages", {}).values():
            for k, v in st["kinds"].items():
                rem = max(v["total"] - done.get(k, 0), 0)
                done[k] = max(done.get(k, 0) - v["total"], 0)
                total += rem * v["cost_each"]
        return total

    def check_deadline(self) -> None:
        if not self.deadline:
            return
        start = deadline_start_epoch()
        hours = (time.time() - start) / 3600
        marks = self.status.setdefault("timing_checkpoints", {})
        rate = self.throughput()
        remaining = self.remaining_cost()
        eta = remaining / rate if rate else None
        projected = None if eta is None else (time.time() + eta - start) / 3600 + 4.0
        for mark in S.TIMING_CHECKPOINT_HOURS:
            if hours >= mark and str(mark) not in marks:
                marks[str(mark)] = {
                    "utc": now_utc(),
                    "projected_total_hours": projected,
                    "remaining_cost_core_hours": remaining / 3600,
                }
        self.write_status(
            force=True,
            eta_seconds=eta,
            projected_total_hours=projected,
            remaining_cost_core_hours=remaining / 3600,
            throughput_core_hours_per_hour=rate,
        )
        if (
            self.pause_on_deadline
            and projected is not None
            and projected > S.DEADLINE_CEILING_HOURS
        ):
            # Pause at the next safe point: stop submitting, drain in-flight jobs.
            self.stop_reason = (
                "PAUSED_DEADLINE",
                f"projected total {projected:.1f} h exceeds the 72 h ceiling",
            )

    # executor ------------------------------------------------------------------------
    def _executor(self, workers: int) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=(str(self.store.root), self.config_hash, self.commit),
        )

    def run_jobs(self, jobs: list[J.FJob]) -> dict:
        seen, unique = set(), []
        for j in jobs:
            jid = j.key().job_id
            if jid not in seen:
                seen.add(jid)
                unique.append(j)
        self._backfill_progress(unique)
        pending = sorted(
            [j for j in unique if self.store.load(j.key()) is None], key=lambda j: -j.cost
        )
        if not pending:
            return {"jobs": len(unique), "ran": 0}
        if self.inprocess:  # dry runs only: patched settings must reach the job code
            global _STORE
            _STORE = self.store
            for job in pending:
                info = _work(job)
                self.progress(job, info)
                self.write_status()
            return {"jobs": len(unique), "ran": len(pending), "inprocess": True}
        for var in S.THREAD_ENV:
            os.environ[var] = "1"
        restarts, ran, t_start = 0, 0, time.time()
        probe_due = self.probe and "workers_decided" not in self.status
        workers = S.WORKERS_TRY if probe_due else self.workers
        probe_cost, probe_t0 = 0.0, time.time()
        queue = list(pending)
        executor = self._executor(workers)
        inflight: dict = {}
        last_done = time.time()
        try:
            while queue or inflight:
                draining = False
                while (
                    queue and len(inflight) < 2 * workers and not draining and not self.stop_reason
                ):
                    if self.free_gb() < S.DISK_MIN_FREE_GB:
                        draining = True
                        self.stop_reason = (
                            "STOPPED_DISK",
                            f"free disk {self.free_gb():.1f} GB < 10 GB",
                        )
                        break
                    job = queue.pop(0)
                    inflight[executor.submit(_work, job)] = job
                if not inflight:
                    if self.stop_reason:
                        raise RunStop(*self.stop_reason)
                    break
                done, _ = wait(list(inflight), timeout=60, return_when=FIRST_COMPLETED)
                if not done:
                    if time.time() - last_done > S.JOB_TIMEOUT_HOURS * 3600:
                        raise BrokenProcessPool("no completion within the hang timeout")
                    self.write_status(force=True)
                    continue
                for fut in done:
                    job = inflight.pop(fut)
                    try:
                        info = fut.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as exc:  # deterministic job error: mechanical bug
                        self.record_error(f"{job}: {type(exc).__name__}: {exc}")
                        raise RunStop("FAILED", f"job {job} failed: {exc}") from exc
                    last_done = time.time()
                    ran += 0 if info["skipped"] else 1
                    self.progress(job, info)
                    if probe_due:
                        probe_cost += job.cost
                self.write_status(stage_jobs={"done": ran, "pending": len(queue) + len(inflight)})
                if probe_due and time.time() - probe_t0 >= S.THROUGHPUT_PROBE_MINUTES * 60:
                    rate = probe_cost / (time.time() - probe_t0)  # P-core-h per wall hour
                    keep = rate > S.BASELINE_CORE_HOURS_PER_HOUR
                    self.workers = S.WORKERS_TRY if keep else S.WORKERS_FALLBACK
                    self.status["workers_decided"] = self.workers
                    self.status["throughput_probe"] = {
                        "utc": now_utc(),
                        "workers_tried": S.WORKERS_TRY,
                        "measured_core_hours_per_hour": rate,
                        "baseline_core_hours_per_hour": S.BASELINE_CORE_HOURS_PER_HOUR,
                        "decision": self.workers,
                        "basis": "computational only",
                    }
                    self.write_status(force=True)
                    probe_due = False
                    if self.workers != workers:  # drain, then rebuild with the new size
                        for fut in list(inflight):
                            info = fut.result()
                            self.progress(inflight.pop(fut), info)
                        executor.shutdown(wait=True)
                        workers = self.workers
                        executor = self._executor(workers)
                self.check_deadline()
                if self.stop_reason:
                    for fut in list(inflight):
                        self.progress(inflight.pop(fut), fut.result())
                    raise RunStop(*self.stop_reason)
        except BrokenProcessPool as exc:
            restarts = self.status.get("executor_restarts", 0) + 1
            self.status["executor_restarts"] = restarts
            self.record_error(f"worker pool broken ({exc}); restart {restarts}")
            self._kill_children(executor)
            if restarts > S.EXECUTOR_MAX_RESTARTS:
                raise RunStop("FAILED", f"worker pool broke {restarts} times") from exc
            return self.run_jobs(jobs)  # resubmits only jobs without checkpoints
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return {
            "jobs": len(unique),
            "ran": ran,
            "wall_seconds": time.time() - t_start,
            "workers": workers,
            "restarts": restarts,
        }

    def _backfill_progress(self, jobs: list[J.FJob]) -> None:
        """A job saved by a worker just before a parent crash has a checkpoint but no
        progress line; record it once (skipped=True, recovered) so counts stay exact."""
        logged = set()
        if self.progress_path.exists():
            for line in self.progress_path.read_text().splitlines():
                try:
                    logged.add(json.loads(line)["job_id"])
                except (ValueError, KeyError):
                    continue
        for job in jobs:
            jid = job.key().job_id
            if jid not in logged and self.store.load(job.key()) is not None:
                self.progress(job, {"job_id": jid, "skipped": True, "cpu": 0.0, "wall": 0.0})

    @staticmethod
    def _kill_children(executor: ProcessPoolExecutor) -> None:
        for proc in list(getattr(executor, "_processes", {}).values()):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001 - best effort during recovery
                pass
