"""Manifest execution with the frozen retry rule, the budget guard, the archive and resume.

Retry rule (P9-001 gate 9):
- Only 429, 529 and 5xx responses are retried, at most 3 times, with exponential
  backoff that honours ``retry-after``.
- No retry for 400, 401, 413, 415 or 422, for a transport error, or for a schema,
  model, provider or semantic-parse failure. Any of these stops the run (P9-001 stop
  conditions) after the failed attempt has been archived.

Every attempt is checked against the budget beforehand and charged afterwards, so a
failed attempt is charged at its worst-case cost.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from complete_journey.systemone import adapters
from complete_journey.systemone import settings as S
from complete_journey.systemone.archive import Archive, RunLock
from complete_journey.systemone.budget import Budget, BudgetStop, worst_case_cost
from complete_journey.systemone.request import CanonicalRequest
from complete_journey.systemone.settings import Phase9ContractError, require

NO_RETRY = frozenset({400, 401, 413, 415, 422})


class RunStop(Phase9ContractError):
    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind


@dataclass
class RunResult:
    done: int = 0
    skipped: int = 0
    attempts: int = 0
    retries: int = 0
    stopped: str | None = None
    latencies: list = field(default_factory=list)
    drift: list = field(default_factory=list)


def backoff(attempt: int, retry_after: str | None) -> float:
    wait = min(S.BACKOFF_MAX_S, S.BACKOFF_BASE_S * (2**attempt))
    try:
        ra = float(retry_after) if retry_after is not None else 0.0
    except ValueError:
        ra = 0.0
    return min(S.BACKOFF_MAX_S, max(wait, ra))


def bind_manifest(root: Path, manifest_sha256: str, policy: str) -> None:
    """An archive root serves one manifest and policy. Anything else is refused.
    Call it while holding the run lock. The reference is created exclusively."""
    root.mkdir(parents=True, exist_ok=True)
    ref = root / "manifest.ref.json"
    want = {
        "manifest_sha256": manifest_sha256,
        "policy": policy,
        "pin": S.POLICY_PINS[policy]["model"],
    }
    tmp = ref.with_name(f".{ref.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(want, sort_keys=True))
    try:
        os.link(tmp, ref)  # atomic and exclusive: the reference is never half-written
        return
    except FileExistsError:
        try:
            have = json.loads(ref.read_text())
        except json.JSONDecodeError:
            raise Phase9ContractError(
                f"unreadable manifest reference {ref}; inspect and remove it by hand"
            ) from None
        require(have == want, "archive root belongs to a different manifest, policy or model pin")
    finally:
        tmp.unlink(missing_ok=True)


def run(
    requests: list[CanonicalRequest],
    provider,
    root: Path,
    budget: Budget,
    tier: str,
    manifest_sha256: str,
    sleep: Callable[[float], None] = time.sleep,
    run_meta: dict | None = None,
    drift_monitor: Callable[[Archive, int], dict] | None = None,
) -> RunResult:
    require(requests, "nothing to run")
    policies = {r.policy for r in requests}
    require(len(policies) == 1, "one policy per run")
    policy = policies.pop()
    hashes = [r.request_hash for r in requests]
    require(len(hashes) == len(set(hashes)), "duplicate request in run")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = RunLock(root)
    lock.acquire()
    res = RunResult()
    try:
        bind_manifest(root, manifest_sha256, policy)
        archive = Archive(root)
        if run_meta is not None:
            meta_path = root / "run_meta.jsonl"
            with open(meta_path, "a") as f:
                f.write(
                    json.dumps(
                        {**run_meta, "utc": time.time(), "manifest_sha256": manifest_sha256},
                        sort_keys=True,
                    )
                    + "\n"
                )
        for rec in archive.successes():  # version and provider check on resume
            require(
                rec["returned_model"] == S.POLICY_PINS[policy]["returned_model"],
                "archived model differs from pin",
            )
            require(
                rec.get("returned_provider") == S.POLICY_PINS[policy]["returned_provider"]
                or S.POLICY_PINS[policy]["transport"] == S.TYPESAFE,
                "archived provider differs from pin",
            )
        for req in requests:
            if archive.done(req.request_hash):
                res.skipped += 1
                continue
            _one(req, provider, archive, budget, tier, sleep, res)
            res.done += 1
            if drift_monitor is not None and req.purpose == "sentinel" and req.timepoint >= 2:
                group = [
                    r for r in requests if r.purpose == "sentinel" and r.timepoint == req.timepoint
                ]
                if all(archive.done(r.request_hash) for r in group):
                    report = drift_monitor(archive, req.timepoint)
                    res.drift.append(report)
                    if report["stop"]:
                        raise RunStop(
                            "drift_bound", f"sentinel drift at time point {req.timepoint}: {report}"
                        )
    except (RunStop, BudgetStop) as exc:
        res.stopped = str(exc)
        raise
    finally:
        lock.release()
    return res


def _one(req, provider, archive: Archive, budget: Budget, tier, sleep, res: RunResult) -> None:
    body = adapters.wire_body(req)
    worst = worst_case_cost(req.policy, body)
    for attempt in range(S.MAX_RETRIES + 1):
        budget.reserve_attempt(req.policy, tier, worst)
        res.attempts += 1
        status, hdrs, payload, latency, raw = provider.send(req)
        res.latencies.append(latency)
        base = {
            "policy": req.policy,
            "requested_model": req.pin["model"],
            "http_status": status,
            "headers": hdrs,
            "latency_s": latency,
            "utc": time.time(),
            "retry_index": attempt,
            "state_key": req.state.key(),
            "template": req.template,
            "purpose": req.purpose,
        }
        if status == 200:
            try:
                parsed = adapters.parse(req, payload)
            except adapters.ResponseError as exc:
                archive.store_attempt(
                    req.request_hash, {**base, "error": exc.kind, "raw": raw, "payload": payload}
                )
                budget.charge(req.policy, tier, worst, success=False)
                raise RunStop(exc.kind, str(exc)) from None
            archive.store_success(
                req.request_hash,
                {
                    **base,
                    "identity": req.identity(),
                    "wire_body": body,
                    "raw_payload": payload,
                    "answers": parsed.answers,
                    "returned_model": parsed.model,
                    "returned_provider": parsed.provider,
                    "usage": {
                        "input_tokens": parsed.input_tokens,
                        "output_tokens": parsed.output_tokens,
                    },
                    "cost_usd": parsed.cost_usd,
                },
            )
            budget.charge(req.policy, tier, parsed.cost_usd, success=True)  # may stop on overrun
            return
        archive.store_attempt(req.request_hash, {**base, "error": f"http_{status}", "raw": raw})
        budget.charge(req.policy, tier, worst, success=False)
        retryable = status in S.RETRY_STATUSES or (isinstance(status, int) and 500 <= status <= 599)
        if status is None or status in NO_RETRY or not retryable:
            raise RunStop(f"http_{status}", f"non-retryable failure on {req.request_hash[:12]}")
        if attempt == S.MAX_RETRIES:
            raise RunStop(
                "retries_exhausted", f"{S.MAX_RETRIES} retries exhausted on {req.request_hash[:12]}"
            )
        res.retries += 1
        sleep(backoff(attempt, (hdrs or {}).get("retry-after")))
