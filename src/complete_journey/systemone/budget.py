"""Hard budget guard (P9-001 budget block; no ``--force``, no bypass).

Before every attempt, the worst-case cost of the attempt must fit what remains of the
policy's tier cap and of the family ceiling (Jev $1.00 across P-J13 and P-J14; Mercury
$0.25). Attempt and success caps are enforced. A failed attempt is charged at its
worst-case cost, because failed-call billing is undocumented. A success is charged
its reconciled cost: TypeSafe input tokens × price, or OpenRouter ``usage.cost``.
State is persisted atomically after every change, so a crash never loses accounting.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

from complete_journey.systemone import settings as S
from complete_journey.systemone.settings import Phase9ContractError, require


class BudgetStop(Phase9ContractError):
    """A budget or request cap would be exceeded: the run stops before the request."""


def estimate_tokens(body: dict, worst: bool = False) -> int:
    chars = len(json.dumps(body, ensure_ascii=False))
    per = S.CHARS_PER_TOKEN_WORST if worst else S.CHARS_PER_TOKEN_EXPECTED
    return math.ceil(chars / per)


def worst_case_cost(policy: str, body: dict) -> float:
    price = S.POLICY_PINS[policy]["usd_per_input_token"]
    return S.WORST_CASE_TOKEN_FACTOR * estimate_tokens(body, worst=True) * price


def family_of(policy: str) -> str:
    require(policy in S.POLICIES, f"unknown policy {policy}")
    return "jev" if policy in S.JEV_POLICIES else "mercury"


def family_ceiling(policy: str) -> float:
    return S.JEV_CEILING_USD if family_of(policy) == "jev" else S.MERCURY_CEILING_USD


class Budget:
    """One persistent ledger per budget root, covering all policies and tiers. Every read
    and write happens under an exclusive ``flock``, and the state is reloaded each time,
    so concurrent runs of different policies cannot race on the ledger."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            self.state = self._load()

    @contextmanager
    def _locked(self):
        lock = self.path.with_name(self.path.name + ".lock")
        with open(lock, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _load(self) -> dict:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            require(data.get("schema") == "phase9-budget-v1", "corrupt budget ledger")
            return data
        return {
            "schema": "phase9-budget-v1",
            "spent": {},
            "attempts": {},
            "successes": {},
            "overruns": [],
        }

    def _save(self) -> None:
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        os.replace(tmp, self.path)

    @staticmethod
    def _k(policy: str, tier: str) -> str:
        return f"{policy}:{tier}"

    def spent(self, policy: str, tier: str | None = None) -> float:
        if tier is not None:
            return self.state["spent"].get(self._k(policy, tier), 0.0)
        return sum(v for k, v in self.state["spent"].items() if k.startswith(f"{policy}:"))

    def family_spent(self, policy: str) -> float:
        fam = family_of(policy)
        return sum(v for k, v in self.state["spent"].items() if family_of(k.split(":")[0]) == fam)

    def count(self, kind: str, policy: str, tier: str) -> int:
        return self.state[kind].get(self._k(policy, tier), 0)

    def check(self, policy: str, tier: str, worst: float) -> None:
        """Refuse the attempt if any cap could be exceeded."""
        with self._locked():
            self.state = self._load()
            self._check_locked(policy, tier, worst)

    def _check_locked(self, policy: str, tier: str, worst: float) -> None:
        require(tier in S.REQUEST_CAPS, f"unknown tier {tier}")
        caps = S.REQUEST_CAPS[tier]
        if self.count("attempts", policy, tier) >= caps["attempts"]:
            raise BudgetStop(f"{policy} {tier}: attempt cap {caps['attempts']} reached")
        if self.count("successes", policy, tier) >= caps["successes"]:
            raise BudgetStop(f"{policy} {tier}: success cap {caps['successes']} reached")
        tier_cap = S.TIER_CAPS_USD[policy][tier]
        if self.spent(policy, tier) + worst > tier_cap + 1e-15:
            raise BudgetStop(f"{policy} {tier}: tier cap ${tier_cap:.2f} would be exceeded")
        ceiling = family_ceiling(policy)
        if self.family_spent(policy) + worst > ceiling + 1e-15:
            raise BudgetStop(f"{family_of(policy)} ceiling ${ceiling:.2f} would be exceeded")

    def reserve_attempt(self, policy: str, tier: str, worst: float) -> None:
        """Check every cap and record the attempt under one lock (atomic across processes)."""
        with self._locked():
            self.state = self._load()
            self._check_locked(policy, tier, worst)
            k = self._k(policy, tier)
            self.state["attempts"][k] = self.state["attempts"].get(k, 0) + 1
            self._save()

    def record_attempt(self, policy: str, tier: str) -> None:
        with self._locked():
            self.state = self._load()
            k = self._k(policy, tier)
            self.state["attempts"][k] = self.state["attempts"].get(k, 0) + 1
            self._save()

    def charge(self, policy: str, tier: str, usd: float, success: bool) -> None:
        """Record a cost. If the actual cost pushes spend past a cap (possible only when a
        response bills more than its worst-case estimate), the overrun is recorded and the
        run stops."""
        require(usd >= 0 and math.isfinite(usd), "cost must be finite and non-negative")
        with self._locked():
            self.state = self._load()
            k = self._k(policy, tier)
            self.state["spent"][k] = self.state["spent"].get(k, 0.0) + usd
            if success:
                self.state["successes"][k] = self.state["successes"].get(k, 0) + 1
            over = []
            if self.spent(policy, tier) > S.TIER_CAPS_USD[policy][tier] + 1e-15:
                over.append(f"{policy} {tier} tier cap")
            if self.family_spent(policy) > family_ceiling(policy) + 1e-15:
                over.append(f"{family_of(policy)} ceiling")
            if over:
                self.state.setdefault("overruns", []).append({"key": k, "usd": usd, "caps": over})
            self._save()
        if over:
            raise BudgetStop(f"actual cost overran: {', '.join(over)}")

    def summary(self) -> dict:
        with self._locked():
            return json.loads(json.dumps(self._load()))
