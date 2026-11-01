"""Canonical provider-neutral request, content-addressed hashing and manifests.

The canonical request carries all of the science: the synthetic state, the questions,
the template, the policy pin, the repeat index and experimental metadata. Adapters may
change only the endpoint, auth header, model string, provider-routing field and
mechanical wrapping (``adapters.py``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from complete_journey.systemone import lattice as L
from complete_journey.systemone import settings as S
from complete_journey.systemone.seeds import rng
from complete_journey.systemone.settings import require
from complete_journey.systemone.state import SemanticState, questions, render_state

MANIFEST_SCHEMA = "phase9-systemone-manifest-v1"


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256(obj) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


@dataclass(frozen=True)
class CanonicalRequest:
    policy: str
    state: SemanticState
    template: str
    question_ids: tuple[str, ...]
    repeat: int = 0
    purpose: str = "production"  # production / placebo / paraphrase / sentinel / smoke-*
    timepoint: int = 0  # sentinel time point (1..4) or 0
    nonce: str | None = None  # smoke nonce-sensitivity test only
    scope: str = "production"
    meta: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        require(self.policy in S.POLICIES, "unknown policy")
        require(self.template in S.TEMPLATES, "unknown template")
        require(self.repeat >= 0, "repeat index must be non-negative")
        require(self.scope in ("smoke", "pilot", "production", "mock"), "unknown scope")

    @property
    def pin(self):
        return S.POLICY_PINS[self.policy]

    def semantic_state(self) -> dict[str, str]:
        st = render_state(self.state, self.template)
        if self.nonce is not None:
            st = {**st, "request_nonce": self.nonce}
        return st

    def semantic_questions(self) -> dict:
        return questions(self.template, self.question_ids)

    def identity(self) -> dict:
        """Everything that defines this request; its hash is the content address."""
        pin = self.pin
        return {
            "policy": self.policy,
            "model": pin["model"],
            "transport": pin["transport"],
            "endpoint": pin["endpoint"],
            "template": self.template,
            "state_key": self.state.key(),
            "state": self.semantic_state(),
            "questions": self.semantic_questions(),
            "repeat": self.repeat,
            "purpose": self.purpose,
            "timepoint": self.timepoint,
            "nonce": self.nonce,
            "scope": self.scope,
        }

    @property
    def request_hash(self) -> str:
        return sha256(self.identity())

    def semantic_body(self) -> dict:
        """The provider-neutral payload; identical across policies apart from ``model``."""
        return {"state": self.semantic_state(), "questions": self.semantic_questions()}


# --- production manifest -----------------------------------------------------------


def production_requests(policy: str, k: int, batched: bool = True) -> list[CanonicalRequest]:
    """The full production request set for one policy (P9-001 as amended by P9-002).

    ``batched=False`` is the predeclared fallback: ``shop`` is asked alone and the
    two diagnostic questions share a second request.
    """
    groups = [(S.Q_SHOP, S.Q_SKIP, S.Q_TYPE)] if batched else [(S.Q_SHOP,), (S.Q_SKIP, S.Q_TYPE)]
    states = L.all_states(k)
    out: list[CanonicalRequest] = []
    for qs in groups:
        for s in states:  # template A, default placebo, 3 repeats
            for r in range(S.REPEATS_A):
                out.append(CanonicalRequest(policy, s, "A", qs, repeat=r, purpose="production"))
        for s in L.placebo_frame(k):  # P9-002 stratified placebo subset, flipped value
            out.append(
                CanonicalRequest(
                    policy, s.with_placebo(S.PLACEBO_VALUES[1]), "A", qs, purpose="placebo"
                )
            )
        for s in states:  # template B paraphrase, full lattice
            out.append(CanonicalRequest(policy, s, "B", qs, purpose="paraphrase"))
        for t in range(1, S.SENTINEL_TIMEPOINTS + 1):  # drift sentinels
            for s in L.sentinels(k):
                out.append(CanonicalRequest(policy, s, "A", qs, purpose="sentinel", timepoint=t))
    hashes = [r.request_hash for r in out]
    require(len(hashes) == len(set(hashes)), "duplicate production request")
    return out


def ordered(requests: list[CanonicalRequest], policy_index: int = 0) -> list[CanonicalRequest]:
    """Seeded random query order, with sentinel time point t placed at the start of the
    t-th quarter of the run (t = 1 first, t = 4 in the final quarter)."""
    body = [r for r in requests if r.purpose != "sentinel"]
    sent = {
        t: [r for r in requests if r.purpose == "sentinel" and r.timepoint == t]
        for t in range(1, 5)
    }
    perm = rng("query_order", 0).permutation(len(body))  # one shared order: co-equal policies
    body = [body[i] for i in perm]
    n = len(body)
    out: list[CanonicalRequest] = []
    for t in range(1, S.SENTINEL_TIMEPOINTS + 1):
        lo, hi = (t - 1) * n // 4, t * n // 4
        out += sent[t] + body[lo:hi]
    require(len(out) == len(requests), "ordering lost requests")
    return out


def smoke_requests(policy: str, k: int) -> list[CanonicalRequest]:
    """Technical smoke set (P9-001 gate 5/6): per designated state, 5 byte-identical
    3-question repeats, 5 nonce repeats and 3 shop-alone repeats (co-presence test)."""
    out = []
    for i, s in enumerate(L.smoke_states(k)):
        qs3 = (S.Q_SHOP, S.Q_SKIP, S.Q_TYPE)
        out += [
            CanonicalRequest(
                policy, s, "A", qs3, repeat=r, purpose="smoke-identical", scope="smoke"
            )
            for r in range(5)
        ]
        g = rng("nonce", k, i)
        for r in range(5):
            nonce = f"{int(g.integers(0, 2**32)):08x}"
            out.append(
                CanonicalRequest(
                    policy, s, "A", qs3, repeat=r, purpose="smoke-nonce", nonce=nonce, scope="smoke"
                )
            )
        out += [
            CanonicalRequest(
                policy, s, "A", (S.Q_SHOP,), repeat=r, purpose="smoke-alone", scope="smoke"
            )
            for r in range(3)
        ]
    return out


def manifest(
    requests: list[CanonicalRequest], scope: str, k: int, extra: dict | None = None
) -> dict:
    require(requests, "empty manifest")
    policies = {r.policy for r in requests}
    require(len(policies) == 1, "a manifest belongs to exactly one policy")
    policy = policies.pop()
    body = {
        "schema": MANIFEST_SCHEMA,
        "run_label": S.RUN_LABEL,
        "scope": scope,
        "policy": policy,
        "pin": plain_pin(policy),
        "k": k,
        "design_decisions": list(S.DESIGN_DECISIONS),
        "budget": manifest_budget_block(policy, scope),
        "requests": [
            {"request_hash": r.request_hash, **{k_: v for k_, v in r.identity().items()}}
            for r in requests
        ],
        "counts": _counts(requests),
        **(extra or {}),
    }
    body["manifest_sha256"] = sha256(body)
    return body


def manifest_budget_block(policy: str, scope: str) -> dict:
    """The budget block stamped into a manifest. Smoke manifests were frozen in P9-003
    and keep the historical P9-003 block (hash-stable). Manifests frozen after P9-004
    carry the operative P9-004 caps (``settings.operative_budget_json``)."""
    request_caps = dict(S.REQUEST_CAPS[scope]) if scope in S.REQUEST_CAPS else None
    if scope == "smoke":
        legacy = S.MANIFEST_BUDGET_P9003
        return {
            "tier_cap_usd": legacy["tier_cap_usd"][policy].get(scope),
            "request_caps": request_caps,
            "jev_ceiling_usd": legacy["jev_ceiling_usd"],
            "mercury_ceiling_usd": legacy["mercury_ceiling_usd"],
        }
    return {
        "tier_cap_usd": S.TIER_CAPS_USD[policy].get(scope),
        "request_caps": request_caps,
        "jev_ceiling_usd": S.JEV_CEILING_USD,
        "mercury_ceiling_usd": S.MERCURY_CEILING_USD,
        "budget_decision": S.BUDGET_DECISION,
    }


def plain_pin(policy: str) -> dict:
    """The JSON form of a frozen policy pin (used in manifests and by ``authorize``)."""
    return _plain(dict(S.POLICY_PINS[policy]))


def verify_manifest(m: dict) -> str:
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    require(sha256(body) == m.get("manifest_sha256"), "manifest hash mismatch")
    return m["manifest_sha256"]


def _counts(requests) -> dict:
    out: dict[str, int] = {}
    for r in requests:
        out[r.purpose] = out.get(r.purpose, 0) + 1
    out["total"] = len(requests)
    return out


def _plain(v):
    if v is None:
        return None
    if hasattr(v, "items"):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    return v


def request_from_identity(entry: dict) -> CanonicalRequest:
    """Rebuild a canonical request from a manifest entry and check that it round-trips."""
    cell, prof, hist, need, placebo = entry["state_key"].split("|")
    state = SemanticState(
        cell,
        None if prof == "U" else float(prof),
        None if hist == "-" else tuple(int(c) for c in hist),
        None if need == "-" else need,
        placebo,
    )
    req = CanonicalRequest(
        entry["policy"],
        state,
        entry["template"],
        tuple(entry["questions"]),
        repeat=entry["repeat"],
        purpose=entry["purpose"],
        timepoint=entry["timepoint"],
        nonce=entry["nonce"],
        scope=entry["scope"],
    )
    require(req.request_hash == entry["request_hash"], "manifest entry does not round-trip")
    return req


__all__ = ["CanonicalRequest", "asdict"]
