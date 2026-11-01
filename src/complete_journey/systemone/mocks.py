"""Deterministic mock policies and a mock provider (no network, no credentials).

Each mock policy is a known artificial mechanism over ``SemanticState``. Together they
exercise every scientific label. The mock provider returns transport-shaped payloads,
so the real adapters and parsers are exercised end to end. It can also script HTTP
failures for the retry, budget and stop tests.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from complete_journey.systemone import settings as S
from complete_journey.systemone.request import CanonicalRequest
from complete_journey.systemone.state import SemanticState

NEED_SCORE = {"low": -1.0, "usual": 0.0, "high": 1.0, None: 0.0}


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _base(s: SemanticState) -> float:
    return logit(s.profile) if s.profile is not None else 0.0


def _hist(s: SemanticState, b1: float, bc: float) -> float:
    if s.history is None:
        return 0.0
    return b1 * (s.history[0] - 0.5) + bc * (s.count - 2) / 2


def _type(s: SemanticState, shift: float = 0.0) -> float:
    f = s.profile if s.profile is not None else 0.5
    return expit(8.0 * (f - 0.5) + shift)


@dataclass(frozen=True)
class MockPolicy:
    """``answer(state, template) -> (shop, skip, usual_type)`` (noise-free)."""

    name: str
    kind: str
    params: dict = field(default_factory=dict)

    def answer(self, s: SemanticState, template: str) -> tuple[float, float, float]:
        k, p = self.kind, self.params
        x = _base(s)
        t_shift = 0.0
        if k in ("history", "both"):
            x += _hist(s, p.get("b1", 1.2), p.get("bc", 0.8))
        if k in ("latent", "both"):
            x += p.get("g", 1.0) * NEED_SCORE[s.need]
        if k == "proxy":
            # history read as evidence about type: strong when the profile is unavailable,
            # weaker when it is stated, weaker again when need is visible (explaining away)
            w = 1.0 if s.profile is None else 0.35
            if s.need is not None:
                w *= 0.5
                x += 0.8 * NEED_SCORE[s.need]
            x += w * _hist(s, 1.5, 1.5)
            t_shift = 1.5 * (0.0 if s.history is None else (s.count - 2) / 2)
        if k == "wording":
            x += 0.05 * _hist(s, 1.0, 0.0)
            x += 0.9 if template == "B" else 0.0
            x += 0.8 if s.placebo != S.PLACEBO_DEFAULT else 0.0
        shop = expit(x)
        skip = 1.0 - shop
        if k == "incoherent":
            skip = min(1.0, 0.45 * (1.0 - shop) + 0.35)
        return shop, skip, _type(s, t_shift)


def mock_policies() -> dict[str, MockPolicy]:
    return {
        "M0": MockPolicy("M0", "base"),
        "M1": MockPolicy("M1", "history"),
        "M2": MockPolicy("M2", "latent"),
        "M3": MockPolicy("M3", "both"),
        "M4": MockPolicy("M4", "proxy"),
        "M5": MockPolicy("M5", "wording"),
        "M6": MockPolicy("M6", "incoherent"),
        "M7": MockPolicy("M7", "both", {"noise_sd": 0.08}),
    }


@dataclass
class MockProvider:
    """Answers canonical requests like the pinned provider would. ``noise_sd`` adds
    seeded per-request noise. ``script`` maps a request hash to a list of statuses or
    failure kinds, consumed one per attempt."""

    policy: MockPolicy
    noise_sd: float = 0.0
    script: dict = field(default_factory=dict)
    calls: int = 0

    def send(self, req: CanonicalRequest):
        self.calls += 1
        h = req.request_hash
        plan = self.script.get(h)
        action = plan.pop(0) if plan else 200
        if isinstance(action, int) and action != 200:
            hdrs = {"retry-after": "0"} if action == 429 else {}
            return action, hdrs, {"error": {"code": action}}, 0.01, f'{{"error":{action}}}'
        shop, skip, typ = self.policy.answer(req.state, req.template)
        sd = self.noise_sd or self.policy.params.get("noise_sd", 0.0)
        if sd:
            seed = int(hashlib.sha256(f"{h}:{self.calls}".encode()).hexdigest()[:16], 16)
            g = np.random.default_rng(seed)
            shop, skip, typ = (float(np.clip(v + g.normal(0, sd), 0, 1)) for v in (shop, skip, typ))
        if req.nonce is not None and self.policy.params.get("nonce_shift"):
            shop = min(1.0, shop + self.policy.params["nonce_shift"])
        vals = {S.Q_SHOP: shop, S.Q_SKIP: skip, S.Q_TYPE: typ}
        pin = req.pin
        answers = {q: {"type": "noul", "noul": round(vals[q], 6)} for q in req.question_ids}
        payload = {
            "model": pin["returned_model"],
            "answers": answers,
            "usage": {"input_tokens": 400, "output_tokens": 3 * len(answers)},
        }
        if pin["transport"] == S.OPENROUTER:
            payload["provider"] = pin["returned_provider"]
            payload["usage"]["cost"] = 400 * pin["usd_per_input_token"]
        if action == "malformed":
            payload = {"model": pin["returned_model"], "answers": "oops"}
        elif action == "missing_noul":
            payload["answers"][req.question_ids[0]] = {"type": "noul"}
        elif action == "invalid_probability":
            payload["answers"][req.question_ids[0]]["noul"] = 1.7
        elif action == "model_mismatch":
            payload["model"] = (
                "jev-latest"
                if pin["transport"] == S.TYPESAFE
                else "inception/mercury-decide-20991231"
            )
        elif action == "provider_mismatch":
            payload["provider"] = "SomeoneElse"
        return 200, {}, payload, 0.01, "<mock>"
