"""Mechanical provider adapters (P9-001: endpoint, auth header, model string,
provider-routing field and wrapping only; no wording, no tuning).

Both TypeSafe (`POST /v1/systemone`) and the OpenRouter Decisions API
(`POST /api/alpha/decisions`) take ``{model, state, questions}`` with Noul questions
and return ``answers[q] = {"type": "noul", "noul": p}`` plus ``model`` and ``usage``.
Parsing is strict. A mismatch or malformed answer is a non-retryable error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from complete_journey.systemone import settings as S
from complete_journey.systemone.request import CanonicalRequest
from complete_journey.systemone.settings import Phase9ContractError, require


class ResponseError(Phase9ContractError):
    """Non-retryable response failure; ``kind`` names the class."""

    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind


SCHEMA, MODEL_MISMATCH, PROVIDER_MISMATCH, SEMANTIC = (
    "schema",
    "model_mismatch",
    "provider_mismatch",
    "semantic_parse",
)


def wire_body(req: CanonicalRequest) -> dict:
    """The exact JSON body for the request's transport."""
    pin = req.pin
    require(
        not any(p in pin["model"] for p in S.PROHIBITED_MODEL_SUBSTRINGS), "aliases are prohibited"
    )
    body = {"model": pin["model"], **req.semantic_body()}
    if pin["transport"] == S.OPENROUTER:
        routing = pin["provider_routing"]
        body["provider"] = {
            "only": list(routing["only"]),
            "allow_fallbacks": routing["allow_fallbacks"],
        }
    else:
        require(pin["transport"] == S.TYPESAFE, "unknown transport")
    return body


def headers(transport: str, api_key: str) -> dict[str, str]:
    require(transport in (S.TYPESAFE, S.OPENROUTER), "unknown transport")
    require(isinstance(api_key, str) and api_key, "missing credential")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


@dataclass(frozen=True)
class Parsed:
    answers: dict[str, float]
    model: str
    provider: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float  # reconciled cost of this successful response


def parse(req: CanonicalRequest, payload) -> Parsed:
    """Validate a 200-response body against the request and the policy pin."""
    pin = req.pin
    if not isinstance(payload, dict):
        raise ResponseError(SCHEMA, "response body is not an object")
    model = payload.get("model")
    if model != pin["returned_model"]:
        raise ResponseError(
            MODEL_MISMATCH, f"returned model {model!r} != pin {pin['returned_model']!r}"
        )
    provider = payload.get("provider")
    if pin["transport"] == S.OPENROUTER and provider != pin["returned_provider"]:
        raise ResponseError(
            PROVIDER_MISMATCH, f"returned provider {provider!r} != {pin['returned_provider']!r}"
        )
    answers = payload.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(req.question_ids):
        raise ResponseError(SCHEMA, "answers missing or keyed differently from the questions")
    out = {}
    for q in req.question_ids:
        a = answers[q]
        if not isinstance(a, dict) or a.get("type") != "noul":
            raise ResponseError(SCHEMA, f"answer {q} is not a noul answer")
        v = a.get("noul")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ResponseError(SEMANTIC, f"answer {q} has no finite noul probability")
        if not 0.0 <= float(v) <= 1.0:
            raise ResponseError(SEMANTIC, f"answer {q} probability {v} outside [0, 1]")
        out[q] = float(v)
    usage = payload.get("usage")
    if not isinstance(usage, dict) or not isinstance(usage.get("input_tokens"), int):
        raise ResponseError(SCHEMA, "usage.input_tokens missing")
    tin = usage["input_tokens"]
    tout = usage.get("output_tokens", 0) if isinstance(usage.get("output_tokens"), int) else 0
    if pin["transport"] == S.OPENROUTER and isinstance(usage.get("cost"), (int, float)):
        cost = float(usage["cost"])
    else:
        cost = tin * pin["usd_per_input_token"] + tout * pin["usd_per_output_token"]
    return Parsed(out, model, provider, tin, tout, cost)
