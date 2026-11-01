"""Smoke analysis, implemented before any response exists (P9-001 gates 5–7).

Technical outputs only: repeatability, nonce sensitivity, the co-presence (batching)
test, the coherence gap, metadata, latency, usage and cost, and schema validity.
No scientific Phase 9 conclusion is drawn from the smoke run.

**Batching fallback criterion** (predeclared): if the mean over smoke states of
|shop alone - shop with all three questions| exceeds 3 × the pooled SD of
byte-identical repeats, production asks ``shop`` alone (``batched=False``).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from complete_journey.systemone import settings as S
from complete_journey.systemone.archive import Archive


def analyze(root, policy: str) -> dict:
    arch = Archive(root)
    by = defaultdict(lambda: defaultdict(list))
    meta, lat, cost, tokens = defaultdict(int), [], 0.0, 0
    for rec in arch.successes():
        ident = rec["identity"]
        by[ident["purpose"]][ident["state_key"]].append(rec["answers"])
        meta[(rec["returned_model"], rec.get("returned_provider"))] += 1
        lat.append(rec["latency_s"])
        cost += rec["cost_usd"]
        tokens += rec["usage"]["input_tokens"]
    attempts_dir = arch.root / "attempts"
    failed = sum(1 for _ in attempts_dir.glob("*/*.json")) if attempts_dir.exists() else 0

    def stats(purpose: str, q: str) -> dict:
        per = {}
        for key, reps in by[purpose].items():
            v = np.array([r[q] for r in reps if q in r], float)
            if len(v):
                per[key] = {
                    "n": len(v),
                    "mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                    "range": float(v.max() - v.min()),
                }
        return per

    out = {"policy": policy, "repeatability": {}, "nonce": {}, "coherence_gap": {}}
    pooled = {}
    for q in (S.Q_SHOP, S.Q_SKIP, S.Q_TYPE):
        ident_stats = stats("smoke-identical", q)
        nonce_stats = stats("smoke-nonce", q)
        out["repeatability"][S.QUESTION_ROLES[q]] = ident_stats
        sds = [v["sd"] for v in ident_stats.values() if v["sd"] is not None]
        pooled[q] = float(np.sqrt(np.mean(np.square(sds)))) if sds else None
        out["nonce"][S.QUESTION_ROLES[q]] = {
            key: {
                "nonce_mean": v["mean"],
                "identical_mean": ident_stats[key]["mean"],
                "effect": v["mean"] - ident_stats[key]["mean"],
                "nonce_sd": v["sd"],
            }
            for key, v in nonce_stats.items()
            if key in ident_stats
        }
    out["pooled_identical_sd"] = {S.QUESTION_ROLES[q]: v for q, v in pooled.items()}
    alone = stats("smoke-alone", S.Q_SHOP)
    shop_ident = stats("smoke-identical", S.Q_SHOP)
    effects = [abs(alone[k]["mean"] - shop_ident[k]["mean"]) for k in alone if k in shop_ident]
    sd = pooled[S.Q_SHOP]
    out["co_presence"] = {
        "per_state_abs_effect": effects,
        "mean_abs_effect": float(np.mean(effects)) if effects else None,
        "threshold": None if sd is None else S.REPEAT_SD_MULTIPLE * sd,
        "fallback_triggered": bool(
            effects and sd is not None and np.mean(effects) > S.REPEAT_SD_MULTIPLE * sd
        ),
    }
    gaps = [r[S.Q_SHOP] + r[S.Q_SKIP] - 1 for reps in by["smoke-identical"].values() for r in reps]
    out["coherence_gap"] = {
        "mean": float(np.mean(gaps)) if gaps else None,
        "mean_abs": float(np.mean(np.abs(gaps))) if gaps else None,
    }
    out["metadata"] = {f"{m}|{p}": c for (m, p), c in meta.items()}
    out["latency_s"] = {
        "n": len(lat),
        "median": float(np.median(lat)) if lat else None,
        "p95": float(np.quantile(lat, 0.95)) if lat else None,
    }
    out["usage"] = {"input_tokens": tokens, "cost_usd": cost}
    out["schema"] = {
        "successes": sum(len(v) for d in by.values() for v in d.values()),
        "failed_attempts": failed,
    }
    out["note"] = (
        "Technical smoke only: no scientific Phase 9 conclusion is drawn from these values."
    )
    return out
