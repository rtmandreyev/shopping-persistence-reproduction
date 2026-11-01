"""Local preflight: the predeclared CI-1 grid check (K), need-grid diagnostics, cost
sizing and manifest hashes. No provider output is used anywhere.

**K rule** (P9-001: the grid resolution comes from a local check that the *ideal
base-rate policy* passes the CI-1 gate):
- The ideal base-rate policy sets pi = the stated grid frequency, with iid weeks.
- Households are assigned by nearest-level rounding of expit(0.213 + 1.540*xi).
- R_K = 20 panels with N = 2,500 and T = 102, on Phase 9 seeds.
- A grid passes if the mean CI-1 over the panels meets the unchanged CI-1 gate
  |mean - real| <= max(0.02, 2 SE_real) on all six components.
- Choose the smallest passing K in (10, 19). If neither passes, STOP.
"""

from __future__ import annotations

import numpy as np

from complete_journey.systemone import lattice as L
from complete_journey.systemone import settings as S
from complete_journey.systemone.budget import estimate_tokens, worst_case_cost
from complete_journey.systemone.closedloop import DGP, need_summary, simulate
from complete_journey.systemone.request import (
    manifest,
    ordered,
    production_requests,
    smoke_requests,
)
from complete_journey.systemone.settings import require

R_K = 20


def ci1_gate(mean: np.ndarray) -> dict:
    from complete_journey.abm import config as abm_config
    from complete_journey.abm.stage1 import real_ci1_target

    real = real_ci1_target()
    out = {}
    for j, c in enumerate(abm_config.CI1_COMPONENTS):
        tol = max(abm_config.CI1_GATE_ABS, 2 * real.se[c])
        out[c] = {
            "value": float(mean[j]),
            "real": real.values[c],
            "abs_diff": float(abs(mean[j] - real.values[c])),
            "tolerance": tol,
            "pass": bool(abs(mean[j] - real.values[c]) <= tol),
        }
    return out


def k_check(r: int = R_K, n: int = S.N_HOUSEHOLDS) -> dict:
    from complete_journey.abm.moments import ci1_fast

    out = {}
    for k in (10, 19):
        g = np.array(L.grid(k))
        table = np.broadcast_to(g[:, None, None], (len(g), 16, 3)).copy()
        vals = np.array(
            [
                ci1_fast(simulate(table, DGP(S.NEITHER), k, (k, j), n=n)["behavior"])
                for j in range(r)
            ]
        )
        gate = ci1_gate(vals.mean(0))
        out[str(k)] = {
            "pass": all(v["pass"] for v in gate.values()),
            "components": gate,
            "max_abs_diff": max(v["abs_diff"] for v in gate.values()),
        }
    # continuous reference (no rounding): the Phase 8 A1 law itself
    from complete_journey.systemone.seeds import rng

    ref = []
    for j in range(r):
        gg = rng("k_check", 99, j)
        p = 1 / (1 + np.exp(-(S.A1_MU + S.A1_S * gg.standard_normal(n))))
        ref.append(
            ci1_fast(
                (gg.random((n, S.BURN_IN + S.T_WEEKS)) < p[:, None]).astype(np.int8)[:, S.BURN_IN :]
            )
        )
    gate = ci1_gate(np.mean(ref, axis=0))
    out["continuous_reference"] = {
        "pass": all(v["pass"] for v in gate.values()),
        "components": gate,
    }
    passing = [k for k in (10, 19) if out[str(k)]["pass"]]
    out["chosen_k"] = min(passing) if passing else None
    out["rule"] = (
        "smallest K in (10, 19) whose ideal base-rate policy passes the CI-1 gate; neither -> STOP"
    )
    return out


def need_grid() -> list[dict]:
    return [need_summary(lam) for lam in S.NEED_LAMBDAS]


def cost_sizing(
    k: int, batched_and_fallback=(True, False), token_scale: dict | None = None
) -> dict:
    """Request counts and token / cost sizing for every policy.

    STATUS: CANDIDATE. The frozen sizing rule (P9-001) uses the input tokens *measured
    in the smoke run*. Without ``token_scale``, tokens are local character-based
    estimates. ``sizing_from_smoke`` supplies the measured per-policy scale.
    Retries are not part of the frozen worst case. Their headroom is reported:
    the number of extra failed attempts, each charged at its worst-case cost, that fit
    between the worst case and the cap."""
    token_scale = token_scale or {}
    out = {
        "k": k,
        "status": "MEASURED (smoke tokens)" if token_scale else "CANDIDATE (local token estimate)",
        "policies": {},
    }
    totals = {"jev_expected": 0.0, "jev_worst": 0.0, "mercury_expected": 0.0, "mercury_worst": 0.0}
    for policy in S.POLICIES:
        rec = {}
        for batched in batched_and_fallback:
            reqs = production_requests(policy, k, batched=batched)
            from complete_journey.systemone.adapters import wire_body

            bodies = [wire_body(r) for r in reqs]
            scale = token_scale.get(policy, 1.0)
            exp_tok = scale * sum(estimate_tokens(b) for b in bodies)
            worst_cost = scale * sum(worst_case_cost(policy, b) for b in bodies)
            price = S.POLICY_PINS[policy]["usd_per_input_token"]
            rec["batched" if batched else "fallback"] = {
                "requests": len(reqs),
                "expected_tokens": exp_tok,
                "mean_tokens_per_request": exp_tok / len(reqs),
                "expected_usd": exp_tok * price,
                "worst_case_usd": worst_cost,
            }
        cap = S.TIER_CAPS_USD[policy]["production"]
        worst = max(v["worst_case_usd"] for v in rec.values())
        rec["production_cap_usd"] = cap
        rec["worst_over_cap"] = worst / cap
        rec["passes_80pct_rule"] = bool(worst <= S.SIZING_FRACTION * cap)
        n_max = max(v["requests"] for v in rec.values() if isinstance(v, dict))
        per_attempt = worst / n_max
        rec["retry_headroom_failed_attempts"] = (
            int((cap - worst) / per_attempt) if per_attempt > 0 else None
        )
        rec["request_cap_ok"] = all(
            v["requests"] <= S.REQUEST_CAPS["production"]["successes"]
            for v in rec.values()
            if isinstance(v, dict)
        )
        out["policies"][policy] = rec
        fam = "jev" if policy in S.JEV_POLICIES else "mercury"
        totals[f"{fam}_expected"] += rec["batched"]["expected_usd"]
        totals[f"{fam}_worst"] += worst
    # family ceilings, including the smoke and pilot tiers at their caps (most conservative)
    jev_worst_all = totals["jev_worst"] + sum(
        S.TIER_CAPS_USD[p]["smoke"] + S.TIER_CAPS_USD[p]["pilot"] for p in S.JEV_POLICIES
    )
    mer_worst_all = (
        totals["mercury_worst"] + S.TIER_CAPS_USD[S.P_M]["smoke"] + S.TIER_CAPS_USD[S.P_M]["pilot"]
    )
    out["totals"] = {
        **totals,
        "jev_worst_incl_smoke_pilot_caps": jev_worst_all,
        "mercury_worst_incl_smoke_pilot_caps": mer_worst_all,
        "jev_ceiling": S.JEV_CEILING_USD,
        "mercury_ceiling": S.MERCURY_CEILING_USD,
        "jev_ceiling_80pct_ok": jev_worst_all <= S.SIZING_FRACTION * S.JEV_CEILING_USD,
        "mercury_ceiling_80pct_ok": mer_worst_all <= S.SIZING_FRACTION * S.MERCURY_CEILING_USD,
    }
    out["all_pass"] = (
        all(p["passes_80pct_rule"] and p["request_cap_ok"] for p in out["policies"].values())
        and out["totals"]["jev_ceiling_80pct_ok"]
        and out["totals"]["mercury_ceiling_80pct_ok"]
    )
    return out


def smoke_manifests(k: int) -> dict:
    out = {}
    for policy in S.POLICIES:
        reqs = smoke_requests(policy, k)
        m = manifest(reqs, "smoke", k, extra={"smoke_states": [s.key() for s in L.smoke_states(k)]})
        from complete_journey.systemone.adapters import wire_body

        worst = sum(worst_case_cost(policy, wire_body(r)) for r in reqs)
        out[policy] = {
            "manifest": m,
            "requests": len(reqs),
            "worst_case_usd": worst,
            "tier_cap_usd": S.TIER_CAPS_USD[policy]["smoke"],
        }
        require(worst <= S.TIER_CAPS_USD[policy]["smoke"], "smoke worst case exceeds its tier cap")
    return out


PRODUCTION_STATUS = (
    "FINAL: production manifest for the P9-005 post-smoke freeze (batched form; the smoke "
    "batching fallback was not triggered). Runs only under an ACTIVE per-policy "
    "production approval."
)
FALLBACK_STATUS = (
    "NOT FROZEN: unbatched fallback form. The smoke batching fallback was not triggered "
    "(P9-005), so this form may not be run."
)


def production_manifest(policy: str, k: int, batched: bool = True) -> dict:
    """The production manifest. Only the batched form is frozen (P9-005); its hash is
    checked against the decision text by ``authorize``."""
    reqs = ordered(production_requests(policy, k, batched=batched), S.POLICIES.index(policy))
    return manifest(
        reqs,
        "production",
        k,
        extra={
            "status": PRODUCTION_STATUS if batched else FALLBACK_STATUS,
            "batched": batched,
            "governing_decisions": list(S.SCOPE_DECISIONS["production"]),
        },
    )


def sizing_from_smoke(archive_roots: dict, k: int) -> dict:
    """The frozen sizing rule with measured tokens. The per-policy scale is the median,
    over smoke successes, of measured usage.input_tokens / local expected estimate."""
    from complete_journey.systemone.adapters import wire_body
    from complete_journey.systemone.archive import Archive
    from complete_journey.systemone.request import request_from_identity

    scale = {}
    for policy, root in archive_roots.items():
        ratios = []
        for rec in Archive(root).successes():
            req = request_from_identity({**rec["identity"], "request_hash": rec["request_hash"]})
            ratios.append(rec["usage"]["input_tokens"] / estimate_tokens(wire_body(req)))
        require(ratios, f"no smoke successes for {policy}")
        scale[policy] = float(np.median(ratios))
    out = cost_sizing(k, token_scale=scale)
    out["measured_token_scale"] = scale
    return out
