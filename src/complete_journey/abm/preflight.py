"""Phase 8 Run B synthetic preflight (SYNTHETIC NON-EVIDENCE; software checks only).

These fixtures exercise the implementation with tiny N, R and grids:
- B1–B10: architecture worlds and operators;
- B11–B12: toy RQ2 verdict records;
- the second-order and higher-order A1′/A2 engineering comparisons;
- the purchase-envelope diagnostic;
- a tiny span check and a tiny perfect-model run.

Nothing here touches real data, builds an emulator or runs production budgets. The
numbers are not research results.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from complete_journey.abm import adequacy as adq
from complete_journey.abm import architectures as arch
from complete_journey.abm import config, mapping, regions, rq2, span
from complete_journey.abm.calibration import calibrate
from complete_journey.abm.evaluate import (
    evaluate_point,
    simulate_reference,
    truth_point_se,
    world_targets,
)
from complete_journey.abm.guards import TargetVector
from complete_journey.abm.moments import ci1_fast, primary_moments
from complete_journey.abm.operators import (
    envelopes,
    historical_forced_endpoint_panel,
    world_panel,
)
from complete_journey.abm.seeds import Role, SeedKey, truth_key
from complete_journey.prediction import moments as p7m
from complete_journey.prediction import protocol as p7p
from complete_journey.prediction.panel import CanonicalPanel

STAGE = 0
HET = arch.Heterogeneity(0.0, 1.0)


def _key(rep: int = 0, role: Role = Role.DATA) -> SeedKey:
    return SeedKey(stage=STAGE, role=role, replicate=rep)


def _ladder(panel: CanonicalPanel, op=config.OPERATOR_O) -> dict:
    lad = {r["component"]: r["estimate"] for r in p7m.vi2_ladder(panel, op)}
    tr = {r["component"]: r["estimate"] for r in p7m.vi1_transitions(panel, op)}
    return {"GAP": tr["GAP"], "M3": lad["O-M3"]}


def _lags(panel: CanonicalPanel) -> dict:
    return {r["component"]: r["estimate"] for r in p7m.vi3_lag_profile(panel, config.OPERATOR_O)}


def _autocorr(x: np.ndarray, k: int) -> float:
    a, b = x[:, :-k].ravel(), x[:, k:].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def b1_a0(n=2000) -> dict:
    w = arch.simulate(arch.a0(), arch.Heterogeneity(0.2, 1.0), _key(), n)
    panel = world_panel(w)
    return {
        "unique_probabilities": int(np.unique(np.round(w.p, 12)).size),
        "rate": float(w.behavior.mean()),
        **_ladder(panel),
        "delta_log_loss_recency": p7p.recency_gain(panel)[p7p.config.RECENCY_GAIN],
    }


def b2_a1(n=2000) -> dict:
    w = arch.simulate(arch.a1(), arch.Heterogeneity(0.2, 1.2), _key(), n)
    panel = world_panel(w)
    return {
        **_ladder(panel),
        "delta_log_loss_recency": p7p.recency_gain(panel)[p7p.config.RECENCY_GAIN],
    }


def b3_a1_prime(n=2000, rho=0.9, sigma=1.0) -> dict:
    w = arch.simulate(arch.a1_prime(rho, sigma), HET, _key(), n)
    panel = world_panel(w)
    return {
        "latent_var": float(w.latent.var()),
        "latent_ac1": _autocorr(w.latent, 1),
        "latent_ac4": _autocorr(w.latent, 4),
        "rho": rho,
        **_ladder(panel),
        "delta_log_loss_recency": p7p.recency_gain(panel)[p7p.config.RECENCY_GAIN],
    }


def b4_a2(n=2000, delta=2.0, lam=0.6) -> dict:
    spec = arch.a2(delta, lam)
    w = arch.simulate(spec, HET, _key(), n)
    panel = world_panel(w)
    return {
        "history_min": float(w.history.min()),
        "history_max": float(w.history.max()),
        "delta1_pp": mapping.delta1(w),
        "delta12_weeks": mapping.delta12(spec, HET, _key(), n),
        **_ladder(panel),
        "delta_log_loss_recency": p7p.recency_gain(panel)[p7p.config.RECENCY_GAIN],
    }


def b5_hmm(n=2000, q=0.05, sigma=1.0) -> dict:
    w = arch.simulate(arch.a1_prime_hmm(q, sigma), HET, _key(), n)
    return {
        "share_high": float((w.latent > 0).mean()),
        "state_ac1": _autocorr(w.latent, 1),
        "expected_ac1": 1 - 2 * q,
        "state_ac4": _autocorr(w.latent, 4),
        "expected_ac4": (1 - 2 * q) ** 4,
        **_ladder(world_panel(w)),
    }


def b6_recency(n=2000, delta=2.0, lam=0.5) -> dict:
    w = arch.simulate(arch.a2_recency(delta, lam), HET, _key(), n)
    weeks = np.arange(1, config.T_WEEKS + 1)[None, :]
    gap = weeks - w.last_purchase
    rates = {}
    for g in (1, 2, 3, 5):
        sel = gap == g
        rates[f"rate_gap_{g}"] = float(w.behavior[sel].mean())
    return rates


def b7_envelope(n=2000) -> dict:
    w = arch.simulate(arch.a1(), arch.Heterogeneity(-0.5, 1.5), _key(), n)
    env = envelopes(world_panel(w))
    length = env["w_len"].fillna(0).to_numpy()
    first_ok = all(w.behavior[i, int(f) - 1] == 1 for i, f in enumerate(env["f"]) if f == f)
    last_ok = all(w.behavior[i, int(l_) - 1] == 1 for i, l_ in enumerate(env["l"]) if l_ == l_)
    return {
        "corr_alpha_interior_length": float(np.corrcoef(w.alpha, length)[0, 1]),
        "first_endpoint_is_purchase": bool(first_ok),
        "last_endpoint_is_purchase": bool(last_ok),
        "status_counts": env["status"].value_counts().to_dict(),
    }


def _availability(n: int, rng_seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)  # synthetic availability fixture (not E-2)
    a = rng.integers(1, 40, n)
    b = rng.integers(60, config.T_WEEKS + 1, n)
    return np.column_stack([a, b])


def b8_sa(n=2000) -> dict:
    av = _availability(n)
    w = arch.simulate(arch.a2(2.0, 0.6), HET, _key(), n, availability=av)
    env = envelopes(world_panel(w))
    weeks = np.arange(1, config.T_WEEKS + 1)[None, :]
    outside = (weeks < av[:, :1]) | (weeks > av[:, 1:])
    has = env["f"].notna().to_numpy()
    shorter = (env["l"] - env["f"]).to_numpy()[has] <= (av[has, 1] - av[has, 0])
    return {
        "purchases_outside_availability": int(w.behavior[outside].sum()),
        "envelope_within_availability": bool(
            (
                (env["f"].to_numpy()[has] >= av[has, 0]) & (env["l"].to_numpy()[has] <= av[has, 1])
            ).all()
        ),
        "envelope_not_longer": bool(shorter.all()),
        "mean_availability_minus_envelope": float(
            np.mean((av[has, 1] - av[has, 0]) - (env["l"] - env["f"]).to_numpy()[has])
        ),
    }


def b9_periodic(n=2000, a=1.0) -> dict:
    out = {}
    for name, spec in (("A1_PRIME", arch.a1_prime(0.9, 0.5)), ("A2", arch.a2(1.0, 0.5))):
        base = arch.simulate(spec, HET, _key(), n)
        per = arch.simulate(spec, HET, _key(), n, periodic_amplitude=a)
        zero = arch.simulate(spec, HET, _key(), n, periodic_amplitude=0.0)
        lags = _lags(world_panel(per))
        out[name] = {
            "zero_amplitude_identical": bool(np.array_equal(base.behavior, zero.behavior)),
            "beta_1": lags["beta_1"],
            "beta_2": lags["beta_2"],
        }
    return out


def b10_operator_trap(n=2000) -> dict:
    """A memoryless world observed under late availability: G-operator M3 is spurious."""
    av = _availability(n)
    w = arch.simulate(arch.a1(), arch.Heterogeneity(0.0, 1.0), _key(), n, availability=av)
    panel = world_panel(w)
    return {
        "M3_O": _ladder(panel)["M3"],
        "M3_G_full_grid": _ladder(panel, config.p7.OPERATOR_DIAGNOSTIC)["M3"],
    }


def b11_false_attribution() -> dict:
    hf, eq = adq.HISTORY_FAVOURED, adq.EQUIVALENT
    recs = []
    for q in range(10):
        recs.append(
            rq2.VerdictRecord("T1", "rho=0.9,HIGH", q, {0.25: hf if q < 1 else eq}, 0.1, 0.02)
        )
        recs.append(rq2.VerdictRecord("T2", "HMM q=0.05", q, {0.25: hf if q < 3 else eq}, 0.1, 0.0))
        recs.append(rq2.VerdictRecord("T0", "A1", q, {0.25: eq}, -0.01, 0.0))
    fa = rq2.fa_proc(recs)
    naive = rq2.fa_naive(recs)
    return {
        "max_t2": fa["max_t2"],
        "max_non_history": fa["max_non_history"],
        "naive": naive["union"],
    }


def b12_mde() -> dict:
    det = {
        (0.5, d): v
        for d, v in zip(config.DELTA1_GRID_PP[1:], (0.1, 0.3, 0.85, 0.7, 0.9, 0.95), strict=True)
    }
    return {
        "nonmonotone": rq2.mde80(det, 0.5, 0.05),
        "inadmissible": rq2.mde80(det, 0.5, 0.2),
        "none_qualifies": rq2.mde80({k: 0.5 for k in det}, 0.5, 0.0),
        "top_unreachable": rq2.mde80(
            {k: v for k, v in det.items() if k[1] <= 15.0}, 0.5, 0.0, frozenset({20.0, 30.0})
        ),
    }


def a2_reachability(n=800) -> dict:
    """Engineering finding behind P8-002: max reachable Δ_1 per persistence-axis λ."""
    out = {}
    for het in (HET, arch.Heterogeneity(-1.2, 0.4)):
        rows = {}
        for lam in regions.persistence_axis():
            peak, top = mapping.delta1_peak(lam, het, _key(0, Role.MODEL), n)
            rows[f"{lam:.3f}"] = {"delta_peak": peak, "max_delta1_pp": top}
        out[f"mu={het.mu},s={het.s}"] = rows
    return out


def _calibrated(spec, target: TargetVector, n: int, r_cal: int = 2):
    return calibrate(spec, target, stage=STAGE, n=n, r_cal=r_cal)


def second_order_and_higher(n=2500, reps=3) -> dict:
    """Matched A1′ / A2 engineering comparison (not the production span check)."""
    a1p = arch.a1_prime(0.9, 1.0)
    ref = arch.simulate(a1p, HET, _key(), n)
    target = TargetVector(
        "synthetic", dict(zip(config.CI1_COMPONENTS, ci1_fast(ref.behavior), strict=True))
    )
    m3_target = _ladder(world_panel(ref))["M3"]
    best = None
    for lam in (0.8, 0.9, 0.95):
        lo, hi = 0.0, 8.0
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            cal = _calibrated(arch.a2(mid, lam), target, n)
            w = arch.simulate(arch.a2(mid, lam), cal.heterogeneity, _key(0, Role.MODEL), n)
            m3 = _ladder(world_panel(w))["M3"]
            lo, hi = (mid, hi) if m3 < m3_target else (lo, mid)
        delta = 0.5 * (lo + hi)
        cal = _calibrated(arch.a2(delta, lam), target, n)
        w = arch.simulate(arch.a2(delta, lam), cal.heterogeneity, _key(0, Role.MODEL), n)
        b8_gap = abs(_lags(world_panel(w))["beta_8"] - _lags(world_panel(ref))["beta_8"])
        if best is None or b8_gap < best[0]:
            best = (b8_gap, lam, delta, cal.heterogeneity)
    _, lam, delta, het_a2 = best
    out = {"a1_prime": {"rho": 0.9, "sigma_z": 1.0, "mu": 0.0, "s": 1.0}}
    out["a2_matched"] = {"lam": lam, "delta": delta, "mu": het_a2.mu, "s": het_a2.s}
    summaries = {}
    for name, spec, het in (("A1_PRIME", a1p, HET), ("A2", arch.a2(delta, lam), het_a2)):
        rows = []
        for r in range(reps):
            w = arch.simulate(spec, het, _key(r, Role.MODEL), n)
            panel = world_panel(w)
            lags = _lags(panel)
            pm = primary_moments(panel).values
            rows.append(
                [
                    lags["beta_1"],
                    lags["beta_2"],
                    lags["beta_4"],
                    lags["beta_8"],
                    lags["gamma_2"],
                    pm["P1"],
                    pm["P4"],
                    pm["P5"],
                ]
            )
        arr = np.array(rows)
        summaries[name] = {"mean": arr.mean(axis=0), "sd": arr.std(axis=0, ddof=1)}
    labels = (
        "beta_1",
        "beta_2",
        "beta_4",
        "beta_8",
        "gamma_2",
        "P1_O-M3",
        "P4_dLL_rec",
        "P5_active_1",
    )
    table = {}
    for k, label in enumerate(labels):
        a, b = summaries["A1_PRIME"], summaries["A2"]
        pooled = float(np.sqrt((a["sd"][k] ** 2 + b["sd"][k] ** 2) / 2))
        diff = float(a["mean"][k] - b["mean"][k])
        table[label] = {
            "A1_PRIME": float(a["mean"][k]),
            "A2": float(b["mean"][k]),
            "diff": diff,
            "diff_over_pooled_sd": diff / pooled if pooled > 0 else None,
        }
    out["table"] = table
    return out


def envelope_diagnostic(n=2000) -> dict:
    het = arch.Heterogeneity(-0.5, 1.5)
    w = arch.simulate(arch.a1(), het, _key(), n)
    endo = envelopes(world_panel(w))["w_len"].fillna(0).to_numpy()
    # Arbitrary assigned windows: another world's envelopes, drawn independently of type.
    other = envelopes(world_panel(arch.simulate(arch.a1(), het, _key(1), n)))
    pairs = other[["f", "l"]].dropna().astype(int).to_numpy()
    assigned = pairs[np.random.default_rng(11).choice(len(pairs), size=n, replace=True)]
    assigned_len = np.clip(np.minimum(assigned[:, 1] - 1, 101) - assigned[:, 0], 0, None)
    forced = historical_forced_endpoint_panel(w, assigned)
    rec = forced.recorded["retailer_purchase"].to_numpy().reshape(n, -1)
    sa = arch.simulate(arch.a1(), het, _key(), n, availability=_availability(n))
    return {
        "corr_alpha_endogenous_length": float(np.corrcoef(w.alpha, endo)[0, 1]),
        "corr_alpha_assigned_length": float(np.corrcoef(w.alpha, assigned_len)[0, 1]),
        "forced_record_cells_without_behaviour": int(((rec == 1) & (w.behavior == 0)).sum()),
        "mean_interior_primary": float(endo.mean()),
        "mean_interior_s_a": float(envelopes(world_panel(sa))["w_len"].fillna(0).mean()),
    }


def tiny_span(n=800, replicates=2) -> dict:
    """Tiny 3 × 3 span check per family (production: full grid, R = 50, real CI-1)."""
    ref = arch.simulate(arch.a1_prime(0.9, 1.0), HET, _key(), n)
    target = TargetVector(
        "synthetic", dict(zip(config.CI1_COMPONENTS, ci1_fast(ref.behavior), strict=True))
    )

    images, degenerate, infeasible = {}, {}, {}
    for family in regions.PRIMARY_FAMILIES:
        grid = regions.build_grid(family, (3, 7, 11), (2, 6, 10))
        means, sds = [], []
        degenerate[family.name] = sum(pt.degenerate for pt in grid)
        for pt in grid:
            if pt.degenerate:
                continue
            spec = arch.Spec(family, "primary", pt.params)
            cal = _calibrated(spec, target, n)
            if not cal.feasible:  # CI-1 gate: infeasible points are recorded and excluded
                infeasible[family.name] = infeasible.get(family.name, 0) + 1
                continue
            m, s = simulate_reference(spec, cal.heterogeneity, STAGE, n, replicates)
            means.append(m[:3])
            sds.append(s[:3])
        images[family.name] = span.SimulatedImage(family.name, np.array(means), np.array(sds))
    res = span.span_check(images["A1_PRIME"], images["A2"])
    return {
        "coverage_a1p_in_a2": res.coverage_a_in_b,
        "coverage_a2_in_a1p": res.coverage_b_in_a,
        "status": res.status,
        "degenerate_points": degenerate,
        "ci1_infeasible_points": infeasible,
        "feasible_points": {k: int(v.means.shape[0]) for k, v in images.items()},
        "note": "3 x 3 engineering grids are far too sparse for coverage; not a span result",
    }


def gate_power_analytic() -> dict:
    """P8-003 recheck: truth-point pass probability, old rule against the corrected one.

    Direct simulation uses R = 200. The emulator case uses the worst-case validated
    approximations, with the true model SD at the truth equal to SE:
    - the emulator SD overstated by 20% and inflated by g⁺, so the bound uses
      1.25·1.2·s;
    - e = 0.1·τ.

    Under the approved formula's own, more conservative design point (the inflation
    applied to the data-side spread too), the joint bound is exactly 0.80 by
    construction.
    """
    z = adq.z_equivalence()
    c = adq.resolution_constant()
    old_direct = adq.truth_pass_probability(2.0 / np.sqrt(1 + 1 / config.R_DIRECT))
    old_emulator = adq.truth_pass_probability(
        2.0 / np.sqrt(1 + 1 / config.R_EMULATOR), sd_inflation=config.G_PLUS
    )
    # Corrected rule at a floor-bound moment (τ = ρ = c·SE), emulator worst case:
    # |d| ≤ (1 − f_e)·c·SE − z·SE·sqrt(1 + (g⁺·1.2)²/R), with d ~ N(0, SE²(1 + 1/R)).
    inflation = np.sqrt(1 + (config.G_PLUS * 1.2) ** 2 / config.R_EMULATOR)
    margin = (1 - config.EMULATOR_MEAN_ERROR_FRACTION) * c - z * inflation
    sd_d = np.sqrt(1 + 1 / config.R_EMULATOR)
    new_emulator = 2 * float(norm.cdf(margin / sd_d)) - 1
    new_direct = (
        2
        * float(
            norm.cdf((c - z * np.sqrt(1 + 1 / config.R_DIRECT)) / np.sqrt(1 + 1 / config.R_DIRECT))
        )
        - 1
    )
    return {
        "c": c,
        "old_rule_single_moment_direct_R200": old_direct,
        "old_rule_single_moment_emulator": old_emulator,
        "new_rule_single_moment_direct_R200": new_direct,
        "new_rule_single_moment_emulator_worst_case": new_emulator,
        "new_rule_joint_lower_bound_emulator_worst_case": new_emulator**config.J_PRIMARY,
        "target_joint": config.GATE_TRUTH_INCLUSION_MIN,
    }


def tiny_perfect_model(n=800, replicates=4, panels=4, pseudo=5) -> dict:
    """Tiny P8-003 perfect-model exercise: an A1′ truth, the inserted truth point, and
    both families on tiny grids. SYNTHETIC; not a gate result."""
    truth = arch.a1_prime(0.9, 1.0)
    pseudo_panels = [
        world_targets(arch.simulate(truth, HET, truth_key(STAGE, "PSEUDO", 0, q), n))
        for q in range(pseudo)
    ]
    ref_se = truth_point_se(pseudo_panels)
    ref_vals = {
        j: float(np.mean([p.primary.values[j] for p in pseudo_panels])) for j in config.PRIMARY_IDS
    }
    cal_a1 = _calibrated(arch.a1(), pseudo_panels[0].ci1, n)
    a1_mean, _ = simulate_reference(arch.a1(), cal_a1.heterogeneity, STAGE, n, replicates)
    tol = adq.build_tolerances(
        TargetVector("synthetic", ref_vals, ref_se),
        dict(zip(config.PRIMARY_IDS, a1_mean, strict=True)),
    )
    worlds = [arch.simulate(truth, HET, truth_key(STAGE, "T1", 0, q), n) for q in range(panels)]
    raw = [world_targets(w) for w in worlds]
    data_se = truth_point_se(raw + pseudo_panels)  # truth-point across-panel SD (P8-003)
    results = []
    for q, w in enumerate(worlds):
        targets = world_targets(w, data_se)
        verdicts = {}
        for family in regions.PRIMARY_FAMILIES:
            grid = regions.build_grid(family, (7, 9), (3, 5))
            outcomes = [
                evaluate_point(
                    pt,
                    arch.Spec(family, "primary", pt.params),
                    targets,
                    tol,
                    STAGE,
                    n,
                    replicates,
                    2,
                )
                for pt in grid
            ]
            verdicts[family.name] = {
                k: adq.family_verdict(family.name, "primary", outcomes, k, tol)
                for k in config.KAPPAS
            }
        truth_point = evaluate_point(None, truth, targets, tol, STAGE, n, replicates, 2)
        comp = {
            k: adq.comparative_verdict(verdicts["A2"][k], verdicts["A1_PRIME"][k]).label
            for k in config.KAPPAS
        }
        results.append(
            {
                "q": q,
                "truth_point_status": truth_point.status,
                "A1_PRIME": {k: v.label for k, v in verdicts["A1_PRIME"].items()},
                "A2": {k: v.label for k, v in verdicts["A2"].items()},
                "kappa_star_A1_PRIME": verdicts["A1_PRIME"][0.25].kappa_star,
                "kappa_star_A2": verdicts["A2"][0.25].kappa_star,
                "comparative": comp,
            }
        )
    return {
        "rho": tol.rho,
        "tau": {k: tol.tau(k).tolist() for k in config.KAPPAS},
        "resolution_limited": {k: tol.resolution_limited(k) for k in config.KAPPAS},
        "kappa_res": tol.kappa_res(),
        "truth_inclusion_rate": {
            k: float(np.mean([r["truth_point_status"].get(k) == adq.ADEQ for r in results]))
            for k in config.KAPPAS
        },
        "panels": results,
    }


SYNTHETIC_FIXTURES = {
    "B1": b1_a0,
    "B2": b2_a1,
    "B3": b3_a1_prime,
    "B4": b4_a2,
    "B5": b5_hmm,
    "B6": b6_recency,
    "B7": b7_envelope,
    "B8": b8_sa,
    "B9": b9_periodic,
    "B10": b10_operator_trap,
    "B11": b11_false_attribution,
    "B12": b12_mde,
    "SECOND_AND_HIGHER_ORDER": second_order_and_higher,
    "A2_REACHABILITY": a2_reachability,
    "GATE_POWER_ANALYTIC": gate_power_analytic,
    "ENVELOPE_DIAGNOSTIC": envelope_diagnostic,
}
