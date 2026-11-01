"""Static estimands on the artificial policy and the frozen validity rule (P9-001, P9-002).

Every history intervention flips whole coherent 4-week vectors. The displayed count
and the last-week line are re-derived from h, so no impossible counterfactual state
exists.

**Estimands.** Each static estimand is a fixed linear combination sum_s c_s *
pi_bar(s) of per-state template-A means (3 repeats). Its value and its repeat-noise
standard error come from the same coefficients:
SE = sigma_rep / sqrt(3) * sqrt(sum_s c_s^2), assuming independent response noise
across states. This is a CANDIDATE analysis rule for the pre-smoke freeze. The label
is POSITIVE or NEGATIVE when |estimate| > 3 SE, NULL otherwise, and UNDEFINED when a
value is undefined under the convention. Values are reported in probability units and
in percentage points; Δ1 uses Phase 8's Δ_1 percentage-point scale.

**Validity rule** (P9-001, with P9-002 §2 for the placebo term). A visibility contrast
is INTERPRETABLE only if its paired per-state |Δp| exceeds all three of the following;
otherwise it is WORDING-DOMINATED.
1. **The paraphrase-induced |Δp|:** a paired mean over the contrast's states, whose
   one-sided 95% bootstrap lower bound (stratified by profile band) must be > 0.
2. **The placebo-induced |Δp|:** paired on placebo-covered states, weighted to the
   contrast's full state population with the P9-002 inverse-inclusion weights. The
   bootstrap is stratified by P9-002 stratum.
3. **3 × the pooled repeat SD:** the mean |Δp| must exceed it.

The per-state exceedance shares are reported alongside, as diagnostics. A value that
is undefined under the convention makes the label UNDEFINED. A value missing from the
archive is an error, never a label.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from complete_journey.systemone import lattice as L
from complete_journey.systemone import settings as S
from complete_journey.systemone.closedloop import DGP, profile_mass, stationary_distribution
from complete_journey.systemone.policy import PolicyData
from complete_journey.systemone.seeds import rng
from complete_journey.systemone.settings import require
from complete_journey.systemone.state import SemanticState

INTERPRETABLE, WORDING_DOMINATED, UNDEFINED = "INTERPRETABLE", "WORDING-DOMINATED", "UNDEFINED"
POSITIVE, NEGATIVE, NULL = "POSITIVE", "NEGATIVE", "NULL"
CONVENTION_DEPENDENT = "CONVENTION-DEPENDENT"
SE_MULTIPLE = 3.0

CONTRASTS = {
    "history_visibility|need_hidden": (S.HISTORY, S.NEITHER),
    "history_visibility|need_visible": (S.BOTH, S.LATENT),
    "need_visibility|history_hidden": (S.LATENT, S.NEITHER),
    "need_visibility|history_visible": (S.BOTH, S.HISTORY),
}


class Lin:
    """A linear combination of per-state values."""

    def __init__(self):
        self.c: dict[SemanticState, float] = defaultdict(float)

    def add(self, s: SemanticState, w: float) -> None:
        self.c[s] += w

    def __sub__(self, other: Lin) -> Lin:
        out = Lin()
        for s, w in self.c.items():
            out.add(s, w)
        for s, w in other.c.items():
            out.add(s, -w)
        return out

    def value(self, fn) -> float:
        return float(sum(w * fn(s) for s, w in self.c.items()))

    def se(self, sigma: float, repeats: int = S.REPEATS_A) -> float:
        return float(sigma / np.sqrt(repeats) * np.sqrt(sum(w * w for w in self.c.values())))


def base_state(s: SemanticState, base_cell: str) -> SemanticState:
    hv, nv = S.CELL_VISIBILITY[base_cell]
    return SemanticState(base_cell, s.profile, s.history if hv else None, s.need if nv else None)


def sign_label(value: float, se: float) -> str:
    if not np.isfinite(value):
        return UNDEFINED
    if not np.isfinite(se):
        return NULL
    return POSITIVE if value > SE_MULTIPLE * se else NEGATIVE if value < -SE_MULTIPLE * se else NULL


def _mass(k: int) -> dict[float, float]:
    return dict(zip(L.grid(k), profile_mass(k).tolist(), strict=True))


def flip1(h: tuple[int, ...], v: int) -> tuple[int, ...]:
    return (v, *h[1:])


# --- estimand coefficient builders ----------------------------------------------------------


def lin_delta1(k: int, cell: str, profile_stated: bool = True) -> Lin:
    """Mean one-week effect with design weights: profile mass x uniform (h2..h4, need)."""
    mass = _mass(k)
    _, nv = S.CELL_VISIBILITY[cell]
    profiles = list(L.grid(k)) if profile_stated else [None]
    terms = [
        (f, h, n)
        for f in profiles
        for h in S.HISTORY_PATTERNS
        if h[0] == 0
        for n in (S.NEED_LEVELS if nv else [None])
    ]
    tot = sum(mass[f] if f is not None else 1.0 for f, _, _ in terms)
    lin = Lin()
    for f, h, n in terms:
        w = (mass[f] if f is not None else 1.0) / tot
        lin.add(SemanticState(cell, f, flip1(h, 1), n), w)
        lin.add(SemanticState(cell, f, h, n), -w)
    return lin


def lin_ahe(pd: PolicyData, cell: str, lam: float | None, convention: str) -> Lin:
    """Average one-week effect over the closed-loop stationary law of (h2..h4, need) at
    each profile level, weighted by profile mass. The stationary weights are taken from
    the policy's own table and treated as fixed."""
    k = pd.k
    mass = profile_mass(k)
    dgp = DGP(cell, lam)
    tab = pd.table(cell, convention)
    _, nv = S.CELL_VISIBILITY[cell]
    lin = Lin()
    for lv, f in enumerate(L.grid(k)):
        st, nn = stationary_distribution(tab, lv, dgp, k)
        ctx = defaultdict(float)
        for hi, h in enumerate(S.HISTORY_PATTERNS):
            for ni in range(nn):
                ctx[(h[1:], ni)] += st[hi * nn + ni]
        for (tail, ni), w in ctx.items():
            n = S.NEED_LEVELS[ni] if nv else None
            lin.add(SemanticState(cell, f, (1, *tail), n), mass[lv] * w)
            lin.add(SemanticState(cell, f, (0, *tail), n), -mass[lv] * w)
    return lin


def ahe_from_table(tab: np.ndarray, k: int, cell: str, lam: float | None) -> float:
    """Installed AHE of any table (the raw, calibrated or surrogate variant)."""
    mass = profile_mass(k)
    dgp = DGP(cell, lam)
    out = 0.0
    for lv in range(len(L.grid(k))):
        st, nn = stationary_distribution(tab, lv, dgp, k)
        for hi, h in enumerate(S.HISTORY_PATTERNS):
            for ni in range(nn):
                j = ni if nn == 3 else 0
                out += (
                    mass[lv]
                    * st[hi * nn + ni]
                    * (
                        tab[lv, S.HISTORY_PATTERNS.index(flip1(h, 1)), j]
                        - tab[lv, S.HISTORY_PATTERNS.index(flip1(h, 0)), j]
                    )
                )
    return float(out)


def lin_delta4(k: int, cell: str) -> Lin:
    mass = _mass(k)
    _, nv = S.CELL_VISIBILITY[cell]
    needs = S.NEED_LEVELS if nv else [None]
    lin = Lin()
    for f in L.grid(k):
        for n in needs:
            w = mass[f] / len(needs)
            lin.add(SemanticState(cell, f, (1, 1, 1, 1), n), w)
            lin.add(SemanticState(cell, f, (0, 0, 0, 0), n), -w)
    return lin


def lin_delta_need(k: int, cell: str) -> Lin:
    """pi(high) - pi(low): profile mass x uniform history (BOTH)."""
    mass = _mass(k)
    hv, _ = S.CELL_VISIBILITY[cell]
    hists = S.HISTORY_PATTERNS if hv else [None]
    lin = Lin()
    for f in L.grid(k):
        for h in hists:
            w = mass[f] / len(hists)
            lin.add(SemanticState(cell, f, h, "high"), w)
            lin.add(SemanticState(cell, f, h, "low"), -w)
    return lin


def type_probe(pd: PolicyData) -> dict:
    """Negative-control outcome: change in usual_type under history interventions with
    the profile stated. Reported overall and by profile level (it is pinned at extreme
    levels, so the levels near 0.5 carry most of the information)."""
    k = pd.k
    mass = _mass(k)
    sigma = pd.repeat_sd(question=S.Q_TYPE)
    out = {"sigma_rep_type": sigma}
    for cell in (S.HISTORY, S.BOTH):
        _, nv = S.CELL_VISIBILITY[cell]
        needs = S.NEED_LEVELS if nv else [None]
        four = Lin()
        by_level = {}
        for f in L.grid(k):
            lv = Lin()
            for n in needs:
                four.add(SemanticState(cell, f, (1, 1, 1, 1), n), mass[f] / len(needs))
                four.add(SemanticState(cell, f, (0, 0, 0, 0), n), -mass[f] / len(needs))
                lv.add(SemanticState(cell, f, (1, 1, 1, 1), n), 1.0 / len(needs))
                lv.add(SemanticState(cell, f, (0, 0, 0, 0), n), -1.0 / len(needs))
            v, se = lv.value(pd.type_probe), lv.se(sigma)
            by_level[f"{f:.2f}"] = {"shift": v, "se": se, "label": sign_label(v, se)}
        val, se = four.value(pd.type_probe), four.se(sigma)
        out[cell] = {"four_week": val, "se": se, "label": sign_label(val, se), "by_level": by_level}
    return out


def base_rate_fidelity(pd: PolicyData, convention: str) -> dict:
    from scipy.stats import spearmanr

    g = np.array(L.grid(pd.k))
    pi = np.array([pd.pi(SemanticState(S.NEITHER, f, None, None), convention) for f in g])
    ok = np.isfinite(pi)
    slope, intercept = np.polyfit(g[ok], pi[ok], 1) if ok.sum() >= 2 else (np.nan, np.nan)
    return {
        "stated": g.tolist(),
        "policy": pi.tolist(),
        "mean_abs_error": float(np.mean(np.abs(pi[ok] - g[ok]))) if ok.any() else float("nan"),
        "max_abs_error": float(np.max(np.abs(pi[ok] - g[ok]))) if ok.any() else float("nan"),
        "slope": float(slope),
        "intercept": float(intercept),
        "spearman": float(spearmanr(g[ok], pi[ok]).statistic) if ok.sum() >= 3 else float("nan"),
    }


def coherence(pd: PolicyData) -> dict:
    gaps = defaultdict(list)
    for s in L.all_states(pd.k):
        gaps[s.cell].append(pd.coherence_gap(s))
    allg = [x for v in gaps.values() for x in v]
    return {
        "mean_gap": float(np.mean(allg)),
        "mean_abs_gap": float(np.mean(np.abs(allg))),
        "max_abs_gap": float(np.max(np.abs(allg))),
        "by_cell": {c: float(np.mean(v)) for c, v in gaps.items()},
    }


# --- validity rule ----------------------------------------------------------------------------


def _strat_boot_lcb(
    d: np.ndarray, strata: list, weights: np.ndarray, seed_coords: tuple[int, ...]
) -> float:
    g = rng("bootstrap", *seed_coords)
    groups = defaultdict(list)
    for i, st in enumerate(strata):
        groups[repr(st)].append(i)
    idx = [np.array(v) for v in groups.values()]
    means = np.empty(S.BOOTSTRAP_DRAWS)
    for b in range(S.BOOTSTRAP_DRAWS):
        pick = np.concatenate([gi[g.integers(0, len(gi), len(gi))] for gi in idx])
        means[b] = np.average(d[pick], weights=weights[pick])
    return float(np.quantile(means, S.BOOTSTRAP_ALPHA))


def validity(
    pd: PolicyData, contrast: str, convention: str, sigma_rep: float, coords: tuple[int, ...] = (0,)
) -> dict:
    cell, base_cell = CONTRASTS[contrast]
    k = pd.k
    states = L.cell_states(cell, k)
    e = np.array(
        [abs(pd.pi(s, convention) - pd.pi(base_state(s, base_cell), convention)) for s in states]
    )
    para = np.array([abs(pd.pi(s, convention) - pd.paraphrase(s, convention)) for s in states])
    weights = L.stratum_weights(L.placebo_frame(k), k)
    cov = [i for i, s in enumerate(states) if s.key() in weights]
    plc = []
    for i in cov:
        v = pd.placebo(states[i], convention)
        require(v is not None, f"placebo answer missing for covered state {states[i].key()}")
        plc.append(v)
    q = np.abs(np.array([pd.pi(states[i], convention) for i in cov]) - np.array(plc))
    out = {
        "states": len(states),
        "covered_states": len(cov),
        "mean_abs_effect": float(np.nanmean(e)),
    }
    if not (
        np.isfinite(e).all()
        and np.isfinite(para).all()
        and np.isfinite(q).all()
        and np.isfinite(sigma_rep)
    ):
        out["label"] = UNDEFINED
        out["reason"] = "a value is undefined under this convention"
        return out
    bands = [L.band(s.profile) for s in states]
    d1 = e - para
    lcb1 = _strat_boot_lcb(d1, bands, np.ones(len(d1)), (*coords, 1))
    ok1 = bool(d1.mean() > 0 and lcb1 > 0)
    d2 = e[cov] - q
    w = np.array([weights[states[i].key()] for i in cov])
    mean2 = float(np.average(d2, weights=w))
    lcb2 = _strat_boot_lcb(d2, [L.stratum(states[i]) for i in cov], w, (*coords, 2))
    ok2 = bool(mean2 > 0 and lcb2 > 0)
    thr = S.REPEAT_SD_MULTIPLE * sigma_rep
    ok3 = bool(e.mean() > thr)
    out.update(
        {
            "label": INTERPRETABLE if (ok1 and ok2 and ok3) else WORDING_DOMINATED,
            "paraphrase": {
                "mean_abs": float(para.mean()),
                "mean_diff": float(d1.mean()),
                "lcb95": lcb1,
                "pass": ok1,
                "share_states_exceeding": float(np.mean(e > para)),
            },
            "placebo": {
                "weighted_mean_diff": mean2,
                "lcb95": lcb2,
                "pass": ok2,
                "weighted_share_states_exceeding": float(np.average(e[cov] > q, weights=w)),
            },
            "repeat": {
                "sigma_rep": sigma_rep,
                "threshold": thr,
                "pass": ok3,
                "share_states_exceeding": float(np.mean(e > thr)),
            },
        }
    )
    return out


# --- assembly ---------------------------------------------------------------------------------------


def static_report(pd: PolicyData, convention: str, lams=S.NEED_LAMBDAS) -> dict:
    k = pd.k
    sigma = pd.repeat_sd(convention)
    dropped = pd.repeat_sd_dropped
    lins = {
        "delta1_history": lin_delta1(k, S.HISTORY),
        "delta1_both": lin_delta1(k, S.BOTH),
        "delta1_history_profile_unavailable": lin_delta1(k, S.HISTORY, profile_stated=False),
        "ahe_history": lin_ahe(pd, S.HISTORY, None, convention),
        **{f"ahe_both_lam{lam:g}": lin_ahe(pd, S.BOTH, lam, convention) for lam in lams},
        "delta4_history": lin_delta4(k, S.HISTORY),
        "delta4_both": lin_delta4(k, S.BOTH),
        "delta_need_latent": lin_delta_need(k, S.LATENT),
        "delta_need_both": lin_delta_need(k, S.BOTH),
    }
    lins["explaining_away_need"] = lins["delta1_history"] - lins["delta1_both"]
    lins["explaining_away_profile"] = (
        lins["delta1_history_profile_unavailable"] - lins["delta1_history"]
    )

    def fn(s):
        return pd.pi(s, convention)

    def lg(s):
        p = min(max(pd.pi(s, convention), 1e-6), 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    est = {name: lin.value(fn) for name, lin in lins.items()}
    se = {name: lin.se(sigma) for name, lin in lins.items()}
    return {
        "convention": convention,
        "sigma_rep": sigma,
        "sigma_rep_states_dropped_undefined": dropped,
        "estimands": est,
        "estimands_pp": {kk: 100 * v for kk, v in est.items()},
        "estimand_se": se,
        "estimand_labels": {name: sign_label(est[name], se[name]) for name in lins},
        "label_rule": f"|estimate| > {SE_MULTIPLE:g} x propagated repeat-noise SE (CANDIDATE)",
        "descriptive_logit_scale": {
            "logit_delta1_history": lins["delta1_history"].value(lg),
            "logit_delta1_both": lins["delta1_both"].value(lg),
            "logit_explaining_away_need": lins["explaining_away_need"].value(lg),
            "logit_explaining_away_profile": lins["explaining_away_profile"].value(lg),
        },
        "validity": {c: validity(pd, c, convention, sigma, (i,)) for i, c in enumerate(CONTRASTS)},
        "type_probe": type_probe(pd),
        "base_rate_fidelity": base_rate_fidelity(pd, convention),
        "coherence": coherence(pd),
    }


def convention_dependence(rep_primary: dict, rep_norm: dict) -> dict:
    """CONVENTION-DEPENDENT wherever a substantive label differs between conventions."""
    out = {}
    for k_, v in rep_primary["estimand_labels"].items():
        w = rep_norm["estimand_labels"].get(k_)
        out[f"estimand:{k_}"] = v if v == w else CONVENTION_DEPENDENT
    for c, v in rep_primary["validity"].items():
        w = rep_norm["validity"][c]["label"]
        out[f"validity:{c}"] = v["label"] if v["label"] == w else CONVENTION_DEPENDENT
    return out
