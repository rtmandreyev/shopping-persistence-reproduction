"""P8-005 representational exclusion, conditional RQ2 reporting and heterogeneity labels.

**Classification (§P8-005.4).** For a truth T with population primary moments M_T
(estimated as the mean over its Q truth panels, Monte Carlo SE s_T,j/√Q) and a
competing family C, search only Θ_C^CI, C's CI-feasible evaluated set. At each
θ ∈ Θ_C^CI the validated emulator gives μ_C,j(θ) and s_C,j(θ) from R_em = 50 replicates,
with error budget e_j = 0.1·τ_j(0.15). With D_j(θ) = |μ_C,j(θ) − M_T,j| and
σ±_pop,j(θ) = √(s_T,j²/Q + (g±·s_C,j(θ))²/R_em):

- REPRESENTABLE: some θ has D_j + e_j + z_0.95·σ⁺_pop,j ≤ τ_j(κ) for every j;
- EXCLUDED: every θ has, for some j, D_j − e_j − z*·σ⁻_pop,j > τ_j(κ),
  z* = Φ⁻¹(1 − 0.05/(2J)); an empty Θ_C^CI makes every truth EXCLUDED;
- INDETERMINATE: otherwise; never forced to a binary answer.

The object compares population moments. It carries no single-panel sampling noise
(SE_j of the finite-sample procedure) and no span-coverage count. Each state is
computed per κ.

**Conditional reporting (§P8-005.5).** FA_PROC_OVERALL/REP, DET_OVERALL/REP and the
REPRESENTABLE / EXCLUDED / INDETERMINATE shares. INDETERMINATE truths are reported
separately and never pooled into REP. MDE_80 is unchanged (``rq2.mde80``);
MDE_80_DIAG is secondary.

**Labels (§P8-005.8).** ROBUST TO FROZEN HETEROGENEITY BRACKET, HETEROGENEITY-DEPENDENT
or HETEROGENEITY BRACKET UNAVAILABLE. No preferred arm is chosen.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from complete_journey.abm import config
from complete_journey.abm.adequacy import HISTORY_FAVOURED, Tolerances, z_equivalence, z_inadequacy
from complete_journey.abm.config import require
from complete_journey.abm.rq2 import NON_HISTORY_CLASSES, UNDEFINED_MDE, VerdictRecord

REPRESENTABLE, EXCLUDED, INDETERMINATE = "REPRESENTABLE", "EXCLUDED", "INDETERMINATE"
STATES = (REPRESENTABLE, EXCLUDED, INDETERMINATE)
INFEASIBLE_TRUTH = "INFEASIBLE TRUTH"
NO_REPRESENTABLE = "NO REPRESENTABLE HISTORY TRUTH"
ROBUST = "ROBUST TO FROZEN HETEROGENEITY BRACKET"
HET_DEPENDENT = "HETEROGENEITY-DEPENDENT"
BRACKET_UNAVAILABLE = "HETEROGENEITY BRACKET UNAVAILABLE"


@dataclass(frozen=True)
class TruthMoments:
    """Population primary moments of one truth point: mean and SD over its Q panels."""

    mean: np.ndarray  # (J,)
    sd: np.ndarray  # (J,)
    q: int

    def __post_init__(self):
        m, s = np.asarray(self.mean, float), np.asarray(self.sd, float)
        require(m.shape == s.shape == (config.J_PRIMARY,), "truth moments must hold P1..P5")
        require(np.isfinite(m).all() and (s >= 0).all(), "truth moments must be finite")
        require(self.q >= 2, "truth moments need at least two panels")
        object.__setattr__(self, "mean", m)
        object.__setattr__(self, "sd", s)


@dataclass(frozen=True)
class CompetitorImage:
    """Emulated population image of the competing family over Θ_C^CI only.

    ``mean`` and ``sd`` are (points, J). ``r_em`` is the emulator replicate count.
    The rows must be CI-1-feasible points; an empty image is allowed (EXCLUDED).
    """

    mean: np.ndarray
    sd: np.ndarray
    r_em: int = config.R_EMULATOR

    def __post_init__(self):
        m = np.asarray(self.mean, float).reshape(-1, config.J_PRIMARY)
        s = np.asarray(self.sd, float).reshape(-1, config.J_PRIMARY)
        require(m.shape == s.shape, "image mean and SD must align")
        require(np.isfinite(m).all() and (s >= 0).all(), "image values must be finite")
        require(self.r_em >= 2, "the emulator needs at least two replicates")
        object.__setattr__(self, "mean", m)
        object.__setattr__(self, "sd", s)


def classify(
    truth: TruthMoments,
    image: CompetitorImage,
    tolerances: Tolerances,
    emulator_error: np.ndarray,
    kappa: float,
) -> str:
    """Population-level three-state classification at one κ."""
    e = np.asarray(emulator_error, float)
    require(e.shape == (config.J_PRIMARY,) and (e >= 0).all(), "e_j must hold P1..P5")
    if image.mean.shape[0] == 0:
        return EXCLUDED
    tau = tolerances.tau(kappa)[None, :]
    dist = np.abs(image.mean - truth.mean[None, :])
    mc_truth = truth.sd[None, :] ** 2 / truth.q
    sig_plus = np.sqrt(mc_truth + (config.G_PLUS * image.sd) ** 2 / image.r_em)
    sig_minus = np.sqrt(mc_truth + (config.G_MINUS * image.sd) ** 2 / image.r_em)
    inside = (dist + e[None, :] + z_equivalence() * sig_plus <= tau).all(axis=1)
    if inside.any():
        return REPRESENTABLE
    outside = (dist - e[None, :] - z_inadequacy() * sig_minus > tau).any(axis=1)
    if outside.all():
        return EXCLUDED
    return INDETERMINATE


def classify_all_kappas(
    truth: TruthMoments, image: CompetitorImage, tolerances: Tolerances, emulator_error
) -> dict[float, str]:
    return {k: classify(truth, image, tolerances, emulator_error, k) for k in config.KAPPAS}


# --- conditional RQ2 reporting -------------------------------------------------------

StateMap = dict[tuple[str, str], dict[float, str]]  # (truth_class, label) -> κ -> state


def _state(states: StateMap, key: tuple[str, str], kappa: float) -> str:
    require(key in states, f"no representational state for truth {key}")
    value = states[key]
    if value == INFEASIBLE_TRUTH:
        return INFEASIBLE_TRUTH
    require(kappa in value, f"truth {key} has no state at κ={kappa}")
    require(value[kappa] in STATES, f"unknown state {value[kappa]}")
    return value[kappa]


def shares(states: StateMap, keys, kappa: float) -> dict:
    """Counts and shares of REPRESENTABLE / EXCLUDED / INDETERMINATE over truth points.

    INFEASIBLE truths are counted separately and are not in the share denominator.
    """
    counts = {s: 0 for s in STATES}
    infeasible = 0
    for key in keys:
        st = _state(states, key, kappa)
        if st == INFEASIBLE_TRUTH:
            infeasible += 1
        else:
            counts[st] += 1
    total = sum(counts.values())
    return {
        "counts": counts,
        "shares": {s: (counts[s] / total if total else None) for s in STATES},
        "assessed_points": total,
        "infeasible_truths": infeasible,
    }


def fa_proc_conditional(
    records: list[VerdictRecord], states: StateMap, kappa: float = config.KAPPA_REFERENCE
) -> dict:
    """FA_PROC_OVERALL and FA_PROC_REP (non-history truths; competing family A2).

    REP keeps only truth points A2 can REPRESENT at κ. INDETERMINATE points are never
    pooled into REP; they appear only in the shares and the overall rate.
    """
    nh = [r for r in records if r.truth_class in NON_HISTORY_CLASSES]
    require(nh, "no non-history truth records")
    groups = defaultdict(list)
    for r in nh:
        groups[(r.truth_class, r.label)].append(r.comparative[kappa] == HISTORY_FAVOURED)
    rates = {key: float(np.mean(v)) for key, v in groups.items()}
    rep = {k: v for k, v in rates.items() if _state(states, k, kappa) == REPRESENTABLE}
    t2 = [v for (c, _), v in rates.items() if c == "T2"]
    t2_rep = [v for (c, _), v in rep.items() if c == "T2"]
    return {
        "kappa": kappa,
        "FA_PROC_OVERALL": {
            "per_point": rates,
            "max_t2": max(t2) if t2 else None,
            "max_non_history": max(rates.values()),
        },
        "FA_PROC_REP": {
            "per_point": rep,
            "max_t2": max(t2_rep) if t2_rep else None,
            "max_non_history": max(rep.values()) if rep else None,
        },
        "shares": shares(
            states,
            sorted(set(rates) | {k for k in states if k[0] in NON_HISTORY_CLASSES}),
            kappa,
        ),
    }


def det_conditional(
    records: list[VerdictRecord],
    states: StateMap,
    kappa: float = config.KAPPA_REFERENCE,
    truth_class: str = "T3",
) -> dict:
    """DET_OVERALL and DET_REP (history truths; competing family A1′)."""
    hist = [r for r in records if r.truth_class == truth_class]
    require(hist, f"no {truth_class} records")
    groups, labels = defaultdict(list), {}
    for r in hist:
        require(r.lam is not None and r.delta1_pp is not None, "history records need Δ_1, λ")
        groups[(r.lam, r.delta1_pp)].append(r.comparative[kappa] == HISTORY_FAVOURED)
        labels[(r.lam, r.delta1_pp)] = (r.truth_class, r.label)
    det = {key: float(np.mean(v)) for key, v in sorted(groups.items())}
    rep = {k: v for k, v in det.items() if _state(states, labels[k], kappa) == REPRESENTABLE}
    return {
        "kappa": kappa,
        "DET_OVERALL": det,
        "DET_REP": rep,
        "shares": shares(
            states,
            sorted(set(labels.values()) | {k for k in states if k[0] == truth_class}),
            kappa,
        ),
    }


def mde80_diag(
    det: dict[tuple[float, float], float],
    lam: float,
    max_fa_non_history: float,
    representable: frozenset[float],
    grid=config.DELTA1_GRID_PP[1:],
) -> float | str:
    """SECONDARY MDE_80_DIAG(λ): the frozen monotone-envelope rule restricted to the
    tested Δ_1 values whose truths A1′ can REPRESENT. Admissibility is unchanged."""
    if max_fa_non_history > config.FA_ADMISSIBILITY:
        return UNDEFINED_MDE
    tested = [d for d in grid if d in representable]
    if not tested:
        return NO_REPRESENTABLE
    missing = [d for d in tested if (lam, d) not in det]
    require(not missing, f"MDE_80_DIAG needs DET at every representable Δ_1; missing {missing}")
    values = [det[(lam, d)] for d in tested]
    for k, d1 in enumerate(tested):
        if all(v >= config.DETECTION_THRESHOLD for v in values[k:]):
            return float(d1)
    return f"> {tested[-1]:g} pp"


def annotate_mde_grid(
    states: StateMap, labels: dict[tuple[float, float], tuple[str, str]], kappa: float
) -> dict[tuple[float, float], str]:
    """Representational status of every MDE grid point (annotation of MDE_80)."""
    return {key: _state(states, label, kappa) for key, label in sorted(labels.items())}


# --- heterogeneity labels -------------------------------------------------------------


def heterogeneity_label(primary: str, mixture: str | None, bracket_available: bool) -> str:
    """Comparative conclusion at the reference κ in each arm → bracket label."""
    if not bracket_available or mixture is None:
        return BRACKET_UNAVAILABLE
    return ROBUST if primary == mixture else HET_DEPENDENT
