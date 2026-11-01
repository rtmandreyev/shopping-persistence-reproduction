"""RQ2 truth configuration, aggregators and gates (design §AB, §AC, §AD–§AG).

Production configurations are **declarations**. Run B never executes Q = 200. The
aggregators operate on supplied verdict records, which are synthetic toy records in
Run B.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from complete_journey.abm import architectures as arch
from complete_journey.abm import config
from complete_journey.abm.adequacy import (
    ADEQUATE,
    EQUIVALENT,
    HISTORY_FAVOURED,
    INADEQUATE,
    INCONCLUSIVE,
    LATENT_FAVOURED,
    NEITHER_FAMILY,
    UNRESOLVED,
    Tolerances,
)
from complete_journey.abm.config import require

NON_HISTORY_CLASSES = ("T0", "T1", "T2")
UNDEFINED_MDE = "UNDEFINED — PROCEDURE NOT ADMISSIBLE"


@dataclass(frozen=True)
class TruthPoint:
    """One known-truth point. Amplitudes are symbolic (LOW/HIGH) until s_A1 is frozen."""

    truth_class: str
    label: str
    kind: str  # A1 | A1_PRIME | HMM | HET_RHO | AR_PERIODIC | A2 | MIXED
    params: tuple[tuple[str, object], ...]
    n: int = config.N_HOUSEHOLDS

    def param(self, name):
        return dict(self.params)[name]


def amplitude(rule: str, s_a1: float) -> float:
    """LOW = 0.5·s_A1 and HIGH = 1.0·s_A1 (§AD.3), with s_A1 from CI-1 calibration only."""
    require(rule in config.AMPLITUDE_RULE, f"unknown amplitude rule {rule}")
    require(s_a1 > 0 and math.isfinite(s_a1), "s_A1 must be a positive calibrated SD")
    return config.AMPLITUDE_RULE[rule] * s_a1


def production_truth_points() -> tuple[TruthPoint, ...]:
    pts = [TruthPoint("T0", "A1", "A1", ())]
    for rho in config.RHO_TRUTH_GRID:
        for amp in ("LOW", "HIGH"):
            pts.append(
                TruthPoint("T1", f"rho={rho},{amp}", "A1_PRIME", (("rho", rho), ("amp", amp)))
            )
    for q in config.HMM_TRUTH_Q:
        pts.append(TruthPoint("T2", f"HMM q={q}", "HMM", (("q", q), ("amp", "HIGH"))))
    for mean in config.HET_RHO_TRUTH_MEANS:
        pts.append(
            TruthPoint(
                "T2",
                f"het-rho mean={mean}",
                "HET_RHO",
                (("rho_mean", mean), ("rho_sd", config.HET_RHO_TRUTH_SD), ("amp", "HIGH")),
            )
        )
    for a in ("LOW", "HIGH"):
        pts.append(
            TruthPoint(
                "T2",
                f"AR+periodic a={a}",
                "AR_PERIODIC",
                (("rho", config.MASKING_RHO), ("amp", "HIGH"), ("periodic", a)),
            )
        )
    for lam in config.LAMBDA_TRUTH_GRID:
        for d1 in config.DELTA1_GRID_PP[1:]:
            pts.append(
                TruthPoint("T3", f"lam={lam},D1={d1}", "A2", (("lam", lam), ("delta1_pp", d1)))
            )
    for d1 in config.DELTA1_GRID_PP[1:]:
        pts.append(
            TruthPoint(
                "T4",
                f"mask D1={d1}",
                "MIXED",
                (
                    ("rho", config.MASKING_RHO),
                    ("amp", config.MASKING_AMPLITUDE),
                    ("lam", config.MASKING_LAMBDA),
                    ("delta1_pp", d1),
                ),
            )
        )
    t5 = (
        ("A2", (("lam", 0.5), ("delta1_pp", 10.0))),
        ("A2", (("lam", 0.5), ("delta1_pp", 30.0))),
        ("A1_PRIME", (("rho", 0.9), ("amp", "HIGH"))),
        (
            "MIXED",
            (("rho", 0.9), ("amp", "HIGH"), ("lam", 0.5), ("delta1_pp", 10.0)),
        ),
    )
    for k, (kind, params) in enumerate(t5, start=1):
        pts.append(TruthPoint("T5", f"T5-{k}", kind, params, n=config.N_SCALING))
    return tuple(pts)


def gate_points() -> tuple[TruthPoint, ...]:
    """Perfect-model gate (§AB.2): T0, T1 and the T3 subset λ∈{0.5,0.8} × Δ_1∈{5,15,30}."""
    return tuple(
        p
        for p in production_truth_points()
        if p.truth_class in ("T0", "T1")
        or (
            p.truth_class == "T3"
            and p.param("lam") in (0.5, 0.8)
            and p.param("delta1_pp") in (5.0, 15.0, 30.0)
        )
    )


PILOT_LABELS = ("A1", "rho=0.9,HIGH", "lam=0.5,D1=10.0", "HMM q=0.05")


def pilot_points() -> tuple[TruthPoint, ...]:
    """Stage 2 engineering pilot (§AI): four predeclared points, Q = 20."""
    by_label = {p.label: p for p in production_truth_points() if p.truth_class != "T5"}
    return tuple(by_label[label] for label in PILOT_LABELS)


def resolve_spec(point: TruthPoint, s_a1: float, delta: float | None = None):
    """Concrete architecture spec of a truth point (delta from the Stage 1 mapping)."""
    kind = point.kind
    if kind == "A1":
        return arch.a1(), 0.0
    if kind == "A1_PRIME":
        return arch.a1_prime(point.param("rho"), amplitude(point.param("amp"), s_a1)), 0.0
    if kind == "HMM":
        return arch.a1_prime_hmm(point.param("q"), amplitude(point.param("amp"), s_a1)), 0.0
    if kind == "HET_RHO":
        spec = arch.a1_prime_heterogeneous_rho(
            point.param("rho_mean"), point.param("rho_sd"), amplitude(point.param("amp"), s_a1)
        )
        return spec, 0.0
    if kind == "AR_PERIODIC":
        spec = arch.a1_prime(point.param("rho"), amplitude(point.param("amp"), s_a1))
        return spec, amplitude(point.param("periodic"), s_a1)
    require(delta is not None, "history truths need the frozen Δ_1 -> delta mapping")
    if kind == "A2":
        return arch.a2(delta, point.param("lam")), 0.0
    require(kind == "MIXED", f"unknown truth kind {kind}")
    sigma = amplitude(point.param("amp"), s_a1)
    return arch.mixed(point.param("rho"), sigma, delta, point.param("lam")), 0.0


# --- verdict records and aggregators -----------------------------------------------


@dataclass(frozen=True)
class VerdictRecord:
    truth_class: str
    label: str
    q: int
    comparative: dict[float, str]  # κ -> comparative verdict
    naive_m3: float
    naive_rec: float
    lam: float | None = None
    delta1_pp: float | None = None


def _rates(records, kappa, predicate) -> dict[str, float]:
    groups = defaultdict(list)
    for r in records:
        groups[(r.truth_class, r.label)].append(predicate(r))
    return {key: float(np.mean(v)) for key, v in groups.items()}


def fa_proc(records: list[VerdictRecord], kappa: float = config.KAPPA_REFERENCE) -> dict:
    """FA_PROC per non-history truth point, plus the maxima over T2 and over T0∪T1∪T2."""
    nh = [r for r in records if r.truth_class in NON_HISTORY_CLASSES]
    require(nh, "no non-history truth records")
    rates = _rates(nh, kappa, lambda r: r.comparative[kappa] == HISTORY_FAVOURED)
    t2 = [v for (c, _), v in rates.items() if c == "T2"]
    return {
        "per_point": rates,
        "max_t2": max(t2) if t2 else None,
        "max_non_history": max(rates.values()),
    }


def naive_flag(r: VerdictRecord) -> bool:
    """FA_NAIVE rule (§AF): NAIVE_M3 OR NAIVE_REC, with fixed archived design scales."""
    return bool(
        r.naive_m3 > config.NAIVE_Z * config.NAIVE_M3_SE
        or r.naive_rec > config.NAIVE_Z * config.NAIVE_REC_SE
    )


def fa_naive(records: list[VerdictRecord]) -> dict:
    nh = [r for r in records if r.truth_class in NON_HISTORY_CLASSES]
    require(nh, "no non-history truth records")
    return {
        "union": _rates(nh, None, naive_flag),
        "m3": _rates(nh, None, lambda r: r.naive_m3 > config.NAIVE_Z * config.NAIVE_M3_SE),
        "rec": _rates(nh, None, lambda r: r.naive_rec > config.NAIVE_Z * config.NAIVE_REC_SE),
    }


def detection(
    records: list[VerdictRecord], kappa: float = config.KAPPA_REFERENCE, truth_class: str = "T3"
) -> dict[tuple[float, float], float]:
    """DET(Δ_1, λ) = P(HISTORY_FAVOURED | history truth) per tested point."""
    hist = [r for r in records if r.truth_class == truth_class]
    groups = defaultdict(list)
    for r in hist:
        require(r.lam is not None and r.delta1_pp is not None, "history records need Δ_1, λ")
        groups[(r.lam, r.delta1_pp)].append(r.comparative[kappa] == HISTORY_FAVOURED)
    return {key: float(np.mean(v)) for key, v in sorted(groups.items())}


NO_REACHABLE = "NO REACHABLE HISTORY TRUTH"


def mde80(
    det: dict[tuple[float, float], float],
    lam: float,
    max_fa_non_history: float,
    infeasible: frozenset[float] = frozenset(),
    grid=config.DELTA1_GRID_PP[1:],
) -> float | str:
    """MDE_80(λ): monotone-envelope rule over the reachable grid, with FA admissibility.

    P8-002 rules:
    - Tested Δ_1 values above the reachable maximum are infeasible truth points. They
      must form the top of the grid, and they are excluded and recorded.
    - If detection never reaches 0.80, the result is "> X pp", where X is the largest
      reachable tested Δ_1 at that λ.
    """
    if max_fa_non_history > config.FA_ADMISSIBILITY:
        return UNDEFINED_MDE
    unknown = [d for d in grid if (lam, d) not in det and d not in infeasible]
    require(not unknown, f"MDE needs every tested Δ_1 at λ={lam}; missing {unknown}")
    reachable = [d for d in grid if d not in infeasible]
    require(
        reachable == list(grid[: len(reachable)]),
        "infeasible Δ_1 values must be the top of the grid (reachability is monotone)",
    )
    if not reachable:
        return NO_REACHABLE
    values = [det[(lam, d)] for d in reachable]
    for k, d1 in enumerate(reachable):
        if all(v >= config.DETECTION_THRESHOLD for v in values[k:]):
            return float(d1)
    return f"> {reachable[-1]:g} pp"


# --- perfect-model and emulator gates -------------------------------------------------


@dataclass(frozen=True)
class GateRecord:
    """One perfect-model truth panel (P8-003 gate inputs).

    - truth_in_acc: κ -> θ_true (inserted as an evaluation point) ∈ Acc_F(κ);
    - family_verdict: the true family's verdict at κ = 0.25 on the production grid plus
      the frozen refinement;
    - truth_in_nr: θ_true ∈ NR_F(0.25), the not-clearly-inadequate set;
    - comparative: the comparative verdict at κ = 0.25 (needed for A1 truths, G5).
    """

    label: str
    truth_class: str
    truth_in_acc: dict[float, bool]
    family_verdict: str
    truth_in_nr: bool
    comparative: str | None = None
    recovery_distance: float | None = None


def perfect_model_gate(records: list[GateRecord]) -> dict:
    """P8-003 hard gates per gate point; the gate passes only if every point passes.

    - G1: P(θ_true ∈ Acc_F(κ)) ≥ 0.80 at every κ (pointwise truth inclusion).
    - G2: the family ADEQUATE rate ≥ 0.80 at κ = 0.25.
    - G3: the true-family INADEQUATE rate ≤ 0.05.
    - G4: P(θ_true ∈ NR_F) ≥ 0.90.
    - G5: under A1 truth, P(HISTORY_FAVOURED) ≤ 0.10.
    """
    raw_family = {ADEQUATE, INADEQUATE, INCONCLUSIVE}
    raw_comparative = {
        HISTORY_FAVOURED,
        LATENT_FAVOURED,
        EQUIVALENT,
        NEITHER_FAMILY,
        UNRESOLVED,
        None,
    }
    groups = defaultdict(list)
    for r in records:
        require(set(r.truth_in_acc) == set(config.KAPPAS), "G1 needs all three κ")
        # Gates count raw verdicts. Qualified labels (e.g. "... WITHIN DESIGN RESOLUTION")
        # are reporting strings and are refused here, so they cannot silently miscount.
        require(r.family_verdict in raw_family, "gate records need raw family verdicts")
        require(r.comparative in raw_comparative, "gate records need raw comparative verdicts")
        groups[(r.truth_class, r.label)].append(r)
    report, passed = {}, True
    for key, rs in groups.items():
        g1 = {k: float(np.mean([r.truth_in_acc[k] for r in rs])) for k in config.KAPPAS}
        g2 = float(np.mean([r.family_verdict == ADEQUATE for r in rs]))
        g3 = float(np.mean([r.family_verdict == INADEQUATE for r in rs]))
        g4 = float(np.mean([r.truth_in_nr for r in rs]))
        checks = {
            "G1": all(v >= config.GATE_TRUTH_INCLUSION_MIN for v in g1.values()),
            "G2": g2 >= config.GATE_FAMILY_ADEQUATE_MIN,
            "G3": g3 <= config.GATE_TRUE_INADEQUATE_MAX,
            "G4": g4 >= config.GATE_NR_INCLUSION_MIN,
        }
        entry = {
            "G1_truth_inclusion": g1,
            "G2_family_adequate": g2,
            "G3_true_inadequate": g3,
            "G4_nr_inclusion": g4,
        }
        if key[0] == "T0":
            g5 = float(np.mean([r.comparative == HISTORY_FAVOURED for r in rs]))
            entry["G5_a1_history_favoured"] = g5
            checks["G5"] = g5 <= config.GATE_A1_HISTORY_MAX
        distances = [r.recovery_distance for r in rs if r.recovery_distance is not None]
        entry["recovery_mean"] = float(np.mean(distances)) if distances else None
        entry["recovery_sd"] = float(np.std(distances, ddof=1)) if len(distances) > 1 else None
        entry["checks"] = checks
        entry["pass"] = bool(all(checks.values()))
        report[key] = entry
        passed = passed and entry["pass"]
    return {"points": report, "pass": passed}


def emulator_gate(
    mu_emulator: np.ndarray,
    mu_direct: np.ndarray,
    s_emulator: np.ndarray,
    s_direct: np.ndarray,
    tolerances: Tolerances,
) -> dict:
    """§AC: |μ_em − μ_direct| ≤ 0.1·τ_j(0.15) and s_em ∈ [0.8, 1.2]·s_direct.

    τ_j(0.15) is derived internally from the frozen tolerances at
    ``config.EMULATOR_GATE_KAPPA``; callers cannot choose it.

    Arrays are (validation points, J). It returns e_j (the maximum absolute mean error
    per moment) for the adequacy rule, plus pass flags.
    """
    mu_e, mu_d = np.asarray(mu_emulator, float), np.asarray(mu_direct, float)
    s_e, s_d = np.asarray(s_emulator, float), np.asarray(s_direct, float)
    tau = tolerances.tau(config.EMULATOR_GATE_KAPPA)
    require(mu_e.shape == mu_d.shape == s_e.shape == s_d.shape, "emulator arrays must align")
    require(tau.shape == (mu_e.shape[1],), "τ must have one entry per moment")
    err = np.abs(mu_e - mu_d)
    mean_ok = err <= config.EMULATOR_MEAN_ERROR_FRACTION * tau[None, :]
    lo, hi = 1.0 - config.EMULATOR_SD_TOLERANCE, 1.0 + config.EMULATOR_SD_TOLERANCE
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(s_d > 0, s_e / s_d, np.where(s_e == 0, 1.0, np.inf))
    sd_ok = (ratio >= lo) & (ratio <= hi)
    return {
        "e": err.max(axis=0),
        "mean_ok": mean_ok.all(axis=0),
        "sd_ok": sd_ok.all(axis=0),
        "pass": bool(mean_ok.all() and sd_ok.all()),
    }
