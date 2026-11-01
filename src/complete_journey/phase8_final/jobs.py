"""Atomic, checkpointed jobs of the Phase 8 final run (P8-006).

Every job is deterministic: its result depends only on its key (seed streams keyed by
role, stage, replicate and truth), never on worker count or order. The model side
uses the S1-P8005 seed semantics (stage key 11, CALIBRATION and MODEL roles), so the
R = 50 → 200 extension continues the imported streams and refinement images are
drawn exactly as S1A drew the coarse ones.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from complete_journey.abm import architectures as arch
from complete_journey.abm import config as abm_config
from complete_journey.abm import mapping, mixture, regions, rq2
from complete_journey.abm import stage1_p8005 as s1p
from complete_journey.abm.calibration import calibrate
from complete_journey.abm.checkpoint import JobKey
from complete_journey.abm.config import Phase8ContractError, require
from complete_journey.abm.emulator import LOGIT_NORMAL, MIXTURE
from complete_journey.abm.guards import TargetVector
from complete_journey.abm.moments import ci1_fast, primary_moments
from complete_journey.abm.operators import world_panel
from complete_journey.abm.representability import INFEASIBLE_TRUTH
from complete_journey.abm.seeds import Family, Role, SeedKey, truth_key
from complete_journey.paths import PROJECT_ROOT
from complete_journey.phase8_final import settings as S
from complete_journey.phase8_final.store import legacy_load

ARMS = (LOGIT_NORMAL, MIXTURE)
FAMILIES = ("A1_PRIME", "A2")
N = abm_config.N_HOUSEHOLDS
FEASIBLE = "FEASIBLE"
OUTSIDE_LATTICE = "θ_true OUTSIDE SEARCH LATTICE (RQ2 truth; labelled, not inserted)"
#: Expected P-core CPU seconds per job (Run C probe and S1-P8005 checkpoints); used only
#: for progress weights and ETA, never for any scientific decision.
COST = {
    ("ext", LOGIT_NORMAL): 160,
    ("ext", MIXTURE): 160,
    ("ref", LOGIT_NORMAL): 113,
    ("ref", MIXTURE): 389,
    ("theta", LOGIT_NORMAL): 214,
    ("theta", MIXTURE): 214,
    ("gpanels", LOGIT_NORMAL): 214,
    ("gpanels", MIXTURE): 214,
    ("truth", LOGIT_NORMAL): 150,
    ("truth", MIXTURE): 550,
    ("panels", LOGIT_NORMAL): 22,
    ("panels", MIXTURE): 22,
    ("calnoise", LOGIT_NORMAL): 112,
    ("calnoise", MIXTURE): 389,
    ("audit", LOGIT_NORMAL): 160,
    ("audit", MIXTURE): 160,
    ("ref2", LOGIT_NORMAL): 113,
    ("ref2", MIXTURE): 389,
    ("sa", LOGIT_NORMAL): 120,
    ("saref", LOGIT_NORMAL): 120,
    ("_eng", LOGIT_NORMAL): 1,
}


@dataclass(frozen=True)
class FJob:
    kind: str
    arm: str
    family: str = ""
    a: float = -1.0
    b: float = -1.0
    index: int = 0
    sub: int = 0
    block: str = ""

    def key(self) -> JobKey:
        point = (
            f"FINAL:{self.kind}:{self.arm}:{self.a:g},{self.b:g}:{self.index}:"
            f"{self.sub}:{self.block}"
        )
        stage = S.RQ2_DATA_STAGE if self.kind == "panels" else S.MODEL_STAGE
        seeds = tuple(
            (abm_config.SEED_NAMESPACE, stage, int(r))
            for r in (Role.DATA, Role.MODEL, Role.CALIBRATION)
        )
        return JobKey(stage, self.family or "TRUTH", point, 0, 1, seeds)

    @property
    def cost(self) -> float:
        return float(COST.get((self.kind, self.arm), 100))

    def label(self) -> dict:
        out = {"kind": self.kind, "arm": self.arm}
        if self.family:
            out["family"] = self.family
        if self.kind in ("truth", "panels", "theta", "gpanels", "calnoise"):
            p = rq2.production_truth_points()[self.index]
            out["truth_class"] = p.truth_class
        return out


# --- shared helpers ----------------------------------------------------------------


def a1_het(arm: str = LOGIT_NORMAL):
    return s1p.het_from_payload(legacy_load(s1p.Job("a1", arm, "A1"))["het"])


def model_rows(spec, het, replicates, periodic=0.0, availability=None):
    """Per-replicate P1..P5 at (spec, het) over MODEL replicates (stage 11 streams)."""
    rows, failed = [], []
    for r in replicates:
        world = arch.simulate(
            spec, het, SeedKey(S.MODEL_STAGE, Role.MODEL, replicate=r), N, availability, periodic
        )
        try:
            p = primary_moments(world_panel(world)).values
        except (Phase8ContractError, ValueError, np.linalg.LinAlgError):
            failed.append(r)
            continue
        rows.append([p[j] for j in abm_config.PRIMARY_IDS])
    return rows, failed


def summarize(rows) -> dict:
    arr = np.asarray(rows, float)
    require(arr.shape[0] >= 2, "fewer than two successful replicates")
    return {
        "primary_mean": dict(zip(abm_config.PRIMARY_IDS, arr.mean(0).tolist(), strict=True)),
        "primary_sd": dict(zip(abm_config.PRIMARY_IDS, arr.std(0, ddof=1).tolist(), strict=True)),
        "replicates_ok": int(arr.shape[0]),
    }


UNDEFINED_PENALTY = 1e12


def sa_calibrate(spec, target: TargetVector, start, availability) -> dict:
    """The frozen primary calibrator (calibration.calibrate) with one mechanical
    completion for the S-a operator: a simplex point whose simulated worlds have no
    CI-1-eligible household (CI-1 undefined) gets the objective +inf (a large finite
    penalty) instead of raising. Wherever CI-1 is defined the objective, and therefore
    the search and its result, are identical to the frozen calibrator (pinned by a test).
    """
    from scipy.optimize import minimize
    from scipy.special import logit

    from complete_journey.abm.calibration import (
        INITIAL_SIMPLEX_SCALES,
        NM_OPTIONS,
        simulated_ci1,
        target_arrays,
    )
    from complete_journey.abm.guards import require_run_stage, require_target_allowed

    require_run_stage(S.MODEL_STAGE)
    require_target_allowed(target, "ci1_calibration")
    values, se = target_arrays(target)
    weights = 1.0 / se**2
    homogeneous = spec.family == Family.A0
    if start is None:
        start = arch.Heterogeneity(float(logit(np.clip(values[0], 1e-3, 1 - 1e-3))), 1.0)
    count = {"n": 0, "undefined": 0}

    def unpack(x):
        return arch.Heterogeneity(float(x[0]), 0.0 if homogeneous else float(np.exp(x[1])))

    def objective(x):
        count["n"] += 1
        try:
            sim = simulated_ci1(
                spec, unpack(x), S.MODEL_STAGE, N, abm_config.R_CAL, availability, 0.0
            )
        except Phase8ContractError:
            count["undefined"] += 1
            return UNDEFINED_PENALTY
        return float(np.sum(weights * (sim - values) ** 2))

    x0 = np.array([start.mu]) if homogeneous else np.array([start.mu, np.log(max(start.s, 1e-3))])
    result = None
    for scale in INITIAL_SIMPLEX_SCALES:
        simplex = np.vstack([x0, x0 + scale * np.eye(len(x0))])
        result = minimize(
            objective, x0, method="Nelder-Mead", options={**NM_OPTIONS, "initial_simplex": simplex}
        )
        x0 = result.x
    het = unpack(result.x)
    try:
        sim = simulated_ci1(spec, het, S.MODEL_STAGE, N, abm_config.R_CAL, availability, 0.0)
        residuals = sim - values
        bounds = np.maximum(abm_config.CI1_GATE_ABS, 2.0 * se)
        feasible = bool(np.all(np.abs(residuals) <= bounds))
    except Phase8ContractError:
        residuals, feasible = np.full(6, np.inf), False
    return {
        "het": s1p.het_payload(het),
        "ci1_residuals": residuals.tolist(),
        "ci1_feasible": feasible,
        "evaluations": count["n"],
        "undefined_evaluations": count["undefined"],
        "d_ci": float(np.max(np.abs(residuals))),
    }


def calibrate_any(
    arm: str, spec, target: TargetVector, periodic: float = 0.0, availability=None
) -> dict:
    """The frozen calibration of an arm (A1 start; mixture from the primary solution),
    with the periodic amplitude or S-a availability windows when the spec needs them."""
    if periodic == 0.0 and availability is None:
        return s1p.calibrate_arm(arm, spec, target, primary_start=a1_het())
    if availability is not None:
        require(arm == LOGIT_NORMAL and periodic == 0.0, "S-a is logit-normal only (P8-006)")
        return sa_calibrate(spec, target, a1_het(), availability)
    cal = calibrate(
        spec,
        target,
        stage=S.MODEL_STAGE,
        n=N,
        r_cal=abm_config.R_CAL,
        start=a1_het(),
        availability=availability,
        periodic_amplitude=periodic,
    )
    prim = {
        "het": s1p.het_payload(cal.heterogeneity),
        "ci1_residuals": cal.residuals.tolist(),
        "ci1_feasible": cal.feasible,
        "evaluations": cal.evaluations,
        "d_ci": float(np.max(np.abs(cal.residuals))),
    }
    if arm == LOGIT_NORMAL:
        return prim
    m = mixture.calibrate_mixture(
        spec,
        target,
        S.MODEL_STAGE,
        cal.heterogeneity,
        n=N,
        r_cal=abm_config.R_CAL,
        availability=availability,
        periodic_amplitude=periodic,
    )
    return {
        "het": s1p.het_payload(m.heterogeneity),
        "ci1_residuals": m.residuals.tolist(),
        "ci1_feasible": m.feasible,
        "d_ci": m.max_abs_residual,
        "evaluations": m.evaluations,
        "mixture_flags": list(m.flags),
        "selection_rule": m.selection_rule,
        "primary_fit": prim,
    }


# --- truths ------------------------------------------------------------------------


def gate_indices() -> set[int]:
    return {i for i, _ in s1p.gate_truths()}


def rq2_indices() -> list[int]:
    return [
        i for i, p in enumerate(rq2.production_truth_points()) if p.truth_class in S.TRUTH_CLASSES
    ]


def build_spec(truth: dict):
    """(spec, periodic amplitude) of a configured truth."""
    name, params = truth["spec"]
    periodic = float(truth.get("periodic", 0.0))
    if name == "A1":
        return arch.a1(), periodic
    if name == "A1_PRIME":
        return arch.a1_prime(params["rho"], params["sigma_z"]), periodic
    if name == "HMM":
        return arch.a1_prime_hmm(params["q"], params["sigma"]), periodic
    if name == "HET_RHO":
        return arch.a1_prime_heterogeneous_rho(
            params["rho_mean"], params["rho_sd"], params["sigma_z"]
        ), periodic
    if name == "A2":
        return arch.a2(params["delta"], params["lam"]), periodic
    require(name == "MIXED", f"unknown truth spec {name}")
    return arch.mixed(params["rho"], params["sigma_z"], params["delta"], params["lam"]), periodic


def _scan_peak(factory, het, key, periodic=0.0):
    values = [
        mapping.delta1(arch.simulate(factory(d), het, key, N, None, periodic))
        for d in abm_config.DELTA_PEAK_SCAN
    ]
    k = int(np.argmax(values))
    rising = np.diff([0.0, *values[: k + 1]])
    require(
        (rising >= -abm_config.DELTA1_MAPPING_TOL_PP).all(),
        "Δ_1 scan is not monotone below its peak (ascending branch undefined)",
    )
    return abm_config.DELTA_PEAK_SCAN[k], values[k]


def solve_ascending(factory, target_pp, het, key, tol=abm_config.DELTA1_MAPPING_TOL_PP):
    """P8-002 ascending-branch bisection for any history spec factory δ -> Spec."""
    peak, top = _scan_peak(factory, het, key)
    require(target_pp <= top + tol, f"Δ_1 = {target_pp} pp unreachable (max {top:.2f} pp)")
    if abs(top - target_pp) <= tol:
        return peak, top
    lo, hi, value, mid = 0.0, peak, top, peak
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value = mapping.delta1(arch.simulate(factory(mid), het, key, N))
        if abs(value - target_pp) <= tol:
            return mid, value
        lo, hi = (mid, hi) if value < target_pp else (lo, mid)
    require(abs(value - target_pp) <= tol, "Δ_1 bisection did not converge")
    return mid, value


def truth_payload(arm: str, index: int, store) -> dict:
    """A configured truth: imported S1-P8005 gate configuration or a final-run job."""
    if index in gate_indices():
        return legacy_load(s1p.Job("truth", arm, "TRUTH", index=index))
    rec = store.load(FJob("truth", arm, index=index).key())
    require(rec is not None, f"missing truth configuration {arm} {index}")
    return rec["payload"]


def run_truth(job: FJob, store) -> dict:
    point = rq2.production_truth_points()[job.index]
    require(job.index not in gate_indices(), "gate truths are imported, not reconfigured")
    s_a1 = s1p.resolution()["s_A1"]
    target = s1p.real_ci1()
    key = SeedKey(S.MODEL_STAGE, Role.MODEL, replicate=0)
    out = {
        "label": point.label,
        "truth_class": point.truth_class,
        "kind": point.kind,
        "arm": job.arm,
        "periodic": 0.0,
    }
    amp = lambda rule: rq2.amplitude(rule, s_a1)  # noqa: E731
    if point.kind in ("HMM", "HET_RHO", "AR_PERIODIC", "A1_PRIME"):
        if point.kind == "HMM":
            spec_t = ("HMM", {"q": point.param("q"), "sigma": amp(point.param("amp"))})
        elif point.kind == "HET_RHO":
            spec_t = (
                "HET_RHO",
                {
                    "rho_mean": point.param("rho_mean"),
                    "rho_sd": point.param("rho_sd"),
                    "sigma_z": amp(point.param("amp")),
                },
            )
        else:
            spec_t = ("A1_PRIME", {"rho": point.param("rho"), "sigma_z": amp(point.param("amp"))})
        if point.kind == "AR_PERIODIC":
            out["periodic"] = amp(point.param("periodic"))
        spec, periodic = build_spec({"spec": spec_t, "periodic": out["periodic"]})
        cal = calibrate_any(job.arm, spec, target, periodic)
        status = FEASIBLE if cal["ci1_feasible"] else INFEASIBLE_TRUTH
        return {**out, "status": status, "spec": spec_t, "het": cal["het"], "calibration": cal}
    require(point.kind in ("A2", "MIXED"), f"unexpected truth kind {point.kind}")
    lam, target_pp = point.param("lam"), point.param("delta1_pp")
    if point.kind == "A2":

        def factory(d):
            return arch.a2(d, lam)

        fam = Family.A2
    else:
        rho, sigma = point.param("rho"), amp(point.param("amp"))

        def factory(d):
            return arch.mixed(rho, sigma, d, lam)

        fam = None
    het = s1p.het_from_payload(legacy_load(s1p.Job("a1", job.arm, "A1"))["het"])
    delta, cal, history = None, None, []
    if job.arm == MIXTURE:
        prim = truth_payload(LOGIT_NORMAL, job.index, store)
        if prim.get("delta") is not None:
            delta = prim["delta"]
            cal = calibrate_any(job.arm, factory(delta), target)
            het = s1p.het_from_payload(cal["het"])
    try:
        for _ in range(s1p.TRUTH_JOINT_MAX_ITER):
            new_delta, realized = solve_ascending(factory, target_pp, het, key)
            cal = calibrate_any(job.arm, factory(new_delta), target)
            het = s1p.het_from_payload(cal["het"])
            history.append({"delta": new_delta, "realized_pp": realized, "het": cal["het"]})
            converged = delta is not None and abs(new_delta - delta) <= (
                s1p.TRUTH_DELTA_RTOL * max(1.0, abs(new_delta))
            )
            delta = new_delta
            if converged:
                break
        final_pp = mapping.delta1(arch.simulate(factory(delta), het, key, N))
    except Phase8ContractError as exc:
        return {
            **out,
            "status": INFEASIBLE_TRUTH,
            "reason": str(exc),
            "history": history,
            "delta": None,
        }
    within = abs(final_pp - target_pp) <= abm_config.DELTA1_MAPPING_TOL_PP
    status = FEASIBLE if (cal["ci1_feasible"] and within) else INFEASIBLE_TRUTH
    spec_name = "A2" if point.kind == "A2" else "MIXED"
    params = {"delta": delta, "lam": lam}
    if point.kind == "MIXED":
        params.update({"rho": rho, "sigma_z": sigma})
    coord = list(regions.coordinate_of(Family.A2, lam, delta)) if fam else None
    lattice = None if coord is None else regions.inside_region(tuple(coord))
    return {
        **out,
        "status": status,
        "spec": (spec_name, params),
        "het": cal["het"],
        "delta": delta,
        "realized_pp": final_pp,
        "mapping_within_tol": within,
        "coord": coord,
        "inside_lattice": lattice,
        "lattice_note": None if lattice in (None, True) else OUTSIDE_LATTICE,
        "calibration": cal,
        "history": history,
    }


def run_panels(job: FJob, store) -> dict:
    truth = truth_payload(job.arm, job.index, store)
    require(truth["status"] == FEASIBLE, "panels exist only for feasible truths")
    spec, periodic = build_spec(truth)
    het = s1p.het_from_payload(truth["het"])
    point = rq2.production_truth_points()[job.index]
    offset = S.POP_Q_OFFSET if job.block == "pop" else 0
    panels, failures = [], []
    for q in range(job.sub * S.PANEL_CHUNK, (job.sub + 1) * S.PANEL_CHUNK):
        qq = q + offset
        world = arch.simulate(
            spec,
            het,
            truth_key(S.RQ2_DATA_STAGE, point.truth_class, job.index, qq),
            N,
            None,
            periodic,
        )
        try:
            p = primary_moments(world_panel(world)).values
        except (Phase8ContractError, ValueError, np.linalg.LinAlgError):
            failures.append(qq)
            continue
        panels.append(
            {
                "q": qq,
                "ci1": ci1_fast(world.behavior).tolist(),
                "primary": [p[j] for j in abm_config.PRIMARY_IDS],
            }
        )
    return {"label": truth["label"], "block": job.block, "panels": panels, "failures": failures}


# --- model table ------------------------------------------------------------------


def run_ext(job: FJob, store) -> dict:
    """Coarse-image extension: MODEL replicates 50..199 at the imported S1A η̂."""
    m = legacy_load(s1p.Job("map", job.arm, job.family, job.a, job.b))
    spec = s1p.spec_for(job.family, job.a, job.b)
    het = s1p.het_from_payload(m["het"])
    rows, failed = model_rows(spec, het, range(abm_config.R_EMULATOR, S.R_COARSE))
    return {
        "rows": rows,
        "failed": failed,
        "replicates": [abm_config.R_EMULATOR, S.R_COARSE],
        "het": m["het"],
    }


def run_ref(job: FJob, store) -> dict:
    """A refinement-point image: frozen calibration to the real CI-1 + R direct replicates."""
    spec = s1p.spec_for(job.family, job.a, job.b)
    cal = calibrate_any(job.arm, spec, s1p.real_ci1())
    het = s1p.het_from_payload(cal["het"])
    rows, failed = model_rows(spec, het, range(S.R_REFINE))
    return {
        "coord": [job.a, job.b],
        "params": dict(spec.params),
        **cal,
        **summarize(rows),
        "rows": rows,
        "failed": failed,
    }


def run_theta(job: FJob, store) -> dict:
    truth = truth_payload(job.arm, job.index, store)
    spec, periodic = build_spec(truth)
    rows, failed = model_rows(
        spec, s1p.het_from_payload(truth["het"]), range(S.R_THETA_TRUE), periodic
    )
    return {
        "label": truth["label"],
        **summarize(rows),
        "failed": failed,
        "coord": truth.get("coord"),
    }


def run_gpanels(job: FJob, store) -> dict:
    return s1p.run_panels(s1p.Job("panels", job.arm, "TRUTH", index=job.index))


def run_calnoise(job: FJob, store) -> dict:
    """Per-panel recalibration of a family at one θ to one RQ2 panel's own CI-1."""
    chunk = store.load(
        FJob("panels", job.arm, index=job.index, sub=job.sub // S.PANEL_CHUNK, block="rq2").key()
    )
    require(chunk is not None, "calibration-noise check needs its RQ2 panel chunk")
    panel = next(p for p in chunk["payload"]["panels"] if p["q"] == job.sub)
    target = TargetVector(
        "synthetic", dict(zip(abm_config.CI1_COMPONENTS, panel["ci1"], strict=True))
    )
    spec = s1p.spec_for(job.family, job.a, job.b)
    cal = calibrate_any(job.arm, spec, target)
    out = {"q": job.sub, "coord": [job.a, job.b], **cal}
    if cal["ci1_feasible"]:
        rows, failed = model_rows(spec, s1p.het_from_payload(cal["het"]), range(S.CALNOISE_R))
        out.update(summarize(rows))
        out["failed"] = failed
    return out


# --- post-reveal model-side jobs (no real data inside the job) ------------------------


def run_audit(job: FJob, store) -> dict:
    """R = 50 → 200 extension at a refinement point (real-panel audit)."""
    rec = store.load(FJob(job.block, job.arm, job.family, job.a, job.b).key())
    require(rec is not None, "audit needs the refinement image")
    spec = s1p.spec_for(job.family, job.a, job.b)
    rows, failed = model_rows(
        spec, s1p.het_from_payload(rec["payload"]["het"]), range(S.R_REFINE, S.R_AUDIT)
    )
    return {"rows": rows, "failed": failed}


def run_ref2(job: FJob, store) -> dict:
    return run_ref(job, store)


def windows_path() -> Path:
    return PROJECT_ROOT / S.WORK / "sa_windows.json"


def run_sa(job: FJob, store) -> dict:
    """S-a operator image (real E-2 availability windows, frozen draw)."""
    data = json.loads(windows_path().read_text())
    windows = np.asarray(data["windows"], int)
    spec = s1p.spec_for(job.family, job.a, job.b)
    cal = calibrate_any(job.arm, spec, s1p.real_ci1(), availability=windows)
    het = s1p.het_from_payload(cal["het"])
    rows, failed = model_rows(spec, het, range(abm_config.R_EMULATOR), availability=windows)
    return {
        "coord": [job.a, job.b],
        **cal,
        **summarize(rows),
        "failed": failed,
        "windows_sha256": data["sha256"],
    }


def run_eng(job: FJob, store) -> dict:
    """Engineering job for runner tests: deterministic; a 'crash' job kills its worker once."""
    marker = store.root.parent / f"eng_crashed_{job.index}"
    if job.block == "crash" and not marker.exists():
        marker.write_text("1")
        import os

        os._exit(13)
    time.sleep(job.sub / 1000.0)
    return {"value": math.sin(job.index) * 1e6}


RUNNERS = {
    "truth": run_truth,
    "panels": run_panels,
    "ext": run_ext,
    "ref": run_ref,
    "theta": run_theta,
    "gpanels": run_gpanels,
    "calnoise": run_calnoise,
    "audit": run_audit,
    "ref2": run_ref2,
    "sa": run_sa,
    "saref": run_sa,
    "_eng": run_eng,
}
