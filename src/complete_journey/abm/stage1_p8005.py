"""Phase 8 replacement Stage 1 under P8-005 (run label S1-P8005, seed stage key 11).

**Namespace.** Everything here is separate from the archived Stage 1 (snapshot
``e59d101b…``):
- seed-tree stage key 11 (archived: 1);
- archive ``docs/phases/PHASE_08_STAGE1_P8005/`` and work directory
  ``outputs/diagnostics/phase8_stage1_p8005/``;
- its own design snapshot (schema v2) and checkpoint store keyed by that snapshot.

No path here can resolve inside the archived Stage 1 namespace. The archived driver
(``stage1``) is bound to its own snapshot and refuses to run once the source drifts.

**Stage order (§P8-005.12).**
- S1-RESOLUTION: A1 under the primary logit-normal layer → τ_j, s_A1, LOW/HIGH; A1
  under the mixture → bracket availability.
- Truth configuration of the frozen gate points per arm (CI-1 matched to the real
  CI-1; T3 δ jointly solved on the ascending branch). INFEASIBLE TRUTH is recorded,
  never replaced. A gate truth outside the search lattice STOPS for researcher review.
- S1A: CI-feasible representational maps, both arms, all lattice points (no span gate).
- S1B: reference tables per arm, P8-004 CRN-paired validation (primary) plus the
  independent R = 200 diagnostic, the predeclared refinement once, and the §AC H2
  direct-procedure check.
- S1C: P8-003 G1–G5 per family and arm on Q = 200 truth panels per gate point.
- S1D: the secondary calibration frontier (never a gate).

**Real data used here.** The real CI-1 (calibration) and the archived real P1..P5
estimates and SEs (A1 tolerance construction only). No A1′ or A2 point is ever
compared with a real dynamic target. Real dynamic targets never reach this module's
search, maps, emulator or gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import posixpath
import subprocess
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

from complete_journey.abm import adequacy as adq
from complete_journey.abm import architectures as arch
from complete_journey.abm import (
    config,
    frontier,
    mapping,
    mixture,
    regions,
    representability,
    rq2,
    span,
    stage1,
)
from complete_journey.abm import emulator as em
from complete_journey.abm import procedure as proc
from complete_journey.abm.calibration import calibrate
from complete_journey.abm.checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from complete_journey.abm.checkpoint import CheckpointStore, JobKey
from complete_journey.abm.config import Phase8ContractError, require
from complete_journey.abm.guards import TargetVector, require_p8005_authorized, verify_real_archive
from complete_journey.abm.moments import ci1_fast, moment_bundle, primary_moments
from complete_journey.abm.operators import world_panel
from complete_journey.abm.seeds import Family, Role, SeedKey, truth_key
from complete_journey.paths import PROJECT_ROOT

STAGE = config.P8005_STAGE_KEY
PROTOCOL_STAGE = 1
RUN_LABEL = "S1-P8005"
ARCHIVE = "docs/phases/PHASE_08_STAGE1_P8005"
SNAPSHOT = f"{ARCHIVE}/design_snapshot.json"
SNAPSHOT_SHA = f"{ARCHIVE}/design_snapshot.sha256"
WORK = "outputs/diagnostics/phase8_stage1_p8005"
SNAPSHOT_SCHEMA = "phase8-stage1-snapshot-v2-p8005"
WORKERS = 6  # M2 Pro: 6 performance + 4 efficiency cores; conservative, never all 10
ARMS = (em.LOGIT_NORMAL, em.MIXTURE)
FAMILIES = ("A1_PRIME", "A2")
LATTICE = tuple(float(i) for i in range(config.GRID_SIZE))
VALIDATION_POINTS = tuple(tuple(float(c) for c in p) for p in stage1.EMULATOR_VALIDATION_POINTS)
MAP_R = config.R_EMULATOR  # direct R = 50 per lattice point (images, frontier, table node 0)
TRUTH_JOINT_MAX_ITER = 8
TRUTH_DELTA_RTOL = 1e-3
H2_PANELS_PER_FAMILY = 10
SOURCE_GLOBS = ("src/complete_journey/abm/*.py", "src/complete_journey/prediction/*.py")
SOURCE_FILES = ("scripts/run_abm.py",)
ARCHIVED_STAGE1_FILES = (
    "design_snapshot.json",
    "design_snapshot.sha256",
    "s1_checkpoint_manifest.json",
    "s1_provenance.json",
    "s1_resolution.json",
    "s1_span.json",
    "s1_span_diagnostics.json",
)
#: SHA-256 of the archived Stage 1 artifacts (verified byte-identical before P8-005).
ARCHIVED_STAGE1_SHA256 = {
    "design_snapshot.json": "e59d101b8900cfcb47a42d0bb7c2fdee52d8ce6160e4664a3eabfa0cfc171718",
    "design_snapshot.sha256": "0444ef598cc4ee994fdacb9a5cf20e1e6ce4e295d7d82a959752d4c8bdcc38aa",
    "s1_checkpoint_manifest.json": (
        "f3235466d53cc2f22ce1fcf1296b7834806f0962566dffced2ca52dd83b9c75c"
    ),
    "s1_provenance.json": "294fb5cb7501946cf75ac84e956c9ad2bf5b85ae068510a950449abdbc2dd5bb",
    "s1_resolution.json": "5ffa52d9b6eb15e438aea1d015c0720c37782f142e88f86766c05ae73dd098ca",
    "s1_span.json": "8b0b0e559261f0d6b58262a5f6f1b5681a83615b72a262d1bb3366fcb644b56b",
    "s1_span_diagnostics.json": "4f720e389a5877cd1ecc91712d5a77b32ef564eee8f18bb7ff1d9b2366e41505",
}
OUTSIDE_REGION = "OUTSIDE SEARCH REGION"
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class ResearcherStop(Phase8ContractError):
    """A frozen STOP rule fired: researcher review needed before continuing."""


# --- namespace ------------------------------------------------------------------------


def _normalized(path: str) -> str:
    """POSIX-normalized relative path; any '..' component or absolute path is refused."""
    raw = Path(path).as_posix()
    require(not raw.startswith("/"), "S1-P8005 paths must be repository-relative")
    require(".." not in Path(raw).parts, "S1-P8005 paths may not contain '..'")
    return posixpath.normpath(raw)


def _inside(path: str, root: str) -> bool:
    p, r = _normalized(path), _normalized(root)
    return p == r or p.startswith(r + "/")


def require_new_namespace(relative: str) -> None:
    """Writes go to the S1-P8005 archive or work directory only, never the old ones."""
    for old in (stage1.ARCHIVE, stage1.WORK):
        require(not _inside(relative, old), f"S1-P8005 may not write into {old}")
    require(
        _inside(relative, ARCHIVE) or _inside(relative, WORK),
        f"S1-P8005 writes only under {ARCHIVE} or {WORK}",
    )


def verify_archived_stage1(root: Path = PROJECT_ROOT) -> dict:
    """The archived (failed) Stage 1 artifacts are byte-identical to their record."""
    out = {}
    for name in ARCHIVED_STAGE1_FILES:
        digest = hashlib.sha256((root / stage1.ARCHIVE / name).read_bytes()).hexdigest()
        require(digest == ARCHIVED_STAGE1_SHA256[name], f"archived Stage 1 file {name} changed")
        out[name] = digest
    return out


# --- snapshot -------------------------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_hashes(root: Path = PROJECT_ROOT) -> dict[str, str]:
    paths = sorted(
        {p for g in SOURCE_GLOBS for p in root.glob(g)} | {root / f for f in SOURCE_FILES}
    )
    return {str(p.relative_to(root)): _sha(p.read_bytes()) for p in paths}


def decision_hashes(root: Path = PROJECT_ROOT) -> dict[str, str]:
    text = (root / "docs/DECISIONS.md").read_text()
    out = {}
    for d in ("P8-001", "P8-002", "P8-003", "P8-004", "P8-005"):
        start = text.index(f"### {d} ")
        nxt = text.find("\n### ", start + 5)
        out[d] = _sha(text[start : nxt if nxt != -1 else len(text)].encode())
    return out


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def build_snapshot(root: Path = PROJECT_ROOT, projected_core_hours=None) -> dict:
    versions = {pkg: metadata.version(pkg) for pkg in ("numpy", "scipy", "pandas", "scikit-learn")}
    base = stage1.build_snapshot(root)  # archived v1 fields (architectures, caps, grid, …)
    return {
        "schema": SNAPSHOT_SCHEMA,
        "run_label": RUN_LABEL,
        "seed_stage_key": STAGE,
        "protocol_stage": PROTOCOL_STAGE,
        "archive": ARCHIVE,
        "work": WORK,
        "archived_stage1": {
            "snapshot_sha256": ARCHIVED_STAGE1_SHA256["design_snapshot.json"],
            "status": "FAILED PREREGISTERED DESIGN GATE (immutable; never reused)",
            "artifact_sha256": ARCHIVED_STAGE1_SHA256,
        },
        "source_commit_at_snapshot": _git_head(root),
        "source_sha256": source_hashes(root),
        "decision_sha256": decision_hashes(root),
        "authorized_stage": config.AUTHORIZED_STAGE,
        "p8005_stage1_authorized": config.P8005_STAGE1_AUTHORIZED,
        "architectures": base["architectures"],
        "caps": base["caps"],
        "grid": base["grid"],
        "primary_moments": base["primary_moments"],
        "p8_003": base["p8_003"],
        "kappas": base["kappas"],
        "kappa_reference": base["kappa_reference"],
        "perfect_model_gates": base["perfect_model_gates"],
        "ci1": {
            "components": list(config.CI1_COMPONENTS),
            "gate": "|sim - target| <= max(0.02, 2 SE_j) for every component",
            "gate_abs": config.CI1_GATE_ABS,
            "boot_se": dict(config.CI1_BOOT_SE),
            "R_cal": config.R_CAL,
            "primary_start": "A1 solution (real CI-1); A1 itself: logit(mean), s = 1",
        },
        "heterogeneity": {
            "primary": "logit-normal alpha_i = mu + s xi_i",
            "mixture": {
                "form": "alpha_i = mu_1 + d B_i + s xi_i, B_i = 1{u_i < 1 - pi}, "
                "u_i from the shared MIXTURE stream (code 8, family 0)",
                "bounds": {
                    "pi": [mixture.PI_LO, mixture.PI_HI],
                    "d": [mixture.D_LO, mixture.D_HI],
                    "mu1": [mixture.MU_LO, mixture.MU1_HI],
                    "mu2_max": mixture.MU_HI,
                    "s": [mixture.S_LO, mixture.S_HI],
                },
                "transforms": "pi = 0.10 + 0.80 expit(u1); mu1 = -5 + 9.75 expit(u2); "
                "d = 0.25 + (min(6, 5 - mu1) - 0.25) expit(u3); s = 0.05 + 2.95 expit(u4)",
                "optimizer": "Nelder-Mead; simplex 0.5 then 0.1 in u; xatol 1e-4, fatol 1e-10, "
                "maxiter 400, maxfev 800",
                "starts": "pi in (0.25, 0.50, 0.75); s = s_hat/2; d = s_hat sqrt(0.75/(pi(1-pi))); "
                "mu1 = mu_hat - (1 - pi) d; projected with margin 0.01",
                "selection": "lowest objective if it passes; else lowest passing; else INFEASIBLE",
                "flags": [mixture.INFEASIBLE, mixture.MIXTURE_BOUNDARY, mixture.UNIMODAL],
                "a1_gate": f"A1 infeasible under the mixture -> {mixture.BRACKET_UNAVAILABLE}; "
                "primary arm continues; mixture specification never altered",
                "diagnostics": "descriptive only: Jacobian condition (steps "
                f"{mixture.JACOBIAN_STEPS}), separation, bound flags, start agreement; "
                f"pi profile on {list(mixture.PI_PROFILE_GRID)} for A1",
                "tau_and_amplitudes": "from the primary arm only",
            },
            "labels": [
                representability.ROBUST,
                representability.HET_DEPENDENT,
                representability.BRACKET_UNAVAILABLE,
            ],
        },
        "representability": {
            "search_set": "Theta_C^CI only (emulated CI-1 gate); no span counts",
            "sigma_pop": "sqrt(s_T^2/Q + (g+- s_C)^2/R_em), R_em = 50",
            "emulator_error": "e_j = 0.1 tau_j(0.15)",
            "states": list(representability.STATES),
            "empty_set": representability.EXCLUDED,
            "per_kappa": list(config.KAPPAS),
        },
        "span_map": {
            "status": span.DESCRIPTIVE_MAP,
            "coords": list(stage1.SPAN_IMAGE_COORDS),
            "box_half_width_sd": config.SPAN_BOX_SD,
            "gate": None,
        },
        "frontier": {
            "status": frontier.STATUS,
            "thresholds": list(frontier.THRESHOLDS),
            "coords": list(frontier.COORDS),
            "is_gate": False,
        },
        "stage_order": [
            "S1-RESOLUTION",
            "TRUTH",
            "S1A",
            "S1B",
            "S1C",
            "S1D",
            "REPRESENTABILITY-PREFLIGHT",
        ],
        "gate_behaviour": {
            "S1A": "never a gate; mixture A1 infeasible -> bracket unavailable",
            "S1B": "CRN emulator gate per family and arm (+ A1 table) and H2; one predeclared "
            "refinement; still failing -> STOP",
            "S1C": "G1-G5 per family and arm over assessable gate points; a family with no "
            "assessable gate point fails; any failure -> STOP",
            "S1D": "never a gate",
            "truth_outside_region": "STOP for researcher review before S1A",
        },
        "emulator": {
            "logit_normal_nodes": "eta_hat + {0, 0.10 e_mu, 0.10 e_s}",
            "mixture_nodes": "eta_hat + {0, 0.10 e_k} for k in (pi, mu1, d, s); -0.10 if a node "
            "would leave the bounds",
            "validation_points": [list(p) for p in VALIDATION_POINTS],
            "validation_offsets": "(+0.03, -0.03, ...) and its negation in eta",
            "table_R": em.TABLE_R,
            "reference": "P8-004: direct, the same 50 MODEL streams (replicates 0..49)",
            "independent_diagnostic": "R = 200, MODEL replicates 50..249 (non-binding)",
            "gate": "|mu_em - mu_crn| <= 0.1 tau_j(0.15); s_em in [0.8, 1.2] s_crn",
            "refinement": "once: failing points' neighbouring cell centres at R = 100; "
            "piecewise-linear triangles inside refined cells",
            "recalibration": "bounded weighted linear least squares on the emulated CI-1 map",
            "h2": f"{H2_PANELS_PER_FAMILY} panels per family, 6x6 subgrid, direct R = "
            f"{proc.H2_R}; agreement >= {proc.H2_AGREEMENT_MIN} at every kappa and no "
            "ADEQ/INADEQ contradiction",
            "a1_table": "single point, same node rule",
        },
        "truths": {
            "gate_points": [p.label for p in rq2.gate_points()],
            "q": config.Q_GATE,
            "data_keys": "truth_key(11, class, point, q): Role.DATA, never MODEL",
            "t3_mapping": "joint: delta on the ascending branch (P8-002) and heterogeneity "
            f"matched to the real CI-1; at most {TRUTH_JOINT_MAX_ITER} iterations, "
            f"relative delta change <= {TRUTH_DELTA_RTOL}",
            "infeasible": "unreachable Delta_1, undefined branch or CI-1 infeasible -> "
            "INFEASIBLE TRUTH (recorded, never replaced)",
        },
        "budgets": base["budgets"],
        "seed_tree": base["seed_tree"] | {"stage_key": STAGE, "mixture_stream": 8},
        "operators": base["operators"],
        "software": {"python": platform.python_version(), **versions},
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "workers": WORKERS,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "projected_core_hours": projected_core_hours,
    }


def write_snapshot(root: Path = PROJECT_ROOT, projected_core_hours=None) -> str:
    require_p8005_authorized()
    require_new_namespace(SNAPSHOT)
    path = root / SNAPSHOT
    require(not path.exists(), "the S1-P8005 design snapshot exists and is never rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(build_snapshot(root, projected_core_hours), indent=2, sort_keys=True) + "\n"
    ).encode()
    path.write_bytes(data)
    digest = _sha(data)
    (root / SNAPSHOT_SHA).write_text(digest + "\n")
    return digest


def check_snapshot(snap: dict, root: Path = PROJECT_ROOT) -> None:
    """Schema, stage key, source and decision hashes of a snapshot match the checkout."""
    require(
        snap["schema"] == SNAPSHOT_SCHEMA and snap["seed_stage_key"] == STAGE,
        "not an S1-P8005 snapshot",
    )
    require(snap["source_sha256"] == source_hashes(root), "source drifted from the snapshot")
    require(snap["decision_sha256"] == decision_hashes(root), "P8 decisions drifted")


def verify_snapshot(root: Path = PROJECT_ROOT) -> str:
    data = (root / SNAPSHOT).read_bytes()
    digest = _sha(data)
    require(digest == (root / SNAPSHOT_SHA).read_text().strip(), "S1-P8005 snapshot tampered")
    check_snapshot(json.loads(data), root)
    require_p8005_authorized()
    return digest


def store(root: Path = PROJECT_ROOT) -> CheckpointStore:
    digest = verify_snapshot(root)
    require_new_namespace(f"{WORK}/checkpoints")
    return CheckpointStore(root / WORK / "checkpoints", digest, _git_head(root))


# --- heterogeneity helpers -------------------------------------------------------------


def het_payload(het) -> dict:
    if isinstance(het, arch.MixtureHeterogeneity):
        return {"arm": em.MIXTURE, **het.as_dict()}
    return {"arm": em.LOGIT_NORMAL, "mu": het.mu, "s": het.s}


def het_from_payload(p: dict):
    if p["arm"] == em.MIXTURE:
        return arch.MixtureHeterogeneity(p["pi"], p["mu1"], p["d"], p["s"])
    return arch.Heterogeneity(p["mu"], p["s"])


def eta_of(het) -> np.ndarray:
    if isinstance(het, arch.MixtureHeterogeneity):
        return np.array([het.pi, het.mu1, het.d, het.s])
    return np.array([het.mu, het.s])


def het_of(arm: str, eta) -> arch.Heterogeneity | arch.MixtureHeterogeneity:
    x = em.clamp_eta(arm, eta)
    if arm == em.MIXTURE:
        return arch.MixtureHeterogeneity(*map(float, x))
    return arch.Heterogeneity(float(x[0]), float(x[1]))


def spec_for(family: str, ip: float, ia: float) -> arch.Spec:
    if family == "A1":
        return arch.a1()
    point = regions.make_point(Family[family], ip, ia, 0)
    return arch.Spec(Family[family], "primary", point.params)


# --- direct simulation ----------------------------------------------------------------


def direct_moments(spec, het, replicates, secondary: bool = False) -> dict:
    """P1..P5 means/SDs and the MODEL-world CI-1 mean over the given MODEL replicates."""
    prim, sec, ci1s, labels, failures = [], [], [], [], 0
    for r in replicates:
        world = arch.simulate(
            spec, het, SeedKey(STAGE, Role.MODEL, replicate=r), config.N_HOUSEHOLDS
        )
        ci1s.append(ci1_fast(world.behavior))
        try:
            if secondary:
                p, s = moment_bundle(world_panel(world))
                sec.append([s[k] for k in sorted(s)])
            else:
                p = primary_moments(world_panel(world))
        except (Phase8ContractError, ValueError, np.linalg.LinAlgError):
            failures += 1
            continue
        prim.append([p.values[j] for j in config.PRIMARY_IDS])
        if spec.family == Family.A2 and secondary:
            labels.append(mapping.delta1(world))
    prim = np.array(prim)
    require(len(prim) >= 2, "fewer than two successful replicates")
    out = {
        "replicates": len(list(replicates)),
        "replicates_ok": len(prim),
        "failures": failures,
        "primary_mean": dict(zip(config.PRIMARY_IDS, prim.mean(0).tolist(), strict=True)),
        "primary_sd": dict(zip(config.PRIMARY_IDS, prim.std(0, ddof=1).tolist(), strict=True)),
        "ci1_model_mean": np.mean(ci1s, axis=0).tolist(),
    }
    if secondary:
        from complete_journey.abm.moments import BUNDLE_SECONDARY

        out["secondary_mean"] = dict(
            zip(sorted(BUNDLE_SECONDARY), np.array(sec).mean(0).tolist(), strict=True)
        )
    if labels:
        out["delta1_pp_label"] = float(np.mean(labels))
    return out


# --- calibration under an arm --------------------------------------------------------


def a1_start() -> arch.Heterogeneity:
    return het_from_payload(load(Job("a1", em.LOGIT_NORMAL, "A1"))["het"])


def calibrate_arm(
    arm: str, spec, target: TargetVector, primary_start=None, primary_solution=None
) -> dict:
    """Frozen CI-1 calibration under one arm; mixture starts from the primary solution."""
    if primary_solution is None:
        cal = calibrate(
            spec,
            target,
            stage=STAGE,
            n=config.N_HOUSEHOLDS,
            r_cal=config.R_CAL,
            start=primary_start,
        )
        primary_solution = cal.heterogeneity
        primary_payload = {
            "het": het_payload(cal.heterogeneity),
            "ci1_residuals": cal.residuals.tolist(),
            "ci1_feasible": cal.feasible,
            "evaluations": cal.evaluations,
        }
    else:
        primary_payload = None
    if arm == em.LOGIT_NORMAL:
        p = primary_payload
        return {**p, "d_ci": float(np.max(np.abs(p["ci1_residuals"])))}
    mcal = mixture.calibrate_mixture(
        spec, target, STAGE, primary_solution, n=config.N_HOUSEHOLDS, r_cal=config.R_CAL
    )
    return {
        "het": het_payload(mcal.heterogeneity),
        "ci1_residuals": mcal.residuals.tolist(),
        "ci1_feasible": mcal.feasible,
        "d_ci": mcal.max_abs_residual,
        "evaluations": mcal.evaluations,
        "primary_start_solution": het_payload(primary_solution),
        "mixture": {
            "flags": list(mcal.flags),
            "selection_rule": mcal.selection_rule,
            "selected_start": mcal.selected_start,
            "fits": [
                {
                    "start": f.start.as_dict(),
                    "fit": f.heterogeneity.as_dict(),
                    "distance": f.distance,
                    "feasible": f.feasible,
                    "evaluations": f.evaluations,
                    "converged": f.converged,
                }
                for f in mcal.fits
            ],
            "diagnostics": mixture.diagnostics(mcal, STAGE, config.N_HOUSEHOLDS, config.R_CAL),
        },
        "primary_fit": primary_payload,
    }


# --- jobs -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    kind: str  # a1 | map | node | validation | centre | truth | panels | h2 | gate_eval
    arm: str
    family: str
    ip: float = -1.0
    ia: float = -1.0
    index: int = 0
    sub: int = 0

    def key(self) -> JobKey:
        point = f"{self.kind}:{self.arm}:{self.ip:g},{self.ia:g}:{self.index}:{self.sub}"
        seeds = tuple(
            (config.SEED_NAMESPACE, STAGE, int(role))
            for role in (Role.DATA, Role.MODEL, Role.CALIBRATION)
        )
        return JobKey(STAGE, self.family, point, 0, 1, seeds)


def load(job: Job, st: CheckpointStore | None = None) -> dict:
    rec = (st or store()).load(job.key())
    require(rec is not None, f"missing S1-P8005 checkpoint for {job}")
    return rec["payload"]


def real_ci1() -> TargetVector:
    return stage1.real_ci1_target()


def run_a1(job: Job) -> dict:
    target = real_ci1()
    spec = arch.a1()
    if job.arm == em.LOGIT_NORMAL:
        cal = calibrate_arm(job.arm, spec, target)
        out = {**cal}
        if cal["ci1_feasible"]:
            out.update(direct_moments(spec, het_from_payload(cal["het"]), range(config.R_DIRECT)))
        return out
    primary = a1_start()
    cal = calibrate_arm(job.arm, spec, target, primary_solution=primary)
    best = het_from_payload(cal["het"])
    cal["pi_profile"] = mixture.pi_profile(
        spec, target, STAGE, best, config.N_HOUSEHOLDS, config.R_CAL, mixture.PI_PROFILE_GRID
    )
    return cal


def run_map(job: Job) -> dict:
    spec = spec_for(job.family, job.ip, job.ia)
    target = real_ci1()
    if job.family == "A1":
        cal = load(Job("a1", job.arm, "A1"))
        cal = {
            k: v
            for k, v in cal.items()
            if k
            not in (
                "primary_mean",
                "primary_sd",
                "ci1_model_mean",
                "replicates",
                "replicates_ok",
                "failures",
                "pi_profile",
            )
        }
    elif job.arm == em.LOGIT_NORMAL:
        cal = calibrate_arm(job.arm, spec, target, primary_start=a1_start())
    else:
        primary = load(Job("map", em.LOGIT_NORMAL, job.family, job.ip, job.ia))
        cal = calibrate_arm(
            job.arm, spec, target, primary_solution=het_from_payload(primary["het"])
        )
    het = het_from_payload(cal["het"])
    point = (
        None if job.family == "A1" else regions.make_point(Family[job.family], job.ip, job.ia, 0)
    )
    out = {
        "arm": job.arm,
        "family": job.family,
        "ip": job.ip,
        "ia": job.ia,
        "params": dict(spec.params),
        "boundary": bool(point.boundary) if point else False,
        **cal,
        **direct_moments(spec, het, range(MAP_R), secondary=True),
    }
    if job.family == "A2":
        out["delta12_weeks_label"] = mapping.delta12(
            spec, het, SeedKey(STAGE, Role.MODEL, replicate=0), config.N_HOUSEHOLDS
        )
    return out


def run_node(job: Job) -> dict:
    base = load(Job("map", job.arm, job.family, job.ip, job.ia))
    nodes = em.design_nodes(job.arm, eta_of(het_from_payload(base["het"])))
    eta = nodes[job.index]
    spec = spec_for(job.family, job.ip, job.ia)
    return {"eta": eta.tolist(), **direct_moments(spec, het_of(job.arm, eta), range(em.TABLE_R))}


def run_validation(job: Job) -> dict:
    spec = spec_for(job.family, job.ip, job.ia)
    target = real_ci1()
    if job.family == "A1":
        cal = load(Job("a1", job.arm, "A1"))
    else:
        cal = calibrate_arm(job.arm, spec, target, primary_start=a1_start())
    eta_hat = eta_of(het_from_payload(cal["het"]))
    out = {
        "eta_hat": eta_hat.tolist(),
        "ci1_feasible": cal["ci1_feasible"],
        "d_ci": cal["d_ci"],
        "offsets": [],
    }
    for k, off in enumerate(em.validation_offsets(job.arm)):
        eta = em.clamp_eta(job.arm, eta_hat + off)
        het = het_of(job.arm, eta)
        crn = direct_moments(spec, het, range(em.TABLE_R))
        ind = direct_moments(
            spec,
            het,
            range(
                em.INDEPENDENT_FIRST_REPLICATE, em.INDEPENDENT_FIRST_REPLICATE + em.INDEPENDENT_R
            ),
        )
        out["offsets"].append({"offset": k, "eta": eta.tolist(), "crn": crn, "independent": ind})
    return out


def run_centre(job: Job) -> dict:
    ip, ia = job.ip + 0.5, job.ia + 0.5
    spec = spec_for(job.family, ip, ia)
    cal = calibrate_arm(job.arm, spec, real_ci1(), primary_start=a1_start())
    nodes = em.design_nodes(job.arm, eta_of(het_from_payload(cal["het"])))
    rows = [direct_moments(spec, het_of(job.arm, e), range(em.REFINED_R)) for e in nodes]
    return {
        "cell": [job.ip, job.ia],
        "eta_design": nodes.tolist(),
        "nodes": rows,
        "calibration": cal,
    }


# --- truths ----------------------------------------------------------------------------


def gate_truths() -> tuple[tuple[int, rq2.TruthPoint], ...]:
    """(production index, point) of the frozen perfect-model gate points."""
    allp = rq2.production_truth_points()
    gate = {p.label for p in rq2.gate_points()}
    return tuple((i, p) for i, p in enumerate(allp) if p.label in gate)


def resolution() -> dict:
    a1 = load(Job("a1", em.LOGIT_NORMAL, "A1"))
    require(a1["ci1_feasible"], "A1 fails the CI-1 gate against the real CI-1: STOP (§F)")
    tol = adq.build_tolerances(stage1.real_primary_reference(), a1["primary_mean"])
    s_a1 = het_from_payload(a1["het"]).s
    return {"a1": a1, "tolerances": tol, "s_A1": s_a1}


def tolerances() -> adq.Tolerances:
    return resolution()["tolerances"]


def bracket_available() -> bool:
    return bool(load(Job("a1", em.MIXTURE, "A1"))["ci1_feasible"])


def run_truth(job: Job) -> dict:
    """Truth configuration of one gate point under one arm (CI-1 matched to the real CI-1)."""
    point = rq2.production_truth_points()[job.index]
    res = resolution()
    s_a1 = res["s_A1"]
    target = real_ci1()
    key = SeedKey(STAGE, Role.MODEL, replicate=0)
    out = {
        "label": point.label,
        "truth_class": point.truth_class,
        "kind": point.kind,
        "arm": job.arm,
    }
    a1_arm = load(Job("a1", job.arm, "A1"))
    if point.kind == "A1":
        return {
            **out,
            "status": "FEASIBLE" if a1_arm["ci1_feasible"] else representability.INFEASIBLE_TRUTH,
            "spec": ("A1", {}),
            "het": a1_arm["het"],
            "coord": [0.0, 0.0],
            "calibration": {"d_ci": a1_arm["d_ci"]},
        }
    if point.kind == "A1_PRIME":
        rho, sigma = point.param("rho"), rq2.amplitude(point.param("amp"), s_a1)
        spec = arch.a1_prime(rho, sigma)
        coord = regions.coordinate_of(Family.A1_PRIME, rho, sigma)
        cal = calibrate_arm(job.arm, spec, target, primary_start=a1_start())
        status = "FEASIBLE" if cal["ci1_feasible"] else representability.INFEASIBLE_TRUTH
        if not regions.inside_region(coord):
            status = OUTSIDE_REGION
        return {
            **out,
            "status": status,
            "spec": ("A1_PRIME", {"rho": rho, "sigma_z": sigma}),
            "het": cal["het"],
            "coord": list(coord),
            "calibration": cal,
        }
    require(point.kind == "A2", f"gate truth kind {point.kind} not in the gate")
    lam, target_pp = point.param("lam"), point.param("delta1_pp")
    het = het_from_payload(a1_arm["het"])
    delta, cal, history = None, None, []
    if job.arm == em.MIXTURE:
        primary_truth = load(Job("truth", em.LOGIT_NORMAL, "TRUTH", index=job.index))
        if primary_truth.get("delta") is not None:
            delta = primary_truth["delta"]
            cal = calibrate_arm(job.arm, arch.a2(delta, lam), target, primary_start=a1_start())
            het = het_from_payload(cal["het"])
    try:
        for _ in range(TRUTH_JOINT_MAX_ITER):
            new_delta, realized = mapping.solve_delta(target_pp, lam, het, key, config.N_HOUSEHOLDS)
            cal = calibrate_arm(job.arm, arch.a2(new_delta, lam), target, primary_start=a1_start())
            het = het_from_payload(cal["het"])
            history.append({"delta": new_delta, "realized_pp": realized, "het": cal["het"]})
            converged = delta is not None and abs(new_delta - delta) <= TRUTH_DELTA_RTOL * max(
                1.0, abs(new_delta)
            )
            delta = new_delta
            if converged:
                break
        final_pp = mapping.delta1(arch.simulate(arch.a2(delta, lam), het, key, config.N_HOUSEHOLDS))
    except Phase8ContractError as exc:
        return {
            **out,
            "status": representability.INFEASIBLE_TRUTH,
            "reason": str(exc),
            "history": history,
            "delta": None,
        }
    within = abs(final_pp - target_pp) <= config.DELTA1_MAPPING_TOL_PP
    status = "FEASIBLE" if (cal["ci1_feasible"] and within) else representability.INFEASIBLE_TRUTH
    coord = regions.coordinate_of(Family.A2, lam, delta)
    if not regions.inside_region(coord):
        status = OUTSIDE_REGION
    return {
        **out,
        "status": status,
        "spec": ("A2", {"delta": delta, "lam": lam}),
        "het": cal["het"],
        "delta": delta,
        "realized_pp": final_pp,
        "mapping_within_tol": within,
        "coord": list(coord),
        "calibration": cal,
        "history": history,
    }


def truth_spec(truth: dict) -> arch.Spec:
    name, params = truth["spec"]
    if name == "A1":
        return arch.a1()
    if name == "A1_PRIME":
        return arch.a1_prime(params["rho"], params["sigma_z"])
    return arch.a2(params["delta"], params["lam"])


def run_panels(job: Job) -> dict:
    truth = load(Job("truth", job.arm, "TRUTH", index=job.index))
    require(truth["status"] == "FEASIBLE", "panels exist only for feasible truths")
    spec, het = truth_spec(truth), het_from_payload(truth["het"])
    point = rq2.production_truth_points()[job.index]
    panels, failures = [], 0
    for q in range(config.Q_GATE):
        world = arch.simulate(
            spec, het, truth_key(STAGE, point.truth_class, job.index, q), config.N_HOUSEHOLDS
        )
        try:
            p = primary_moments(world_panel(world)).values
        except (Phase8ContractError, ValueError, np.linalg.LinAlgError):
            failures += 1
            continue
        panels.append(
            {
                "q": q,
                "ci1": ci1_fast(world.behavior).tolist(),
                "primary": [p[j] for j in config.PRIMARY_IDS],
            }
        )
    return {"label": truth["label"], "panels": panels, "failures": failures}


def panel_objects(payload: dict) -> list[proc.Panel]:
    arr = np.array([p["primary"] for p in payload["panels"]])
    se = dict(zip(config.PRIMARY_IDS, arr.std(axis=0, ddof=1).tolist(), strict=True))
    return [
        proc.Panel(np.array(p["ci1"]), dict(zip(config.PRIMARY_IDS, p["primary"], strict=True)), se)
        for p in payload["panels"]
    ]


def h2_panel_slots(arm: str, family: str) -> list[tuple[int, int]]:
    """(truth index, panel position) of the 10 predeclared H2 panels of a family."""
    want = "T1" if family == "A1_PRIME" else "T3"
    feasible = [
        i
        for i, p in gate_truths()
        if p.truth_class == want
        and load(Job("truth", arm, "TRUTH", index=i))["status"] == "FEASIBLE"
    ]
    require(feasible, f"no feasible {family} gate truth for H2 under {arm}")
    return [(feasible[k % len(feasible)], k // len(feasible)) for k in range(H2_PANELS_PER_FAMILY)]


def h2_subgrid() -> tuple[tuple[float, float], ...]:
    idx = config.SENSITIVITY_SUBGRID_INDEX
    return tuple((float(i), float(j)) for i in idx for j in idx if i <= 11 and j <= 11)


def run_h2(job: Job) -> dict:
    truth_index, position = h2_panel_slots(job.arm, job.family)[job.index]
    payload = load(Job("panels", job.arm, "TRUTH", index=truth_index))
    panel = payload["panels"][position]
    ip, ia = h2_subgrid()[job.sub]
    spec = spec_for(job.family, ip, ia)
    target = TargetVector("synthetic", dict(zip(config.CI1_COMPONENTS, panel["ci1"], strict=True)))
    cal = calibrate_arm(job.arm, spec, target, primary_start=a1_start())
    out = {"truth_index": truth_index, "position": position, "coord": [ip, ia], **cal}
    if cal["ci1_feasible"]:
        out.update(direct_moments(spec, het_from_payload(cal["het"]), range(proc.H2_R)))
    return out


# --- tables -------------------------------------------------------------------------


def build_table(
    arm: str, family: str, st: CheckpointStore | None = None, centres: bool = True
) -> em.ReferenceTable:
    st = st or store()
    k = em.eta_dimension(arm)
    if family == "A1":
        coords_i, coords_a = (0.0,), (0.0,)
    else:
        coords_i, coords_a = LATTICE, LATTICE
    ni, na = len(coords_i), len(coords_a)
    design = np.zeros((ni, na, k + 1, k))
    mean = np.zeros((ni, na, k + 1, config.J_PRIMARY))
    sd = np.zeros_like(mean)
    ci1 = np.zeros((ni, na, k + 1, 6))
    for a, ip in enumerate(coords_i):
        for b, ia in enumerate(coords_a):
            base = load(
                Job(
                    "map",
                    arm,
                    family,
                    ip if family != "A1" else -1.0,
                    ia if family != "A1" else -1.0,
                ),
                st,
            )
            nodes = em.design_nodes(arm, eta_of(het_from_payload(base["het"])))
            design[a, b] = nodes
            rows = [base] + [
                load(
                    Job(
                        "node",
                        arm,
                        family,
                        ip if family != "A1" else -1.0,
                        ia if family != "A1" else -1.0,
                        index=n,
                    ),
                    st,
                )
                for n in range(1, k + 1)
            ]
            for n, row in enumerate(rows):
                mean[a, b, n] = [row["primary_mean"][j] for j in config.PRIMARY_IDS]
                sd[a, b, n] = [row["primary_sd"][j] for j in config.PRIMARY_IDS]
                ci1[a, b, n] = row["ci1_model_mean"]
    table = em.ReferenceTable(family, arm, (coords_i, coords_a), design, mean, sd, ci1)
    if centres and family != "A1":
        found = {}
        for i in range(len(LATTICE) - 1):
            for j in range(len(LATTICE) - 1):
                rec = st.load(Job("centre", arm, family, float(i), float(j)).key())
                if rec is None:
                    continue
                p = rec["payload"]
                found[(i, j)] = (
                    np.array(p["eta_design"]),
                    np.array(
                        [[r["primary_mean"][x] for x in config.PRIMARY_IDS] for r in p["nodes"]]
                    ),
                    np.array(
                        [[r["primary_sd"][x] for x in config.PRIMARY_IDS] for r in p["nodes"]]
                    ),
                    np.array([r["ci1_model_mean"] for r in p["nodes"]]),
                )
        if found:
            table = em.with_centres(table, found)
    return table


def validation_records(
    arm: str, family: str, table: em.ReferenceTable, st: CheckpointStore | None = None
) -> list[em.ValidationRecord]:
    st = st or store()
    points = ((0.0, 0.0),) if family == "A1" else VALIDATION_POINTS
    out = []
    for ip, ia in points:
        job = Job(
            "validation",
            arm,
            family,
            ip if family != "A1" else -1.0,
            ia if family != "A1" else -1.0,
        )
        p = load(job, st)
        for off in p["offsets"]:
            mu, sd = em.predict(table, ip, ia, off["eta"])
            out.append(
                em.ValidationRecord(
                    family,
                    (ip, ia),
                    off["offset"],
                    np.array(off["eta"]),
                    mu,
                    sd,
                    np.array([off["crn"]["primary_mean"][j] for j in config.PRIMARY_IDS]),
                    np.array([off["crn"]["primary_sd"][j] for j in config.PRIMARY_IDS]),
                    np.array([off["independent"]["primary_mean"][j] for j in config.PRIMARY_IDS]),
                    np.array([off["independent"]["primary_sd"][j] for j in config.PRIMARY_IDS]),
                )
            )
    return out


# --- gate evaluation (S1C) -----------------------------------------------------------


def run_gate_eval(job: Job) -> dict:
    st = store()
    tol = tolerances()
    truth = load(Job("truth", job.arm, "TRUTH", index=job.index), st)
    point = rq2.production_truth_points()[job.index]
    panels = panel_objects(load(Job("panels", job.arm, "TRUTH", index=job.index), st))
    tables = {f: build_table(job.arm, f, st) for f in ("A1", *FAMILIES)}
    true_family = {"A1": "A1", "A1_PRIME": "A1_PRIME", "A2": "A2"}[point.kind]
    theta_true = tuple(truth["coord"])
    records = []
    for panel in panels:
        evals = {}
        for f in FAMILIES:
            evals[f] = proc.evaluate_family(
                tables[f], panel, tol, theta_true if f == true_family else None
            )
        a1_eval = proc.evaluate_single(tables["A1"], panel, tol)
        tf = a1_eval if true_family == "A1" else evals[true_family]
        g = proc.gate_inputs(tf, evals["A1_PRIME"], evals["A2"], tol)
        records.append(
            {
                **g,
                "truth_in_acc": {str(k): v for k, v in g["truth_in_acc"].items()},
                "refinement_points": {f: evals[f].refinement_points for f in FAMILIES},
                "family_verdicts": {
                    f: {str(k): evals[f].verdict(k, tol).label for k in config.KAPPAS}
                    for f in FAMILIES
                },
            }
        )
    return {
        "label": point.label,
        "truth_class": point.truth_class,
        "true_family": true_family,
        "records": records,
    }


def gate_records(payload: dict) -> list[rq2.GateRecord]:
    out = []
    for r in payload["records"]:
        out.append(
            rq2.GateRecord(
                payload["label"],
                payload["truth_class"],
                {float(k): v for k, v in r["truth_in_acc"].items()},
                r["family_verdict"],
                r["truth_in_nr"],
                r["comparative"],
                r["recovery_distance"],
            )
        )
    return out


# --- job lists and execution -------------------------------------------------------


def map_jobs(arm: str) -> list[Job]:
    jobs = [Job("map", arm, "A1")]
    for f in FAMILIES:
        jobs += [Job("map", arm, f, ip, ia) for ip in LATTICE for ia in LATTICE]
    return jobs


def node_jobs(arm: str) -> list[Job]:
    k = em.eta_dimension(arm)
    return [
        Job("node", arm, j.family, j.ip, j.ia, index=n)
        for j in map_jobs(arm)
        for n in range(1, k + 1)
    ]


def validation_jobs(arm: str) -> list[Job]:
    jobs = [Job("validation", arm, "A1")]
    for f in FAMILIES:
        jobs += [Job("validation", arm, f, ip, ia) for ip, ia in VALIDATION_POINTS]
    return jobs


def truth_jobs(arm: str) -> list[Job]:
    return [Job("truth", arm, "TRUTH", index=i) for i, _ in gate_truths()]


def feasible_truth_indices(arm: str, st=None) -> list[int]:
    return [
        i
        for i, _ in gate_truths()
        if load(Job("truth", arm, "TRUTH", index=i), st)["status"] == "FEASIBLE"
    ]


def panel_jobs(arm: str) -> list[Job]:
    return [Job("panels", arm, "TRUTH", index=i) for i in feasible_truth_indices(arm)]


def h2_jobs(arm: str) -> list[Job]:
    n_sub = len(h2_subgrid())
    return [
        Job("h2", arm, f, index=k, sub=s)
        for f in FAMILIES
        for k in range(H2_PANELS_PER_FAMILY)
        for s in range(n_sub)
    ]


def gate_eval_jobs(arm: str) -> list[Job]:
    return [Job("gate_eval", arm, "TRUTH", index=i) for i in feasible_truth_indices(arm)]


RUNNERS = {
    "a1": run_a1,
    "map": run_map,
    "node": run_node,
    "validation": run_validation,
    "centre": run_centre,
    "truth": run_truth,
    "panels": run_panels,
    "h2": run_h2,
    "gate_eval": run_gate_eval,
}


def run_job(job: Job) -> dict:
    t0, c0 = time.time(), time.process_time()
    payload = RUNNERS[job.kind](job)
    payload["runtime_seconds"] = time.time() - t0
    payload["cpu_seconds"] = time.process_time() - c0
    return payload


def _worker(job: Job) -> tuple[str, str]:
    st = store()
    key = job.key()
    if st.load(key) is not None:
        return key.job_id, "skipped"
    record = st.save(key, run_job(job))
    return key.job_id, record["output_sha256"]


def run_jobs(jobs: list[Job], workers: int = WORKERS, limit: int | None = None) -> dict:
    """Run pending jobs in a spawn pool; completed checkpoints are skipped (resume)."""
    from multiprocessing import get_context

    st = store()
    seen, unique = set(), []
    for j in jobs:
        if j.key().job_id not in seen:
            seen.add(j.key().job_id)
            unique.append(j)
    pending = [j for j in unique if st.load(j.key()) is None]
    if limit is not None:
        pending = pending[:limit]
    t0 = time.time()
    ran = 0
    for var in THREAD_ENV:  # one BLAS/OpenMP thread per worker: no oversubscription and a
        os.environ[var] = "1"  # fixed summation order across resumes (spawned children inherit)
    if pending:
        if workers <= 1:
            for j in pending:
                _worker(j)
                ran += 1
        else:
            with get_context("spawn").Pool(workers) as pool:
                for _ in pool.imap_unordered(_worker, pending):
                    ran += 1
    return {
        "jobs": len(unique),
        "ran": ran,
        "pending_before": len(pending),
        "wall_seconds": time.time() - t0,
    }


# --- reports ------------------------------------------------------------------------


def write_json(relative: str, payload: dict, root: Path = PROJECT_ROOT) -> str:
    require_new_namespace(relative)
    require_p8005_authorized()
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(v):
        if isinstance(v, float) and not math.isfinite(v):
            return str(v)
        return v.tolist() if hasattr(v, "tolist") else str(v)

    def keys(v):
        if isinstance(v, dict):
            return {
                ("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): keys(x)
                for k, x in v.items()
            }
        if isinstance(v, (list, tuple)):
            return [keys(x) for x in v]
        return v

    data = (json.dumps(keys(payload), indent=2, sort_keys=True, default=default) + "\n").encode()
    path.write_bytes(data)
    return _sha(data)


def resolution_report() -> dict:
    res = resolution()
    tol, a1 = res["tolerances"], res["a1"]
    mix = load(Job("a1", em.MIXTURE, "A1"))
    return {
        "run_label": RUN_LABEL,
        "real_quantities_accessed": {
            "CI-1": {
                "estimates": real_ci1().values,
                "se": real_ci1().se,
                "use": "heterogeneity calibration (both arms)",
            },
            "P1..P5": {
                "estimates": stage1.real_primary_reference().values,
                "se": stage1.real_primary_reference().se,
                "use": "tolerance construction only (E_j, rho_j; A1 primary arm)",
            },
        },
        "archive_sha256": verify_real_archive(),
        "c": adq.resolution_constant(),
        "a1_primary": {
            k: a1[k]
            for k in (
                "het",
                "ci1_residuals",
                "ci1_feasible",
                "d_ci",
                "primary_mean",
                "primary_sd",
                "replicates_ok",
            )
        },
        "s_A1": res["s_A1"],
        "amplitudes": {r: rq2.amplitude(r, res["s_A1"]) for r in config.AMPLITUDE_RULE},
        "explanandum": tol.explanandum,
        "rho": tol.rho,
        "kappa_res": tol.kappa_res(),
        "tau": {
            str(k): dict(zip(config.PRIMARY_IDS, tol.tau(k).tolist(), strict=True))
            for k in config.KAPPAS
        },
        "resolution_limited": {str(k): list(tol.resolution_limited(k)) for k in config.KAPPAS},
        "a1_mixture": {
            k: mix.get(k)
            for k in ("het", "ci1_residuals", "ci1_feasible", "d_ci", "mixture", "pi_profile")
        },
        "bracket_available": bool(mix["ci1_feasible"]),
        "bracket_label": None if mix["ci1_feasible"] else representability.BRACKET_UNAVAILABLE,
    }


def truth_report(arms) -> dict:
    out = {}
    for arm in arms:
        rows = []
        for i, _p in gate_truths():
            t = load(Job("truth", arm, "TRUTH", index=i))
            rows.append(
                {
                    k: t.get(k)
                    for k in (
                        "label",
                        "truth_class",
                        "status",
                        "coord",
                        "het",
                        "delta",
                        "realized_pp",
                        "mapping_within_tol",
                        "reason",
                    )
                }
                | {"index": i, "d_ci": (t.get("calibration") or {}).get("d_ci")}
            )
        out[arm] = rows
    outside = [
        (arm, r["label"])
        for arm, rows in out.items()
        for r in rows
        if r["status"] == OUTSIDE_REGION
    ]
    return {"truths": out, "outside_region": outside}


def require_truths_inside(report: dict) -> None:
    if report["outside_region"]:
        raise ResearcherStop(
            "frozen STOP rule: gate truths outside the search lattice "
            f"{report['outside_region']}; researcher decision needed before S1A"
        )


def _summ(values):
    arr = np.array([v for v in values if v is not None], float)
    if arr.size == 0:
        return None
    return {
        "min": float(arr.min()),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def maps_report(arms) -> dict:
    out = {"status": "S1A REPRESENTATIONAL MAPS (no span gate)", "arms": {}}
    for arm in arms:
        arm_out, images = {}, {}
        for f in FAMILIES:
            rows = [load(j) for j in map_jobs(arm) if j.family == f]
            feas = [r for r in rows if r["ci1_feasible"]]
            entry = {
                "total_points": len(rows),
                "ci1_feasible": len(feas),
                "d_ci": _summ([r["d_ci"] for r in rows]),
                "d_ci_infeasible": _summ([r["d_ci"] for r in rows if not r["ci1_feasible"]]),
                "image_range_feasible": {
                    j: [
                        min(r["primary_mean"][j] for r in feas),
                        max(r["primary_mean"][j] for r in feas),
                    ]
                    if feas
                    else None
                    for j in config.PRIMARY_IDS
                },
                "image_range_all": {
                    j: [
                        min(r["primary_mean"][j] for r in rows),
                        max(r["primary_mean"][j] for r in rows),
                    ]
                    for j in config.PRIMARY_IDS
                },
                "boundary_points_feasible": int(sum(r["boundary"] for r in feas)),
                "points": [
                    {
                        "ip": r["ip"],
                        "ia": r["ia"],
                        "feasible": r["ci1_feasible"],
                        "d_ci": r["d_ci"],
                        "het": r["het"],
                        "ci1_residuals": r["ci1_residuals"],
                        "primary_mean": r["primary_mean"],
                        "primary_sd": r["primary_sd"],
                        "delta1_pp_label": r.get("delta1_pp_label"),
                        "delta12_weeks_label": r.get("delta12_weeks_label"),
                        "mixture_flags": (r.get("mixture") or {}).get("flags"),
                    }
                    for r in rows
                ],
            }
            if arm == em.MIXTURE:
                diags = [r["mixture"]["diagnostics"] for r in rows]
                entry["mixture_identification"] = {
                    "jacobian_condition": _summ([d["jacobian_condition"] for d in diags]),
                    "d_over_s": _summ([d["separation"]["d_over_s"] for d in diags]),
                    "unimodal_points": int(sum(d["separation"]["unimodal"] for d in diags)),
                    "boundary_points": int(sum(bool(d["boundary_parameters"]) for d in diags)),
                    "boundary_counts": {
                        b: int(sum(b in d["boundary_parameters"] for d in diags))
                        for b in (
                            "pi_lo",
                            "pi_hi",
                            "mu1_lo",
                            "mu1_hi",
                            "d_lo",
                            "d_hi",
                            "mu2_hi",
                            "s_lo",
                            "s_hi",
                        )
                    },
                    "selection_rules": {
                        rule: int(sum(d["selection_rule"] == rule for d in diags))
                        for rule in sorted({d["selection_rule"] for d in diags})
                    },
                    "passing_starts": _summ(
                        [d["start_agreement"]["passing_starts"] for d in diags]
                    ),
                    "objective_range": _summ(
                        [d["start_agreement"]["objective_range"] for d in diags]
                    ),
                }
            arm_out[f] = entry
            if feas:
                images[f] = span.SimulatedImage(
                    f,
                    np.array(
                        [[r["primary_mean"][c] for c in stage1.SPAN_IMAGE_COORDS] for r in feas]
                    ),
                    np.array(
                        [[r["primary_sd"][c] for c in stage1.SPAN_IMAGE_COORDS] for r in feas]
                    ),
                )
        if len(images) == 2:
            m = span.descriptive_map(images["A1_PRIME"], images["A2"])
            arm_out["span_map"] = {k: v for k, v in m.items() if k not in ("a_in_b", "b_in_a")}
        else:
            arm_out["span_map"] = {
                "status": span.DESCRIPTIVE_MAP,
                "note": "a family has no CI-feasible point",
            }
        a1 = load(Job("map", arm, "A1"))
        arm_out["A1"] = {
            "ci1_feasible": a1["ci1_feasible"],
            "d_ci": a1["d_ci"],
            "het": a1["het"],
            "primary_mean": a1["primary_mean"],
        }
        out["arms"][arm] = arm_out
    return out


def emulator_report(arm: str, tol: adq.Tolerances) -> dict:
    out = {}
    for f in ("A1", *FAMILIES):
        table = build_table(arm, f)
        recs = validation_records(arm, f, table)
        gate = em.crn_gate(recs, tol)
        diag = em.independent_diagnostic(recs, tol)
        ratio = np.array([r.emulated_sd / np.where(r.crn_sd > 0, r.crn_sd, np.nan) for r in recs])
        out[f] = {
            "table_points": int(np.prod([len(a) for a in table.lattice])),
            "nodes_per_point": table.k + 1,
            "refined_cells": sorted(map(list, table.centres)),
            "validation_records": len(recs),
            "crn_gate": gate,
            "max_abs_mean_error": dict(zip(config.PRIMARY_IDS, gate["e"].tolist(), strict=True)),
            "error_budget": dict(
                zip(config.PRIMARY_IDS, proc.emulator_budget(tol).tolist(), strict=True)
            ),
            "sd_ratio_range": [float(np.nanmin(ratio)), float(np.nanmax(ratio))],
            "independent_diagnostic": diag,
            "failing_points": em.failing_points(recs, tol),
        }
    return out


def h2_report(arm: str, tol: adq.Tolerances) -> dict:
    out = {}
    tables = {f: build_table(arm, f) for f in FAMILIES}
    for f in FAMILIES:
        pairs = []
        for job in [j for j in h2_jobs(arm) if j.family == f]:
            p = load(job)
            panels = panel_objects(load(Job("panels", arm, "TRUTH", index=p["truth_index"])))
            panel = panels[p["position"]]
            coord = tuple(p["coord"])
            if p["ci1_feasible"]:
                ev = adq.evidence_from_simulation(
                    panel.primary,
                    panel.primary_se,
                    np.array([p["primary_mean"][j] for j in config.PRIMARY_IDS]),
                    np.array([p["primary_sd"][j] for j in config.PRIMARY_IDS]),
                    proc.H2_R,
                )
                status = {k: adq.point_status(ev, tol.tau(k)) for k in config.KAPPAS}
                direct = adq.PointOutcome(coord, True, False, status, math.inf)
            else:
                direct = adq.PointOutcome(coord, False, False, {}, math.inf)
            emulated, _ = proc.evaluate_emulated(tables[f], coord, panel, tol)
            pairs.append((direct, emulated))
        out[f] = proc.h2_agreement(pairs)
    return out


def perfect_model_report(arm: str) -> dict:
    out, passed = {"families": {}}, True
    by_family = {"A1": "T0", "A1_PRIME": "T1", "A2": "T3"}
    truths = {i: load(Job("truth", arm, "TRUTH", index=i)) for i, _ in gate_truths()}
    for fam, cls in by_family.items():
        idx = [i for i, p in gate_truths() if p.truth_class == cls]
        infeasible = [truths[i]["label"] for i in idx if truths[i]["status"] != "FEASIBLE"]
        assessable = [i for i in idx if truths[i]["status"] == "FEASIBLE"]
        records = []
        for i in assessable:
            records += gate_records(load(Job("gate_eval", arm, "TRUTH", index=i)))
        gate = rq2.perfect_model_gate(records) if records else {"points": {}, "pass": False}
        fam_pass = bool(gate["pass"] and assessable)
        out["families"][fam] = {
            "gate_points": [truths[i]["label"] for i in idx],
            "infeasible_truths": infeasible,
            "assessable_points": len(assessable),
            "gate": gate,
            "pass": fam_pass,
        }
        passed = passed and fam_pass
    out["pass"] = passed
    return out


def frontier_report(arms) -> dict:
    out = {}
    for arm in arms:
        out[arm] = {}
        for f in FAMILIES:
            rows = [load(j) for j in map_jobs(arm) if j.family == f]
            out[arm][f] = frontier.frontier(
                [
                    {
                        "ip": r["ip"],
                        "ia": r["ia"],
                        "ci1_residuals": r["ci1_residuals"],
                        "ci1_feasible": r["ci1_feasible"],
                        "images": {c: r["primary_mean"][c] for c in frontier.COORDS},
                    }
                    for r in rows
                ]
            )
    return out


def representability_preflight(arm: str, tol: adq.Tolerances) -> dict:
    """Classifier on perfect-model truths (engineering check; not RQ2, no real targets)."""
    tables = {f: build_table(arm, f) for f in FAMILIES}
    e = proc.emulator_budget(tol)
    se6 = proc.ci1_se()
    competitor = {"T0": "A2", "T1": "A2", "T3": "A1_PRIME"}
    out = {}
    for i, p in gate_truths():
        truth = load(Job("truth", arm, "TRUTH", index=i))
        if truth["status"] != "FEASIBLE":
            out[p.label] = representability.INFEASIBLE_TRUTH
            continue
        panels = load(Job("panels", arm, "TRUTH", index=i))["panels"]
        prim = np.array([q["primary"] for q in panels])
        ci1 = np.mean([q["ci1"] for q in panels], axis=0)
        tm = representability.TruthMoments(prim.mean(0), prim.std(0, ddof=1), len(panels))
        table = tables[competitor[p.truth_class]]
        means, sds = [], []
        for ip in LATTICE:
            for ia in LATTICE:
                eta = em.recalibrate(table, ip, ia, ci1, se6)
                resid = em.predict_ci1(table, ip, ia, eta) - ci1
                if np.all(np.abs(resid) <= np.maximum(config.CI1_GATE_ABS, 2 * se6)):
                    mu, sd = em.predict(table, ip, ia, eta)
                    means.append(mu)
                    sds.append(np.maximum(sd, 0))
        image = representability.CompetitorImage(np.array(means), np.array(sds))
        out[p.label] = {
            "competitor": competitor[p.truth_class],
            "competitor_ci_feasible_points": len(means),
            "states": {
                str(k): v
                for k, v in representability.classify_all_kappas(tm, image, tol, e).items()
            },
        }
    return out


# --- orchestration ------------------------------------------------------------------


def available_arms() -> tuple[str, ...]:
    return ARMS if bracket_available() else (em.LOGIT_NORMAL,)


def emulator_gate_with_refinement(arm: str, tol: adq.Tolerances, workers: int) -> dict:
    """CRN gate per family; one predeclared refinement for failing points; then final."""
    report = emulator_report(arm, tol)
    failing = {f: r["failing_points"] for f, r in report.items() if not r["crn_gate"]["pass"]}
    refined = {}
    if failing:
        jobs = []
        for f, pts in failing.items():
            if f == "A1":
                continue  # a single-point table has no cells to refine
            table = build_table(arm, f, centres=False)
            cells = sorted({c for p in pts for c in em.neighbouring_cells(table, *p)})
            refined[f] = cells
            jobs += [Job("centre", arm, f, float(i), float(j)) for i, j in cells]
        if jobs:
            run_jobs(jobs, workers)
        report = emulator_report(arm, tol)
    return {
        "families": report,
        "refinement_applied": bool(failing),
        "refined_cells": refined,
        "pass": all(r["crn_gate"]["pass"] for r in report.values()),
    }


def execute(workers: int = WORKERS, through: str = "ALL") -> dict:
    """Run the S1-P8005 stage order with checkpoints, writing aggregate artifacts.

    Raises ``ResearcherStop`` when a frozen STOP rule fires (truth outside the region,
    S1B or S1C failure); artifacts written up to that point stay on disk.
    """
    log = {"run_label": RUN_LABEL, "steps": []}

    def step(name, info):
        log["steps"].append(
            {"step": name, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **info}
        )

    step("a1-logit", run_jobs([Job("a1", em.LOGIT_NORMAL, "A1")], workers))
    step("a1-mixture", run_jobs([Job("a1", em.MIXTURE, "A1")], workers))
    res = resolution_report()
    write_json(f"{ARCHIVE}/s1p_resolution.json", res)
    arms = available_arms()
    tol = tolerances()
    step("truth-logit", run_jobs(truth_jobs(em.LOGIT_NORMAL), workers))
    if em.MIXTURE in arms:
        step("truth-mixture", run_jobs(truth_jobs(em.MIXTURE), workers))
    truths = truth_report(arms)
    write_json(f"{ARCHIVE}/s1p_truths.json", truths)
    require_truths_inside(truths)
    if through == "TRUTH":
        return log
    for arm in arms:
        step(f"maps-{arm}", run_jobs(map_jobs(arm), workers))
    maps = maps_report(arms)
    maps["bracket_available"] = em.MIXTURE in arms
    maps["bracket_label"] = None if em.MIXTURE in arms else representability.BRACKET_UNAVAILABLE
    write_json(f"{ARCHIVE}/s1p_maps.json", maps)
    if through == "S1A":
        return log
    emu = {}
    for arm in arms:
        step(f"nodes-{arm}", run_jobs(node_jobs(arm), workers))
        step(f"validation-{arm}", run_jobs(validation_jobs(arm), workers))
        emu[arm] = emulator_gate_with_refinement(arm, tol, workers)
    for arm in arms:
        step(f"panels-{arm}", run_jobs(panel_jobs(arm), workers))
        step(f"h2-{arm}", run_jobs(h2_jobs(arm), workers))
        emu[arm]["h2"] = h2_report(arm, tol)
        emu[arm]["pass"] = bool(
            emu[arm]["pass"] and all(v["pass"] for v in emu[arm]["h2"].values())
        )
    emu["pass"] = all(emu[a]["pass"] for a in arms)
    write_json(f"{ARCHIVE}/s1p_emulator.json", emu)
    if not emu["pass"]:
        raise ResearcherStop(
            "S1B emulator/reference validation failed after the predeclared "
            "refinement: STOP (no S1C, no Stage 2)"
        )
    for arm in arms:
        step(f"gate-eval-{arm}", run_jobs(gate_eval_jobs(arm), workers))
    perfect = {arm: perfect_model_report(arm) for arm in arms}
    perfect["pass"] = all(perfect[a]["pass"] for a in arms)
    write_json(f"{ARCHIVE}/s1p_perfect_model.json", perfect)
    if not perfect["pass"]:
        raise ResearcherStop("S1C perfect-model gate failed: STOP (no Stage 2)")
    write_json(f"{ARCHIVE}/s1p_frontier.json", frontier_report(arms))
    write_json(
        f"{ARCHIVE}/s1p_representability_preflight.json",
        {arm: representability_preflight(arm, tol) for arm in arms},
    )
    log["arms"] = list(arms)
    return log


def verify_resume() -> dict:
    """Resume audit on real S1-P8005 checkpoints (deterministic, idempotent, guarded)."""
    from complete_journey.abm.checkpoint import _canonical_json, _to_builtin

    st = store()
    candidates = [j for a in ARMS for j in node_jobs(a)] + [j for a in ARMS for j in map_jobs(a)]
    done = [j for j in candidates if st.load(j.key()) is not None][:2]
    require(len(done) == 2, "need two completed S1-P8005 checkpoints to audit")
    matches = {}
    for job in reversed(done):  # recompute in reverse order: order invariance
        rec = st.load(job.key())
        fresh = run_job(job) | {
            "runtime_seconds": rec["payload"]["runtime_seconds"],
            "cpu_seconds": rec["payload"]["cpu_seconds"],
        }
        matches[job.key().job_id] = (
            _sha(_canonical_json(_to_builtin(fresh))) == rec["output_sha256"]
        )
    foreign = CheckpointStore(st.root, "0" * 64, st.source_commit)
    try:
        foreign.load(done[0].key())
        foreign_refused = False
    except Phase8ContractError:
        foreign_refused = True
    snap = json.loads((PROJECT_ROOT / SNAPSHOT).read_bytes())
    tampered = {
        **snap,
        "source_sha256": {**snap["source_sha256"], "src/complete_journey/abm/config.py": "0" * 64},
    }
    try:
        check_snapshot(tampered)
        source_refused = False
    except Phase8ContractError:
        source_refused = True
    files = list(st.root.glob("*.json"))
    return {
        "jobs_audited": [str(j) for j in done],
        "deterministic_output_hash_match": all(matches.values()),
        "order_invariant_recompute": all(matches.values()),
        "skipped_on_resume": _worker(done[0])[1] == "skipped",
        "foreign_config_refused": foreign_refused,
        "source_mismatch_refused": source_refused,
        "checkpoint_files": len(files),
        "unique_job_ids": len({f.stem for f in files}),
        "no_duplicates": len(files) == len({f.stem for f in files}),
        "snapshot_sha256": verify_snapshot(),
    }
