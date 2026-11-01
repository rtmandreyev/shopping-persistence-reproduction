"""Phase 8 Stage 1 driver: design snapshot, resolution metadata, span parity (S1-A).

Stage 1 is authorized by ``config.AUTHORIZED_STAGE = 1`` (researcher instruction,
committed in the Stage 1 authorization commit).

**Real data Stage 1 may access** (archived in S1-RESOLUTION):
- the real CI-1 vector and its SEs, for heterogeneity calibration;
- the archived real P1..P5 estimates and SEs, for tolerance construction only:
  E_j = m_j^real − μ_j^A1 and ρ_j = c·SE_j.

**Real data Stage 1 never accesses.** No A1′ or A2 point is compared with a real
dynamic moment, and no real acceptance set, κ* or verdict exists. The span check uses
simulated images and simulated scales only.

**Execution.** Every job is keyed by deterministic seeds (stage 1) and checkpointed
atomically, so results do not depend on worker count or order. Large tables stay under
the ignored ``outputs/diagnostics/phase8_stage1/``. Aggregate, identifier-free
artifacts go to ``docs/phases/PHASE_08_STAGE1/``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

from complete_journey.abm import adequacy as adq
from complete_journey.abm import architectures as arch
from complete_journey.abm import config, mapping, regions, span
from complete_journey.abm.calibration import calibrate
from complete_journey.abm.checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from complete_journey.abm.checkpoint import CheckpointStore, JobKey, _canonical_json, _to_builtin
from complete_journey.abm.config import Phase8ContractError, require
from complete_journey.abm.guards import TargetVector, archive_contract_values, verify_real_archive
from complete_journey.abm.moments import BUNDLE_SECONDARY, moment_bundle
from complete_journey.abm.operators import envelopes, world_panel
from complete_journey.abm.seeds import Family, Role, SeedKey
from complete_journey.paths import PROJECT_ROOT

STAGE = 1
ARCHIVE = "docs/phases/PHASE_08_STAGE1"
SNAPSHOT = f"{ARCHIVE}/design_snapshot.json"
SNAPSHOT_SHA = f"{ARCHIVE}/design_snapshot.sha256"
WORK = "outputs/diagnostics/phase8_stage1"
WORKERS = 5  # M2 Pro: 6 performance + 4 efficiency cores; conservative
SPAN_IMAGE_COORDS = ("P1", "P2", "P3")  # frozen span coordinates (§AB.1)
SPAN_R = config.R_EMULATOR  # direct R = 50 per point (§AB.1)
SOURCE_GLOBS = ("src/complete_journey/abm/*.py", "src/complete_journey/prediction/*.py")
SOURCE_FILES = ("scripts/run_abm.py",)

#: Emulator mechanical details fixed here, before any validation (P8-001 §AQ.5/§AC).
EMULATOR_ETA_OFFSETS = ((0.0, 0.0), (0.10, 0.0), (0.0, 0.10))  # (Δμ, Δs) design nodes
EMULATOR_VALIDATION_OFFSETS = ((0.03, -0.03), (-0.03, 0.03))
EMULATOR_VALIDATION_POINTS = (
    *((i, j) for i in (0, 3, 6, 9) for j in (0, 3, 6, 9)),  # frozen 4 × 4 lattice
    (1.5, 1.5),
    (4.5, 7.5),
    (7.5, 4.5),
    (9.5, 9.5),  # frozen rule: 4 cell centres
    (11, 11),
    (11, 0),
    (0, 11),
    (11, 6),
    (6, 11),
    (10.5, 10.5),  # supplementary caps
)


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
    for d in ("P8-001", "P8-002", "P8-003"):
        start = text.index(f"### {d} ")
        nxt = text.find("\n### ", start + 5)
        out[d] = _sha(text[start : nxt if nxt != -1 else len(text)].encode())
    return out


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def build_snapshot(root: Path = PROJECT_ROOT) -> dict:
    versions = {pkg: metadata.version(pkg) for pkg in ("numpy", "scipy", "pandas", "scikit-learn")}
    return {
        "schema": "phase8-stage1-snapshot-v1",
        "source_commit_at_snapshot": _git_head(root),
        "source_sha256": source_hashes(root),
        "decision_sha256": decision_hashes(root),
        "authorized_stage": config.AUTHORIZED_STAGE,
        "architectures": {
            "A0": "logit p = mu",
            "A1": "logit p = alpha_i, alpha_i = mu + s*xi_i",
            "A1_PRIME": "z_t = rho z_{t-1} + sqrt(1-rho^2) eps_t, z_0 ~ N(0,1); "
            "logit p = alpha_i + sigma_z z_t",
            "A2": "H_{t+1} = lam H_t + (1-lam) y_t, H_start = 0; logit p = alpha_i + delta H_t",
        },
        "caps": {
            "half_life_weeks": config.HALF_LIFE_CAP_WEEKS,
            "persistence_cap": config.PERSISTENCE_CAP,
            "sigma_z_cap": config.SIGMA_Z_CAP,
            "a2_delta_cap": config.A2_DELTA_CAP,
        },
        "grid": {
            "size": config.GRID_SIZE,
            "persistence_axis": list(regions.persistence_axis()),
            "half_life_axis": list(regions.half_life_axis()),
            "amplitude_axis_a1_prime": list(regions.amplitude_axis(Family.A1_PRIME)),
            "amplitude_axis_a2": list(regions.amplitude_axis(Family.A2)),
            "refinement_max_levels": config.REFINEMENT_MAX_LEVELS,
            "refinement_max_points": config.REFINEMENT_MAX_POINTS,
        },
        "primary_moments": [list(m) for m in config.PRIMARY_MOMENTS],
        "p8_003": {
            "alpha": config.ALPHA,
            "z_eq": adq.z_equivalence(),
            "z_star": adq.z_inadequacy(),
            "q_J": adq.per_moment_power_quantile(),
            "c": adq.resolution_constant(),
            "g_plus": config.G_PLUS,
            "g_minus": config.G_MINUS,
            "emulator_mean_error_fraction": config.EMULATOR_MEAN_ERROR_FRACTION,
            "emulator_sd_tolerance": config.EMULATOR_SD_TOLERANCE,
        },
        "kappas": list(config.KAPPAS),
        "kappa_reference": config.KAPPA_REFERENCE,
        "perfect_model_gates": {
            "G1_truth_inclusion_min_every_kappa": config.GATE_TRUTH_INCLUSION_MIN,
            "G2_family_adequate_min": config.GATE_FAMILY_ADEQUATE_MIN,
            "G3_true_inadequate_max": config.GATE_TRUE_INADEQUATE_MAX,
            "G4_nr_inclusion_min": config.GATE_NR_INCLUSION_MIN,
            "G5_a1_history_max": config.GATE_A1_HISTORY_MAX,
        },
        "span_gate": {
            "coords": list(SPAN_IMAGE_COORDS),
            "box_half_width_sd": config.SPAN_BOX_SD,
            "coverage_min": config.SPAN_COVERAGE_MIN,
            "retain_min": config.SPAN_RETAIN_MIN,
            "direct_R": SPAN_R,
        },
        "emulator": {
            "type": "local-affine in (mu, s) from 3 design nodes; bilinear in grid index",
            "eta_design_offsets": [list(o) for o in EMULATOR_ETA_OFFSETS],
            "R_table": config.R_EMULATOR,
            "validation_points": [list(p) for p in EMULATOR_VALIDATION_POINTS],
            "validation_eta_offsets": [list(o) for o in EMULATOR_VALIDATION_OFFSETS],
            "validation_direct_R": config.R_DIRECT,
            "gate": "|mu_em - mu_direct| <= 0.1 * tau_j(0.15); s_em in [0.8, 1.2] * s_direct",
        },
        "budgets": {
            "R_cal": config.R_CAL,
            "R_direct": config.R_DIRECT,
            "R_final_max": config.R_FINAL_MAX,
            "Q_gate": config.Q_GATE,
            "Q_truth": config.Q_TRUTH,
            "N": config.N_HOUSEHOLDS,
        },
        "seed_tree": {
            "entropy": config.SEED_ENTROPY,
            "namespace": config.SEED_NAMESPACE,
            "version": "spawn_key=(8, stage, truth_class, truth_point, q, role, family, "
            "grid_point, replicate, stream)",
        },
        "operators": {
            "primary": config.OPERATOR_PRIMARY,
            "s_a": config.OPERATOR_SA,
            "analysis_operator": config.OPERATOR_O,
            "min_interior_weeks": config.MIN_INTERIOR_WEEKS,
            "burn_in_weeks": config.BURN_IN_WEEKS,
        },
        "software": {"python": platform.python_version(), **versions},
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "workers": WORKERS,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
    }


def write_snapshot(root: Path = PROJECT_ROOT) -> str:
    path = root / SNAPSHOT
    require(not path.exists(), "the Stage 1 design snapshot exists and is never rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(build_snapshot(root), indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    digest = _sha(data)
    (root / SNAPSHOT_SHA).write_text(digest + "\n")
    return digest


def verify_snapshot(root: Path = PROJECT_ROOT) -> str:
    """The committed snapshot hash matches and the source still matches the snapshot."""
    data = (root / SNAPSHOT).read_bytes()
    digest = _sha(data)
    require(digest == (root / SNAPSHOT_SHA).read_text().strip(), "Stage 1 snapshot tampered")
    snap = json.loads(data)
    require(snap["source_sha256"] == source_hashes(root), "source drifted from the snapshot")
    require(snap["decision_sha256"] == decision_hashes(root), "P8 decisions drifted")
    require(config.AUTHORIZED_STAGE >= STAGE, "Stage 1 is not authorized")
    return digest


def store(root: Path = PROJECT_ROOT) -> CheckpointStore:
    digest = verify_snapshot(root)
    return CheckpointStore(root / WORK / "checkpoints", digest, _git_head(root))


# --- real Stage 1 inputs (exact, archived accesses) -----------------------------------


def real_ci1_target() -> TargetVector:
    archive = archive_contract_values()
    values = {c: archive[("CI-1", c)]["estimate"] for c in config.CI1_COMPONENTS}
    se = {c: archive[("CI-1", c)]["se"] for c in config.CI1_COMPONENTS}
    return TargetVector("real", values, se)


def real_primary_reference() -> TargetVector:
    """Real P1..P5 estimates and SEs: tolerance construction only (P8-001 §T, P8-003)."""
    archive = archive_contract_values()
    values = {p8: archive[(m, c)]["estimate"] for p8, m, c in config.PRIMARY_MOMENTS}
    se = {p8: archive[(m, c)]["se"] for p8, m, c in config.PRIMARY_MOMENTS}
    return TargetVector("real", values, se)


# --- jobs -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    kind: str  # "span" | "a1"
    family: str
    ip: float
    ia: float

    def key(self) -> JobKey:
        point = f"{self.kind}:{self.ip:g},{self.ia:g}"
        seeds = (
            (config.SEED_NAMESPACE, STAGE, int(Role.CALIBRATION)),
            (config.SEED_NAMESPACE, STAGE, int(Role.MODEL)),
        )
        reps = SPAN_R if self.kind == "span" else config.R_DIRECT
        return JobKey(STAGE, self.family, point, 0, reps, seeds)


def _envelope_stats(world: arch.World) -> dict:
    env = envelopes(world_panel(world))
    length = env["w_len"].fillna(0).to_numpy()
    status = env["status"].value_counts(normalize=True).to_dict()
    return {
        "mean_interior_weeks": float(length.mean()),
        "share_eligible": float(status.get("eligible", 0.0)),
        "share_short": float(status.get("short", 0.0)),
        "share_single": float(status.get("single_purchase", 0.0)),
        "share_no_purchase": float(status.get("no_purchase", 0.0)),
        "corr_alpha_interior": float(np.corrcoef(world.alpha, length)[0, 1])
        if length.std() > 0
        else None,
    }


def simulate_bundle(spec: arch.Spec, het: arch.Heterogeneity, replicates: int) -> dict:
    """Means/SDs of P1..P5 and the secondary summaries over MODEL-role replicates."""
    prim, sec, failures, labels = [], [], 0, []
    envelope = None
    for r in range(replicates):
        world = arch.simulate(
            spec, het, SeedKey(STAGE, Role.MODEL, replicate=r), config.N_HOUSEHOLDS
        )
        if r == 0:
            envelope = _envelope_stats(world)
        try:
            p, s = moment_bundle(world_panel(world))
        except (Phase8ContractError, ValueError, np.linalg.LinAlgError):
            failures += 1
            continue
        prim.append([p.values[j] for j in config.PRIMARY_IDS])
        sec.append([s[k] for k in sorted(s)])
        if spec.family == Family.A2:
            labels.append(mapping.delta1(world))
    prim, sec = np.array(prim), np.array(sec)
    require(len(prim) >= 2, "fewer than two successful replicates")
    keys = sorted(BUNDLE_SECONDARY)
    out = {
        "replicates_ok": len(prim),
        "failures": failures,
        "primary_mean": dict(zip(config.PRIMARY_IDS, prim.mean(0).tolist(), strict=True)),
        "primary_sd": dict(zip(config.PRIMARY_IDS, prim.std(0, ddof=1).tolist(), strict=True)),
        "secondary_mean": dict(zip(keys, sec.mean(0).tolist(), strict=True)),
        "secondary_sd": dict(zip(keys, sec.std(0, ddof=1).tolist(), strict=True)),
        "envelope": envelope,
    }
    if spec.family == Family.A2:
        out["delta1_pp_label"] = float(np.mean(labels))
        out["delta12_weeks_label"] = mapping.delta12(
            spec, het, SeedKey(STAGE, Role.MODEL, replicate=0), config.N_HOUSEHOLDS
        )
    return out


def run_job(job: Job) -> dict:
    t0 = time.time()
    target = real_ci1_target()
    if job.kind == "a1":
        spec = arch.a1()
        params = ()
        replicates = config.R_DIRECT
        boundary = False
    else:
        family = Family[job.family]
        point = regions.make_point(family, job.ip, job.ia, 0)
        spec = arch.Spec(family, "primary", point.params)
        params = point.params
        replicates = SPAN_R
        boundary = point.boundary
    cal = calibrate(spec, target, stage=STAGE, n=config.N_HOUSEHOLDS, r_cal=config.R_CAL)
    payload = {
        "kind": job.kind,
        "family": job.family,
        "ip": job.ip,
        "ia": job.ia,
        "params": dict(params),
        "boundary": boundary,
        "mu": cal.heterogeneity.mu,
        "s": cal.heterogeneity.s,
        "ci1_sim": cal.ci1_sim.tolist(),
        "ci1_residuals": cal.residuals.tolist(),
        "ci1_feasible": cal.feasible,
        "calibration_evaluations": cal.evaluations,
    }
    if cal.feasible:
        payload.update(simulate_bundle(spec, cal.heterogeneity, replicates))
    payload["runtime_seconds"] = time.time() - t0
    return payload


def _worker(job: Job) -> tuple[str, str]:
    st = store()
    key = job.key()
    if st.load(key) is not None:
        return key.job_id, "skipped"
    record = st.save(key, run_job(job))
    return key.job_id, record["output_sha256"]


def span_jobs() -> list[Job]:
    jobs = []
    for family in regions.PRIMARY_FAMILIES:
        for pt in regions.build_grid(family):
            jobs.append(Job("span", family.name, float(pt.ip), float(pt.ia)))
    return jobs


def run_jobs(jobs: list[Job], workers: int = WORKERS, limit: int | None = None) -> dict:
    """Run pending jobs in a process pool; completed checkpoints are skipped (resume)."""
    from multiprocessing import get_context

    st = store()
    pending = [j for j in jobs if st.load(j.key()) is None]
    if limit is not None:
        pending = pending[:limit]
    t0 = time.time()
    results = []
    if pending:
        with get_context("spawn").Pool(workers) as pool:
            for item in pool.imap_unordered(_worker, pending):
                results.append(item)
    return {"ran": len(results), "pending_before": len(pending), "wall_seconds": time.time() - t0}


def load_payloads(jobs: list[Job]) -> list[dict]:
    st = store()
    out = []
    for j in jobs:
        rec = st.load(j.key())
        require(rec is not None, f"missing checkpoint for {j}")
        out.append(rec["payload"])
    return out


# --- resolution metadata (S1-RESOLUTION) ----------------------------------------------


def resolution_report() -> dict:
    a1 = load_payloads([Job("a1", "A1", -1.0, -1.0)])[0]
    require(a1["ci1_feasible"], "A1 fails the CI-1 gate against the real CI-1: STOP (§F)")
    reference = real_primary_reference()
    tol = adq.build_tolerances(reference, a1["primary_mean"])
    c = adq.resolution_constant()
    return {
        "real_quantities_accessed": {
            "CI-1": {
                "estimates": real_ci1_target().values,
                "se": real_ci1_target().se,
                "use": "heterogeneity calibration",
            },
            "P1..P5": {
                "estimates": reference.values,
                "se": reference.se,
                "use": "tolerance construction only (E_j, rho_j)",
            },
        },
        "archive_sha256": verify_real_archive(),
        "c": c,
        "a1": {
            "mu": a1["mu"],
            "s": a1["s"],
            "ci1_residuals": a1["ci1_residuals"],
            "primary_mean": a1["primary_mean"],
            "primary_sd": a1["primary_sd"],
            "replicates_ok": a1["replicates_ok"],
        },
        "s_A1": a1["s"],
        "amplitudes": {
            "LOW": config.AMPLITUDE_RULE["LOW"] * a1["s"],
            "HIGH": config.AMPLITUDE_RULE["HIGH"] * a1["s"],
        },
        "explanandum": tol.explanandum,
        "rho": tol.rho,
        "kappa_res": tol.kappa_res(),
        "tau": {
            str(k): dict(zip(config.PRIMARY_IDS, tol.tau(k).tolist(), strict=True))
            for k in config.KAPPAS
        },
        "resolution_limited": {str(k): list(tol.resolution_limited(k)) for k in config.KAPPAS},
    }


# --- span analysis (S1-SPAN) ----------------------------------------------------------


def span_report() -> dict:
    payloads = load_payloads(span_jobs())
    by_family = {
        f.name: [p for p in payloads if p["family"] == f.name] for f in regions.PRIMARY_FAMILIES
    }
    images, summary = {}, {}
    for name, rows in by_family.items():
        feas = [r for r in rows if r["ci1_feasible"]]
        means = np.array([[r["primary_mean"][c] for c in SPAN_IMAGE_COORDS] for r in feas])
        sds = np.array([[r["primary_sd"][c] for c in SPAN_IMAGE_COORDS] for r in feas])
        images[name] = (feas, span.SimulatedImage(name, means, sds))
        summary[name] = {
            "total_points": len(rows),
            "ci1_feasible": len(feas),
            "image_range": {
                c: [float(means[:, k].min()), float(means[:, k].max())]
                for k, c in enumerate(SPAN_IMAGE_COORDS)
            },
        }
    (fa, ia_), (fb, ib) = images["A1_PRIME"], images["A2"]
    res = span.span_check(ia_, ib)
    for name, feas, keep in (("A1_PRIME", fa, res.keep_a), ("A2", fb, res.keep_b)):
        s = summary[name]
        s["unique_points"] = int((~keep).sum())
        s["common_points"] = int(keep.sum())
        s["boundary_points_feasible"] = int(sum(r["boundary"] for r in feas))
        s["boundary_points_unique"] = int(
            sum(r["boundary"] for r, k in zip(feas, keep, strict=True) if not k)
        )
        long_mem = [i for i, r in enumerate(feas) if r["ip"] >= 9]
        s["long_memory_points"] = len(long_mem)
        s["long_memory_unique"] = int(sum(not keep[i] for i in long_mem))
        common = np.array(
            [
                [r["primary_mean"][c] for c in SPAN_IMAGE_COORDS]
                for r, k in zip(feas, keep, strict=True)
                if k
            ]
        )
        s["common_image_range"] = (
            {
                c: [float(common[:, k].min()), float(common[:, k].max())]
                for k, c in enumerate(SPAN_IMAGE_COORDS)
            }
            if len(common)
            else None
        )
        s["trim_mask"] = [[r["ip"], r["ia"], bool(k)] for r, k in zip(feas, keep, strict=True)]
    return {
        "coverage_a1p_in_a2": res.coverage_a_in_b,
        "coverage_a2_in_a1p": res.coverage_b_in_a,
        "status": res.status,
        "families": summary,
        "linear_vs_nonlinear": linear_comparison(by_family),
    }


def linear_comparison(by_family: dict) -> dict:
    """Linear-approximation proxies: ρ1 ≈ β_1 and per-week decay φ ≈ (β_8/β_1)^(1/7)."""
    out = {}
    for name, rows in by_family.items():
        pts = []
        for r in rows:
            if not r["ci1_feasible"]:
                continue
            b1, b8 = r["secondary_mean"]["beta_1"], r["primary_mean"]["P2"]
            phi = (b8 / b1) ** (1 / 7) if b1 > 0 and b8 > 0 else None
            pts.append((b1, phi))
        b1s = [p[0] for p in pts]
        phis = [p[1] for p in pts if p[1] is not None]
        out[name] = {
            "beta_1_range": [min(b1s), max(b1s)] if b1s else None,
            "decay_phi_range": [min(phis), max(phis)] if phis else None,
            "points_with_positive_b1_b8": len(phis),
        }
    return out


def write_json(relative: str, payload: dict, root: Path = PROJECT_ROOT) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(v):
        if isinstance(v, float) and not math.isfinite(v):
            return str(v)
        return v.tolist() if hasattr(v, "tolist") else str(v)

    data = (json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n").encode()
    path.write_bytes(data)
    return _sha(data)


def verify_resume() -> dict:
    """Resume audit on real Stage 1 checkpoints.

    It checks four things:
    - a completed job is skipped, not re-run;
    - recomputing it reproduces the checkpointed output hash (determinism);
    - a different config hash is refused;
    - the snapshot still matches the source.
    """
    st = store()
    done = [j for j in span_jobs() if st.load(j.key()) is not None]
    require(done, "no completed Stage 1 span checkpoint to audit")
    job = done[0]
    rec = st.load(job.key())
    payload = run_job(job) | {"runtime_seconds": rec["payload"]["runtime_seconds"]}
    recomputed_sha = _sha(_canonical_json(_to_builtin(payload)))
    foreign = CheckpointStore(st.root, "0" * 64, st.source_commit)
    try:
        foreign.load(job.key())
        foreign_refused = False
    except Phase8ContractError:
        foreign_refused = True
    files = list(st.root.glob("*.json"))
    return {
        "job": f"{job.family} {job.ip:g},{job.ia:g}",
        "deterministic_output_hash_match": recomputed_sha == rec["output_sha256"],
        "skipped_on_resume": _worker(job)[1] == "skipped",
        "foreign_config_refused": foreign_refused,
        "checkpoint_files": len(files),
        "unique_job_ids": len({f.stem for f in files}),
        "snapshot_sha256": verify_snapshot(),
    }
