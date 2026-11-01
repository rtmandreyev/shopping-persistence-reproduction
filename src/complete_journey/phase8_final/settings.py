"""Frozen P8-006 settings for the Phase 8 final run (authority: docs/DECISIONS.md P8-006).

Everything scientific that is not listed here is inherited unchanged from P8-001 to
P8-005 through ``complete_journey.abm.config`` (architectures, caps, grid, moments,
CI-1 gate, P8-003 rule, κ, truth grid, seeds).
"""

from __future__ import annotations

from types import MappingProxyType

from complete_journey.abm import config as abm_config

RUN_LABEL = "P8-FINAL"
ARCHIVE = "docs/phases/PHASE_08_FINAL"
WORK = "outputs/diagnostics/phase8_final"
SNAPSHOT = f"{ARCHIVE}/design_snapshot.json"
SNAPSHOT_SHA = f"{ARCHIVE}/design_snapshot.sha256"
SNAPSHOT_SCHEMA = "phase8-final-snapshot-v1-p8006"
#: Researcher authorization of the final run (set only in the authorization commit).
FINAL_RUN_AUTHORIZED = True  # researcher authorization (P8-006 final run)

# --- deadline governance (P8-006) ----------------------------------------------------
DEADLINE_START_UTC = "2026-10-19T13:35:07Z"
DEADLINE_PREFERRED_HOURS = 48.0
DEADLINE_CEILING_HOURS = 72.0
TIMING_CHECKPOINT_HOURS = (36.0, 48.0, 60.0)

# --- model table (plug-in): the real-CI-1 calibration of every family point --------
#: Seed-tree stage key of the model table. It is the S1-P8005 key, so the imported S1A
#: calibrations and R = 50 images are continued, never redrawn.
MODEL_STAGE = abm_config.P8005_STAGE_KEY  # 11
R_COARSE = 200  # coarse-grid images: S1A replicates 0..49 imported + 50..199 extended
R_REFINE = 50  # first-level refinement points and real-panel level-2 (secondary)
R_THETA_TRUE = 200  # perfect-model θ_true images
R_AUDIT = 200  # real-panel audits (κ* argmin + accepted points)
AUDIT_MAX_ACCEPTED = 10  # frozen §AC: up to 10 accepted points, smallest κ* first
REFINEMENT_LEVELS_CONFIRMATORY = 1  # frozen first level; level 2 = real-panel secondary
#: Exact pruning: a cell whose four corners are all CI-1-infeasible is never refined by
#: the frozen rule (ADEQ needs a feasible point; the κ* argmin needs a finite κ*).

# --- known-truth experiment ------------------------------------------------------
Q = abm_config.Q_TRUTH  # 200
Q_POP = 200
POP_Q_OFFSET = 10_000  # population block uses panel indices 10000..10199 (disjoint)
RQ2_DATA_STAGE = 3  # frozen: RQ2 truth panels use Stage 3 DATA keys
GATE_DATA_STAGE = abm_config.P8005_STAGE_KEY  # S1-P8005 gate panels (Stage 1 keys)
PANEL_CHUNK = 20
TRUTH_CLASSES = ("T0", "T1", "T2", "T3", "T4")  # T5 deferred
#: Truth configuration uses the model-table calibration semantics (stage 11, real CI-1,
#: A1 start), so the true family at θ_true carries exactly the truth's heterogeneity.

# --- calibration-noise checks (predeclared; the frozen Stage 2 pilot points) --------
CALNOISE_THETA = tuple((float(i), float(j)) for i in (1, 5, 9) for j in (1, 5, 9))
CALNOISE_PLAN = (
    # (arm, family, truth labels, panels per truth)
    ("logit_normal", "A2", ("rho=0.9,HIGH", "HMM q=0.05", "lam=0.5,D1=10.0"), 20),
    ("mixture", "A2", ("HMM q=0.05", "lam=0.5,D1=10.0"), 10),
    ("mixture", "A1_PRIME", ("HMM q=0.05",), 10),
)
CALNOISE_R = 50  # compared with the R = 50 imported image (replicates 0..49): CRN
CALNOISE_SHIFT_P90_MAX = 0.25  # p90 |Δμ_j| / τ_j(0.25) over feasible pairs, any moment
CALNOISE_DISAGREE_MAX = 0.05  # status disagreement (any κ) and CI-1 feasibility mismatch rate
CALNOISE_LABEL = "CALIBRATION-NOISE SENSITIVE"

# --- reporting ----------------------------------------------------------------------
CI_LEVEL = 0.95  # exact Clopper-Pearson, two-sided; one-sided 95% bound for MDE bracket
RQ2_LABELS = MappingProxyType(
    {"robust": "ROBUST TO FROZEN HETEROGENEITY BRACKET", "dependent": "HETEROGENEITY-DEPENDENT"}
)

# --- real comparison and S-a ----------------------------------------------------------
S_A_ARMS = ("logit_normal",)  # mixture S-a deferred
REAL_SECONDARY_LEVEL2 = True  # second-level refinement on the real panel only (secondary)

# --- execution and reliability ------------------------------------------------------
WORKERS_TRY = 7
WORKERS_FALLBACK = 6
THROUGHPUT_PROBE_MINUTES = 15.0
BASELINE_CORE_HOURS_PER_HOUR = 107.458 / 19.027  # S1-P8005, 6 workers (5.65)
DISK_MIN_FREE_GB = 10.0
JOB_TIMEOUT_HOURS = 3.0  # no completion for this long while jobs are in flight = hang
EXECUTOR_MAX_RESTARTS = 3
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
STAGES = (
    "SETUP",
    "STAGE1",
    "RQ2_POP",
    "RQ2",
    "RQ2_ARCHIVE",
    "REAL_RQ1",
    "S_A",
    "SYNTHESIS",
    "VALIDATION",
    "COMPLETE",
)
