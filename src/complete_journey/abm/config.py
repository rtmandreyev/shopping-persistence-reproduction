"""Frozen P8-001 constants. Every number of the Phase 8 design lives here.

Authority: docs/phases/PHASE_08_ABM_SUFFICIENCY_DETECTABILITY.md (P8-001) and
docs/DECISIONS.md P8-001. Changing any scientific value here is a design change. It
requires a new researcher decision and, from Stage 3 on, a full RQ2 restart.

Production budgets are DECLARATIONS ONLY in Run B: nothing in Run B executes them.
"""

from __future__ import annotations

import math
from types import MappingProxyType

from complete_journey.prediction import config as p7


class Phase8ContractError(ValueError):
    """A P8-001 protocol, timing, guard or authorization rule was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase8ContractError(message)


# --- Calendar, horizon and operator (design §W, §Y, §Z) -----------------------------
N_HOUSEHOLDS = 2500
T_WEEKS = p7.LAST_WEEK - p7.FIRST_WEEK + 1  # 102 retained empirical weeks
BURN_IN_WEEKS = 156
FIRST_SIM_WEEK = p7.FIRST_WEEK - BURN_IN_WEEKS  # week -155
TOTAL_SIM_WEEKS = BURN_IN_WEEKS + T_WEEKS  # 258 simulated weeks (-155 ... 102)
RESPONSE_WEEKS = p7.RESPONSE_WEEKS  # Phase 7 convention: responses 2-101
MIN_INTERIOR_WEEKS = p7.MIN_INTERIOR_WEEKS  # |W_i| >= 8
OPERATOR_O = p7.OPERATOR_OFFLINE  # identical Phase 7 analysis operator code
OPERATOR_PRIMARY = "primary_endogenous_envelope"  # §W.2
OPERATOR_SA = "S-a_availability_bracket"  # §X
PHASE8_OPERATORS = (OPERATOR_PRIMARY, OPERATOR_SA)

# --- Parameter regions (design §V) ---------------------------------------------------
HALF_LIFE_CAP_WEEKS = 25.0
PERSISTENCE_CAP = 2.0 ** (-1.0 / HALF_LIFE_CAP_WEEKS)  # rho_max = lambda_max ~ 0.972655
SIGMA_Z_CAP = 2.5  # A1' latent SD, logit scale
#: P8-002: the A2 (and A2-recency) amplitude axis is raw delta on the logit scale, capped
#: so that the history term's maximum SD (delta/2, since H in [0, 1]) equals the A1'
#: sigma_z cap. Delta_1 / Delta_12 become per-point interpretive labels. The P8-001
#: "Delta_1 <= 40 pp" cap is RETIRED: the one-week effect of the normalized stock is
#: non-monotone in delta and unreachable at moderate or long memory.
A2_DELTA_CAP = 2.0 * SIGMA_Z_CAP  # 5.0
GRID_SIZE = 12
HALF_LIFE_MIN_WEEKS = 0.5
#: 11 log-spaced half-lives h_k = 0.5 * 50^(k/10), k = 0..10 (0.5 ... 25 weeks).
HALF_LIFE_AXIS = tuple(HALF_LIFE_MIN_WEEKS * 50.0 ** (k / 10) for k in range(11))
SENSITIVITY_SUBGRID_INDEX = (1, 3, 5, 7, 9, 11)  # 6 x 6 sensitivity / H2 subgrid
PERIODIC_AMPLITUDES = (0.25, 0.5, 1.0, 2.0)
PERIODIC_CAP = 2.5
REFINEMENT_MAX_LEVELS = 2
REFINEMENT_MAX_POINTS = 144

# --- Heterogeneity calibration (design §F, §P) --------------------------------------
CI1_COMPONENTS = ("mean", "q10", "q25", "q50", "q75", "q90")
CI1_QUANTILES = p7.CI1_QUANTILES
CI1_GATE_ABS = 0.02  # gate: |sim - target| <= max(0.02, 2 SE_j)
#: Archived Run C bootstrap SEs of CI-1 (a calibration object; pinned against the archive
#: in tests). Used for the calibration distance weights and the CI-1 gate only.
CI1_BOOT_SE = MappingProxyType(
    {
        "mean": 0.00556934648198798,
        "q10": 0.005476840777654468,
        "q25": 0.0074781272734876295,
        "q50": 0.008602230721040705,
        "q75": 0.008025183283748042,
        "q90": 0.00544566315990625,
    }
)

# --- Moments (design §Q, §R) ---------------------------------------------------------
#: P1..P5: (phase-8 id, Phase 7 moment id, Phase 7 component).
PRIMARY_MOMENTS = (
    ("P1", "VI-2", "O-M3"),
    ("P2", "VI-3", "beta_8"),
    ("P3", "VI-3", "gamma_2"),
    ("P4", "VI-6", "delta_log_loss_recency"),
    ("P5", "VI-4", "active_1"),
)
PRIMARY_IDS = tuple(m[0] for m in PRIMARY_MOMENTS)
J_PRIMARY = len(PRIMARY_MOMENTS)
SECONDARY_MOMENTS = (
    ("VI-1", "P11"),
    ("VI-1", "P01"),
    ("VI-2", "O-M0"),
    ("VI-1", "GAP"),
    ("VI-2", "O-M1"),
    ("VI-2", "O-M2"),
    ("VI-3", "beta_1"),
    ("VI-3", "beta_2"),
    ("VI-3", "beta_4"),
    ("VI-4", "active_2"),
    ("VI-4", "active_3-4"),
    ("VI-4", "active_5+"),
    ("VI-4", "inactive_1"),
    ("VI-4", "inactive_2"),
    ("VI-4", "inactive_3-4"),
    ("VI-4", "inactive_5+"),
    ("VI-6", "delta_brier_recency"),
    ("VI-5", "spearman"),
)
DIAGNOSTIC_MOMENTS = (
    ("D-2", "S-operator versions"),
    ("D-3", "G-operator versions"),
    ("K-1", "between_share"),
    ("K-2", "gini"),
    ("K-2", "top10_share"),
    ("CI-1", "fit quality"),
    ("E-1", "weekly prevalence"),
    ("E-2", "envelope start/end/length"),
    ("VI-3", "beta_2_minus_beta_1"),
    ("DIST", "mahalanobis"),
    ("MAP", "acceptance-set map"),
    ("D-4", "shuffle check"),
)

# --- Adequacy rule (design §S-§U) ---------------------------------------------------
ALPHA = 0.05
KAPPAS = (0.15, 0.25, 0.35)
KAPPA_REFERENCE = 0.25
#: P8-003: the P8-001 "2 * SE" tolerance floor is RETIRED (it double-counted sampling
#: noise under TOST). The design-resolution floor rho_j = c * SE_j is computed in
#: ``adequacy.resolution_constant`` from ALPHA, J_PRIMARY, GATE_TRUTH_INCLUSION_MIN,
#: G_PLUS, EMULATOR_SD_TOLERANCE, R_EMULATOR and EMULATOR_MEAN_ERROR_FRACTION (c ~ 4.16).
EMULATOR_SD_TOLERANCE = 0.20  # s_emulator within +-20% of s_direct
G_PLUS = 1.0 / (1.0 - EMULATOR_SD_TOLERANCE)  # 1.25
G_MINUS = 1.0 / (1.0 + EMULATOR_SD_TOLERANCE)  # 1/1.2
EMULATOR_MEAN_ERROR_FRACTION = 0.1  # |mu_em - mu_direct| <= 0.1 * tau_j(0.15)
EMULATOR_GATE_KAPPA = min(KAPPAS)

# --- RQ2 (design §AD-§AG) -----------------------------------------------------------
DELTA1_GRID_PP = (0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0)
LAMBDA_TRUTH_GRID = (0.0, 0.5, 0.8)
RHO_TRUTH_GRID = (0.5, 0.8, 0.9, 0.95)
AMPLITUDE_RULE = MappingProxyType({"LOW": 0.5, "HIGH": 1.0})  # sigma_z = factor * s_A1
MASKING_RHO = 0.9
MASKING_AMPLITUDE = "HIGH"
MASKING_LAMBDA = 0.5
HMM_TRUTH_Q = (0.10, 0.05)
HET_RHO_TRUTH_MEANS = (0.8, 0.9)
HET_RHO_TRUTH_SD = 0.10
DELTA1_MAPPING_TOL_PP = 0.25
#: P8-002: RQ2 history truths use the ascending branch of Delta_1(delta). The peak is
#: located on this predeclared geometric delta scan; unreachable targets are infeasible.
DELTA_PEAK_SCAN = tuple(0.25 * 2.0 ** (k / 4) for k in range(37))  # 0.25 ... 128
DETECTION_THRESHOLD = 0.80
FA_ADMISSIBILITY = 0.10
MDE_TOP_PP = 30.0
#: FA_NAIVE (design §AF): OR of the two rules; archived real SEs as fixed design scales.
NAIVE_Z = 1.96
NAIVE_M3_SE = 0.004504060591558743
NAIVE_REC_SE = 0.0014917094999764902
#: Perfect-model gate (§AB.2 as revised by P8-003). All hard; failure = STOP.
#: G1 pointwise truth inclusion in Acc_F(kappa) at every kappa;
#: G2-G5 at the reference kappa.
GATE_TRUTH_INCLUSION_MIN = 0.80  # G1 (also the power P in the resolution constant)
GATE_FAMILY_ADEQUATE_MIN = 0.80  # G2: production grid + frozen refinement
GATE_TRUE_INADEQUATE_MAX = 0.05  # G3
GATE_NR_INCLUSION_MIN = 0.90  # G4: truth in the not-clearly-inadequate set
GATE_A1_HISTORY_MAX = 0.10  # G5: A1 truth classified HISTORY_FAVOURED
SPAN_COVERAGE_MIN = 0.90
SPAN_RETAIN_MIN = 0.50
SPAN_BOX_SD = 2.0  # image boxes of half-width 2 * s_j (simulation-only scale)

# --- Replication budgets (design §AK; production declarations only) ---------------
R_CAL = 20
R_EMULATOR = 50
R_DIRECT = 200
R_FINAL_MAX = 500
Q_TRUTH = 200
Q_GATE = 200
N_SCALING = 10_000

# --- Seed tree (design §AA) ---------------------------------------------------------
SEED_ENTROPY = p7.SEED  # 20261015, reused under the Phase 8 namespace key
SEED_NAMESPACE = 8
SYNTHETIC_KEY_OFFSET = p7.SYNTHETIC_KEY_OFFSET

# --- Authorization (Run B: none) ----------------------------------------------------
#: Highest stage whose real-data or heavy operations are authorized. Run B = Stage 0:
#: no real CI-1 fit, no tolerance construction from the archive, no emulator build,
#: no RQ2, no real comparison. Raising this is a later, explicit researcher decision.
AUTHORIZED_STAGE = 1  # Stage 1 authorized by the researcher (Stage 1 authorization commit)

#: P8-005 replacement Stage 1 (run label S1-P8005). Its seed-tree stage key is 11, so its
#: streams can never coincide with the archived Stage 1 (key 1) or any later stage.
#: Computations stamped with key 11 are protocol Stage 1 and need this separate
#: authorization constant, set only by the researcher-authorized S1-P8005 commit.
P8005_STAGE_KEY = 11
P8005_STAGE1_AUTHORIZED = True  # researcher authorization (P8-005 combined Run C + Stage 1)


def half_life_to_persistence(h: float) -> float:
    """rho (or lambda) with autocorrelation / weight half-life h weeks; h = 0 -> 0."""
    require(h >= 0 and math.isfinite(h), "half-life must be finite and non-negative")
    return 0.0 if h == 0 else 2.0 ** (-1.0 / h)


def persistence_to_half_life(rho: float) -> float:
    """h = ln 0.5 / ln rho for 0 < rho < 1; rho = 0 -> 0."""
    require(0.0 <= rho < 1.0, "persistence must lie in [0, 1)")
    return 0.0 if rho == 0 else math.log(0.5) / math.log(rho)
