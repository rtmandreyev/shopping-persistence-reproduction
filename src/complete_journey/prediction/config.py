"""Frozen Phase 7 constants (P7-003). Every magic number of the Phase 7 design lives here.

Authority: docs/phases/PHASE_07_PREDICTIVE_BENCHMARK.md and docs/MOMENT_REGISTER.md.
Changing any value here is a design change and requires a researcher decision.
"""

from __future__ import annotations

from types import MappingProxyType

# --- Calendar -----------------------------------------------------------------------
FIRST_WEEK, LAST_WEEK = 1, 102
#: Responses never use partial week 102; week 1 may only supply lag information.
RESPONSE_WEEKS = (2, 101)

# --- Target -------------------------------------------------------------------------
#: Canonical accepted participation variable (any recorded line in the week; P4-001).
TARGET_SOURCE = "retailer_purchase"

# --- Observation operators (P7-002, implemented by P7-003) --------------------------
OPERATOR_OFFLINE = "O_interior_window"  # primary offline ABM comparison operator
OPERATOR_SENSITIVITY = "S_fixed_20_101"  # responses 20-101, full-grid lags
OPERATOR_DIAGNOSTIC = "G_full_grid"  # responses 2-101, full grid
OPERATORS = (OPERATOR_OFFLINE, OPERATOR_SENSITIVITY, OPERATOR_DIAGNOSTIC)
SENSITIVITY_RESPONSES = (20, 101)
MIN_INTERIOR_WEEKS = 8
#: Real-time predictive population (no future information); never an ABM operator.
POPULATION_REALTIME = "P_point_in_time"

# --- Temporal split (by response week) ----------------------------------------------
SPLITS = MappingProxyType({"train": (2, 68), "validation": (69, 84), "test": (85, 101)})
#: VI-6 recency gain: no tuning, so it trains on r <= 68 and evaluates on 69-101.
RECENCY_TRAIN = (2, 68)
RECENCY_EVAL = (69, 101)

# --- Features -----------------------------------------------------------------------
TIER0 = ("T0_tenure_log", "T0_rate_logit", "T0_rate_x_tenure")
TIER1 = (*TIER0, "T1_y_t", "T1_recent4_share", "T1_weeks_since_last")
RECENT_WINDOW = 4
WEEKS_SINCE_CAP = 12
FEATURE_INVENTORY = "docs/phases/PHASE_07_FEATURE_INVENTORY.csv"

# --- Models -------------------------------------------------------------------------
SEED = 20261015
B1_FIXED_C = 1.0  # VI-6 and Tier 0/1: fixed, no tuning
B1_TIER2_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
B1_SOLVER = "lbfgs"
B1_TOL = 1e-8
B1_MAX_ITER = 5000
B2_GRID = MappingProxyType(
    {
        "max_leaf_nodes": (15, 31, 63),
        "learning_rate": (0.05, 0.1),
        "min_samples_leaf": (50, 200),
    }
)
B2_MAX_ITER = 300
MODEL_LADDER = ("B0", "B1", "B2")

# --- Metrics ------------------------------------------------------------------------
PRIMARY_METRIC = "log_loss"
SECONDARY_METRIC = "brier"
DESCRIPTIVE_METRICS = ("roc_auc",)
CALIBRATION_BINS = 10
PROBABILITY_CLIP = 1e-15
RECENCY_GAIN = "delta_log_loss_recency"  # LL(Tier 0) - LL(Tier 1); positive = recency helps
RECENCY_GAIN_SECONDARY = "delta_brier_recency"

# --- Moments ------------------------------------------------------------------------
CI1_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
LAG_SET = (1, 2, 4, 8)
SPELL_BINS = ((1, 1), (2, 2), (3, 4), (5, None))
HALF_MIN_WEEKS = 16
BOOTSTRAP_PRODUCTION_B = 500
SHUFFLE_REPLICATIONS = 20
SHUFFLE_ATOL = 1e-12  # calibration invariance up to floating summation order
TWFE_TOL = 1e-12
TWFE_MAX_ITER = 10000

# --- Synthetic namespace ------------------------------------------------------------
#: Synthetic household keys live far above the real 1..2500 key space.
SYNTHETIC_KEY_OFFSET = 1_000_000_000

# --- Real-data guard ----------------------------------------------------------------
#: Run C authorization (P7-003 design and the Run B synthetic preflight were accepted by
#: the researcher). The token alone is NOT sufficient: real fitting additionally needs
#: the process-local Run C context entered by ``prediction.run_c``. That context checks
#: the committed design snapshot and the one-time test ledger. No CLI flag or
#: environment variable can enable real fitting.
REAL_FIT_AUTHORIZATION: str | None = "P7-RUN-C-AUTHORIZED-ONE-TIME-REAL-REVEAL"
REAL_FIT_TOKEN = "P7-RUN-C-AUTHORIZED-ONE-TIME-REAL-REVEAL"
