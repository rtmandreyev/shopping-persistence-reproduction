"""Frozen Phase 9 settings (authority: ``docs/DECISIONS.md`` P9-001 as amended by P9-002).

Items marked OPEN are candidate values prepared for the researcher's pre-smoke freeze.
They are not frozen until that decision is recorded.
"""

from __future__ import annotations

from types import MappingProxyType

from complete_journey.paths import PROJECT_ROOT

RUN_LABEL = "P9-SYSTEMONE"


class Phase9ContractError(ValueError):
    """A P9-001 / P9-002 design, governance, budget or authorization rule was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase9ContractError(message)


# --- policies (P9-001 table; P9-002 §1: the Jev versions are co-equal) ---------------
P_J13, P_J14, P_M = "P-J13", "P-J14", "P-M"
POLICIES = (P_J13, P_J14, P_M)
JEV_POLICIES = (P_J13, P_J14)  # co-equal frozen Jev policies (P9-002 §1)
TYPESAFE, OPENROUTER = "typesafe", "openrouter"
POLICY_PINS = MappingProxyType(
    {
        P_J13: MappingProxyType(
            {
                "transport": TYPESAFE,
                "model": "jev-1.13.0",
                "endpoint": "https://api.typesafe.ai/v1/systemone",
                "returned_model": "jev-1.13.0",
                "returned_provider": None,
                "usd_per_input_token": 0.042e-6,
                "usd_per_output_token": 0.0,  # documented free
            }
        ),
        P_J14: MappingProxyType(
            {
                "transport": TYPESAFE,
                "model": "jev-1.14.0",
                "endpoint": "https://api.typesafe.ai/v1/systemone",
                "returned_model": "jev-1.14.0",
                "returned_provider": None,
                "usd_per_input_token": 0.042e-6,
                "usd_per_output_token": 0.0,  # documented free
            }
        ),
        P_M: MappingProxyType(
            {
                "transport": OPENROUTER,
                "model": "inception/mercury-decide-20260930",
                "endpoint": "https://openrouter.ai/api/alpha/decisions",
                "returned_model": "inception/mercury-decide-20260930",
                "returned_provider": "Inception",
                # paid endpoint, never ":free" (P9-001)
                "usd_per_input_token": 0.02e-6,
                "usd_per_output_token": 0.0,  # documented free
                "provider_routing": MappingProxyType(
                    {"only": ("inception",), "allow_fallbacks": False}
                ),
            }
        ),
    }
)
PROHIBITED_MODEL_SUBSTRINGS = ("latest", "preview", "~", ":free")  # aliases are prohibited

# --- information cells (P9-001 2x2) ---------------------------------------------------
NEITHER, HISTORY, LATENT, BOTH = "NEITHER", "HISTORY", "LATENT", "BOTH"
CELLS = (NEITHER, HISTORY, LATENT, BOTH)
CELL_VISIBILITY = MappingProxyType(
    {NEITHER: (False, False), HISTORY: (True, False), LATENT: (False, True), BOTH: (True, True)}
)  # (history visible, need visible)
NEED_LEVELS = ("low", "usual", "high")
HISTORY_WEEKS = 4
HISTORY_PATTERNS = tuple(
    tuple((k >> (HISTORY_WEEKS - 1 - j)) & 1 for j in range(HISTORY_WEEKS))
    for k in range(2**HISTORY_WEEKS)
)  # h = (h1, h2, h3, h4); h1 = last week

# --- stable profile grid (P9-001; K decided by the predeclared local CI-1 check) ------
GRID_10 = tuple(round(0.05 + 0.1 * k, 2) for k in range(10))
GRID_19 = tuple(round(0.05 * k, 2) for k in range(1, 20))
GRIDS = MappingProxyType({10: GRID_10, 19: GRID_19})
A1_MU, A1_S = 0.213, 1.540  # Phase 8 A1 logit-normal fit (local household assignment only)
PROFILE_BANDS = (("low", 0.0, 0.35), ("mid", 0.35, 0.65), ("high", 0.65, 1.0))  # P9-002 §2

# --- questions (three nouls; ids are neutral) ------------------------------------------
Q_SHOP, Q_SKIP, Q_TYPE = "q1", "q2", "q3"
QUESTION_ROLES = MappingProxyType({Q_SHOP: "shop", Q_SKIP: "skip", Q_TYPE: "usual_type"})
TEMPLATES = ("A", "B")
#: OPEN (researcher choice at the pre-smoke freeze): the placebo field. Two finalists are
#: implemented; switching ``PLACEBO_CANDIDATE`` is the only change needed. For each
#: candidate: (value 1, value 2), template-A key and sentence, template-B key and sentence.
PLACEBO_CANDIDATES = MappingProxyType(
    {
        "record_format_version": MappingProxyType(
            {
                "values": ("version 3", "version 4"),
                "A": ("record_format", "Record format {v}."),
                "B": ("record_layout", "Layout {v}."),
            }
        ),
        "record_id": MappingProxyType(
            {
                "values": ("4821", "7356"),
                "A": ("record_id", "Record ID {v}."),
                "B": ("record_reference", "Reference number {v}."),
            }
        ),
    }
)
PLACEBO_CANDIDATE = (
    "record_id"  # researcher choice (2026-10-21); frozen only when P9-003 is recorded
)
PLACEBO_VALUES = PLACEBO_CANDIDATES[PLACEBO_CANDIDATE]["values"]
PLACEBO_DEFAULT = PLACEBO_VALUES[0]

# --- experiment shape (P9-001; P9-002 §2) -----------------------------------------------
REPEATS_A = 3
SENTINELS = 24
SENTINEL_TIMEPOINTS = 4
PLACEBO_FRACTION = 0.25
PLACEBO_MIN_PER_STRATUM = 4
N_HOUSEHOLDS = 2500
T_WEEKS = 102
BURN_IN = 26
R_PANELS = 200
Q_POP = 200  # OPEN (Level-2 admissibility details): separate population block, as in RQ2

# --- need Markov chain (OPEN: candidate grid; symmetric 3-state, uniform switching) -----
#: lambda = second eigenvalue = lag-1 autocorrelation of the need state index;
#: stay probability q = (2*lambda + 1) / 3; expected spell 1 / (1 - q) weeks.
NEED_LAMBDAS = (0.5, 0.8, 0.95)  # OPEN: matches the Phase 8 T1 rho values 0.5 / 0.8 / 0.95

# --- seeds (new Phase 9 root) -----------------------------------------------------------
SEED_ROOT = 20261021
SEED_PURPOSE = MappingProxyType(
    {
        "placebo_frame": 1,
        "smoke_states": 2,
        "query_order": 3,
        "closed_loop": 4,
        "bootstrap": 5,
        "nonce": 6,
        "k_check": 7,
        "mock_noise": 8,
        "shuffle": 9,
        "calibration": 10,
        "pop_block": 11,
        "p9007_calibration": 12,  # P9-007 CALIBRATION-role common random numbers
        "p9007_panel": 13,  # P9-007 recovery and population panels
    }
)

# --- amended P-M drift rule (P9-006; P-M only) -------------------------------------------
#: sigma_c = max(sigma_c_smoke, sigma_pooled); single-shift bound b_c = max(0.10, z*sqrt(2)*sigma_c);
#: median rule unchanged (max(0.02, 3*sigma_pooled)). Inputs: frozen P-M smoke identical repeats.
DRIFT_P9006_POLICIES = ("P-M",)
DRIFT_P9006_Z = 3.80
DRIFT_P9006_SIGMA_POOLED = 0.027323927795960528
DRIFT_P9006_SMOKE_SD = MappingProxyType(
    {
        BOTH: 0.049993755886152354,
        HISTORY: 0.01709407780240743,
        LATENT: 0.0,
        NEITHER: 0.013957256079778203,
    }
)

# --- validity rule (P9-001; P9-002 §2) ----------------------------------------------
REPEAT_SD_MULTIPLE = 3.0
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_ALPHA = 0.05  # one-sided lower bound

# --- budget (P9-004 governance; hard guards, no bypass) ------------------------------
#: Operative dollar caps (P9-004). They are runaway-billing guards sized never to bind
#: under any legitimate execution of the frozen manifests (see P9-004 for the
#: arithmetic). Outer bounds from the researcher's available balances: OpenRouter $5,
#: Jev/TypeSafe $4 for all of Phase 9. Request counts, attempt caps, retry limits,
#: duplicate prevention and the archive are the primary runaway protection.
JEV_CEILING_USD = 4.00
MERCURY_CEILING_USD = 5.00
TIER_CAPS_USD = MappingProxyType(
    {
        P_J13: MappingProxyType({"smoke": 0.25, "pilot": 0.50, "production": 1.50}),
        P_J14: MappingProxyType({"smoke": 0.25, "pilot": 0.50, "production": 1.50}),
        P_M: MappingProxyType({"smoke": 0.25, "pilot": 0.50, "production": 4.00}),
    }
)
BUDGET_DECISION = "P9-004"
#: The budget block stamped into the P9-003 smoke manifests is this historical block, so
#: those frozen manifests (and their hashes) never change. Later manifests carry the
#: operative caps (``request.manifest_budget_block``).
MANIFEST_BUDGET_P9003 = MappingProxyType(
    {
        "jev_ceiling_usd": 1.00,
        "mercury_ceiling_usd": 0.25,
        "tier_cap_usd": MappingProxyType(
            {
                P_J13: MappingProxyType({"smoke": 0.01, "pilot": 0.02, "production": 0.40}),
                P_J14: MappingProxyType({"smoke": 0.01, "pilot": 0.02, "production": 0.40}),
                P_M: MappingProxyType({"smoke": 0.01, "pilot": 0.02, "production": 0.22}),
            }
        ),
    }
)


def operative_budget_json() -> str:
    """Canonical one-line statement of the operative caps. It must appear verbatim in the
    governing budget decision for ``authorize`` to pass."""
    import json

    body = {
        "jev_ceiling_usd": JEV_CEILING_USD,
        "mercury_ceiling_usd": MERCURY_CEILING_USD,
        "tier_caps_usd": {p: dict(v) for p, v in TIER_CAPS_USD.items()},
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


REQUEST_CAPS = MappingProxyType(
    {
        "smoke": MappingProxyType({"successes": 200, "attempts": 260}),
        "pilot": MappingProxyType({"successes": 300, "attempts": 390}),
        "production": MappingProxyType({"successes": 16000, "attempts": 20000}),
    }
)
WORST_CASE_TOKEN_FACTOR = 2.0
SIZING_FRACTION = 0.80  # P9-001 sizing rule: worst case <= 80% of every cap
CHARS_PER_TOKEN_EXPECTED = 4.0
CHARS_PER_TOKEN_WORST = 3.0

# --- retry engine (P9-001 governance gate 9) --------------------------------------------
RETRY_STATUSES = frozenset({429, 529, 500, 502, 503, 504})
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 60.0
REQUEST_TIMEOUT_S = 30.0

# --- governance / authorization -----------------------------------------------------
DECISIONS = PROJECT_ROOT / "docs" / "DECISIONS.md"
DESIGN_DECISIONS = ("P9-001", "P9-002")
#: Decisions that must be recorded before each external scope:
#: - P9-003 is the pre-smoke instrument and manifest freeze;
#: - P9-004 is the budget-governance amendment (operative dollar caps);
#: - P9-005 is the post-smoke production-manifest freeze and unlock.
#: A manifest's hash must appear in one of the scope's decisions from P9-003 on.
SCOPE_DECISIONS = MappingProxyType(
    {
        "smoke": ("P9-001", "P9-002", "P9-003", "P9-004"),
        "pilot": ("P9-001", "P9-002", "P9-003", "P9-004", "P9-005"),
        "production": ("P9-001", "P9-002", "P9-003", "P9-004", "P9-005"),
    }
)
MANIFEST_FREEZE_DECISIONS = ("P9-003", "P9-004", "P9-005")
EXTERNAL_ENV = "SYSTEMONE_EXTERNAL_SCOPE"
APPROVALS_DIR = PROJECT_ROOT / "docs" / "phases" / "PHASE_09" / "approvals"
ARCHIVE_ROOT = PROJECT_ROOT / "outputs" / "decision_logs" / "phase9"  # B-009
MOCK_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "systemone_mock"
PREFLIGHT_DIR = PROJECT_ROOT / "docs" / "phases" / "PHASE_09"
