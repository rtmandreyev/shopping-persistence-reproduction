"""Phase-5-specific Run E researcher decisions and exact publication paths."""

from types import MappingProxyType

FINAL_OUTPUTS = MappingProxyType(
    {
        "F11": "outputs/figures/choice/final/f11_history_and_purchased_composition.png",
        "T6": "outputs/tables/choice/final/t6_core_purchased_composition_results.csv",
        "T7": "outputs/tables/choice/final/t7_purchased_composition_robustness.csv",
    }
)
PUBLICATION_PROVENANCE = (
    "outputs/tables/choice/exploratory/phase5_run_e_publication_provenance.json"
)
D2_PATH = "outputs/tables/choice/exploratory/phase5_run_d2_focal_covariance_diagnostic.json"
D2_SHA256 = "41df65bf2808c440f74388471bc9a9e1a9c88618236af44ff798026ad72f6e77"
FINAL_IGNORE_APPEND = """

# --- Phase 5 Run E: exactly F11/T6/T7; exploratory archives remain ignored ---
!outputs/figures/choice/
outputs/figures/choice/**
!outputs/figures/choice/final/
outputs/figures/choice/final/**
!outputs/figures/choice/final/f11_history_and_purchased_composition.png
!outputs/tables/choice/final/
outputs/tables/choice/final/**
!outputs/tables/choice/final/t6_core_purchased_composition_results.csv
!outputs/tables/choice/final/t7_purchased_composition_robustness.csv
"""
PROMOTED_IDS = ("P5-RC-01", "P5-RC-02", "P5-RC-03", "P5-RC-04")
APPROVED_STATEMENTS = MappingProxyType(
    {
        "P5-RC-01": "Recorded Private-only SPAGHETTI DRY purchased composition differs sharply across recorded prior pure-type composition histories.",
        "P5-RC-02": "Large history-composition contrasts remain after additive recorded-store and broad calendar adjustment and remain qualitatively robust to the predeclared continuous-history representation and later-calendar response sample.",
        "P5-RC-03": "The observed history-composition gradient cannot be interpreted as causal loyalty, habit formation, or structural state dependence because the design does not remove persistent household-specific composition tendencies.",
        "P5-RC-04": "The full high-dimensional multiway covariance matrices are materially indefinite, while the predeclared history and history-plus-depth focal covariance subspaces are positive definite and independently reconciled. Phase 5 therefore restricts native multiway inference to those predeclared focal coefficients and prohibits omnibus/full-model inference.",
    }
)
FOCAL_TERMS = ("H0", "HB", "HP", "log1p_prior_pure")
INFERENCE_SCOPE = MappingProxyType(
    {
        "decision": "P5-002",
        "model": "M1",
        "sample": "baseline_pure_SPAGHETTI_DRY",
        "covariance": "household_week",
        "terms": FOCAL_TERMS,
        "reference_df": 101,
        "joint_inference": False,
        "covariance_repair": False,
        "full_covariance_status": "indefinite",
        "R1_R2_authorized": False,
        "arbitrary_nuisance_authorized": False,
    }
)
P5_002_ENTRY = """

### P5-002 — Baseline focal scalar inference and result promotion (researcher decision)

**Authority and scope.** Following researcher review of Phase 5 Runs C, D and D2
at `06520ccbc55d20b7f9498e17f2887af0ba316475`, primary paper-facing covariance
for the baseline Model 1 remains the precommitted two-way household × response-week
covariance. Native scalar inference is authorized ONLY for H0 versus HN, HB versus
HN, HP versus HN and the secondary log(1+n_prior_pure) coefficient on the baseline
6,544 pure-event sample. Use the frozen Student-t reference df101 and native
archived variances/SEs. This is a Phase-5-specific researcher decision, not a generic
repository-wide scalar-exception policy.

**Rationale.** Each focal native variance is finite and strictly positive. The
native 3×3 H0/HB/HP block and 4×4 history-plus-depth block are positive definite.
All six predeclared D2 history contrast variances are finite positive. Focal blocks
independently reconstruct from cluster-score aggregation. The household×week focal
component geometry is positive definite; intersection subtraction does not overwhelm
a focal history direction. Full-matrix negative eigendirections are overwhelmingly
store-coordinate oriented in the original coding. Loading mass is a coordinate-
dependent diagnostic, not a causal explanation of the source of indefiniteness.

**Boundaries.** The full 169×169 covariance remains materially indefinite and is
NOT declared PSD or generally valid. No covariance repair or eigenvalue clipping,
arbitrary nuisance-coefficient inference, omnibus/full-model Wald or other joint
inference is authorized. No dependence-complete coverage claim. Do not generalize
this decision beyond the four baseline focal coefficients; no new pairwise-contrast
scalar inference. R1 and R2 scalar inference remains gated. Baseline household×store
is required transparency sensitivity with diagnostic focal SEs and full-matrix
indefiniteness visible, not a switched primary. Household-only remains a PSD
diagnostic benchmark with diagnostic intervals; it omits common-week/shared-store
cross-household dependence. Neither alternative is selected by width or significance.

**Promotion and publication.** Promote exactly P5-RC-01 through P5-RC-04 under the
researcher's approved wording: descriptive history composition, robustness of
adjusted contrasts, the central household-heterogeneity interpretation limitation,
and the restricted-inference qualification. Depth is secondary adjustment, not a
separate substantive finding. Publish exactly F11/T6/T7 with explicit per-path
allowlisting, historical evidence preserved, and a complete paper-writing handoff.
Phase 5 remains ACTIVE with researcher closure pending. Phase 6 is not begun.
"""


def authorized_scope(model, sample, kind, terms):
    """Exact baseline scalar scope only; no fitting or generic exception interface."""
    return (
        model == INFERENCE_SCOPE["model"]
        and sample == INFERENCE_SCOPE["sample"]
        and kind == INFERENCE_SCOPE["covariance"]
        and tuple(terms) == FOCAL_TERMS
    )


# Closure-aware ledger semantics (P7-002). The protected Phase 5 history is the exact
# pre-P5-002 ledger (pinned in contracts.PROTECTED_INPUTS), then P5-002, then the exact
# P5-003 closure append. Later phases begin at the Phase 6 heading and are not pinned
# here, so future appends never require a Phase 5 code change.
RUN_E_LEDGER_SHA256 = "e6daaa708afdc8349d4d300fb293e5baddf396135b4312788f8aaf0f17f0808d"
P5_003_APPEND_SHA256 = "2e8ea2fa7be1de5d37a4e3dfb233fdffe504c0dbf50f3e70ff33af98082cb98b"
P5_003_HEADING = "\n\n### P5-003 —"
LATER_LEDGER_BOUNDARY = "\n\n## Phase 6 ("
# Immutable historical Run E publication record (ignored local evidence; never rewritten).
ACCEPTED_RUN_E_PROVENANCE_SHA256 = (
    "9be7c1b33c3845a52c3fa140d4a44ac3aed57f72c798557ce1540e0cc55652a8"
)
# Current authorized implementation for Run E revalidation after closure: exact hashes of
# source files whose bytes differ from the historical Run E generating source. The ledger
# is governed by the closure rule above; this module is pinned by its integrity test.
AUTHORIZED_SOURCE_REVISIONS = MappingProxyType(
    {
        # P7-003 scikit-learn addition; every previously locked package version unchanged.
        "pyproject.toml": "9c5b346f74b17e3d30d7f1fbdf71e061aa17c6ab180966338e6e324024491059",
        "uv.lock": "9797b69ca0735437b3680dfdb3375c8b09c95b9b190f243c8135f9fba4cc2e8d",
        "scripts/publish_choice.py": "13264696702332d2543491186200f755e634df2b9612f5bb233655fdb8c3dbaa",
        "scripts/run_choice.py": "7bb248de1e8334c5dc5cdd191df37eceeb3d13ee3d0b1f45edea93849aa400cf",
        # Portability revision after an independent audit (2026-10-30): the
        # private TMPDIR fallback became tempfile.gettempdir(); no computation changed.
        "src/complete_journey/choice/artifacts.py": "92787ab7662ea449876637c73bc7694b04da0dbc52fb249de30b47151563df41",
        "src/complete_journey/choice/publication.py": "0642f0891f5d7827b12a7fe56a1aac0c59b83481fa1a9ccd08a3907379c18d3b",
        "src/complete_journey/choice/run_c.py": "fb9a3e959c155e253be01dbd8592c8b086f81a55c9e0242f2a314dddb33b7011",
        "src/complete_journey/choice/run_d.py": "8973da41ad5446d9bb145799549ca17038afb8c42fbb7c10cd0fa4e74f03ddaf",
    }
)
CLOSURE_EXEMPT_SOURCES = frozenset(
    {"docs/DECISIONS.md", "src/complete_journey/choice/final_contracts.py"}
)


def phase5_ledger_state(payload: bytes, prefix_sha256: str) -> str:
    """Classify the ledger as preparation, run_e or closed; refuse anything else.

    Exact bytes are compared, never heading counts: the pinned pre-P5-002 prefix,
    the exact P5-002 entry and the exact P5-003 closure, in that order, each once.
    Any text after the closure must begin at the Phase 6 section heading.
    """
    import hashlib

    from complete_journey.choice.contracts import require

    if hashlib.sha256(payload).hexdigest() == prefix_sha256:
        return "preparation"
    entry = P5_002_ENTRY.encode()
    require(payload.count(entry) == 1, "exact P5-002 ledger entry required")
    end = payload.index(entry) + len(entry)
    require(
        hashlib.sha256(payload[: payload.index(entry)]).hexdigest() == prefix_sha256,
        "exact P5-002 ledger entry required: protected prefix changed",
    )
    if end == len(payload):
        return "run_e"
    closure, _, _ = payload[end:].partition(LATER_LEDGER_BOUNDARY.encode())
    require(
        closure.startswith(P5_003_HEADING.encode())
        and hashlib.sha256(closure).hexdigest() == P5_003_APPEND_SHA256
        and payload.count(closure) == 1,
        "exact P5-002 ledger entry required: P5-003 closure missing or changed",
    )
    return "closed"
