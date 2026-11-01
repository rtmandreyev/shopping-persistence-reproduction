"""Researcher-authorized Run E publication contract; no empirical computation."""

REVIEWED_COMMIT = "1f22d89a54b4582fb36dafeeac808f99af4a1ed9"
FINAL_PATHS = (
    "outputs/figures/panel/final/f10_panel_adjustment_and_calendar_sensitivity.png",
    "outputs/tables/panel/final/t4_core_panel_results.csv",
    "outputs/tables/panel/final/t5_panel_robustness.csv",
)
RESULT_STATUSES = {
    "RC-01": "PROMOTED",
    "RC-02": "PROMOTED",
    "RC-03": "CHALLENGED",
    "RC-04": "PROMOTED",
    "RC-05": "PROMOTED",
}
APPROVED_WORDING = {
    "RC-01": "Accounting for stable household differences reduces the raw association between prior and current retailer activity substantially more than accounting for common week conditions alone.",
    "RC-02": "After accounting for stable household differences and common week conditions, prior-week retailer participation remains positively associated with current retailer participation, unconditional retailer revenue, and recorded basket frequency.",
    "RC-03": "Revenue numerator contributions are more concentrated than regressor denominator contributions, but no individual household dominates the baseline slope enough to justify an exclusion sensitivity.",
    "RC-04": "Estimated uncertainty is materially sensitive to the assumed dependence structure: predeclared Driscoll-Kraay standard errors are approximately two to 2.6 times the original two-way-cluster standard errors, although all predeclared baseline intervals remain above zero.",
    "RC-05": "The positive association is directionally stable but materially weaker after the early observed-support-formation period.",
}
PAPER_COVARIANCE = "Driscoll-Kraay Bartlett bandwidth 8"
DESIGN_COVARIANCE = "two-way household/response-week clustered"
P4_003_APPEND = """

### P4-003 — Primary paper-facing TWFE uncertainty is predeclared DK8 (researcher decision)

**Decision.** After researcher review of Runs C/D at
`1f22d89a54b4582fb36dafeeac808f99af4a1ed9`, the main Model-3 TWFE
coefficients use the **already-estimated Driscoll-Kraay covariance, Bartlett
kernel, bandwidth 8, as PRIMARY PAPER-FACING uncertainty**. The frozen TWFE
point estimates, outcomes, history, samples, FE and weights do not change.
This is a presentation decision over accepted evidence, not new estimation.

The original two-way household × response-week clustered covariance remains
the **DESIGN BENCHMARK**, is transparently reported, and remains part of the
robustness/inference documentation. DK4 also remains reported. Household-only
clustering remains a sensitivity, not the preferred uncertainty measure: it
allows within-household serial dependence but does not address the strong
cross-household temporal score dependence documented in Runs C/D. M0–M2 retain
their accepted original two-way uncertainty; they are not rerun under DK8.
Post-formation, population and log1p checks retain the covariance actually
estimated for those rows; no additional DK interval is invented for them.

**Reasoning.** Aggregate-score lag-1 correlations are 0.970833 for participation,
0.959429 for revenue and 0.975392 for trips; lag-8 correlations remain 0.745221,
0.757585 and 0.826805. The predeclared DK4/DK8 standard errors materially widen
uncertainty. DK8 was predeclared before sensitivity results were observed; it is
the more conservative of the two predeclared bandwidths in these accepted
comparisons, and substantial score dependence remains at longer lags. It does
not reverse the qualitative result. It is **NOT selected because it preserves
statistical significance**; no new bandwidth may be searched.

**Boundary.** DK8 is assumption-dependent, not definitively correct or
dependence-complete. Bartlett weights downweight included lags; dependence
remains visible at lag 8; finite T = 101 limits asymptotic certainty. Two-way
clustering and DK rely on different dependence assumptions/asymptotics. The
native univariate Student t(df_resid) convention and all preflighted debias/FE
settings are retained. Covariance choice does not supply exogeneity, identify
structural state dependence or correct dynamic FE bias. P4-002 remains intact:
no repair or joint/Wald inference from indefinite native matrices, and stop for
nonpositive/nonfinite focal variance rather than replacing a failed covariance.

**Promotion and publication.** The researcher promotes RC-01, RC-02 (with its
noncausal boundary), RC-04 (inference/limitation) and RC-05 using the approved
wording. RC-03 remains CHALLENGED / DIAGNOSTIC, not a headline; no household
deletion is authorized. Only F10/T4/T5 receive exact final-path allowlists.

**Historical integrity and run boundary.** Run B/C/D numerical artifacts and
provenance stay byte-identical. Their validators refer to the immutable accepted
generating states, not a retroactively rewritten source identity. Only bounded
publication/control-plane validation evolves, with the prior decision ledger
and ignore policy preserved except this exact appended decision and the three
explicit final-path exceptions. Final artifacts select and serialize existing
aggregate evidence only. No new empirical search or result is authorized.
Phase 4 remains ACTIVE, closure pending; Phase 5 is not begun.
"""
IGNORE_OLD_COMMENT = (
    "# No Phase 4 result/model/figure allowlist; local customer diagnostics stay ignored."
)
IGNORE_HISTORICAL_COMMENT = (
    "# Historical Run B boundary; the exact researcher-approved Run E exceptions are below."
)
FINAL_IGNORE_APPEND = """

# --- Phase 4 Run E: exactly three researcher-approved final artifacts ---
!outputs/figures/panel/
outputs/figures/panel/**
!outputs/figures/panel/final/
outputs/figures/panel/final/**
!outputs/figures/panel/final/f10_panel_adjustment_and_calendar_sensitivity.png
!outputs/tables/panel/
outputs/tables/panel/**
!outputs/tables/panel/final/
outputs/tables/panel/final/**
!outputs/tables/panel/final/t4_core_panel_results.csv
!outputs/tables/panel/final/t5_panel_robustness.csv
"""


def approved_ignore_bytes(historical: bytes) -> bytes:
    """Exact authorized policy delta, not an unrestricted source exemption."""
    return (
        historical.replace(IGNORE_OLD_COMMENT.encode(), IGNORE_HISTORICAL_COMMENT.encode(), 1)
        + FINAL_IGNORE_APPEND.encode()
    )


P4_004_APPEND_SHA256 = "1f49390730f52bf9ac0e1ae0ee61ea78a7f207e867d61595427393d84dcd6b7f"


def phase4_ledger_prefix(payload: bytes) -> bytes:
    """Validate the exact accepted Phase 4 boundary, permitting later append-only phases."""
    import hashlib

    from complete_journey.econometrics.samples import require

    decision = P4_003_APPEND.encode()
    require(
        payload.count(b"### P4-003") == 1 and payload.count(decision) == 1,
        "P4-003 missing, changed or duplicated",
    )
    end = payload.index(decision) + len(decision)
    if end == len(payload):
        return payload  # Historical publication state, before valid closure.
    require(payload.count(b"### P4-004") == 1, "P4-004 missing or duplicated")
    suffix = payload[end:]
    closure, separator, _ = suffix.partition(b"\n---\n\n## Phase 5")
    require(closure.startswith("\n\n### P4-004 —".encode()), "P4-004 closure boundary changed")
    require(hashlib.sha256(closure).hexdigest() == P4_004_APPEND_SHA256, "P4-004 changed closure")
    require(
        not separator or suffix[len(closure) :].startswith(b"\n---\n\n## Phase 5"),
        "P4-004 invalid later ledger boundary",
    )
    return payload[:end] + closure


PHASE5_PREPARATION_IGNORE_APPEND = "\n\n# --- Phase 5 Run B: aggregate preparation artifacts only; no final results ---\n!outputs/tables/choice/\noutputs/tables/choice/**\n!outputs/tables/choice/contracts.json\n!outputs/tables/choice/qa_summary.csv\n!outputs/tables/choice/phase5_preflight_summary.csv\n!outputs/tables/choice/provenance.json\n"
