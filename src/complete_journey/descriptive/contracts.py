"""Frozen Phase 3 descriptive semantic contracts and accepted reconciliation anchors.

Everything here is metadata: frozen measurement conventions, the accepted Phase 2
reconciliation anchors, the campaign vocabulary boundaries, and the output formats.
No statistic is computed and no result is stored.

**Authority.** The conventions encode the frozen Phase 3 plan
(``docs/phases/PHASE_03_DESCRIPTIVE_ANALYTICS.md``, sections 4 and 5) and the
accepted Phase 1/2 semantics they inherit. The anchors are **accepted Phase 2
reconciliation anchors, not Phase 3 findings**: they were verified inside Phase 2
(162/162 hard invariants) and exist so that a Phase 3 run fails loudly if its
analytical inputs do not reconcile, rather than silently reporting a number that
cannot be traced back to the accepted build.

Deliberately not here: any KPI value, any household-level detail, any normalized
discount or price measure (``DECISIONS.md`` P2-003), any brand-share variable
(``P2-005``), and any Phase 4+ specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from complete_journey.data import contracts as phase2_contracts
from complete_journey.paths import PROCESSED_DIR, PROJECT_ROOT

# --- Frozen populations and grid conventions (plan section 4) -----------------

#: Observed transaction-universe households (S0 universe).
HOUSEHOLD_UNIVERSE: Final[int] = 2500

#: Observed retailer weeks.
WEEKS: Final[int] = 102

#: Complete household x week retailer-observation grid.
HOUSEHOLD_WEEK_GRID: Final[int] = HOUSEHOLD_UNIVERSE * WEEKS

#: Active household-week definition. A non-active cell means "no recorded
#: purchase from THIS retailer that week"; it does not mean zero household
#: grocery consumption.
ACTIVE_FLAG_COLUMN: Final[str] = "retailer_purchase"
ACTIVE_FLAG_VALUE: Final[int] = 1
NON_ACTIVE_SEMANTICS: Final[str] = (
    "retailer_purchase = 0 means no recorded purchase from this retailer that week; "
    "it does NOT mean zero household grocery consumption"
)

#: Semantics labels that must travel with any statistic built on these measures.
SALES_VALUE_SEMANTICS: Final[str] = (
    "retailer SALES_VALUE (retailer revenue) -- not grocery expenditure, customer spend, or price"
)
TRIPS_SEMANTICS: Final[str] = "basket count"
UNITS_SEMANTICS: Final[str] = "units (SUM(QUANTITY)); not the meaning of a bundle"
DISCOUNT_SEMANTICS: Final[str] = (
    "signed accepted discount aggregates and named magnitudes only; no normalized "
    "discount/promotion share exists unless a later decision names the denominator"
)

#: Category/department penetration denominator and its required label.
PENETRATION_DENOMINATOR: Final[int] = HOUSEHOLD_UNIVERSE
PENETRATION_LABEL: Final[str] = "observed-household penetration"
PENETRATION_FORBIDDEN_LABEL: Final[str] = "market penetration"

#: Accepted Phase 1 week mapping: WEEK_NO = floor((DAY + 1) / 7) + 1. Week 1 holds
#: 5 observed days and week 102 holds 6; complete weekly series keep them and
#: annotate them, and no statistic is silently normalized per day.
WEEK_MAPPING: Final[str] = "WEEK_NO = floor((DAY + 1) / 7) + 1"
PARTIAL_WEEK_DAYS: Final[dict[int, int]] = {1: 5, 102: 6}
FULL_WEEK_RANGE: Final[tuple[int, int]] = (2, 101)

#: Sample semantics: S0 is the headline population; S1/S2 are sensitivity only and
#: are never causal or predictive eligibility rules (DECISIONS.md P2-001).
HEADLINE_SAMPLE_FLAG: Final[str] = "sample_full"
SENSITIVITY_SAMPLE_FLAGS: Final[tuple[str, str]] = (
    "sample_sustained_support",
    "sample_higher_support",
)
SAMPLE_SEMANTICS: Final[str] = (
    "S0 (sample_full) is the headline population; S1/S2 are sensitivity populations only"
)

#: Campaign vocabulary boundaries (plan section 10). A later identification phase
#: may define the forbidden terms explicitly; Phase 3 never uses them.
#: Vocabulary that may never appear in a published figure's text or metadata: a
#: customer identifier, or a causal/campaign reading of what the figures show.
FIGURE_FORBIDDEN_VOCABULARY: Final[frozenset[str]] = frozenset(
    {
        "household_key",
        "customer spend",
        "grocery expenditure",
        "price",
        "treated",
        "control group",
        "exposed",
        "campaign effect",
    }
)

CAMPAIGN_ALLOWED_VOCABULARY: Final[frozenset[str]] = frozenset(
    {"listed", "listed pair", "listed household-week", "recorded redemption"}
)
CAMPAIGN_FORBIDDEN_VOCABULARY: Final[frozenset[str]] = frozenset(
    {"treated", "control", "exposed", "take-up", "treatment effect"}
)

# --- Exploratory output formats (plan section 11) ----------------------------

#: Formats for future runs. Working and final figures are PNG via matplotlib; no
#: plotting-theme package and no reporting framework is introduced.
EXPLORATORY_TABLE_FORMAT: Final[str] = "csv"
EXPLORATORY_FIGURE_FORMAT: Final[str] = "png"
FIGURE_BACKEND: Final[str] = "matplotlib"
DEFAULT_FIGURE_DPI: Final[int] = 150

# --- Accepted input locations ------------------------------------------------

PANEL_ARTIFACT_DIR: Final[Path] = PROJECT_ROOT / "outputs" / "tables" / "panels"
GENERATED_CONTRACTS_JSON: Final[Path] = PANEL_ARTIFACT_DIR / "contracts.json"
GENERATED_SAMPLE_FLAG_COUNTS: Final[Path] = PANEL_ARTIFACT_DIR / "sample_flag_counts.csv"

#: Phase 3 descriptive SQL (one file per output).
DESCRIPTIVE_SQL_DIR: Final[Path] = PROJECT_ROOT / "sql" / "descriptive"

#: Accepted analytical tables. Derived from the accepted Phase 2 contract metadata
#: so a future accepted table is picked up automatically rather than by list edit.
PHASE2_TABLE_SPECS: Final[dict[str, dict[str, object]]] = {
    **phase2_contracts.CANONICAL_TABLES,
    **phase2_contracts.PRIVATE_TABLES,
    **phase2_contracts.PHASE2B_TABLES,
    **phase2_contracts.PHASE2C_TABLES,
}


def accepted_parquet_path(table: str) -> Path:
    """Return the local Parquet path of an accepted Phase 2 table.

    Raises:
        ValueError: if the table is not an accepted Phase 2 table, so a typo or an
            invented table name cannot silently resolve to a path.
    """
    spec = PHASE2_TABLE_SPECS.get(table)
    if spec is None:
        raise ValueError(f"{table!r} is not an accepted Phase 2 table; refusing to resolve a path")
    if spec["scope"] == "private":
        return PROCESSED_DIR / "private" / f"{table}.parquet"
    return PROCESSED_DIR / f"{table}.parquet"


# --- Accepted reconciliation anchors -----------------------------------------


@dataclass(frozen=True)
class ReconciliationAnchors:
    """Accepted Phase 2 headline totals used to gate Phase 3.

    Every value is an accepted Phase 2 fact (QA-verified inside the build), not a
    Phase 3 result. Tolerances exist because ``SUM(DOUBLE)`` depends on summation
    order: Phase 2 pinned single-threaded aggregation so the Parquet is
    byte-reproducible, and the residuals observed against the accepted total are
    around 1e-6 on an 8.06M total (relative ~1e-13).
    """

    households: int = HOUSEHOLD_UNIVERSE
    weeks: int = WEEKS
    household_weeks: int = HOUSEHOLD_WEEK_GRID
    active_household_weeks: int = 123976
    baskets: int = 276484
    units: int = 260685622
    retailer_sales_value: float = 8057463.08
    sample_full_households: int = 2500
    sample_sustained_support_households: int = 2115
    sample_higher_support_households: int = 1807
    campaigns: int = 30
    listed_pairs: int = 7208
    listed_households: int = 1584
    listed_household_weeks: int = 735216
    redemption_rows: int = 2318
    redemption_households: int = 434
    sales_value_abs_tolerance: float = 1e-6
    sales_value_rel_tolerance: float = 1e-9

    @property
    def listed_household_weeks_derived(self) -> int:
        """Structurally implied listed household-weeks: listed pairs x weeks."""
        return self.listed_pairs * self.weeks


#: The accepted anchors used by every Phase 3 run.
ACCEPTED_ANCHORS: Final[ReconciliationAnchors] = ReconciliationAnchors()

#: Provenance note that must accompany any report of these numbers.
ANCHOR_PROVENANCE: Final[str] = (
    "accepted Phase 2 reconciliation anchors (re-verified here), not Phase 3 findings"
)


def is_partial_week(week_no: int) -> bool:
    """True if the observed week is shorter than a full week (weeks 1 and 102)."""
    return week_no in PARTIAL_WEEK_DAYS


def is_full_week(week_no: int) -> bool:
    """True if the week lies in the full-week range (weeks 2-101)."""
    low, high = FULL_WEEK_RANGE
    return low <= week_no <= high
