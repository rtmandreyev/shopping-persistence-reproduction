"""Phase 3 synthesis: the paper-facing KPI table (T1) and the accepted-evidence inventory.

This module discovers nothing new. It assembles the accepted Phase 3 evidence into the compact
paper-facing KPI table (T1) from already accepted/reconciled sources, and it exposes the promoted /
challenged / failed inventory straight from the challenge register so the synthesis record cannot
drift from the register.

One expectation table lives here: ``ACCEPTED_KPI_EXPECTATIONS``. It restates figures that the
researcher already approved in the register's promotion notes (Question A/B/C/D decisions) so T1
fails loudly if a recomputed KPI ever disagrees with an accepted one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd

from complete_journey.descriptive import (
    challenges,
    contracts,
    metrics,
    question_b,
    question_c,
    question_d,
    reconcile,
)

T1_SQL_FILENAME: Final[str] = "22_core_kpis.sql"

#: KPI rows the paper-facing table must carry. The brief's target band is 15-25 rows.
T1_MIN_ROWS: Final[int] = 15
T1_MAX_ROWS: Final[int] = 25

#: Domains T1 covers, in the order they appear in the paper-facing table.
T1_DOMAINS: Final[tuple[str, ...]] = (
    "SAMPLE",
    "RETAILER VALUE",
    "HOUSEHOLD",
    "PRODUCT",
    "CAMPAIGN",
)

#: Figures already accepted by the researcher, with the finding that carries them. T1 is checked
#: against these rather than the other way round, and a mismatch stops the run.
ACCEPTED_KPI_EXPECTATIONS: Final[dict[str, tuple[str, float, int]]] = {
    # metric: (accepted finding ID, accepted value, decimal places to compare)
    "top-10% household share of retailer SALES_VALUE": ("QB-01", 34.29, 2),
    "top-10% household share of trips": ("QB-01", 33.64, 2),
    "top-10% household share of active weeks": ("QB-01", 18.21, 2),
    "first-half vs second-half household retailer SALES_VALUE correlation": ("QB-02", 0.816, 3),
    "GROCERY share of retailer SALES_VALUE": ("QC-01", 50.81, 2),
    "median accepted listed-recipient overlap": ("QD-02", 85.77, 2),
}

#: Accepted counts T1 must reproduce exactly. Every Phase 2 figure is read from the single accepted
#: anchor object in contracts.py rather than restated here, so the anchor lives in exactly one place.
_ANCHORS = contracts.ACCEPTED_ANCHORS
ACCEPTED_KPI_COUNTS: Final[dict[str, float]] = {
    "households": _ANCHORS.sample_full_households,
    "observed weeks": _ANCHORS.weeks,
    "household-week grid": _ANCHORS.household_weeks,
    "active household-weeks": _ANCHORS.active_household_weeks,
    "trips (baskets)": _ANCHORS.baskets,
    "total retailer SALES_VALUE": _ANCHORS.retailer_sales_value,
    "total recorded units": _ANCHORS.units,
    "department labels": 44,  # Phase 3 department derivation: 43 labelled + 1 unlabelled sentinel
    "commodity labels": 308,  # Phase 3 commodity object: all labels retained, sentinel included
    "campaigns": _ANCHORS.campaigns,
    "households listed at least once": _ANCHORS.listed_households,
    "median campaigns per listed household": 4,  # median over the accepted listed households
}


class SynthesisError(RuntimeError):
    """Raised when T1 disagrees with an accepted figure or violates the publication contract."""


@dataclass
class Phase3Inventory:
    """The accepted Phase 3 evidence inventory, read from the challenge register."""

    promoted: list[str]
    challenged: list[str]
    failed: list[str]
    questions: dict[str, list[str]] = field(default_factory=dict)

    @property
    def promoted_count(self) -> int:
        return len(self.promoted)


def load_register_inventory() -> Phase3Inventory:
    """Read the promoted / challenged / failed sets straight from the register."""
    records = challenges.read_register()
    promoted = [record.candidate_id for record in challenges.promoted_records(records)]
    challenged = [record.candidate_id for record in records if record.status == "CHALLENGED"]
    failed = [record.candidate_id for record in records if record.status == "FAILED"]
    questions: dict[str, list[str]] = {}
    for record in records:
        questions.setdefault(record.question, []).append(record.candidate_id)
    return Phase3Inventory(
        promoted=promoted, challenged=challenged, failed=failed, questions=questions
    )


def register_synthesis_views(con) -> None:
    """Register the accepted views T1 reads, including the Question C/D derived objects."""
    reconcile.register_accepted_views(con)
    question_c.register_question_c_views(con)
    question_c.load_department_view(con)  # creates the descriptive_department_week view
    department_level = question_c.load_department_level(con)
    con.register("department_level", department_level)
    con.register("commodity_level", question_c.load_commodity_level(con))
    question_d.register_question_d_views(con)
    con.register("campaign_level", question_d.load_campaign_level(con))


#: The three household-concentration rows are carried from the accepted Question B metric route
#: (count-based selection of the k largest values) rather than re-derived, so T1 reproduces the
#: accepted figures exactly.
ACCEPTED_CONCENTRATION_ROWS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "top-10% household share of retailer SALES_VALUE",
        "total_sales_value",
        "retailer revenue; not welfare or income inequality (QB-01)",
    ),
    (
        "top-10% household share of trips",
        "total_baskets",
        "trips are baskets recorded at this retailer (QB-01)",
    ),
    (
        "top-10% household share of active weeks",
        "active_weeks",
        "active weeks are weeks with a recorded purchase at this retailer (QB-01)",
    ),
)


def accepted_concentration_rows(con) -> pd.DataFrame:
    """The three top-10% household shares from the accepted Question B metric route."""
    household = question_b.load_household_level(con)
    rows = []
    for metric, column, note in ACCEPTED_CONCENTRATION_ROWS:
        share = metrics.top_activity_share(household[column].to_numpy(dtype=float), 0.10) * 100.0
        rows.append(
            {
                "domain": "HOUSEHOLD",
                "metric": metric,
                # Six decimals so the accepted two-decimal figure can be reproduced by rounding
                # once rather than twice.
                "value": round(share, 6),
                "unit": "%",
                "population_or_denominator": "top 250 of 2,500 households (QB-01)",
                "notes": note,
            }
        )
    return pd.DataFrame(rows)


def build_t1(con) -> pd.DataFrame:
    """Assemble the compact paper-facing KPI table from accepted sources only."""
    register_synthesis_views(con)
    sql = (contracts.DESCRIPTIVE_SQL_DIR / T1_SQL_FILENAME).read_text(encoding="utf-8")
    frame = con.execute(sql).fetchdf()
    frame = pd.concat([frame, accepted_concentration_rows(con)], ignore_index=True)
    frame = frame.sort_values(["domain", "metric"], kind="stable").reset_index(drop=True)
    return frame


def check_t1(frame: pd.DataFrame) -> list[str]:
    """Verify T1 against the accepted figures and the publication contract."""
    checks: list[str] = []

    def record(label: str, ok: bool, detail: str = "") -> None:
        checks.append(f"{'ok  ' if ok else 'FAIL'} {label}{'' if not detail else ': ' + detail}")
        if not ok:
            raise SynthesisError(f"T1 contract violated: {label} {detail}")

    record(
        "row count is inside the declared band",
        T1_MIN_ROWS <= len(frame) <= T1_MAX_ROWS,
        f"rows={len(frame)} band={T1_MIN_ROWS}-{T1_MAX_ROWS}",
    )
    record(
        "schema is exactly the declared long format",
        list(frame.columns)
        == ["domain", "metric", "value", "unit", "population_or_denominator", "notes"],
        f"{list(frame.columns)}",
    )
    record(
        "domains are the declared ones",
        sorted(set(frame["domain"])) == sorted(T1_DOMAINS),
        f"{sorted(set(frame['domain']))}",
    )
    record("no metric is repeated", frame["metric"].is_unique)
    record(
        "every row carries a unit, a denominator and a note",
        bool(
            (
                frame["unit"].str.strip().ne("")
                & frame["population_or_denominator"].str.strip().ne("")
                & frame["notes"].str.strip().ne("")
            ).all()
        ),
    )
    record(
        "every SALES_VALUE row says retailer revenue or retailer currency",
        bool(
            frame.loc[frame["metric"].str.contains("SALES_VALUE", case=False), "notes"]
            .str.lower()
            .str.contains("retailer revenue|retailer currency")
            .all()
        ),
    )
    record(
        "the unit row carries the QA-03 caveat",
        "QA-03" in frame.loc[frame["metric"] == "total recorded units", "notes"].iloc[0],
    )

    for metric, (finding, expected, places) in ACCEPTED_KPI_EXPECTATIONS.items():
        row = frame.loc[frame["metric"] == metric]
        record(f"metric present: {metric}", len(row) == 1)
        observed = float(row["value"].iloc[0])
        record(
            f"{metric} still matches accepted {finding} ({expected})",
            round(observed, places) == round(expected, places),
            f"observed={observed} accepted={expected}",
        )

    for metric, expected in ACCEPTED_KPI_COUNTS.items():
        row = frame.loc[frame["metric"] == metric]
        record(f"metric present: {metric}", len(row) == 1)
        observed = float(row["value"].iloc[0])
        record(
            f"{metric} reproduces the accepted value ({expected})",
            abs(observed - expected) < 1e-6,
            f"observed={observed} accepted={expected}",
        )
    return checks


def accepted_sources_note() -> str:
    """Provenance line that must accompany T1 wherever it is described."""
    return (
        "T1 is assembled only from already accepted/reconciled Phase 1-3 evidence "
        f"({contracts.ANCHOR_PROVENANCE}); it introduces no new substantive analysis."
    )


def write_local_inventory(inventory: Phase3Inventory, path: Path) -> Path:
    """Write the inventory for the run log (public IDs only)."""
    import json

    path.write_text(
        json.dumps(
            {
                "promoted": inventory.promoted,
                "challenged": inventory.challenged,
                "failed": inventory.failed,
                "per_question": {key: sorted(value) for key, value in inventory.questions.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
