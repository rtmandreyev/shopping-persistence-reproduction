"""Challenge records for candidate Phase 3 findings.

A candidate exploratory pattern does not become a promoted Phase 3 finding until a
challenge record has been completed for it (frozen plan, section 7). This module is
the machine-readable shape of that record plus the register it lives in.

It deliberately does **not** enforce the substance of a challenge (whether a check
was answered honestly). It enforces the mechanical contract: the field set, the
allowed statuses, and that a promoted record carries promotion notes and no failure
reason. The register ships empty; Run 1 records no candidate finding.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from complete_journey.paths import PROJECT_ROOT


class ChallengeStatus(StrEnum):
    """Lifecycle of a candidate finding. Failed candidates stay recorded."""

    EXPLORATORY = "EXPLORATORY"
    CHALLENGED = "CHALLENGED"
    PROMOTED = "PROMOTED"
    FAILED = "FAILED"


#: Ordered field set of a challenge record. The nine check fields are the frozen
#: challenge card from the Phase 3 plan.
CHALLENGE_FIELD_NAMES: Final[tuple[str, ...]] = (
    "candidate_id",
    "question",
    "candidate_statement",
    "evidence_artifacts",
    "denominator_check",
    "composition_check",
    "outlier_check",
    "participation_check",
    "product_mix_check",
    "coverage_check",
    "construction_check",
    "campaign_overlap_check",
    "retailer_observation_check",
    "independent_recomputation",
    "status",
    "failure_reason",
    "promotion_notes",
)

#: The nine challenge items a promoted finding must have addressed.
CHALLENGE_ITEMS: Final[tuple[str, ...]] = (
    "denominator_check",
    "composition_check",
    "outlier_check",
    "participation_check",
    "product_mix_check",
    "coverage_check",
    "construction_check",
    "campaign_overlap_check",
    "retailer_observation_check",
)

#: Researcher-approved promotions: the Question A review (QA-01..QA-03) and the
#: Question B finalization (QB-01..QB-04). This is the only place the approval set is
#: stated: extending or changing it requires a new explicit researcher decision, and the
#: entry-point check compares the register against it.
APPROVED_PROMOTED_IDS: Final[tuple[str, ...]] = (
    "QA-01",
    "QA-02",
    "QA-03",
    "QB-01",
    "QB-02",
    "QB-03",
    "QB-04",
    "QC-01",
    "QC-02",
    "QC-03",
    "QC-04",
    "QC-06",
    "QD-01",
    "QD-02",
    "QD-04",
)

#: Tracked register of challenge records. Ships with a header and zero records.
REGISTER_PATH: Final[Path] = PROJECT_ROOT / "docs" / "phases" / "PHASE_03_CHALLENGE_REGISTER.csv"


class ChallengeRecordError(ValueError):
    """Raised when a challenge record violates the schema or its status rules."""


@dataclass(frozen=True)
class ChallengeRecord:
    """One candidate finding and the outcome of its challenge record."""

    candidate_id: str
    question: str = ""
    candidate_statement: str = ""
    evidence_artifacts: str = ""
    denominator_check: str = ""
    composition_check: str = ""
    outlier_check: str = ""
    participation_check: str = ""
    product_mix_check: str = ""
    coverage_check: str = ""
    construction_check: str = ""
    campaign_overlap_check: str = ""
    retailer_observation_check: str = ""
    independent_recomputation: str = ""
    status: str = ChallengeStatus.EXPLORATORY.value
    failure_reason: str = ""
    promotion_notes: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ChallengeRecordError("candidate_id is required")
        try:
            status = ChallengeStatus(self.status)
        except ValueError as exc:
            allowed = sorted(member.value for member in ChallengeStatus)
            raise ChallengeRecordError(
                f"status {self.status!r} is not allowed; expected one of {allowed}"
            ) from exc
        if status is ChallengeStatus.PROMOTED:
            if not self.promotion_notes.strip():
                raise ChallengeRecordError("a PROMOTED record requires promotion_notes")
            if self.failure_reason.strip():
                raise ChallengeRecordError("a PROMOTED record cannot carry a failure_reason")
        if status is ChallengeStatus.FAILED and not self.failure_reason.strip():
            raise ChallengeRecordError("a FAILED record requires a failure_reason")

    def to_row(self) -> tuple[str, ...]:
        """Return the record as a row in :data:`CHALLENGE_FIELD_NAMES` order."""
        data = asdict(self)
        return tuple(str(data[name]) for name in CHALLENGE_FIELD_NAMES)

    @property
    def status_kind(self) -> ChallengeStatus:
        return ChallengeStatus(self.status)


def read_register(path: Path = REGISTER_PATH) -> tuple[ChallengeRecord, ...]:
    """Read the challenge register, validating the header and every record.

    Raises:
        ChallengeRecordError: if the header does not match
            :data:`CHALLENGE_FIELD_NAMES`, or if any record fails validation.
    """
    if not path.is_file():
        raise ChallengeRecordError(f"challenge register not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:  # pragma: no cover - defensive
            raise ChallengeRecordError(f"challenge register {path} is empty") from exc
        if tuple(header) != CHALLENGE_FIELD_NAMES:
            raise ChallengeRecordError(
                f"challenge register header mismatch:\n  expected {list(CHALLENGE_FIELD_NAMES)}\n"
                f"  found    {header}"
            )
        records: list[ChallengeRecord] = []
        for line_number, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(CHALLENGE_FIELD_NAMES):
                raise ChallengeRecordError(
                    f"line {line_number}: expected {len(CHALLENGE_FIELD_NAMES)} fields, got {len(row)}"
                )
            try:
                records.append(
                    ChallengeRecord(**dict(zip(CHALLENGE_FIELD_NAMES, row, strict=True)))
                )
            except ChallengeRecordError as exc:
                raise ChallengeRecordError(f"line {line_number}: {exc}") from exc
    return tuple(records)


def write_register(path: Path, records: Iterable[ChallengeRecord]) -> Path:
    """Write the register (header plus records). Used by later Phase 3 runs."""
    materialised = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        # LF line endings, matching every other tracked CSV in this repository.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CHALLENGE_FIELD_NAMES)
        for record in materialised:
            writer.writerow(record.to_row())
    return path


def promoted_records(records: Sequence[ChallengeRecord]) -> tuple[ChallengeRecord, ...]:
    """Promoted findings only — the set that may appear as memo findings."""
    return tuple(record for record in records if record.status_kind is ChallengeStatus.PROMOTED)
