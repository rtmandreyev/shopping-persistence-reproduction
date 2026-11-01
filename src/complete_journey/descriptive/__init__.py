"""Phase 3 descriptive-analysis infrastructure.

This package is the reusable Phase 3 analysis layer: the accepted-input
reconciliation gate, pure descriptive helpers, the neutral department-level
derivation, output/publication safety, and candidate-finding challenge records.

**What this package is not.** It contains no estimation code, no canonical data
model, and no Phase 2 data-construction code. It queries the accepted Phase 2
tables (and, where Run 0 explicitly authorized it, the accepted raw source
interface plus the product hierarchy). It contains no substantive result, no
hardcoded finding, and no reference to the Phase 2C predictive feature layer,
which Phase 3 is forbidden to use.

The frozen specification this package implements is
``docs/phases/PHASE_03_DESCRIPTIVE_ANALYTICS.md``.
"""

from __future__ import annotations

from complete_journey.descriptive import (
    artifacts,
    challenges,
    contracts,
    department,
    metrics,
    reconcile,
)

__all__ = [
    "artifacts",
    "challenges",
    "contracts",
    "department",
    "metrics",
    "reconcile",
]
