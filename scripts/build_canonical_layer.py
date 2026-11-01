#!/usr/bin/env python
"""Build the Phase 2A canonical analytical data layer.

THE documented reproduction command for the canonical layer:

    .venv/bin/python scripts/build_canonical_layer.py

It verifies the raw manifest, registers the verified strict Phase 1 raw views,
builds the canonical Parquet tables under ``data/processed/``, runs the QA check
suite and aggregate eligibility summaries, validates every output contract, and
writes the tracked contracts/provenance/lineage documents. Raw data is never
modified. A HARD_INVARIANT failure aborts before any artifact is written.

Exit codes: 0 success, 1 hard invariant or contract failure, 2 raw-data error.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from complete_journey.data.canonical import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
