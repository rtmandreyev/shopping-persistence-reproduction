"""Recompute the in-family known-truth verdicts (Section 6, Appendix D) from stored moments.

Reads ``artifacts/known_truth_panels.json.gz`` (the five dynamic moments of every simulated
panel, the real-panel moments and standard errors, and the simulated moments at every family
grid point) and recomputes the comparative verdicts, family verdicts and naive-rule flags. The
counts are compared with the frozen record ``docs/phases/PHASE_08_FINAL/rq2_per_truth.json``.
With ``all_points=True`` it runs the grid-resolution diagnostic of Appendix D.7, which is not
part of the recorded design.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from repro.adequacy import KAPPAS, Family, Rule, comparative

NAIVE_Z = 1.96


def load(art: Path) -> dict:
    return json.loads(gzip.decompress((art / "known_truth_panels.json.gz").read_bytes()))


def _truth(args):
    doc, arm, label, all_points = args
    rule = Rule(doc["E"], doc["floor"])
    fams = {f: Family(doc["tables"][arm][f], rule) for f in ("A1_PRIME", "A2")}
    rows = np.asarray(doc["rq2_panels"][arm][label])
    se = rows.std(0, ddof=1)  # a known truth's SE: SD across its 200 panels (recorded design)
    real_se = np.asarray(doc["real_se"])
    comp = {k: Counter() for k in KAPPAS}
    fam = {f: {k: Counter() for k in KAPPAS} for f in fams}
    naive = 0
    for row in rows:
        v = {f: fams[f].evaluate(row, se, all_points) for f in fams}
        for k in KAPPAS:
            comp[k][comparative(v["A2"][k], v["A1_PRIME"][k])] += 1
            for f in fams:
                fam[f][k][v[f][k]] += 1
        naive += bool(row[0] > NAIVE_Z * real_se[0] or row[3] > NAIVE_Z * real_se[3])
    return arm, label, {
        "comparative": {str(k): dict(c) for k, c in comp.items()},
        "family": {f: {str(k): dict(c) for k, c in d.items()} for f, d in fam.items()},
        "naive": naive,
        "panels": len(rows),
    }


def recompute(art: Path, all_points: bool = False, workers: int | None = None) -> dict:
    doc = load(art)
    jobs = [(doc, arm, label, all_points) for arm in doc["rq2_panels"] for label in doc["rq2_panels"][arm]]
    out: dict = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for arm, label, res in ex.map(_truth, jobs):
            out.setdefault(arm, {})[label] = res
    return out


def compare(art: Path, recomputed: dict) -> list[str]:
    """Mismatches against the frozen per-truth record; an empty list means agreement."""
    stored = json.loads((art.parent / "docs/phases/PHASE_08_FINAL/rq2_per_truth.json").read_text())
    bad = []
    for arm, truths in recomputed.items():
        for label, r in truths.items():
            s = stored[arm][label]
            if r["comparative"] != s["comparative_counts"]:
                bad.append(f"{arm}/{label}: comparative counts")
            for f in ("A1_PRIME", "A2"):
                if r["family"][f] != s["family_verdict_counts"][f]:
                    bad.append(f"{arm}/{label}: {f} family counts")
            if r["naive"] != s["naive"]["union"]["k"]:
                bad.append(f"{arm}/{label}: naive count")
    expected = sum(1 for arm in stored for t in stored[arm].values() if t.get("comparative_counts"))
    got = sum(len(v) for v in recomputed.values())
    if got != expected:
        bad.append(f"recomputed {got} truth cells, frozen record has {expected}")
    return bad
