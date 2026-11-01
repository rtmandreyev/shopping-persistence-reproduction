"""Cross-policy reporting semantics (P9-002 §1) and exploratory isolation.

- **P-J13, P-J14 and P-M are reported separately.**
- **Generic "Jev".** A label is reported for Jev only when P-J13 and P-J14 give the
  same label. Otherwise it is VERSION-SENSITIVE.
- **Cross-vendor.** A Jev-generic label is compared with P-M as AGREE or DISAGREE. It
  is NOT-COMPARABLE if the Jev label is version-sensitive.
- **Exploratory.** The top-k |pi_a - pi_b| disagreement table lives in its own
  function. Its output is never read by any labelling function.
"""

from __future__ import annotations

import numpy as np

from complete_journey.systemone import lattice as L
from complete_journey.systemone.policy import PolicyData

VERSION_SENSITIVE = "VERSION-SENSITIVE"
AGREE, DISAGREE, NOT_COMPARABLE = "AGREE", "DISAGREE", "NOT-COMPARABLE"


def flat_labels(policy_result: dict) -> dict[str, str]:
    """The substantive labels of one policy result (confirmatory keys only)."""
    out = {}
    st = policy_result["static"]["primary"]
    for k, v in st["estimand_labels"].items():
        out[f"estimand:{k}"] = v
    for c, v in st["validity"].items():
        out[f"validity:{c}"] = v["label"]
    for k, v in policy_result.get("convention_dependence", {}).items():
        out[f"convention:{k}"] = v
    for name, rec in policy_result.get("recovery", {}).items():
        out[f"admissibility:{name}"] = rec["level2"]["admissibility"]["label"]
    return out


def jev_generic(j13: dict[str, str], j14: dict[str, str]) -> dict[str, str]:
    keys = sorted(set(j13) | set(j14))
    return {
        k: (j13[k] if j13.get(k) == j14.get(k) and k in j13 else VERSION_SENSITIVE) for k in keys
    }


def cross_vendor(jev: dict[str, str], merc: dict[str, str]) -> dict[str, str]:
    out = {}
    for k in sorted(set(jev) | set(merc)):
        if jev.get(k) in (None, VERSION_SENSITIVE) or k not in merc:
            out[k] = NOT_COMPARABLE
        else:
            out[k] = AGREE if jev[k] == merc[k] else DISAGREE
    return out


def exploratory_disagreement(
    a: PolicyData, b: PolicyData, top: int = 10, convention: str = "primary"
) -> dict:
    """EXPLORATORY ONLY: states with the largest |pi_a - pi_b|. Never feeds a label."""
    rows = []
    for s in L.all_states(a.k):
        pa, pb = a.pi(s, convention), b.pi(s, convention)
        rows.append((abs(pa - pb), s.key(), pa, pb))
    rows.sort(reverse=True)
    diffs = np.array([r[0] for r in rows])
    return {
        "status": "EXPLORATORY",
        "distribution": {
            "mean": float(diffs.mean()),
            "median": float(np.median(diffs)),
            "max": float(diffs.max()),
        },
        "top": [
            {"state": k, a.policy: pa, b.policy: pb, "abs_diff": d} for d, k, pa, pb in rows[:top]
        ],
    }
