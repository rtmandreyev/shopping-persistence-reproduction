"""Drift sentinels: a mid-run stop on version drift (P9-001 stop conditions; P-M amended
by P9-006).

There are 24 predeclared sentinel states, queried at 4 time points (``request.ordered``
places time point t at the start of the t-th quarter of the run). Once every sentinel
of a time point t >= 2 is archived, its ``shop`` answers are compared with time
point 1.

CANDIDATE bound (pre-smoke freeze). Here sigma_smoke is the pooled SD of byte-identical
smoke repeats for the same policy. The bound is computed from the smoke run, never
from production output. STOP if either holds:
- the median over sentinels of |pi_t - pi_1| > max(0.02, 3 * sigma_smoke);
- any single |pi_t - pi_1| > max(0.10, 6 * sigma_smoke).

Each comparison is a difference of two single responses (SD = sqrt(2) * sigma). The
floors come from the vendor-documented repeatability: a mean SD of 0.010 and ranges of
up to 0.10 on uncertain items.

P9-006 (P-M only): per-cell single-shift bounds b_c = max(0.10, 3.80*sqrt(2)*sigma_c),
sigma_c = max(sigma_c_smoke, sigma_pooled), calibrated to a 1% family-wise false-stop
probability over the 24 x 3 comparisons; the median rule is unchanged.
"""

from __future__ import annotations

import numpy as np

from complete_journey.systemone import lattice as L
from complete_journey.systemone import settings as S
from complete_journey.systemone.archive import Archive
from complete_journey.systemone.settings import require


def bound(sigma_smoke: float) -> dict:
    require(np.isfinite(sigma_smoke) and sigma_smoke >= 0, "sigma_smoke must be a finite SD")
    return {"median": max(0.02, 3 * sigma_smoke), "max": max(0.10, 6 * sigma_smoke)}


def stratified_rule(smoke_sd: dict, sigma_pooled: float, z: float) -> dict:
    """The P9-006 rule: per-cell single-shift bounds b_c = max(0.10, z*sqrt(2)*sigma_c)
    with sigma_c = max(sigma_c_smoke, sigma_pooled); the median rule is unchanged."""
    require(set(smoke_sd) == set(S.CELLS), "the stratified rule needs one SD per cell")
    sigma_c = {c: max(float(smoke_sd[c]), float(sigma_pooled)) for c in S.CELLS}
    return {
        "type": "stratified",
        "decision": "P9-006",
        "z": float(z),
        "sigma_pooled": float(sigma_pooled),
        "sigma_c": sigma_c,
        "median": max(0.02, 3 * float(sigma_pooled)),
        "single_by_cell": {
            c: float(max(0.10, float(z) * np.sqrt(2) * s)) for c, s in sigma_c.items()
        },
    }


def p9006_rule() -> dict:
    """The frozen amended P-M rule (P9-006)."""
    return stratified_rule(S.DRIFT_P9006_SMOKE_SD, S.DRIFT_P9006_SIGMA_POOLED, S.DRIFT_P9006_Z)


def sentinel_values(archive: Archive, t: int) -> dict[str, float]:
    out = {}
    for rec in archive.successes():
        ident = rec["identity"]
        if (
            ident["purpose"] == "sentinel"
            and ident["timepoint"] == t
            and S.Q_SHOP in rec["answers"]
        ):
            out[ident["state_key"]] = rec["answers"][S.Q_SHOP]
    return out


def check(
    archive: Archive, k: int, t: int, sigma_smoke: float | None = None, rule: dict | None = None
) -> dict:
    """P9-005 rule from ``sigma_smoke``; or, if ``rule`` is given, the P9-006 stratified
    rule (which must equal ``p9006_rule()``)."""
    require(t >= 2, "drift is measured against time point 1")
    sents = sorted(L.sentinels(k), key=lambda s: s.key())
    base, now = sentinel_values(archive, 1), sentinel_values(archive, t)
    require(
        all(s.key() in base and s.key() in now for s in sents),
        f"sentinel time points 1 and {t} must be complete",
    )
    diffs = np.array([abs(now[s.key()] - base[s.key()]) for s in sents])
    if rule is None:
        b = bound(sigma_smoke)
        stop = bool(np.median(diffs) > b["median"] or diffs.max() > b["max"])
        return {
            "timepoint": t,
            "median_abs_shift": float(np.median(diffs)),
            "max_abs_shift": float(diffs.max()),
            "bound": b,
            "stop": stop,
        }
    require(rule == p9006_rule(), "only the frozen P9-006 stratified rule is allowed")
    per_cell = {}
    for s, d in zip(sents, diffs, strict=True):
        per_cell[s.cell] = max(per_cell.get(s.cell, 0.0), float(d))
    single = any(per_cell[c] > rule["single_by_cell"][c] for c in per_cell)
    return {
        "timepoint": t,
        "rule": "P9-006 stratified",
        "median_abs_shift": float(np.median(diffs)),
        "max_abs_shift": float(diffs.max()),
        "max_abs_shift_by_cell": per_cell,
        "bound": {"median": rule["median"], "single_by_cell": rule["single_by_cell"]},
        "stop": bool(np.median(diffs) > rule["median"] or single),
    }
