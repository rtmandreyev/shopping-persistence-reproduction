"""Phase 8 recovery tests on artificial panels (Phase 8 code reused read-only).

**Level 1.** The frozen P1–P5 moment engine (``abm.moments.primary_moments``, operator
O) and the Phase 8 naive persistence rule (``phase8_final.analysis.naive_flags``) on
every artificial panel.

**Level 2.** The frozen Phase 8 plug-in classification, with artificial panels treated
as additional RQ2-style truths:
- **Admissibility.** The mean CI-1 of the separate population block must pass the
  unchanged CI-1 gate against the real CI-1:
  |mean - real| <= max(0.02, 2 SE_real) for all six components.
  Otherwise the DGP is labelled INFEASIBLE ARTIFICIAL TRUTH. It is never dropped.
- **Rates.** Each of the R panels is classified against the frozen real-calibrated
  model tables, in both heterogeneity arms, with ``evaluate_panel`` and
  ``comparatives``. The history-favoured rate carries an exact Clopper–Pearson
  interval.
- **Representability.** The state is classified on the population block against the
  competitor family's images, exactly as in RQ2: A2 for no-history truths, A1' for
  history truths.

**Level 3** (per-panel recalibrated family fits) is deferred and not implemented.
Nothing here writes to a Phase 8 location.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np

from complete_journey.systemone import settings as S
from complete_journey.systemone.closedloop import DGP, simulate
from complete_journey.systemone.settings import require

INFEASIBLE_ARTIFICIAL_TRUTH = "INFEASIBLE ARTIFICIAL TRUTH"
FEASIBLE_ARTIFICIAL_TRUTH = "FEASIBLE ARTIFICIAL TRUTH"
HISTORY_TRUTH, NO_HISTORY_TRUTH = "HISTORY-INSTALLED", "NO-HISTORY"


def panel_moments(behavior: np.ndarray) -> dict | None:
    """P1..P5 and CI-1 for one artificial panel (None if the frozen engine refuses it)."""
    from complete_journey.abm.config import Phase8ContractError
    from complete_journey.abm.moments import ci1_fast, primary_moments
    from complete_journey.abm.operators import PHASE8_PANEL_TAG, long_frame
    from complete_journey.prediction.panel import CanonicalPanel

    frame = long_frame(behavior)
    panel = CanonicalPanel(
        frame,
        origin="synthetic",
        behavior=frame.copy(),
        label=f"{PHASE8_PANEL_TAG}:primary:SYSTEMONE",
    )
    try:
        p = primary_moments(panel).values
    except (Phase8ContractError, ValueError, np.linalg.LinAlgError):
        return None
    return {
        "primary": [p[j] for j in ("P1", "P2", "P3", "P4", "P5")],
        "ci1": ci1_fast(behavior).tolist(),
    }


def _panel_job(args):
    table, dgp, k, coords, n = args
    sim = simulate(table, dgp, k, coords, n=n)
    out = panel_moments(sim["behavior"])
    if out is not None:
        out["coords"] = list(coords)
        out["mean_rate"] = float(sim["behavior"].mean())
    return out


def run_block(
    table: np.ndarray,
    dgp: DGP,
    k: int,
    prefix: tuple[int, ...],
    panels: int,
    n: int = S.N_HOUSEHOLDS,
    workers: int = 1,
) -> dict:
    """``panels`` closed-loop panels for one DGP; seeds = (*prefix, replicate)."""
    jobs = [(table, dgp, k, (*prefix, r), n) for r in range(panels)]
    if workers > 1:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_panel_job, jobs, chunksize=max(1, panels // (4 * workers))))
    else:
        res = [_panel_job(j) for j in jobs]
    ok = [r for r in res if r is not None]
    return {"panels": ok, "failures": len(res) - len(ok)}


def level1(block: dict) -> dict:
    from complete_journey.phase8_final.analysis import naive_flags
    from complete_journey.phase8_final.evaluate import rate

    rows = np.array([p["primary"] for p in block["panels"]], float)
    require(len(rows) >= 2, "level 1 needs at least two successful panels")
    flags = [naive_flags(r) for r in rows]
    return {
        "q": len(rows),
        "failures": block["failures"],
        "moments_mean": dict(
            zip(("P1", "P2", "P3", "P4", "P5"), rows.mean(0).tolist(), strict=True)
        ),
        "moments_sd": dict(
            zip(("P1", "P2", "P3", "P4", "P5"), rows.std(0, ddof=1).tolist(), strict=True)
        ),
        "naive": {
            "union": rate([a or b for a, b in flags]),
            "m3": rate([a for a, _ in flags]),
            "rec": rate([b for _, b in flags]),
        },
    }


def ci1_admissible(pop_block: dict) -> dict:
    from complete_journey.abm import config as abm_config
    from complete_journey.abm.stage1 import real_ci1_target

    real = real_ci1_target()
    ci1 = np.array([p["ci1"] for p in pop_block["panels"]], float)
    mean = ci1.mean(0)
    comps = abm_config.CI1_COMPONENTS
    gate = {}
    for j, c in enumerate(comps):
        tol = max(abm_config.CI1_GATE_ABS, 2 * real.se[c])
        gate[c] = {
            "artificial": float(mean[j]),
            "real": real.values[c],
            "tolerance": tol,
            "pass": bool(abs(mean[j] - real.values[c]) <= tol),
        }
    ok = all(v["pass"] for v in gate.values())
    return {
        "label": FEASIBLE_ARTIFICIAL_TRUTH if ok else INFEASIBLE_ARTIFICIAL_TRUTH,
        "components": gate,
    }


@lru_cache(maxsize=2)
def phase8_tables(arm: str):
    """Read-only load of the frozen Phase 8 model table (real-CI-1 plug-in images)."""
    from complete_journey.abm import stage1_p8005 as s1p
    from complete_journey.paths import PROJECT_ROOT
    from complete_journey.phase8_final import analysis as A
    from complete_journey.phase8_final import settings as P8
    from complete_journey.phase8_final.snapshot import checkpoint_config_hash
    from complete_journey.phase8_final.store import ShardedStore

    store = ShardedStore(
        PROJECT_ROOT / P8.WORK / "checkpoints", checkpoint_config_hash(), "systemone-read-only"
    )
    return A.build_tables(store, arm), s1p.tolerances()


def level2(rows_block: dict, pop_block: dict, truth: str, arms=("logit_normal", "mixture")) -> dict:
    from collections import Counter

    from complete_journey.abm import config as abm_config
    from complete_journey.abm.representability import HISTORY_FAVOURED
    from complete_journey.phase8_final import analysis as A
    from complete_journey.phase8_final.evaluate import classify_direct, rate

    require(truth in (HISTORY_TRUTH, NO_HISTORY_TRUTH), "unknown artificial truth class")
    adm = ci1_admissible(pop_block)
    out = {"admissibility": adm, "truth": truth, "arms": {}}
    if adm["label"] == INFEASIBLE_ARTIFICIAL_TRUTH:
        return out
    rows, se = A.panel_arrays(rows_block["panels"])
    pop, pop_sd = A.panel_arrays(pop_block["panels"])
    competitor = "A2" if truth == NO_HISTORY_TRUTH else "A1_PRIME"
    for arm in arms:
        tables, tol = phase8_tables(arm)
        image = list(tables[competitor]["coarse"].values()) + list(
            tables[competitor]["refined"].values()
        )
        states = {
            str(k): classify_direct(pop.mean(0), pop_sd, len(pop), image, tol, k)
            for k in abm_config.KAPPAS
        }
        hf = {k: [] for k in abm_config.KAPPAS}
        counts = {k: Counter() for k in abm_config.KAPPAS}
        for row in rows:
            res, _ = A.evaluate_panel(tables, row, se, tol)
            comp = A.comparatives(res, tol)
            for k in abm_config.KAPPAS:
                hf[k].append(comp[k] == HISTORY_FAVOURED)
                counts[k][comp[k]] += 1
        out["arms"][arm] = {
            "competitor": competitor,
            "states": states,
            "history_favoured": {str(k): rate(v) for k, v in hf.items()},
            "comparative_counts": {str(k): dict(v) for k, v in counts.items()},
        }
    return out
