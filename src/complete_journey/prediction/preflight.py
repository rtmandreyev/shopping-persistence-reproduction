"""Run B checks: contracts, real-data preflight without fits, synthetic preflight.

Real-data work here is limited to schema, counts, eligibility, split support,
prevalence (authorized in P7-003 Run A), feature-timing verification, and
reconciliation of the ladder adapter to already-published Phase 4 aggregates. No
predictive model touches real outcomes, and no new dynamic moment is computed on
real data.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd

from complete_journey.paths import PROJECT_ROOT
from complete_journey.prediction import config, protocol, twfe
from complete_journey.prediction import features as feat
from complete_journey.prediction import modeling as mdl
from complete_journey.prediction import moments as mom
from complete_journey.prediction import synthetic as syn
from complete_journey.prediction.operators import (
    draw_empirical_windows,
    empirical_window_pairs,
    offline_rows,
    realtime_rows,
)
from complete_journey.prediction.panel import load_real_panel, require

#: Published Phase 4 values (T4 / Run B transitions) used only to reconcile the adapter.
PUBLISHED_PHASE4 = {
    "M0": 0.423973,
    "M1": 0.396100,
    "M2": 0.227728,
    "M3": 0.174384,
    "P11": 0.708473499,
    "P01": 0.284500767,
}
PUBLISHED_TOLERANCE = 5e-7
LEAKAGE_CUTOFFS = (34, 68, 84, 100)


def check() -> dict:
    from complete_journey.data.source_validation import validate_current_source
    from complete_journey.prediction import panel as panel_module

    require(
        panel_module._RUN_C_MODE is None and not panel_module._authorized(),
        "check/preflight must run outside any Run C real-fit context",
    )
    require(importlib.util.find_spec("sklearn") is not None, "scikit-learn missing")
    import pyarrow.parquet as pq
    import sklearn

    store_path = PROJECT_ROOT / "data/processed/predictive_household_week_features.parquet"
    store_columns = pq.read_schema(store_path).names if store_path.is_file() else None
    if store_columns is not None:
        feat.validate_store_columns(store_columns)
    tiers = {r["tier"] for r in feat.inventory()}
    require(not (tiers & {"D"}), "inventory contains a Tier D feature")
    excluded = [r["name"] for r in feat.inventory() if r["tier"] == "E" and r["tier2"] == "yes"]
    require(not excluded, f"Tier E admitted to Tier 2: {excluded}")
    source = validate_current_source()
    return {
        "mode": "check",
        "scikit_learn": sklearn.__version__,
        "real_fit_authorized_outside_run_c": False,
        "tier2_stored_features": len(feat.stored_tier2_allowlist()),
        "tier2_total_features": len(feat.tier2_columns()),
        "feature_store_columns_classified": store_columns is not None,
        "phase2_source_classes": {k: len(v) for k, v in source["source_classes"].items()},
    }


def real_preflight() -> dict:
    report = check()
    report["mode"] = "preflight"
    panel = load_real_panel()
    p_rows = realtime_rows(panel).rows
    o_rows = offline_rows(panel, config.OPERATOR_OFFLINE).rows
    s_rows = offline_rows(panel, config.OPERATOR_SENSITIVITY).rows
    support = {}
    for name, rows in (("P", p_rows), ("O", o_rows), ("S", s_rows)):
        parts = mdl.temporal_split(rows)
        support[name] = {
            split: {
                "rows": int(len(f)),
                "households": int(f["household_key"].nunique()),
                "prevalence": round(float(f["y"].mean()), 4),
            }
            for split, f in parts.items()
        }
    report["split_support"] = support
    store = pd.read_parquet(
        PROJECT_ROOT / "data/processed/predictive_household_week_features.parquet"
    )
    pre_entry = int((store["ever_purchased_as_of_week"] == 0).sum())
    report["pre_first_purchase_store_rows_excluded"] = pre_entry
    require(pre_entry == 22788, "pre-first-purchase row count drifted from the Run A audit")
    require(len(p_rows) == 227212 and len(o_rows) == 216162, "population counts drifted from Run A")
    timing = {
        c: feat.truncation_invariant(panel, p_rows, c, seed=config.SEED) for c in LEAKAGE_CUTOFFS
    }
    require(all(timing.values()), f"derived feature truncation test failed: {timing}")
    report["derived_feature_truncation"] = timing
    report["tier2_store_truncation"] = _store_truncation()
    tier2 = feat.attach_tier2(feat.build_features(panel, p_rows, tier=1), store)
    report["tier2_matrix_columns"] = len(feat.tier2_columns())
    require(
        set(feat.tier2_columns()) <= set(tier2.columns), "Tier 2 matrix missing allowlisted columns"
    )
    report["published_reconciliation"] = _reconcile_published(panel)
    pairs = empirical_window_pairs(panel)
    agents = config.SYNTHETIC_KEY_OFFSET + np.arange(1, len(pairs) + 1)
    drawn = draw_empirical_windows(pairs, agents, np.random.default_rng(config.SEED))
    require(
        sorted(zip(drawn["f"], drawn["l"], strict=True))
        == sorted(zip(pairs["f"], pairs["l"], strict=True)),
        "empirical window draw does not reproduce the E-2 multiset",
    )
    report["e2_window_pairs"] = {"pairs": len(pairs), "draw_reproduces_multiset": True}
    return report


def _store_truncation() -> dict:
    spec = importlib.util.spec_from_file_location(
        "verify_predictive_leakage", PROJECT_ROOT / "scripts/verify_predictive_leakage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = {}
    for cutoff in LEAKAGE_CUTOFFS:
        code = module.main(["verify_predictive_leakage.py", str(cutoff)])
        require(code == 0, f"stored feature truncation test failed at cutoff {cutoff}")
        results[cutoff] = "PASS"
    return results


def _reconcile_published(panel) -> dict:
    """Reproduce accepted Phase 4 S0 numbers (responses 2-102) with the new adapter."""
    mart = pd.read_parquet(
        PROJECT_ROOT / "data/processed/econometrics_household_week.parquet",
        columns=["household_key", "week_no", "retailer_purchase", "lag1_retailer_purchase"],
    )
    frame = mart.loc[mart["week_no"].between(2, 102)].rename(
        columns={"retailer_purchase": "y", "lag1_retailer_purchase": "lag1"}
    )
    out = {}
    for model in twfe.MODELS:
        out[model] = twfe.fe_coefficients(frame, "y", ["lag1"], model)["lag1"]
    out["P11"] = float(frame.loc[frame["lag1"] == 1, "y"].mean())
    out["P01"] = float(frame.loc[frame["lag1"] == 0, "y"].mean())
    for key, published in PUBLISHED_PHASE4.items():
        require(
            abs(out[key] - published) <= PUBLISHED_TOLERANCE,
            f"adapter does not reproduce published Phase 4 {key}: {out[key]} vs {published}",
        )
    return {k: round(v, 9) for k, v in out.items()}


def synthetic_preflight(seed: int = config.SEED) -> dict:
    report = {"mode": "synthetic", "fixtures": {}}
    for key, spec in syn.FIXTURES.items():
        panel, truth = syn.generate(spec, seed)
        records = protocol.evaluate_world(panel, key, 0, seed, operators=(config.OPERATOR_OFFLINE,))
        values = {f"{r['moment_id']}:{r['component']}": r["estimate"] for r in records}
        naive = mom.vi2_ladder(panel, config.OPERATOR_DIAGNOSTIC)
        report["fixtures"][key] = {
            "notes": spec.notes,
            "true_carryover_pp": round(truth["true_carryover_pp"], 4),
            "recency_gain_nats": round(values[f"VI-6:{config.RECENCY_GAIN}"], 5),
            "O_M3": round(values["VI-2:O-M3"], 4),
            "G_M3_full_grid": round(
                next(r["estimate"] for r in naive if r["component"] == "O-M3"), 4
            ),
            "lag_profile": [round(values[f"VI-3:beta_{k}"], 4) for k in config.LAG_SET],
            "gamma_2": round(values["VI-3:gamma_2"], 4),
            "transition_gap": round(values["VI-1:GAP"], 4),
        }
    leak_panel, _ = syn.generate(syn.FIXTURES["S1"], seed)
    try:
        feat.validate_store_columns(syn.s7_feature_store(leak_panel).columns)
        s7 = "NOT REJECTED"
    except ValueError:
        s7 = "rejected (unclassified future column never admitted)"
    require(s7.startswith("rejected"), "S7 leakage trap was not rejected")
    report["fixtures"]["S7"] = {"leak_column": syn.S7_LEAK_COLUMN, "result": s7}
    return report


def render(report: dict) -> str:
    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    return json.dumps(report, indent=2, default=default)
