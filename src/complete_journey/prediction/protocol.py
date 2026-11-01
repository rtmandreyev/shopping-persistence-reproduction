"""VI-6 recency gain and the per-world evaluation protocol (P7-003).

The recency gain is a VALIDATION statistic, not a causal diagnostic. Latent
non-history persistence (A1′-type) can produce a positive gain just as genuine
history dependence (A2-type) can.

``evaluate_world`` applies the frozen pipeline to one panel. The predictor is
**retrained inside every world**; no model trained on real data is ever applied to a
simulated world. Run B accepts synthetic panels only for anything that fits a model.
"""

from __future__ import annotations

import pandas as pd

from complete_journey.prediction import config
from complete_journey.prediction import features as feat
from complete_journey.prediction import modeling as mdl
from complete_journey.prediction import moments as mom
from complete_journey.prediction.operators import offline_rows, realtime_rows
from complete_journey.prediction.panel import CanonicalPanel, require, require_fit_allowed


def recency_rows(panel: CanonicalPanel, operator: str):
    """Operator rows on which Tier 0/1 are defined: origin at or after the first purchase.

    Operator O satisfies this by construction. S and G also contain origins before a
    household's first recorded purchase, where Tier 0 (counted from the first purchase)
    is undefined. Those rows are excluded; the rule is point-in-time (f_i <= t is known
    at t) and uses no future information.
    """
    from complete_journey.prediction.panel import household_support

    rows = offline_rows(panel, operator).rows
    first = household_support(panel.recorded)["f"]
    return rows.loc[rows["origin_week"].to_numpy() >= first.loc[rows["household_key"]].to_numpy()]


def recency_gain(panel: CanonicalPanel, operator: str = config.OPERATOR_OFFLINE) -> dict:
    """ΔLL_rec = LL(B1 | Tier 0) − LL(B1 | Tier 1) on operator rows (positive = recency helps).

    Train on responses 2–68, evaluate on 69–101, B1 at the frozen fixed penalty, both
    tiers on identical rows. The secondary is ΔBrier_rec, defined the same way.
    """
    require_fit_allowed(panel)
    rows = recency_rows(panel, operator)
    rows = feat.build_features(panel, rows, tier=1)
    windows = {"train": config.RECENCY_TRAIN, "eval": config.RECENCY_EVAL}
    parts = mdl.temporal_split(rows, windows)
    train, evaluate = parts["train"], parts["eval"]
    require(len(train) and len(evaluate), "recency gain needs train and evaluation rows")
    losses = {}
    for tier, columns in (("tier0", config.TIER0), ("tier1", config.TIER1)):
        model = mdl.fit_fixed_b1(panel, train, columns)
        p = model.predict(evaluate)
        losses[tier] = (mdl.log_loss(evaluate["y"], p), mdl.brier(evaluate["y"], p))
    return {
        config.RECENCY_GAIN: mdl.recency_gain(losses["tier0"][0], losses["tier1"][0]),
        config.RECENCY_GAIN_SECONDARY: mdl.recency_gain(losses["tier0"][1], losses["tier1"][1]),
        "log_loss_tier0": losses["tier0"][0],
        "log_loss_tier1": losses["tier1"][0],
        "n_train": int(len(train)),
        "n_eval": int(len(evaluate)),
        "operator": operator,
    }


def vi6_records(panel: CanonicalPanel, operator: str = config.OPERATOR_OFFLINE) -> list[dict]:
    gain = recency_gain(panel, operator)
    rows = recency_rows(panel, operator)
    return [
        mom.record(
            "VI-6",
            name,
            gain[name],
            "nats" if name == config.RECENCY_GAIN else "Brier",
            operator,
            gain["n_eval"],
            rows["household_key"].nunique(),
            panel,
            status="secondary",
        )
        for name in (config.RECENCY_GAIN, config.RECENCY_GAIN_SECONDARY)
    ]


def benchmark_ladder(
    panel: CanonicalPanel, tier: int = 1, store: pd.DataFrame | None = None
) -> dict:
    """P benchmark with the frozen split (design §H/§J).

    Tiers 0/1: B0 and B1 at the fixed penalty (no tuning). Tier 2 (requires the audited
    feature store): B0, B1 over the C grid, and the single B2 ceiling over its grid.
    Selection uses validation only; test is scored once per model.
    """
    require_fit_allowed(panel)
    require(tier in (0, 1, 2), "tier must be 0, 1 or 2")
    rows = feat.build_features(panel, realtime_rows(panel).rows, tier=1 if tier == 2 else tier)
    if tier == 2:
        require(store is not None, "Tier 2 needs the audited feature store")
        rows = feat.attach_tier2(rows, store)
    columns = {0: config.TIER0, 1: config.TIER1, 2: feat.tier2_columns()}[tier]
    parts = mdl.temporal_split(rows)
    report = {}
    report["B0"] = mdl.fit_selected(panel, "B0", parts["train"], parts["validation"], columns)
    if tier == 2:
        report["B1"] = mdl.fit_selected(panel, "B1", parts["train"], parts["validation"], columns)
        report["B2"] = mdl.fit_selected(panel, "B2", parts["train"], parts["validation"], columns)
    else:
        train_val = pd.concat([parts["train"], parts["validation"]])
        report["B1"] = mdl.fit_fixed_b1(panel, train_val, columns)
    return {
        name: {"params": fitted.params, "test": mdl.score(fitted, parts["test"])}
        for name, fitted in report.items()
    }


def evaluate_world(
    panel: CanonicalPanel,
    architecture: str,
    replication: int,
    seed: int,
    operators=(config.OPERATOR_OFFLINE,),
) -> list[dict]:
    """Full register for one world: operators → rows → features → split → retrained B1."""
    require(panel.origin == "synthetic", "evaluate_world runs on simulated/synthetic worlds")
    records = []
    for operator in operators:
        records.extend(mom.moment_vector(panel, operator))
        records.extend(mom.e1_records(panel, operator))
        records.extend(vi6_records(panel, operator))
    for r in records:
        r.update(
            simulation_architecture=architecture, simulation_replication=replication, seed=seed
        )
    return mom.validate_records(records)


def as_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(mom.validate_records(records), columns=mom.MOMENT_FIELDS)
