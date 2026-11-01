"""Factual preparation and outcome-free support gate; no empirical fit path."""

from __future__ import annotations

import io
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from complete_journey.choice import artifacts
from complete_journey.choice.contracts import (
    BASELINE,
    CONDITION_ATOL,
    DESIGN_COLUMNS,
    EXPECTED,
    GROUP,
    INDEPENDENT_ATOL,
    LATER,
    LOCAL_PATHS,
    OUTPUTS,
    REMAINING_ATOL,
    SCOPE,
    SINGULAR,
    SINGULAR_ATOL,
    SUPPORT,
    SUPPORT_ATOL,
    THIRDS,
    require,
)
from complete_journey.data.raw_tables import register_views
from complete_journey.paths import PROJECT_ROOT

QA_EXPECTED = {
    **EXPECTED,
    "single_product_events": 6314,
    "same_type_multi_product_events": 230,
    "product_unmatched": 0,
    "basket_unmatched": 0,
    "join_fanout": 0,
    "duplicate_lines": 0,
    "accepted_positive_line_route_agrees": 1,
    "independent_history_route_agrees": 1,
}
CLUSTERS = {
    "households": 1560,
    "stores": 163,
    "weeks": 102,
    "household_store_pairs": 2428,
    "multi_store_households": 586,
    "household_week_cells": 6422,
    "multi_event_household_week_cells": 120,
    "events_in_multi_event_household_week_cells": 242,
}


def load_inputs(
    root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Raw recorded line projection plus accepted basket context; no transaction values."""
    with duckdb.connect() as con:
        con.execute("SET threads=1; SET temp_directory=''")
        register_views(con, root / "data/raw")
        products = con.execute(
            "SELECT PRODUCT_ID AS product_id, BRAND AS brand, SUB_COMMODITY_DESC AS sub_commodity_desc FROM product"
        ).df()
        require(not products.product_id.duplicated().any(), "product source key duplicates")
        # Semi-join filters the exact group without allowing product fan-out.
        lines = con.execute(
            """SELECT household_key,BASKET_ID AS basket_id,PRODUCT_ID AS product_id,
            STORE_ID AS store_id,WEEK_NO AS week_no,DAY AS day,QUANTITY AS quantity
            FROM transaction_data WHERE PRODUCT_ID IN
            (SELECT PRODUCT_ID FROM product WHERE SUB_COMMODITY_DESC=?)""",
            [GROUP],
        ).df()
        baskets = con.execute(
            """SELECT household_key,basket_id,store_id,week_no,day
            FROM read_parquet(?) WHERE (household_key,basket_id) IN
            (SELECT household_key,BASKET_ID FROM transaction_data WHERE PRODUCT_ID IN
            (SELECT PRODUCT_ID FROM product WHERE SUB_COMMODITY_DESC=?))""",
            [str(root / "data/processed/basket_panel.parquet"), GROUP],
        ).df()
        accepted = con.execute(
            """SELECT household_key,basket_id,product_id,store_id,week_no,day,quantity
            FROM read_parquet(?) WHERE product_id IN
            (SELECT PRODUCT_ID FROM product WHERE SUB_COMMODITY_DESC=?)""",
            [str(root / "data/processed/choice_line_price_candidates.parquet"), GROUP],
        ).df()
    return lines, products, baskets, accepted


def independent_history(ledger: pd.DataFrame) -> None:
    """Literal earlier-week iteration independent of SQL cumulative windows."""
    for _, events in ledger.groupby("household_key", sort=True):
        counts = np.zeros(3, dtype=np.int64)
        for _, week in events.groupby("week_no", sort=True):
            expected = [
                int(counts[0]),
                int(counts[1]),
                int(counts[2]),
                int(counts.sum()),
                int(counts[:2].sum()),
            ]
            cols = [
                "prior_national_pure",
                "prior_private_pure",
                "prior_mixed",
                "prior_total_group",
                "prior_pure",
            ]
            require(
                np.array_equal(week[cols].to_numpy(), np.tile(expected, (len(week), 1))),
                "independent earlier-week history mismatch",
            )
            states = week.composition.value_counts()
            counts += np.array(
                [states.get(s, 0) for s in ("NATIONAL_ONLY", "PRIVATE_ONLY", "MIXED")],
                dtype=np.int64,
            )


def accounting(frame: pd.DataFrame, full: bool = True) -> dict[str, int]:
    # Response is read ONLY here, never passed to any matrix/support function.
    result = {
        "rows": len(frame),
        **{h: int(frame.history_category.eq(h).sum()) for h in ("H0", "HN", "HB", "HP")},
        "mixed_only": int(frame.prior_mixed_only.sum()),
        "national": int(frame.private_only.eq(0).sum()),
        "private": int(frame.private_only.eq(1).sum()),
    }
    if full:
        result.update(
            households=int(frame.household_key.nunique()),
            stores=int(frame.store_id.nunique()),
            weeks=int(frame.week_no.nunique()),
        )
    return result


def reconcile_samples(pure: pd.DataFrame) -> dict:
    require(accounting(pure) == BASELINE, "baseline response/history accounting mismatch")
    thirds = {name: accounting(pure.loc[pure.calendar_third.eq(name)], False) for name in THIRDS}
    require(thirds == THIRDS, "pure history by calendar-third mismatch")
    later = pure.loc[pure.week_no.ge(35)].copy()
    require(accounting(later) == LATER, "later response/history accounting mismatch")
    # Selection never recomputes counters; original earlier pure events remain counted.
    early = pure.loc[pure.week_no.le(34)].groupby("household_key").size()
    inherited = int((later.household_key.map(early).fillna(0) > 0).sum())
    require(inherited == 3244, "later responses lost weeks 1-34 history")
    share = pure.loc[pure.prior_pure.gt(0), "prior_private_share"]
    require(
        len(share) == 4959 and pure.prior_private_share.isna().eq(pure.prior_pure.eq(0)).all(),
        "nullable prior share support mismatch",
    )
    proportions = np.array(
        [share.eq(0).mean(), share.eq(1).mean(), share.between(0, 1, inclusive="neither").mean()]
    )
    require(
        np.allclose(proportions, [0.2890, 0.2948, 0.4162], atol=5.1e-5, rtol=0),
        "continuous share support mismatch",
    )
    return {
        "baseline": accounting(pure),
        "later": accounting(later),
        **thirds,
        "share_support": {
            "history_bearing_events": 4959,
            "zero_share": float(proportions[0]),
            "one_share": float(proportions[1]),
            "interior_share": float(proportions[2]),
        },
        "later_preservation": {"responses_with_early_pure_history": inherited},
    }


def cluster_structure(frame: pd.DataFrame) -> dict[str, int]:
    require(
        tuple(frame.columns) == DESIGN_COLUMNS, "cluster structure requires outcome-free projection"
    )
    sizes = frame.groupby(["household_key", "week_no"]).size()
    multi = sizes.loc[sizes.gt(1)]
    return {
        "households": int(frame.household_key.nunique()),
        "stores": int(frame.store_id.nunique()),
        "weeks": int(frame.week_no.nunique()),
        "household_store_pairs": len(frame[["household_key", "store_id"]].drop_duplicates()),
        "multi_store_households": int(
            frame.groupby("household_key").store_id.nunique().gt(1).sum()
        ),
        "household_week_cells": len(sizes),
        "multi_event_household_week_cells": len(multi),
        "events_in_multi_event_household_week_cells": int(multi.sum()),
    }


def build_payloads(root: Path = PROJECT_ROOT) -> dict[str, bytes]:
    from complete_journey.choice.events import LEDGER_COLUMNS, PURE_COLUMNS, build_events

    artifacts.assert_policy(root)
    before = artifacts.verify_inputs(root)
    identity = artifacts.source_identity(root)
    lines, products, baskets, accepted = load_inputs(root)
    ledger, pure, qa = build_events(lines, products, baskets)
    sort = ["household_key", "basket_id", "product_id"]
    cols = list(lines.columns)
    left = lines.loc[lines.quantity.gt(0), cols].sort_values(sort).reset_index(drop=True)
    right = accepted[cols].sort_values(sort).reset_index(drop=True)
    require(left.equals(right), "raw/accepted positive line routes disagree")
    qa["accepted_positive_line_route_agrees"] = 1
    independent_history(ledger)
    qa["independent_history_route_agrees"] = 1
    reconcile_samples(pure)
    require(qa == QA_EXPECTED, "event construction accepted anchors mismatch")
    require(
        artifacts.source_identity(root) == identity and artifacts.verify_inputs(root) == before,
        "sources moved during build",
    )
    payloads = {}
    tables = {}
    for name, table in (("ledger", ledger), ("pure", pure)):
        handle = io.BytesIO()
        table.to_parquet(handle, index=False, engine="pyarrow", compression="snappy")
        payloads[LOCAL_PATHS[name]] = handle.getvalue()
        import hashlib

        tables[name] = {
            "path": LOCAL_PATHS[name],
            "rows": len(table),
            "columns": list(table.columns),
            "sha256": hashlib.sha256(handle.getvalue()).hexdigest(),
        }
    contract = {"format_version": 1, "scope": SCOPE, "group": GROUP, "tables": tables}
    artifacts.validate_contract(contract, {"ledger": LEDGER_COLUMNS, "pure": PURE_COLUMNS})
    payloads[OUTPUTS["contract"]] = artifacts.canonical_json(contract)
    payloads[OUTPUTS["qa"]] = artifacts.qa_bytes(qa, QA_EXPECTED)
    return payloads


def build(root: Path = PROJECT_ROOT) -> None:
    payloads = build_payloads(root)
    # Reproduce accepted bytes without claiming current code generated historical evidence.
    artifacts.write_preparation(payloads, root)
    artifacts.validate_saved(root)


def support_assertions(report: dict, later: bool = False) -> None:
    n, k, c = (4807, 151, 148) if later else (6544, 169, 166)
    require(
        report["models"]["M0"]["rank"] == 5 and report["models"]["M0"]["columns"] == 5,
        "M0 rank mismatch",
    )
    main = report["models"]["M1"]
    require(
        (main["rows"], main["columns"], main["rank"]) == (n, k, k),
        "M1 matrix rank/dimension mismatch",
    )
    require(report["context_depth"]["rank"] == c, "context/depth rank mismatch")
    history = report["residualized_history"]
    require(history["rank"] == 3, "history block rank mismatch")
    expected_ss = {"H0": 295.195111, "HB": 743.878263, "HP": 781.993186} if later else SUPPORT
    singular = (34.238303, 20.756399, 14.764047) if later else SINGULAR
    remaining = (0.5218, 0.6440, 0.9025) if later else (0.4161, 0.6044, 0.9144)
    for h, share in zip(("H0", "HB", "HP"), remaining, strict=True):
        term = history["terms"][h]
        require(
            abs(term["residual_ss"] - expected_ss[h]) <= SUPPORT_ATOL,
            f"accepted {h} residual SS mismatch",
        )
        require(
            abs(term["remaining_centered_ss_share"] - share) <= REMAINING_ATOL,
            f"accepted {h} remaining SS mismatch",
        )
    require(
        np.allclose(history["singular_values"], singular, atol=SINGULAR_ATOL, rtol=0),
        "history singular values mismatch",
    )
    require(
        abs(history["condition_number"] - (2.3190 if later else 2.1401)) <= CONDITION_ATOL,
        "history condition mismatch",
    )
    require(
        report["independent_residualization"]["maximum_absolute_difference"] <= INDEPENDENT_ATOL,
        "independent QR/absorption mismatch",
    )


def _flatten_numeric(value: dict, prefix: str = "") -> dict[str, float | int]:
    result = {}
    for name, v in value.items():
        label = {"household_key": "households", "store_id": "stores", "week_no": "weeks"}.get(
            name, name
        )
        key = f"{prefix}.{label}" if prefix else label
        if type(v) is dict:
            result.update(_flatten_numeric(v, key))
        elif type(v) in (int, float):
            result[key] = v
        elif type(v) is bool:
            result[key] = int(v)
        elif type(v) is list:
            require(all(type(x) in (int, float) for x in v), "non-numeric preflight list")
            result.update({f"{key}.{i + 1}": x for i, x in enumerate(v)})
        elif type(v) is str:
            pass  # Closed convention labels are documented, not emitted as data.
        else:
            require(False, "unknown support report value type")
    return result


def preflight_payloads(root: Path = PROJECT_ROOT) -> dict[str, bytes]:
    from complete_journey.choice.covariance import run_synthetic_preflight
    from complete_journey.choice.design import support_report
    from complete_journey.choice.schema import allowed_summary_keys

    require(artifacts.validate_saved(root) is not None, "build preparation first")
    before = artifacts.source_identity(root)
    pure = pd.read_parquet(root / LOCAL_PATHS["pure"])
    samples = reconcile_samples(pure)
    projection = pure.loc[:, list(DESIGN_COLUMNS)].copy()
    reports = {
        "baseline": support_report(projection),
        "later": support_report(projection.loc[projection.week_no.ge(35)].copy()),
    }
    for name, report in reports.items():
        support_assertions(report, later=name == "later")
    clusters = cluster_structure(projection)
    require(clusters == CLUSTERS, "cluster/intersection anchor mismatch")
    synthetic = run_synthetic_preflight()
    rows = []

    def add(sample, domain, metrics):
        for metric, value in metrics.items():
            rows.append(
                {
                    "sample": sample,
                    "domain": domain,
                    "metric": metric,
                    "value": value,
                    "scope": SCOPE,
                }
            )

    for sample, counts in samples.items():
        add(sample, "accounting", counts)
    add("baseline", "clusters", clusters)
    for sample, report in reports.items():
        add(sample, "design", _flatten_numeric(report))
    add("synthetic", "implementation", _flatten_numeric(synthetic))
    require(artifacts.source_identity(root) == before, "source moved during preflight")
    return {OUTPUTS["summary"]: artifacts.summary_bytes(rows, allowed_summary_keys())}


def preflight(root: Path = PROJECT_ROOT) -> None:
    payloads = preflight_payloads(root)
    artifacts.write_preparation(payloads, root)
    artifacts.validate_saved(root)


def check(root: Path = PROJECT_ROOT) -> None:
    saved = artifacts.validate_saved(root)
    if saved is not None:
        from complete_journey.choice.events import LEDGER_COLUMNS, PURE_COLUMNS
        from complete_journey.choice.schema import validate_summary_csv

        ledger = pd.read_parquet(root / LOCAL_PATHS["ledger"])
        pure = pd.read_parquet(root / LOCAL_PATHS["pure"])
        require(
            tuple(ledger.columns) == LEDGER_COLUMNS and tuple(pure.columns) == PURE_COLUMNS,
            "local canonical schema drift",
        )
        independent_history(ledger)
        reconcile_samples(pure)
        require(
            (root / OUTPUTS["qa"]).read_bytes() == artifacts.qa_bytes(QA_EXPECTED, QA_EXPECTED),
            "QA schema/value drift",
        )
        if (root / OUTPUTS["summary"]).exists():
            validate_summary_csv((root / OUTPUTS["summary"]).read_bytes())
