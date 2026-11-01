"""Guarded factual basket events and strictly earlier-week exact-group histories.

Inputs are projections, not arbitrary analytical frames. No input/output files are
written; the committed SQL executes in a single-threaded, spill-disabled database.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from complete_journey.paths import PROJECT_ROOT

LINE_COLUMNS = (
    "household_key",
    "basket_id",
    "product_id",
    "store_id",
    "week_no",
    "day",
    "quantity",
)
PRODUCT_COLUMNS = ("product_id", "brand", "sub_commodity_desc")
BASKET_COLUMNS = ("household_key", "basket_id", "store_id", "week_no", "day")
EVENT_KEY = ("household_key", "basket_id")
LINE_KEY = (*EVENT_KEY, "product_id")
CONTEXT_COLUMNS = ("store_id", "week_no", "day")
LEDGER_COLUMNS = (
    "household_key",
    "basket_id",
    "week_no",
    "store_id",
    "national_lines",
    "private_lines",
    "composition",
    "prior_national_pure",
    "prior_private_pure",
    "prior_mixed",
    "prior_total_group",
    "prior_pure",
    "history_category",
    "log1p_prior_pure",
    "prior_private_share",
    "calendar_third",
    "prior_mixed_only",
)
PURE_COLUMNS = (*LEDGER_COLUMNS, "private_only")
# Ordered pandas dtype contracts, including empty outputs. Text storage is pinned
# to Python strings rather than inheriting a process-wide pandas storage option.
LEDGER_DTYPES = tuple(
    (
        column,
        "string"
        if column in ("composition", "history_category", "calendar_third")
        else "float64"
        if column in ("log1p_prior_pure", "prior_private_share")
        else "bool"
        if column == "prior_mixed_only"
        else "int64",
    )
    for column in LEDGER_COLUMNS
)
PURE_DTYPES = (*LEDGER_DTYPES, ("private_only", "int64"))
EVENT_SQL = PROJECT_ROOT / "sql" / "choice" / "01_purchased_composition_event.sql"


def _project(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> pd.DataFrame:
    if frame.columns.has_duplicates or set(frame.columns) != set(columns):
        raise ValueError(f"{name}: factual projection schema mismatch")
    result = frame.loc[:, list(columns)].copy()
    if result.isna().any().any():
        raise ValueError(f"{name}: null factual field")
    return result


def _integers(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    for column in columns:
        values = frame[column]
        if (
            not pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_bool_dtype(values)
            or not np.isfinite(values.to_numpy(dtype=float)).all()
            or not values.eq(np.floor(values)).all()
        ):
            raise ValueError(f"{name}: {column} must contain finite integers")
        frame[column] = values.astype("int64")
    if "week_no" in columns and not frame.week_no.between(1, 102).all():
        raise ValueError(f"{name}: week_no outside frozen calendar 1-102")


def _unique(frame: pd.DataFrame, key: tuple[str, ...], name: str) -> None:
    if frame.duplicated(list(key)).any():
        raise ValueError(f"{name}: duplicate key; many-to-one cardinality/fanout refused")


def _join(left: pd.DataFrame, right: pd.DataFrame, key: tuple[str, ...], name: str) -> pd.DataFrame:
    joined = left.merge(
        right,
        on=list(key),
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("", "_basket"),
    )
    if len(joined) != len(left):
        raise ValueError(f"{name}: join fanout/line count changed")
    if not joined._merge.eq("both").all():
        raise ValueError(f"{name}: unmatched factual keys")
    return joined.drop(columns="_merge")


def build_events(
    lines: pd.DataFrame, products: pd.DataFrame, baskets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Return complete positive-event ledger, pure responses, and scalar-only QA.

    Line keys are household/basket/product, product keys are product, and basket
    keys are household/basket. All joins preserve line count and require complete
    coverage. Context must agree on every recorded line, including nonpositive
    lines. Histories use all positive group events; neither purity nor a later
    response-calendar restriction is applied before constructing histories.
    QA's excluded_zero_only_events counts recorded baskets with no positive group
    lines, including any negative-only basket; source anomalies remain unchanged.
    """
    lines = _project(lines, LINE_COLUMNS, "lines")
    products = _project(products, PRODUCT_COLUMNS, "products")
    baskets = _project(baskets, BASKET_COLUMNS, "baskets")
    _integers(lines, LINE_COLUMNS[:-1], "lines")
    _integers(products, ("product_id",), "products")
    _integers(baskets, BASKET_COLUMNS, "baskets")
    if (
        not pd.api.types.is_numeric_dtype(lines.quantity)
        or pd.api.types.is_bool_dtype(lines.quantity)
        or not np.isfinite(lines.quantity.to_numpy(dtype=float)).all()
    ):
        raise ValueError("lines: quantity must be finite numeric")
    _unique(lines, LINE_KEY, "lines")
    _unique(products, ("product_id",), "products")
    _unique(baskets, EVENT_KEY, "baskets")

    joined = _join(lines, products, ("product_id",), "product")
    joined = _join(joined, baskets, EVENT_KEY, "basket")
    for column in CONTEXT_COLUMNS:
        if not joined[column].eq(joined[f"{column}_basket"]).all():
            raise ValueError(f"basket: incoherent {column} context")
    group = joined.loc[joined.sub_commodity_desc.eq("SPAGHETTI DRY")].copy()
    if not group.brand.isin(("National", "Private")).all():
        raise ValueError("group: unknown label type")
    positive = group.loc[group.quantity.gt(0)].copy()
    all_recorded_events = len(group.drop_duplicates(list(EVENT_KEY)))
    positive_events = len(positive.drop_duplicates(list(EVENT_KEY)))

    with duckdb.connect(":memory:", config={"threads": 1, "temp_directory": ""}) as con:
        con.register("positive_group_lines", positive)
        ledger = con.execute(EVENT_SQL.read_text()).df()
    if tuple(ledger.columns) != LEDGER_COLUMNS:
        raise ValueError("ledger: declared schema mismatch")
    ledger = ledger.astype(
        {
            column: pd.StringDtype(storage="python") if dtype == "string" else dtype
            for column, dtype in LEDGER_DTYPES
        }
    )
    _unique(ledger, EVENT_KEY, "ledger")
    if len(ledger) != positive_events:
        raise ValueError("ledger: event totality mismatch")
    if int(ledger.national_lines.sum() + ledger.private_lines.sum()) != len(positive):
        raise ValueError("ledger: positive line accounting mismatch")
    counter_columns = (
        "prior_national_pure",
        "prior_private_pure",
        "prior_mixed",
        "prior_total_group",
        "prior_pure",
    )
    if (ledger.loc[:, list(counter_columns)] < 0).any().any():
        raise ValueError("ledger: negative history count")
    if not ledger.prior_pure.eq(ledger.prior_national_pure + ledger.prior_private_pure).all():
        raise ValueError("ledger: pure history accounting mismatch")
    if not ledger.prior_total_group.eq(ledger.prior_pure + ledger.prior_mixed).all():
        raise ValueError("ledger: group history accounting mismatch")
    if (
        ledger.groupby(["household_key", "week_no"])[list(counter_columns)]
        .nunique()
        .gt(1)
        .any()
        .any()
    ):
        raise ValueError("ledger: same-week history mismatch")
    categories = pd.DataFrame(
        {
            "H0": ledger.prior_pure.eq(0),
            "HN": ledger.prior_pure.gt(0) & ledger.prior_private_pure.eq(0),
            "HB": ledger.prior_national_pure.gt(0) & ledger.prior_private_pure.gt(0),
            "HP": ledger.prior_pure.gt(0) & ledger.prior_national_pure.eq(0),
        }
    )
    if not categories.sum(axis=1).eq(1).all():
        raise ValueError("ledger: history categories not exhaustive/exclusive")
    for category in categories:
        if not ledger.loc[categories[category], "history_category"].eq(category).all():
            raise ValueError("ledger: history category definition mismatch")
    ledger["log1p_prior_pure"] = np.log1p(ledger.prior_pure.to_numpy(dtype=float))
    if not np.isfinite(ledger.log1p_prior_pure).all():
        raise ValueError("ledger: nonfinite history depth")
    if not ledger.prior_private_share.isna().eq(ledger.prior_pure.eq(0)).all():
        raise ValueError("ledger: undefined share semantics mismatch")
    has_history = ledger.prior_pure.gt(0)
    if not np.array_equal(
        ledger.loc[has_history, "prior_private_share"].to_numpy(),
        (
            ledger.loc[has_history, "prior_private_pure"] / ledger.loc[has_history, "prior_pure"]
        ).to_numpy(),
    ):
        raise ValueError("ledger: pure share accounting mismatch")

    pure = ledger.loc[ledger.composition.ne("MIXED")].copy().reset_index(drop=True)
    pure["private_only"] = pure.composition.eq("PRIVATE_ONLY").astype("int64")
    composition_counts = ledger.composition.value_counts()
    qa = {
        "positive_group_lines": len(positive),
        "all_recorded_group_events": all_recorded_events,
        "positive_group_events": positive_events,
        "excluded_zero_only_events": all_recorded_events - positive_events,
        "national_only": int(composition_counts.get("NATIONAL_ONLY", 0)),
        "private_only": int(composition_counts.get("PRIVATE_ONLY", 0)),
        "mixed": int(composition_counts.get("MIXED", 0)),
        "pure": len(pure),
        "single_product_events": int((ledger.national_lines + ledger.private_lines).eq(1).sum()),
        "same_type_multi_product_events": int(
            (pure.national_lines + pure.private_lines).gt(1).sum()
        ),
        "product_unmatched": 0,
        "basket_unmatched": 0,
        "join_fanout": len(joined) - len(lines),
        "duplicate_lines": 0,
    }
    if qa["pure"] + qa["mixed"] != positive_events:
        raise ValueError("ledger: pure/mixed accounting mismatch")
    return ledger, pure, qa
