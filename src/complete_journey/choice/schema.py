"""Closed aggregate preflight metric universe, independent of any empirical outcome."""

from __future__ import annotations

import csv
import io
import math

from complete_journey.choice.contracts import (
    BASELINE,
    LATER,
    SCOPE,
    SUMMARY_FIELDS,
    THIRDS,
    require,
)

MATRIX_SCALARS = (
    "rows",
    "columns",
    "rank",
    "rank_tolerance",
    "condition_number",
    "unit_column_condition_number",
)
TERM_SCALARS = (
    "residual_ss",
    "raw_centered_ss",
    "remaining_centered_ss_share",
    "mean",
    "variance_n",
    "minimum",
    "maximum",
    "nonzero_rows",
    "ss_partialling_other_history",
)
CONCENTRATION_SCALARS = (
    "observed_groups",
    "nonzero_support_groups",
    "largest_share",
    "effective_groups_inverse_hhi",
)
CLUSTER_METRICS = (
    "households",
    "stores",
    "weeks",
    "household_store_pairs",
    "multi_store_households",
    "household_week_cells",
    "multi_event_household_week_cells",
    "events_in_multi_event_household_week_cells",
)


def design_metric_keys(later: bool = False) -> set[str]:
    result = set()
    for prefix, width in (
        ("models.M0", 5),
        ("models.M1", 151 if later else 169),
        ("context_depth", 148 if later else 166),
        ("residualized_history", 3),
    ):
        result.update(f"{prefix}.{m}" for m in MATRIX_SCALARS)
        result.update(f"{prefix}.singular_values.{i + 1}" for i in range(width))
    for h in ("H0", "HB", "HP"):
        result.update(f"residualized_history.terms.{h}.{m}" for m in TERM_SCALARS)
    for domain, terms in (
        ("concentration", ("H0", "HB", "HP", "joint", "joint_subspace_leverage")),
        ("partial_history_concentration", ("H0", "HB", "HP")),
    ):
        for axis in ("households", "stores", "weeks"):
            for h in terms:
                prefix = f"{domain}.{axis}.{h}"
                result.update(f"{prefix}.{m}" for m in CONCENTRATION_SCALARS)
                result.update(
                    f"{prefix}.top_fraction_shares.{fraction}"
                    for fraction in ("0.01", "0.05", "0.10")
                )
    result.update(
        f"independent_residualization.{m}"
        for m in ("maximum_absolute_difference", "frobenius_difference", "agrees")
    )
    result.add("conventions.nonzero_tolerance")
    return result


def allowed_summary_keys() -> set[tuple[str, str, str]]:
    from complete_journey.choice.covariance import SYNTHETIC_METRICS

    result = set()
    for sample, metrics in {
        "baseline": BASELINE,
        "later": LATER,
        **THIRDS,
        "share_support": dict.fromkeys(
            ("history_bearing_events", "zero_share", "one_share", "interior_share")
        ),
        "later_preservation": {"responses_with_early_pure_history": None},
    }.items():
        result.update((sample, "accounting", m) for m in metrics)
    result.update(("baseline", "clusters", m) for m in CLUSTER_METRICS)
    for sample in ("baseline", "later"):
        result.update((sample, "design", m) for m in design_metric_keys(sample == "later"))
    result.update(("synthetic", "implementation", m) for m in SYNTHETIC_METRICS)
    return result


def validate_summary_csv(payload: bytes) -> None:
    reader = csv.DictReader(io.StringIO(payload.decode()))
    require(
        tuple(reader.fieldnames or ()) == SUMMARY_FIELDS,
        "preflight CSV header/identifier/result alias refused",
    )
    rows = list(reader)
    allowed = allowed_summary_keys()
    require(len(rows) == len(allowed), "preflight CSV row count")
    keys = []
    for row in rows:
        require(
            set(row) == set(SUMMARY_FIELDS) and row["scope"] == SCOPE, "preflight CSV schema/scope"
        )
        key = (row["sample"], row["domain"], row["metric"])
        keys.append(key)
        require(key in allowed, "unapproved preflight metric")
        try:
            value = float(row["value"])
        except (ValueError, TypeError):
            require(False, "preflight scalar numeric value required")
        require(math.isfinite(value), "nonfinite preflight value")
    require(
        len(set(keys)) == len(keys) and set(keys) == allowed,
        "preflight CSV metric universe/duplicates",
    )
