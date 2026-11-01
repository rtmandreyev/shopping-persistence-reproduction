"""Pure Run E presentation of accepted aggregate evidence; never estimate a model."""

from __future__ import annotations

import io
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from complete_journey.descriptive.contracts import FIGURE_FORBIDDEN_VOCABULARY
from complete_journey.econometrics import core_artifacts, sensitivity_artifacts
from complete_journey.econometrics.artifacts import (
    OUTPUTS,
    RUN_C_PATHS,
    RUN_D_PATHS,
    assert_policy,
    sha256,
    source_identity,
)
from complete_journey.econometrics.final_contract import (
    DESIGN_COVARIANCE,
    FINAL_PATHS,
    PAPER_COVARIANCE,
)
from complete_journey.econometrics.samples import SAMPLES, require
from complete_journey.paths import PROJECT_ROOT

OUTCOMES = ("retailer_purchase", "weekly_sales_value", "trips")
VARIANTS = ("baseline", "S1", "S2", "full_week_pair", "post_formation", "household", "dk4", "dk8")
CORE_KEYS = tuple(
    (outcome, model)
    for outcome in OUTCOMES
    for model in (("M0", "M1", "M2", "M3") if outcome != "trips" else ("M3",))
)
SENSITIVITY_KEYS = {(o, v) for o in OUTCOMES for v in VARIANTS} | {("weekly_sales_value", "log1p")}
CORE_TABLE_FIELDS = tuple(
    "outcome model coefficient paper_facing_se paper_facing_ci_low paper_facing_ci_high paper_facing_covariance original_two_way_se nobs households response_weeks household_fe week_fe coefficient_unit interpretation_note".split()
)
ROBUSTNESS_TABLE_FIELDS = tuple(
    "outcome robustness_domain variant coefficient se ci_low ci_high population response_weeks covariance transformation change_from_baseline interpretation_note".split()
)
UNITS = {
    "retailer_purchase": "probability points",
    "weekly_sales_value": "retailer currency units per household-week",
    "trips": "recorded baskets per household-week",
}
PRESENTATION_DOMAINS = (
    ("population", (("S0", "baseline"), ("S1", "S1"), ("S2", "S2"))),
    ("calendar", tuple((v, v) for v in ("baseline", "full_week_pair", "post_formation"))),
    (
        "covariance",
        (("two_way", "baseline"), ("household", "household"), ("dk4", "dk4"), ("dk8", "dk8")),
    ),
)
MIXED_INFERENCE_NOTE = (
    "Noncausal descriptive panel associations; adjustment follows the stated model. Native outcome units. Mixed uncertainty: "
    "DK8 is primary solely for baseline M3; M0–M2 retain original two-way intervals. "
    "Original two-way design benchmark retained; DK4 and household-only are sensitivities. "
    "DK8 remains assumption-dependent; Bartlett weights downweight included lags; "
    "dependence remains visible at lag 8 and finite T=101 limits asymptotic certainty."
)
DOMAIN_NOTES = {
    "population": "S0 is the full baseline; S1/S2 are predeclared activity-selected sensitivities. Descriptive sample dependence, not a preferred population. Original two-way uncertainty retained.",
    "calendar": "Predeclared calendar sensitivity; magnitudes are calendar-dependent. Original two-way uncertainty retained; the later window is not verified enrollment.",
    "covariance": "Same baseline M3 point and rows; DK8 is primary paper-facing uncertainty only here. Two-way is the retained design benchmark; DK4 and household-only are sensitivities. Dependence assumptions differ; DK8 remains assumption-dependent.",
    "functional_form": "Qualitative positive revenue-history comparison only; log1p changes units and estimand. Original two-way uncertainty retained; no comparable levels change.",
}


def guard_text(blocks):
    """Refuse identifiers, prohibited claims and literal zero-probability prose."""
    refused = FIGURE_FORBIDDEN_VOCABULARY | {
        "household_id",
        "customer_id",
        "customer_key",
        "state dependence",
        "state-dependence",
        "habit formation",
        "loyalty effect",
        "behavioral persistence parameter",
        "campaign/promotion effect",
        "abm transition parameter",
        "elasticity",
        "percent effect",
    }
    for text in blocks:
        require(type(text) is str, "paper-facing text must be a string")
        lowered = text.lower()
        require(
            not any(term in lowered for term in refused),
            "paper-facing vocabulary/identifier refusal",
        )
        require(re.search(r"\bcausal\b", lowered) is None, "paper-facing causal vocabulary refusal")
        require(
            re.search(r"\bp(?:[ _-]values?)?\s*[:=]\s*0(?:\.0*)?(?![\d.eE])", lowered) is None,
            "literal zero p-value claim refused",
        )


def _schema(rows, fields, name):
    require(
        all(type(r) is dict and set(r) == set(fields) for r in rows),
        f"{name} exact schema/identifier mismatch",
    )
    require(
        all(type(v) is str for r in rows for v in r.values()),
        f"{name} must preserve accepted CSV strings",
    )


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _scalar_inference(row):
    fields = ("coefficient", "standard_error", "ci_lower", "ci_upper")
    require(all(_finite(row[k]) for k in fields), "nonfinite accepted scalar inference")
    require(
        float(row["standard_error"]) > 0
        and float(row["ci_lower"]) <= float(row["coefficient"]) <= float(row["ci_upper"]),
        "invalid accepted scalar interval",
    )


def _same_values(left, right, fields, message):
    # Only equality verification; never derive or substitute any published value.
    require(all(left[k] == right[k] for k in fields), message)


def _validated_evidence(core_rows, sensitivity_rows):
    """Closed evidence universe and invariant checks without recomputing inference."""
    _schema(core_rows, core_artifacts.RESULT_FIELDS, "core")
    _schema(sensitivity_rows, sensitivity_artifacts.MATRIX_FIELDS, "sensitivity")
    require(
        len(core_rows) == len(CORE_KEYS)
        and {(r["outcome"], r["model_id"]) for r in core_rows} == set(CORE_KEYS),
        "wrong or duplicate core-result universe",
    )
    require(
        len(sensitivity_rows) == len(SENSITIVITY_KEYS)
        and {(r["outcome"], r["variant"]) for r in sensitivity_rows} == SENSITIVITY_KEYS,
        "wrong or duplicate sensitivity universe",
    )
    core = {(r["outcome"], r["model_id"]): r for r in core_rows}
    sensitivity = {(r["outcome"], r["variant"]): r for r in sensitivity_rows}
    require(
        len({r["row_key_sha256"] for r in core_rows}) == 1
        and len({r["source_state_sha256"] for r in core_rows}) == 1,
        "core row/source identity invariance failed",
    )
    for row in core_rows:
        model = row["model_id"]
        spec = SAMPLES[0]
        require(
            row["sample"] == spec.name and row["status"] == "UNREVIEWED",
            "unauthorized historical core sample/status",
        )
        for field, expected in (
            ("nobs", spec.cells),
            ("households", spec.households),
            ("response_weeks", spec.last_response - spec.first_response + 1),
            ("household_clusters", spec.households),
            ("week_clusters", spec.last_response - spec.first_response + 1),
        ):
            require(row[field] == str(expected), f"core specification {field} mismatch")
        require(
            row["household_fe"] == str(model in ("M2", "M3"))
            and row["week_fe"] == str(model in ("M1", "M3")),
            "core FE specification mismatch",
        )
        require(
            row["covariance_type"] == DESIGN_COVARIANCE,
            "core original two-way covariance relabeled",
        )
        require(
            re.fullmatch(r"[0-9a-f]{64}", row["row_key_sha256"]) is not None
            and re.fullmatch(r"[0-9a-f]{64}", row["source_state_sha256"]) is not None,
            "core invalid identity hash",
        )
        _scalar_inference(row)
    scalar_fields = ("coefficient", "standard_error", "ci_lower", "ci_upper")
    for outcome in OUTCOMES:
        baseline = sensitivity[outcome, "baseline"]
        original = core[outcome, "M3"]
        _same_values(
            baseline,
            original,
            (
                *scalar_fields,
                "row_key_sha256",
                "nobs",
                "households",
                "response_weeks",
                "household_fe",
                "week_fe",
            ),
            "Run D baseline differs from accepted Run C",
        )
    for row in sensitivity_rows:
        variant, outcome = row["variant"], row["outcome"]
        baseline = sensitivity[outcome, "baseline"]
        spec = SAMPLES[{"full_week_pair": 1, "post_formation": 2, "S1": 3, "S2": 4}.get(variant, 0)]
        covariance = variant if variant in ("household", "dk4", "dk8") else "two_way"
        domain = (
            "baseline"
            if variant == "baseline"
            else "population"
            if variant in ("S1", "S2")
            else "calendar"
            if variant in ("full_week_pair", "post_formation")
            else "functional_form"
            if variant == "log1p"
            else "covariance"
        )
        expected_text = {
            "sample": spec.name,
            "population": spec.population,
            "response_first": str(spec.first_response),
            "response_last": str(spec.last_response),
            "response_weeks": str(spec.last_response - spec.first_response + 1),
            "households": str(spec.households),
            "nobs": str(spec.cells),
            "covariance": covariance,
            "check_type": domain,
            "transformation": "log(1 + retailer SALES_VALUE)" if variant == "log1p" else "levels",
            "model_id": "M4" if variant == "log1p" else "M3",
            "household_fe": "True",
            "week_fe": "True",
            "status": "UNREVIEWED",
            "kernel": "bartlett" if variant in ("dk4", "dk8") else "",
            "bandwidth": variant[2:] if variant in ("dk4", "dk8") else "",
        }
        require(
            all(row[k] == value for k, value in expected_text.items()),
            "sensitivity specification mismatch",
        )
        _scalar_inference(row)
        require(
            re.fullmatch(r"[0-9a-f]{64}", row["row_key_sha256"]) is not None,
            "sensitivity invalid row identity",
        )
        for field in scalar_fields:
            require(
                row["reference_baseline_" + field] == baseline[field],
                "sensitivity baseline comparison drift",
            )
        if spec is SAMPLES[0]:
            require(
                row["row_key_sha256"] == baseline["row_key_sha256"],
                "baseline row identity invariance failed",
            )
        if variant in ("baseline", "household", "dk4", "dk8"):
            require(
                row["coefficient"] == baseline["coefficient"]
                and row["signed_coefficient_change"] == "0",
                "covariance-only point estimate/change invariance failed",
            )
        elif variant == "log1p":
            require(row["signed_coefficient_change"] == "", "log1p change must be blank")
        else:
            require(
                _finite(row["signed_coefficient_change"]),
                "missing accepted signed coefficient change",
            )
        if covariance == "two_way":
            require(
                row["covariance_type"] == DESIGN_COVARIANCE,
                "sensitivity original two-way covariance relabeled",
            )
    # Same declared population/calendar job must carry the same row identity across outcomes.
    for variant in VARIANTS:
        require(
            len({sensitivity[o, variant]["row_key_sha256"] for o in OUTCOMES}) == 1,
            "sensitivity cross-outcome row identity drift",
        )
    return core, sensitivity


def _response_range(row):
    return f"{row['response_first']}–{row['response_last']}"


def _guard_table(rows, fields, expected_count):
    _schema(rows, fields, "final table")
    require(len(rows) == expected_count, "final table row count mismatch")
    guard_text([" ".join(fields), *(value for row in rows for value in row.values())])


def build_core_table(core_rows, sensitivity_rows):
    """T4: nine copied points; primary DK8 only for accepted baseline M3."""
    core, sensitivity = _validated_evidence(core_rows, sensitivity_rows)
    rows = []
    for outcome, model in CORE_KEYS:
        original = core[outcome, model]
        inference = sensitivity[outcome, "dk8"] if model == "M3" else original
        rows.append(
            {
                "outcome": outcome,
                "model": model,
                "coefficient": original["coefficient"],
                "paper_facing_se": inference["standard_error"],
                "paper_facing_ci_low": inference["ci_lower"],
                "paper_facing_ci_high": inference["ci_upper"],
                "paper_facing_covariance": PAPER_COVARIANCE
                if model == "M3"
                else original["covariance_type"],
                "original_two_way_se": original["standard_error"],
                "nobs": original["nobs"],
                "households": original["households"],
                "response_weeks": _response_range(sensitivity[outcome, "baseline"]),
                "household_fe": original["household_fe"],
                "week_fe": original["week_fe"],
                "coefficient_unit": UNITS[outcome],
                "interpretation_note": MIXED_INFERENCE_NOTE,
            }
        )
    _guard_table(rows, CORE_TABLE_FIELDS, 9)
    return rows


def build_robustness_table(core_rows, sensitivity_rows):
    """T5: duplicate shared baselines by domain; copy signed changes, not ratios."""
    _, sensitivity = _validated_evidence(core_rows, sensitivity_rows)
    selections = [
        (o, domain, display, source)
        for domain, variants in PRESENTATION_DOMAINS
        for o in OUTCOMES
        for display, source in variants
    ]
    selections.append(("weekly_sales_value", "functional_form", "log1p", "log1p"))
    rows = []
    for outcome, domain, display, source in selections:
        accepted = sensitivity[outcome, source]
        rows.append(
            {
                "outcome": outcome,
                "robustness_domain": domain,
                "variant": display,
                "coefficient": accepted["coefficient"],
                "se": accepted["standard_error"],
                "ci_low": accepted["ci_lower"],
                "ci_high": accepted["ci_upper"],
                "population": accepted["population"],
                "response_weeks": _response_range(accepted),
                "covariance": accepted["covariance"],
                "transformation": accepted["transformation"],
                "change_from_baseline": accepted["signed_coefficient_change"],
                "interpretation_note": "Noncausal adjusted descriptive associations. "
                + DOMAIN_NOTES[domain],
            }
        )
    _guard_table(rows, ROBUSTNESS_TABLE_FIELDS, 31)
    require(
        len({(r["outcome"], r["robustness_domain"], r["variant"]) for r in rows}) == 31,
        "duplicate final robustness presentation key",
    )
    return rows


def _figure(core_rows, sensitivity_rows):
    """Build the two-panel figure in memory; all intervals are source endpoints."""
    core, sensitivity = _validated_evidence(core_rows, sensitivity_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.30, top=0.77, wspace=0.38)
    dark, light, original_color = "#173f5f", "#91c5dc", "#778899"
    labels = [
        "M0 pooled",
        "M1 week FE",
        "M2 household FE",
        "M3 TWFE baseline",
        "M3 TWFE post-formation",
    ]
    for ax, outcome, title in zip(
        axes, OUTCOMES[:2], ("A. Retailer participation", "B. Retailer SALES_VALUE"), strict=True
    ):
        for position, model in enumerate(("M0", "M1", "M2", "M3", "post_formation")):
            point = (
                sensitivity[outcome, model] if model == "post_formation" else core[outcome, model]
            )
            primary = sensitivity[outcome, "dk8"] if model == "M3" else point
            ax.hlines(
                position,
                float(primary["ci_lower"]),
                float(primary["ci_upper"]),
                color=dark if model == "M3" else original_color,
                linewidth=2.7,
                zorder=2,
            )
            if model == "M3":
                ax.hlines(
                    position,
                    float(point["ci_lower"]),
                    float(point["ci_upper"]),
                    color=light,
                    linewidth=6,
                    zorder=3,
                )
            # Exactly one point on the baseline M3 row: nested interval, not duplicate estimate.
            ax.plot(float(point["coefficient"]), position, "o", color=dark, markersize=6, zorder=4)
        ax.set_yticks(range(5), labels)
        ax.invert_yaxis()
        ax.set_ylim(4.6, -0.6)
        ax.axvline(0, color="#b0b0b0", linewidth=0.8, zorder=0)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel(UNITS[outcome].capitalize(), fontsize=10)
        ax.grid(axis="x", alpha=0.18)
        ax.tick_params(axis="y", labelsize=10)
    fig.suptitle(
        "Panel adjustment and calendar sensitivity\nNoncausal descriptive panel associations",
        fontsize=14,
        y=0.98,
    )
    fig.legend(
        handles=[
            Line2D(
                [0], [0], color=dark, linewidth=2.7, label="Baseline M3: primary DK8 95% interval"
            ),
            Line2D(
                [0],
                [0],
                color=light,
                linewidth=6,
                label="Baseline M3: nested two-way design benchmark",
            ),
            Line2D(
                [0],
                [0],
                color=original_color,
                linewidth=2.7,
                label="M0–M2 and post-formation M3: original two-way 95% intervals",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        frameon=False,
        fontsize=10,
    )
    caption = (
        "M0–M3 add week and household adjustment on the same S0 response weeks 2–102.\n"
        "Post-formation M3 uses S0 response weeks 21–101 and its already-estimated two-way interval; calendar magnitudes differ.\n"
        "Baseline M3 has one point: dark primary DK8 (Bartlett, bandwidth 8), with a light nested two-way design benchmark.\n"
        "M0–M2 retain original two-way uncertainty. DK8 is assumption-dependent; all intervals are native 95% intervals.\n"
        "These are descriptive within-panel associations, not an identified behavioral mechanism."
    )
    fig.text(0.035, 0.185, caption, ha="left", va="top", fontsize=10, linespacing=1.65)
    return fig


def figure_text_blocks(fig):
    """Scan all title locations, labels, ticks, legends, annotations and captions."""
    blocks = [text.get_text() for text in fig.texts]
    legends = list(fig.legends)
    for ax in fig.axes:
        blocks.extend(ax.get_title(loc=loc) for loc in ("left", "center", "right"))
        blocks.extend((ax.get_xlabel(), ax.get_ylabel()))
        blocks.extend(
            text.get_text() for text in (*ax.texts, *ax.get_xticklabels(), *ax.get_yticklabels())
        )
        blocks.extend(
            (ax.xaxis.get_offset_text().get_text(), ax.yaxis.get_offset_text().get_text())
        )
        if ax.get_legend() is not None:
            legends.append(ax.get_legend())
    for legend in legends:
        blocks.append(legend.get_title().get_text())
        blocks.extend(text.get_text() for text in legend.get_texts())
    return blocks


def figure_bytes(core_rows, sensitivity_rows):
    fig = _figure(core_rows, sensitivity_rows)
    try:
        fig.canvas.draw()
        guard_text(figure_text_blocks(fig))
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=160,
            metadata={"Software": "complete-journey Run E accepted-evidence publisher"},
        )
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _snapshot(root):
    # Aggregate files only. Raw/Parquet verification belongs to the outer bounded CLI.
    paths = (*OUTPUTS.values(), *RUN_C_PATHS, *RUN_D_PATHS)
    require(all((root / p).is_file() for p in paths), "accepted aggregate archive missing: STOP")
    return {
        "source": source_identity(root),
        "aggregate_hashes": {p: sha256(root / p) for p in paths},
    }


def _payloads(core_rows, sensitivity_rows):
    core = build_core_table(core_rows, sensitivity_rows)
    robustness = build_robustness_table(core_rows, sensitivity_rows)
    # Recheck at the serialization boundary even if a builder is substituted by a caller.
    _guard_table(core, CORE_TABLE_FIELDS, 9)
    _guard_table(robustness, ROBUSTNESS_TABLE_FIELDS, 31)
    return (
        figure_bytes(core_rows, sensitivity_rows),
        core_artifacts.csv_bytes(core, CORE_TABLE_FIELDS),
        core_artifacts.csv_bytes(robustness, ROBUSTNESS_TABLE_FIELDS),
    )


def _prepare(root):
    assert_policy(root)
    before = _snapshot(root)
    core = core_artifacts.validate_saved(root)
    require(core is not None, "accepted Run C core missing: STOP; no fitting fallback")
    sensitivity = sensitivity_artifacts.validate_existing(root)
    require(
        sensitivity is not None, "accepted Run D sensitivity missing: STOP; no fitting fallback"
    )
    first = _payloads(core, sensitivity)
    require(first == _payloads(core, sensitivity), "nondeterministic final build/render: STOP")
    require(
        before == _snapshot(root),
        "source or accepted aggregate archive changed during final render: STOP",
    )
    return first, before


def publish_final(root=PROJECT_ROOT):
    """Write exactly F10/T4/T5 after all guards and both deterministic builds pass."""
    present = [(root / p).is_file() for p in FINAL_PATHS]
    require(not any(present) or all(present), "partial final publication: STOP")
    # An explicit publication invocation rebuilds from current guarded source.
    # validate_final remains read-only and refuses stale/tampered output bytes.
    payloads, snapshot = _prepare(root)
    require(
        all(not (root / relative).is_symlink() for relative in FINAL_PATHS),
        "final path must not be a symlink",
    )
    for relative, payload in zip(FINAL_PATHS, payloads, strict=True):
        path = root / relative
        # All payloads and guards were completed before this first write.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    for relative, payload in zip(FINAL_PATHS, payloads, strict=True):
        require((root / relative).read_bytes() == payload, "published final bytes differ")
    require(
        snapshot == _snapshot(root),
        "source or accepted archive changed during final publication: STOP",
    )
    assert_policy(root)
    return {relative: sha256(root / relative) for relative in FINAL_PATHS}


def validate_final(root=PROJECT_ROOT):
    """Rebuild accepted-evidence presentation and compare bytes; never refit or rewrite."""
    present = [(root / p).is_file() for p in FINAL_PATHS]
    require(all(present), "final artifacts missing or partial: STOP")
    payloads, snapshot = _prepare(root)
    for relative, expected in zip(FINAL_PATHS, payloads, strict=True):
        require(
            (root / relative).read_bytes() == expected, f"final artifact bytes differ: {relative}"
        )
    require(
        snapshot == _snapshot(root),
        "source or accepted archive changed during final validation: STOP",
    )
    return {relative: sha256(root / relative) for relative in FINAL_PATHS}
