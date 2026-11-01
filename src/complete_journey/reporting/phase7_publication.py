"""Phase 7 Run D: paper-facing F12 / T8 / T9 from the frozen Run C archive only.

This module performs no estimation. It never imports a model, fit or bootstrap
routine, never reads household-level files, and never touches the consumed test
period. Its inputs:
- the Run C aggregate archive, pinned by hash and cross-checked against the Run C
  provenance;
- the design snapshot and the consumed test ledger;
- the accepted Phase 4 final tables (historical comparison only);
- the researcher-approved result register (every candidate must be PROMOTED).

Final artifacts live in ``docs/phases/PHASE_07_FINAL/``, a closed, explicitly
allowlisted set. They are not under ``outputs/``, because ``.gitignore`` is
hash-pinned by the accepted Phase 4/5 integrity guards.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

from complete_journey.paths import PROJECT_ROOT

RUN_C = "docs/phases/PHASE_07_RUN_C"
ACCEPTED_INPUTS = {
    f"{RUN_C}/benchmark_results.json": "21c1713687711d8b8c778090560cf318490161cab80b5aba0c2c7acfc9ed1877",
    f"{RUN_C}/bootstrap_summary.json": "45c729725fce2efe713410bc9285ed473714727b7a66a1695f02562551ce3816",
    f"{RUN_C}/design_snapshot.json": "e3f443fc147c57f50ea28ea255123895b5c6ac67ae07c78dcd5e89d136b33b8d",
    f"{RUN_C}/design_snapshot.sha256": "eca4993f747021b02819da3fa24f488463702bdd7ca1613463db060a12009777",
    f"{RUN_C}/moment_register_real.csv": "e9ac415fe77b9aac658755c56f1a242ab6e9e0de234198a47e32f30bda1b0590",
    f"{RUN_C}/run_c_provenance.json": "27fc7baff01947ea060ed6ba89a718b733b959d7cd248d3bb6c2eeac65e97078",
    f"{RUN_C}/test_ledger.json": "3d69b40b3745d5aebcd8277c82ee36f57946a4df1c49c40475786b9269bae40e",
    "outputs/tables/panel/final/t4_core_panel_results.csv": "8f9c7bebc93591f3fc8e017c772178ffaf008291f9f9f1390947599ac60b61b1",
    "outputs/tables/panel/final/t5_panel_robustness.csv": "856cf3b3af7abf42f7d27812cd86c331a0334c25b9e4d28fbe06f43a84a46936",
}
REGISTER = "docs/phases/PHASE_07_RESULT_REGISTER.csv"
PROMOTED_IDS = ("P7-RC-01", "P7-RC-02", "P7-RC-03", "P7-RC-04")
FINAL_DIR = "docs/phases/PHASE_07_FINAL"
FINAL_PATHS = {
    "F12": f"{FINAL_DIR}/f12_predictive_information_ladder.png",
    "T8": f"{FINAL_DIR}/t8_dynamic_targets_and_operator_comparison.csv",
    "T9": f"{FINAL_DIR}/t9_predictive_benchmark.csv",
}
PROVENANCE = f"{FINAL_DIR}/phase7_publication_provenance.json"
ALLOWLIST = frozenset({*FINAL_PATHS.values(), PROVENANCE})

OP_O, OP_S, OP_G = "O_interior_window", "S_fixed_20_101", "G_full_grid"
T8_FIELDS = (
    "moment_id",
    "component",
    "estimate",
    "unit",
    "operator",
    "uncertainty_low",
    "uncertainty_high",
    "uncertainty_method",
    "phase8_role",
    "interpretation_note",
)
T9_FIELDS = (
    "information_tier",
    "model",
    "selected_hyperparameter_summary",
    "test_log_loss",
    "test_brier",
    "test_auc",
    "calibration_intercept",
    "calibration_slope",
    "test_rows",
    "test_prevalence",
    "role",
)
#: Affirmative claims refused in rendered artifact text (negated boundary wording
#: is written explicitly and checked separately).
BANNED = (
    "household_key",
    "proves",
    "optimal",
    "bayes",
    "winner",
    "structural state dependence is",
    "habit formation",
    "loyalty effect",
    "caused by observation",
)


class PublicationError(ValueError):
    """The frozen archive or the approved statuses do not support publication."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_inputs(root: Path = PROJECT_ROOT) -> dict:
    """Verify every frozen input byte-for-byte and the archive's internal consistency."""
    raw = {}
    for relative, digest in ACCEPTED_INPUTS.items():
        path = root / relative
        _require(path.is_file(), f"frozen input missing: {relative}")
        raw[relative] = path.read_bytes()
        _require(_sha(raw[relative]) == digest, f"frozen input drifted: {relative}")
    provenance = json.loads(raw[f"{RUN_C}/run_c_provenance.json"])
    for relative, digest in provenance["archive_sha256"].items():
        _require(_sha(raw[relative]) == digest, f"Run C archive inconsistent: {relative}")
    ledger = json.loads(raw[f"{RUN_C}/test_ledger.json"])
    _require(ledger["status"] == "TEST_CONSUMED", "Run C test ledger must be consumed")
    snapshot = _sha(raw[f"{RUN_C}/design_snapshot.json"])
    _require(
        ledger["design_snapshot_sha256"] == snapshot == provenance["design_snapshot_sha256"],
        "design snapshot hash disagreement",
    )
    with (root / REGISTER).open(newline="") as handle:
        register = {r["result_id"]: r for r in csv.DictReader(handle)}
    _require(tuple(sorted(register)) == PROMOTED_IDS, "Phase 7 register universe")
    _require(
        all(register[i]["status"] == "PROMOTED" for i in PROMOTED_IDS),
        "publication requires the researcher-approved PROMOTED statuses",
    )
    moments = list(csv.DictReader(io.StringIO(raw[f"{RUN_C}/moment_register_real.csv"].decode())))
    return {
        "benchmark": json.loads(raw[f"{RUN_C}/benchmark_results.json"]),
        "moments": {(r["operator"], r["moment_id"], r["component"]): r for r in moments},
        "t4": list(
            csv.DictReader(
                io.StringIO(raw["outputs/tables/panel/final/t4_core_panel_results.csv"].decode())
            )
        ),
        "t5": list(
            csv.DictReader(
                io.StringIO(raw["outputs/tables/panel/final/t5_panel_robustness.csv"].decode())
            )
        ),
        "ledger": ledger,
        "snapshot_sha256": snapshot,
    }


def _fmt(value) -> str:
    return "" if value in (None, "") else repr(float(value))


def _moment_row(m, operator, moment_id, component, unit, role, note) -> dict:
    r = m[(operator, moment_id, component)]
    return {
        "moment_id": moment_id,
        "component": component,
        "estimate": _fmt(r["estimate"]),
        "unit": unit,
        "operator": operator,
        "uncertainty_low": _fmt(r["uncertainty_low"]),
        "uncertainty_high": _fmt(r["uncertainty_high"]),
        "uncertainty_method": "household bootstrap B=500; percentile 95%",
        "phase8_role": role,
        "interpretation_note": note,
    }


def t8_rows(inputs: dict) -> list[dict]:
    m = inputs["moments"]
    adj = "Adjusted descriptive association under the stated operator; not a structural or causal parameter"
    rows = [
        _moment_row(
            m,
            OP_O,
            "CI-1",
            c,
            "share of interior weeks",
            "calibration (order-invariant)",
            "Household active-rate distribution; order-invariant by construction",
        )
        for c in ("mean", "q10", "q25", "q50", "q75", "q90")
    ]
    rows += [
        _moment_row(
            m, OP_O, "VI-1", "P11", "probability", "validation (primary)", "P(buy_t | buy_t-1 = 1)"
        ),
        _moment_row(
            m, OP_O, "VI-1", "P01", "probability", "validation (primary)", "P(buy_t | buy_t-1 = 0)"
        ),
        _moment_row(
            m,
            OP_O,
            "VI-1",
            "GAP",
            "probability points",
            "validation (primary)",
            "Raw transition gap; mixes heterogeneity and dynamics; equals O-M0",
        ),
    ]
    for c in ("O-M0", "O-M1", "O-M2", "O-M3"):
        rows.append(
            _moment_row(m, OP_O, "VI-2", c, "probability points", "validation (primary)", adj)
        )
    for k in (1, 2, 4, 8):
        role = (
            "validation (primary; lags 1-2 may become anchors under D1-(b))"
            if k <= 2
            else "validation (primary)"
        )
        rows.append(
            _moment_row(
                m,
                OP_O,
                "VI-3",
                f"beta_{k}",
                "probability points",
                role,
                f"Within-household TWFE association at lag {k} (common lag-8 sample)",
            )
        )
    rows.append(
        _moment_row(
            m,
            OP_O,
            "VI-3",
            "gamma_2",
            "probability points",
            "validation (primary)",
            "Lag-2 association conditional on lag 1; descriptive, not proof of a latent state",
        )
    )
    for state in ("active", "inactive"):
        for b in ("1", "2", "3-4", "5+"):
            rows.append(
                _moment_row(
                    m,
                    OP_O,
                    "VI-4",
                    f"{state}_{b}",
                    "share of interior spells",
                    "validation (primary)",
                    "Interior spells only; window-boundary spells excluded",
                )
            )
    rows.append(
        _moment_row(
            m,
            OP_O,
            "VI-5",
            "spearman",
            "rank correlation",
            "validation (secondary)",
            "Half-to-half stationarity check",
        )
    )
    rows.append(
        _moment_row(
            m,
            OP_O,
            "VI-6",
            "delta_log_loss_recency",
            "nats",
            "validation (secondary)",
            "LL(Tier0) - LL(Tier1); predictive information, not behavioural memory",
        )
    )
    rows.append(
        _moment_row(
            m,
            OP_O,
            "VI-6",
            "delta_brier_recency",
            "Brier",
            "validation (secondary)",
            "Brier(Tier0) - Brier(Tier1)",
        )
    )
    s0 = next(r for r in inputs["t4"] if r["outcome"] == "retailer_purchase" and r["model"] == "M3")
    rc05 = next(
        r
        for r in inputs["t5"]
        if r["outcome"] == "retailer_purchase" and r["variant"] == "post_formation"
    )
    comparison = [
        {
            "moment_id": "OPCOMP",
            "component": "M3_historical_phase4_S0",
            "estimate": _fmt(s0["coefficient"]),
            "unit": "probability points",
            "operator": "accepted Phase 4 S0 (responses 2-102)",
            "uncertainty_low": _fmt(s0["paper_facing_ci_low"]),
            "uncertainty_high": _fmt(s0["paper_facing_ci_high"]),
            "uncertainty_method": "accepted Phase 4 DK8 (P4-003)",
            "phase8_role": "diagnostic (historical; unchanged)",
            "interpretation_note": "Accepted Phase 4 result; not replaced",
        },
        _moment_row(
            m,
            OP_G,
            "VI-2",
            "O-M3",
            "probability points",
            "diagnostic (full grid)",
            "Full grid, responses 2-101; reproduces the accepted S0 ladder",
        ),
        _moment_row(
            m,
            OP_O,
            "VI-2",
            "O-M3",
            "probability points",
            "validation (primary)",
            "Symmetric interior purchase-window operator",
        ),
        _moment_row(
            m,
            OP_S,
            "VI-2",
            "O-M3",
            "probability points",
            "sensitivity",
            "Responses 20-101, full-grid lags",
        ),
        {
            "moment_id": "OPCOMP",
            "component": "M3_historical_RC05_post_formation",
            "estimate": _fmt(rc05["coefficient"]),
            "unit": "probability points",
            "operator": "accepted Phase 4 RC-05 (responses 21-101)",
            "uncertainty_low": _fmt(rc05["ci_low"]),
            "uncertainty_high": _fmt(rc05["ci_high"]),
            "uncertainty_method": "accepted Phase 4 two-way covariance",
            "phase8_role": "diagnostic (historical; unchanged)",
            "interpretation_note": "Historical convention differs from S by one response week",
        },
    ]
    for row in comparison[1:4]:
        row["moment_id"] = "OPCOMP"
        row["component"] = f"M3_{row['operator']}"
    rows += comparison
    for r in rows:
        _require(tuple(r) == T8_FIELDS, "T8 closed schema")
    return rows


TIER_ROWS = (
    ("B0|base_rate", "none (base rate)", "B0 base rate", "reference"),
    ("B1|tier0", "Tier 0: order-free own history", "B1 L2 logistic", "stable-propensity benchmark"),
    ("B1|tier1", "Tier 1: Tier 0 + recency", "B1 L2 logistic", "recency benchmark"),
    (
        "B1|tier2",
        "Tier 2: full point-in-time features",
        "B1 L2 logistic",
        "bounded full-information reference",
    ),
    (
        "B2|tier2",
        "Tier 2: full point-in-time features",
        "B2 histogram gradient boosting",
        "single nonlinear check",
    ),
)


def _params(params: dict) -> str:
    return "; ".join(f"{k}={v}" for k, v in sorted(params.items())) or "none"


def t9_rows(inputs: dict) -> list[dict]:
    b = inputs["benchmark"]
    rows = []
    for key, tier, model, role in TIER_ROWS:
        test = b["test"][key]
        cal = test["calibration"]
        selected = _params(b["selected"][key])
        if key.startswith("B1|tier") and key != "B1|tier2":
            selected += " (fixed, no tuning)"
        elif key in ("B1|tier2", "B2|tier2"):
            selected += " (validation log loss)"
        rows.append(
            {
                "information_tier": tier,
                "model": model,
                "selected_hyperparameter_summary": selected,
                "test_log_loss": _fmt(test["log_loss"]),
                "test_brier": _fmt(test["brier"]),
                "test_auc": _fmt(test["roc_auc"]) if key != "B0|base_rate" else "",
                "calibration_intercept": _fmt(cal["intercept"]),
                "calibration_slope": _fmt(cal["slope"]),
                "test_rows": str(test["n"]),
                "test_prevalence": _fmt(test["prevalence"]),
                "role": role,
            }
        )
    return rows


def _csv(rows, fields) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


def _check_text(texts) -> None:
    """Whole-word match, so e.g. "improves" never trips "proves"."""
    joined = " ".join(texts).lower()
    hits = [w for w in BANNED if re.search(rf"\b{re.escape(w)}\b", joined)]
    _require(not hits, f"refused wording or identifier in artifact: {hits}")


def f12_bytes(inputs: dict) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    b = inputs["benchmark"]["test"]
    gain = inputs["moments"][(OP_O, "VI-6", "delta_log_loss_recency")]
    steps = [
        ("No information\n(base rate)", b["B0|base_rate"]["log_loss"]),
        ("Tier 0\nown history\n(count, tenure)", b["B1|tier0"]["log_loss"]),
        ("Tier 1\n+ recency", b["B1|tier1"]["log_loss"]),
        ("Tier 2\nfull point-in-time\n(logistic)", b["B1|tier2"]["log_loss"]),
        ("Tier 2\nnonlinear check\n(gradient boosting)", b["B2|tier2"]["log_loss"]),
    ]
    fig, ax = plt.subplots(figsize=(11, 7.2))
    x = list(range(len(steps)))
    y = [v for _, v in steps]
    ax.plot(x[:4], y[:4], color="#4b78aa", linewidth=1.6, marker="o", markersize=8)
    ax.plot(
        [3, 4],
        [y[3], y[4]],
        color="#9aa7b4",
        linewidth=1.2,
        linestyle="--",
        marker="o",
        markersize=7,
    )
    for xi, yi in zip(x, y, strict=True):
        ax.text(xi, yi + 0.008, f"{yi:.3f}", ha="center", fontsize=10)
    labels = (
        (0.62, 0.625, "information from own history"),
        (1.5, (y[1] + y[2]) / 2 - 0.028, f"+ recency: test log loss -{y[1] - y[2]:.3f}"),
        (2.5, (y[2] + y[3]) / 2 - 0.028, f"+ fuller information: -{y[2] - y[3]:.3f}"),
        (3.5, (y[3] + y[4]) / 2 - 0.028, f"nonlinear model: -{y[3] - y[4]:.3f}"),
    )
    for lx, ly, label in labels:
        ax.text(lx, ly, label, ha="center" if lx > 1 else "left", fontsize=9, color="#3b4a59")
    ax.set_xticks(x, [s for s, _ in steps], fontsize=9)
    ax.set_ylabel("Test log loss (lower is better)")
    ax.set_ylim(0.45, 0.72)
    ax.set_xlim(-0.5, 4.5)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title(
        "F12. What additional information buys: next-week retailer participation",
        loc="left",
        fontsize=12,
    )
    captions = (
        "Target: recorded retailer participation in week t+1 from information through week t (point-in-time "
        "population; no rows before a household's first purchase).",
        "Test weeks 85-101 were held out and evaluated once, after selection on validation weeks 69-84 "
        "(design snapshot e3f443fc; one-time test ledger).",
        f"Recent behaviour improves prediction beyond longer-run propensity. Primary recency gain within purchase "
        f"windows: {float(gain['estimate']):.4f} nats [{float(gain['uncertainty_low']):.4f}, "
        f"{float(gain['uncertainty_high']):.4f}].",
        "Predictive gain is not evidence of causal state dependence: a synthetic hidden-state world with no "
        "purchase memory produces a similar gain.",
        "Bounded benchmark under a frozen model family; not a limit on predictability.",
    )
    for i, text in enumerate(captions):
        fig.text(0.07, 0.17 - i * 0.032, text, fontsize=8.6, ha="left")
    texts = [
        ax.get_title(loc="left"),
        ax.get_ylabel(),
        *(t.get_text() for t in ax.texts),
        *(t.get_text() for t in ax.get_xticklabels()),
        *(t.get_text() for t in fig.texts),
    ]
    _check_text(texts)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.36)
    handle = io.BytesIO()
    fig.savefig(
        handle, format="png", dpi=160, metadata={"Software": "complete_journey Phase 7 Run D F12"}
    )
    plt.close(fig)
    return handle.getvalue()


def payloads(root: Path = PROJECT_ROOT) -> dict[str, bytes]:
    inputs = load_inputs(root)
    t8, t9 = t8_rows(inputs), t9_rows(inputs)
    _check_text([v for r in t8 + t9 for v in r.values()])
    return {
        FINAL_PATHS["F12"]: f12_bytes(inputs),
        FINAL_PATHS["T8"]: _csv(t8, T8_FIELDS),
        FINAL_PATHS["T9"]: _csv(t9, T9_FIELDS),
    }


def _register_sha(root: Path) -> str:
    """The approved register is recorded, so any later edit makes check() fail."""
    return _sha((root / REGISTER).read_bytes())


def _provenance(out: dict[str, bytes], root: Path) -> bytes:
    record = {
        "scope": "PHASE_7_RUN_D_FINAL_PUBLICATION",
        "inputs_sha256": dict(ACCEPTED_INPUTS),
        "artifacts_sha256": {p: _sha(b) for p, b in sorted(out.items())},
        "promoted": list(PROMOTED_IDS),
        "register_sha256": _register_sha(root),
        "estimation": "none (archive-only selection and serialization)",
    }
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()


def publish(root: Path = PROJECT_ROOT) -> dict:
    first, second = payloads(root), payloads(root)
    _require(first == second, "non-deterministic Phase 7 publication")
    out = {**first, PROVENANCE: _provenance(first, root)}
    _require(set(out) == ALLOWLIST, "publication outputs must equal the allowlist")
    (root / FINAL_DIR).mkdir(parents=True, exist_ok=True)
    for relative, data in out.items():
        (root / relative).write_bytes(data)
    return check(root)


def check(root: Path = PROJECT_ROOT) -> dict:
    expected = payloads(root)
    expected[PROVENANCE] = _provenance(expected, root)
    present = {str(p.relative_to(root)) for p in (root / FINAL_DIR).rglob("*") if p.is_file()}
    _require(
        present == ALLOWLIST,
        f"final directory must contain exactly the allowlist: {sorted(present ^ ALLOWLIST)}",
    )
    for relative, data in expected.items():
        _require(
            (root / relative).read_bytes() == data,
            f"final artifact differs from rebuild: {relative}",
        )
    return {p: _sha(d) for p, d in sorted(expected.items())}
