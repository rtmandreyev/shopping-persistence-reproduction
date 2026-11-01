#!/usr/bin/env python
"""Phase 8 final synthesis: paper-facing F13 / F14 / T10 / T11 from the frozen archive.

Performs no simulation. Inputs: the final-run archive (``docs/phases/PHASE_08_FINAL``),
pinned by hash in the provenance record; the archived real P1..P5 (Run C register, read
only after the reveal ledger exists) for per-point κ* in F13. Outputs are a closed,
allowlisted, deterministic set in ``docs/phases/PHASE_08_FINAL/final/``. Identifier-free.

    uv run python scripts/phase8_final_synthesis.py           # write
    uv run python scripts/phase8_final_synthesis.py --check   # verify bytes reproduce
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARCHIVE = ROOT / "docs/phases/PHASE_08_FINAL"
FINAL = ARCHIVE / "final"
INPUTS = (
    "design_snapshot.json",
    "design_snapshot.amend-1.json",
    "import_manifest.json",
    "stage1_gates.json",
    "rq2_truths.json",
    "rq2_per_truth.json",
    "rq2_results.json",
    "calibration_noise.json",
    "rq2_freeze.json",
    "real_reveal_ledger.json",
    "real_rq1.json",
    "s_a_bracket.json",
)
PATHS = {
    "F13": "f13_real_architecture_comparison.png",
    "F14": "f14_known_truth_operating_characteristics.png",
    "T10": "t10_real_adequacy_across_arms.csv",
    "T11": "t11_known_truth_operating_characteristics.csv",
}
PROVENANCE = "phase8_final_provenance.json"
ALLOWLIST = frozenset({*PATHS.values(), PROVENANCE})
BANNED = (
    "household_key",
    "proves",
    "optimal",
    "winner",
    "habit formation",
    "loyalty effect",
    "causal memory",
    "true latent state",
    "structural state dependence is",
)
ARMS = ("logit_normal", "mixture")
FAMILIES = ("A1_PRIME", "A2")
ARM_NAME = {"logit_normal": "logit-normal (primary)", "mixture": "mixture (bracket)"}
FAM_NAME = {"A1_PRIME": "A1′ latent AR state", "A2": "A2 purchase history"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load() -> dict:
    out = {}
    for name in INPUTS:
        out[name] = json.loads((ARCHIVE / name).read_text())
    return out


def check_text(texts) -> None:
    joined = " ".join(str(t) for t in texts).lower()
    hits = [w for w in BANNED if re.search(rf"\b{re.escape(w)}\b", joined)]
    if hits:
        raise SystemExit(f"refused wording in artifact: {hits}")


def csv_bytes(rows: list[dict], fields: list[str]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
    check_text(v for r in rows for v in r.values())
    return buf.getvalue().encode()


def fmt(x, nd=4):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, float) and not math.isfinite(x):
        return "inf"
    return f"{x:.{nd}f}"


# --- T10 ------------------------------------------------------------------------


def t10(inp: dict) -> bytes:
    real, sa = inp["real_rq1.json"], inp["s_a_bracket.json"]
    rows = []
    for arm in ARMS:
        comp = real[arm]["comparative"]
        for fam in FAMILIES:
            fk = real[arm]["families"][fam]["kappas"]
            for k in ("0.15", "0.25", "0.35"):
                v = fk[k]
                rows.append(
                    {
                        "operator": "primary purchase envelope",
                        "heterogeneity_arm": arm,
                        "family": fam,
                        "kappa": k,
                        "verdict": v["verdict"],
                        "label": v["label"],
                        "kappa_star_F": fmt(v["kappa_star_F"]),
                        "argmin_grid_index": ""
                        if v["argmin"] is None
                        else f"{v['argmin'][0]:g},{v['argmin'][1]:g}",
                        "binding_moment": v["binding_moment"] or "",
                        "boundary_dependent": v["boundary_dependent"],
                        "resolution_limited": " ".join(v["resolution_limited"]),
                        "ci1_feasible_points": v["ci1_feasible_points"],
                        "acceptance_set_size": len(v["acceptance_set"]),
                        "comparative_verdict": comp["verdict"][k],
                        "kappa_robustness": comp["kappa_robustness"],
                        "robustness_label": real["heterogeneity_label"],
                    }
                )
    for fam in FAMILIES:
        fk = sa["families"][fam]["kappas"]
        for k in ("0.15", "0.25", "0.35"):
            v = fk[k]
            rows.append(
                {
                    "operator": "S-a availability bracket (secondary)",
                    "heterogeneity_arm": "logit_normal",
                    "family": fam,
                    "kappa": k,
                    "verdict": v["verdict"],
                    "label": v["label"],
                    "kappa_star_F": fmt(v["kappa_star_F"]),
                    "argmin_grid_index": ""
                    if v["argmin"] is None
                    else f"{v['argmin'][0]:g},{v['argmin'][1]:g}",
                    "binding_moment": v["binding_moment"] or "",
                    "boundary_dependent": v["boundary_dependent"],
                    "resolution_limited": " ".join(v["resolution_limited"]),
                    "ci1_feasible_points": v["ci1_feasible_points"],
                    "acceptance_set_size": len(v["acceptance_set"]),
                    "comparative_verdict": sa["comparative"]["verdict"][k],
                    "kappa_robustness": sa["comparative"]["kappa_robustness"],
                    "robustness_label": sa["operator_label"],
                }
            )
    fields = [
        "operator",
        "heterogeneity_arm",
        "family",
        "kappa",
        "verdict",
        "label",
        "kappa_star_F",
        "argmin_grid_index",
        "binding_moment",
        "boundary_dependent",
        "resolution_limited",
        "ci1_feasible_points",
        "acceptance_set_size",
        "comparative_verdict",
        "kappa_robustness",
        "robustness_label",
    ]
    return csv_bytes(rows, fields)


# --- T11 ------------------------------------------------------------------------


def t11(inp: dict) -> bytes:
    per = inp["rq2_per_truth.json"]
    rows = []
    for arm in ARMS:
        for label, r in per[arm].items():
            if r.get("status") == "INFEASIBLE TRUTH":
                rows.append(
                    {
                        "heterogeneity_arm": arm,
                        "truth_class": r["truth_class"],
                        "truth": label,
                        "status": "INFEASIBLE TRUTH",
                    }
                )
                continue
            hf, nv = r["hf"]["0.25"], r["naive"]["union"]
            rows.append(
                {
                    "heterogeneity_arm": arm,
                    "truth_class": r["truth_class"],
                    "truth": label,
                    "status": "FEASIBLE",
                    "q": r["q"],
                    "q_pop": r["q_pop"],
                    "competitor": r["competitor"],
                    "state_kappa_0.25": r["states"]["0.25"],
                    "history_favoured_rate": fmt(hf["rate"], 3),
                    "history_favoured_ci95": f"{hf['ci95'][0]:.3f}-{hf['ci95'][1]:.3f}",
                    "naive_rate": fmt(nv["rate"], 3),
                    "naive_ci95": f"{nv['ci95'][0]:.3f}-{nv['ci95'][1]:.3f}",
                    "comparative_counts_kappa_0.25": " ".join(
                        f"{k}={v}" for k, v in sorted(r["comparative_counts"]["0.25"].items())
                    ),
                }
            )
    fields = [
        "heterogeneity_arm",
        "truth_class",
        "truth",
        "status",
        "q",
        "q_pop",
        "competitor",
        "state_kappa_0.25",
        "history_favoured_rate",
        "history_favoured_ci95",
        "naive_rate",
        "naive_ci95",
        "comparative_counts_kappa_0.25",
    ]
    return csv_bytes(rows, fields)


# --- figures ----------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.size": 8, "svg.hashsalt": "phase8"})
    return plt


def f13(inp: dict) -> bytes:
    """κ*(θ) of each family on the real panel (coarse grid + first level), per arm."""
    from complete_journey.abm import stage1_p8005 as s1p
    from complete_journey.phase8_final.evaluate import Entry, outcome

    plt = _plt()
    tol = s1p.tolerances()
    from complete_journey.abm import stage1

    target = stage1.real_primary_reference()
    data = [target.values[j] for j in ("P1", "P2", "P3", "P4", "P5")]
    se = [target.se[j] for j in ("P1", "P2", "P3", "P4", "P5")]
    real = inp["real_rq1.json"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.0))
    fig.subplots_adjust(hspace=0.42, wspace=0.28, left=0.07, right=0.86, top=0.93, bottom=0.11)
    texts = []
    for r, arm in enumerate(ARMS):
        for c, fam in enumerate(FAMILIES):
            ax = axes[r][c]
            imgs = real[arm]["families"][fam]["images"]
            xs, ys, ks, feas = [], [], [], []
            for key, im in imgs.items():
                ip, ia = map(float, key.split(","))
                e = Entry(im["mean"], im["sd"], im["R"], im["feasible"])
                o, _ = outcome((ip, ia), e, data, se, tol)
                xs.append(ip)
                ys.append(ia)
                feas.append(o.feasible)
                ks.append(min(o.kappa_star, 2.0) if o.feasible else float("nan"))
            sc = ax.scatter(
                [x for x, f in zip(xs, feas, strict=True) if f],
                [y for y, f in zip(ys, feas, strict=True) if f],
                c=[k for k, f in zip(ks, feas, strict=True) if f],
                cmap="viridis_r",
                vmin=0,
                vmax=2.0,
                s=26,
                marker="s",
            )
            ax.scatter(
                [x for x, f in zip(xs, feas, strict=True) if not f],
                [y for y, f in zip(ys, feas, strict=True) if not f],
                c="lightgrey",
                s=10,
                marker="x",
                label="CI-1 infeasible",
            )
            acc = real[arm]["families"][fam]["kappas"]["0.25"]["acceptance_set"]
            if acc:
                ax.scatter(
                    [a[0] for a in acc],
                    [a[1] for a in acc],
                    facecolors="none",
                    edgecolors="red",
                    s=60,
                    linewidths=1.0,
                    label="adequate at κ=0.25",
                )
            v = real[arm]["families"][fam]["kappas"]["0.25"]
            title = (
                f"{FAM_NAME[fam]} — {ARM_NAME[arm]}\n{v['label']}, "
                f"κ*_F = {fmt(v['kappa_star_F'], 3)}"
            )
            ax.set_title(title, fontsize=8, loc="left")
            ax.set_xlabel("persistence index (half-life 0 … 25 weeks)")
            ax.set_ylabel("amplitude index (σ_z or δ)")
            ax.set_xlim(-0.6, 11.6)
            ax.set_ylim(-0.6, 11.6)
            texts += [title]
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(lab, h)
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="lower center",
        ncol=2,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.46, 0.035),
    )
    cax = fig.add_axes([0.89, 0.3, 0.02, 0.4])
    fig.colorbar(sc, cax=cax, label="κ*(θ) (capped at 2)")
    caption = (
        "Real-panel model adequacy under the frozen P8-003 rule (direct simulation). "
        "Model-family sufficiency is not mechanism identification."
    )
    fig.text(0.02, 0.01, caption, fontsize=7)
    check_text(texts + [caption])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, metadata={"Software": "complete_journey P8 F13"})
    plt.close(fig)
    return buf.getvalue()


def f14(inp: dict) -> bytes:
    """False attribution (non-history truths) and detection (history truths) with exact
    95% intervals, both heterogeneity arms."""
    plt = _plt()
    per = inp["rq2_per_truth.json"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    texts = []
    ax = axes[0]
    labels = [
        lab for lab, r in per["logit_normal"].items() if r.get("truth_class") in ("T0", "T1", "T2")
    ]
    for j, arm in enumerate(ARMS):
        for i, lab in enumerate(labels):
            r = per[arm].get(lab, {})
            if r.get("status") == "INFEASIBLE TRUTH" or "hf" not in r:
                continue
            hf, nv = r["hf"]["0.25"], r["naive"]["union"]
            y = i + (0.18 if j else -0.18)
            ax.errorbar(
                hf["rate"],
                y,
                xerr=[[hf["rate"] - hf["ci95"][0]], [hf["ci95"][1] - hf["rate"]]],
                fmt="o" if j == 0 else "s",
                ms=3.5,
                color="C0" if j == 0 else "C1",
                label=("procedure, " + ARM_NAME[arm]) if i == 0 else None,
            )
            if j == 0:
                ax.plot(nv["rate"], i, "k|", ms=8, label="naive rule" if i == 0 else None)
    ax.axvline(0.10, color="grey", ls=":", lw=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("P(HISTORY_FAVOURED) under a non-history truth (κ = 0.25)")
    ax.set_title("False attribution (T0–T2), exact 95% intervals", loc="left", fontsize=8.5)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax = axes[1]
    markers = {0.0: "o", 0.5: "s", 0.8: "^"}
    for j, arm in enumerate(ARMS):
        for lam, mk in markers.items():
            pts = sorted(
                (r["delta1_pp"], r["hf"]["0.25"])
                for r in per[arm].values()
                if r.get("truth_class") == "T3" and r.get("lam") == lam and "hf" in r
            )
            if not pts:
                continue
            x = [p[0] + (0.25 if j else -0.25) for p in pts]
            yv = [p[1]["rate"] for p in pts]
            err = [
                [p[1]["rate"] - p[1]["ci95"][0] for p in pts],
                [p[1]["ci95"][1] - p[1]["rate"] for p in pts],
            ]
            ax.errorbar(
                x,
                yv,
                yerr=err,
                fmt=mk,
                ms=3.5,
                color="C0" if j == 0 else "C1",
                label=f"λ={lam}, {ARM_NAME[arm]}",
            )
    ax.axhline(0.80, color="grey", ls=":", lw=0.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("one-week history effect Δ_1 (percentage points; reachable truths)")
    ax.set_ylabel("detection rate P(HISTORY_FAVOURED)")
    ax.set_title("Detection (T3), exact 95% intervals; points not joined", loc="left", fontsize=8.5)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    caption = (
        "Known-truth operating characteristics conditional on the empirical population "
        "calibration (P8-006 plug-in). Q = 200 panels per truth point."
    )
    fig.text(0.02, 0.005, caption, fontsize=7)
    texts += labels + [caption]
    check_text(texts)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, wspace=0.28)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, metadata={"Software": "complete_journey P8 F14"})
    plt.close(fig)
    return buf.getvalue()


def payloads(inp: dict) -> dict[str, bytes]:
    return {
        PATHS["F13"]: f13(inp),
        PATHS["F14"]: f14(inp),
        PATHS["T10"]: t10(inp),
        PATHS["T11"]: t11(inp),
    }


def provenance(inp: dict, outs: dict[str, bytes]) -> bytes:
    rec = {
        "inputs_sha256": {n: sha((ARCHIVE / n).read_bytes()) for n in INPUTS},
        "outputs_sha256": {n: sha(b) for n, b in sorted(outs.items())},
        "producer": "scripts/phase8_final_synthesis.py",
        "note": "no simulation; aggregate, identifier-free; deterministic bytes",
    }
    return (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    inp = load()
    outs = payloads(inp)
    outs[PROVENANCE] = provenance(inp, outs)
    if args.check:
        bad = [n for n, b in outs.items() if (FINAL / n).read_bytes() != b]
        present = {q.name for q in FINAL.iterdir()}
        if bad or present != ALLOWLIST:
            print(f"MISMATCH {bad} {sorted(present ^ ALLOWLIST)}")
            return 1
        print("reproduced byte-identically")
        return 0
    FINAL.mkdir(parents=True, exist_ok=True)
    for n, b in outs.items():
        (FINAL / n).write_bytes(b)
    print(json.dumps({n: sha(b) for n, b in sorted(outs.items())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
