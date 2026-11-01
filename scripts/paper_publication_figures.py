#!/usr/bin/env python
"""Publication-clean renderings of frozen Phase 8 and Phase 9 results for the manuscript.

    cd ~/complete-journey && uv run python scripts/paper_publication_figures.py
    cd ~/complete-journey && uv run python scripts/paper_publication_figures.py --check
    cd ~/complete-journey && uv run python scripts/paper_publication_figures.py --out DIR

This is a rendering pass only. It reads committed, hash-pinned artifacts, refuses to run
if any input hash differs from its pin, and redraws three figures without internal codes,
snapshot hashes or engineering titles. No number is recomputed: every plotted value is
read as stored. Outputs go to ``paper/figures/`` with ``PUBLICATION_MANIFEST.json``.

``--check`` never writes to ``paper/figures/``: it renders into a temporary directory and
compares those bytes with the committed figures (exit 1 on any difference). ``--out DIR``
renders into DIR instead of ``paper/figures/`` and writes no manifest.

- ``fig1_known_truth_operating_characteristics.pdf``: Phase 8 false attribution and
  detection (from T11).
- ``fig2_naive_flag_by_mechanism.pdf``: the naive persistence flag on anchored
  learned-dynamics truths, logit-normal arm (from the Phase 9 final summary).
- ``fig3_in_family_vs_out_of_family.pdf``: Phase 8 in-family detection against the
  out-of-family learned own-history truths (from T11 and the Phase 9 final summary).

No simulation, no network, no credentials.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
INPUTS = {
    "t11": (
        ROOT / "docs/phases/PHASE_08_FINAL/final/t11_known_truth_operating_characteristics.csv",
        "c62dcd7d3c3c45dd7cd8a50e0a944f8606b15c1aa921295b141e0cf2c6d2443e",
    ),
    "p9_final": (
        ROOT / "docs/phases/PHASE_09/final/final_summary.json",
        "5735fa4ee108b19ee876350e0c87d4acd764f5dd48a171c98bc22dc42ce2834d",
    ),
    "p9_truths": (
        ROOT / "docs/phases/PHASE_09/results_p9007/table_truths.md",
        "f251a5e1c8ad589cf3a17df2b77b8b58acb33c7e3899eb203838c512633ded42",
    ),
}
POLS = ("P-J13", "P-J14", "P-M")
NAMES = {"P-J13": "Jev 1.13", "P-J14": "Jev 1.14", "P-M": "Mercury"}
# Categorical slots validated for CVD separation; every series also has its own marker.
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"
POL_STYLE = {"P-J13": (BLUE, "o"), "P-J14": (ORANGE, "s"), "P-M": (AQUA, "^")}
ARM_STYLE = {"logit_normal": (BLUE, "o", "logit-normal"), "mixture": (ORANGE, "s", "mixture")}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_inputs() -> dict[str, str]:
    got = {}
    for key, (path, pin) in INPUTS.items():
        h = sha(path)
        if h != pin:
            raise SystemExit(f"input hash mismatch for {path.relative_to(ROOT)}: {h} != {pin}")
        got[str(path.relative_to(ROOT))] = h
    return got


def plt_setup():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.hashsalt": "paper",
        }
    )
    return plt


TARGET = OUT  # where save() writes; main() points it at a scratch directory for --check


def save(fig, name: str) -> None:
    fig.savefig(
        TARGET / name,
        bbox_inches="tight",
        metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None},
    )


# --- Figure 1: Phase 8 operating characteristics ------------------------------------

def nonhistory_label(cls: str, truth: str) -> str:
    if truth == "A1":
        return "Memoryless heterogeneity"
    if truth.startswith("rho="):
        rho, amp = truth[4:].split(",")
        return f"Latent AR(1), $\\rho$ = {rho}, {amp.lower()} amplitude"
    if truth.startswith("het-rho"):
        return f"Latent AR(1), household $\\rho_i$ (mean {truth.split('=')[1]})"
    if truth.startswith("HMM"):
        return f"Two-state hidden Markov, switch prob. {truth.split('=')[1]}"
    if truth.startswith("AR+periodic"):
        return f"Latent AR(1) + periodic routine, {truth.split('=')[1].lower()}"
    raise ValueError(truth)


def fig1(plt, rows):
    order = []
    for r in rows:
        if r["heterogeneity_arm"] == "logit_normal" and r["truth_class"] in ("T0", "T1", "T2"):
            order.append((r["truth_class"], r["truth"]))
    cls_rank = {"T0": 0, "T1": 1, "T2": 2}
    order.sort(key=lambda x: (cls_rank[x[0]], x[1]))
    index = {k: i for i, k in enumerate(order)}

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 4.9), gridspec_kw={"width_ratios": [1.1, 0.85]})
    ys = list(range(len(order)))[::-1]
    for arm, dy in (("logit_normal", 0.14), ("mixture", -0.14)):
        col, mk, lab = ARM_STYLE[arm]
        first = True
        for r in rows:
            if r["heterogeneity_arm"] != arm or r["truth_class"] not in ("T0", "T1", "T2"):
                continue
            y = ys[index[(r["truth_class"], r["truth"])]] + dy
            lo, hi = (float(v) for v in r["history_favoured_ci95"].split("-"))
            a.plot([lo, hi], [y, y], color=col, lw=1.0)
            a.plot(float(r["history_favoured_rate"]), y, mk, color=col, ms=4,
                   label=f"procedure, {lab}" if first else None)
            a.plot(float(r["naive_rate"]), y, "|", color=INK, ms=7, mew=1.2,
                   label="naive persistence rule" if (first and arm == "logit_normal") else None)
            first = False
    a.set_yticks(ys)
    a.set_yticklabels([nonhistory_label(c, t) for c, t in order], fontsize=7.5)
    a.set_xlim(-0.03, 1.03)
    a.set_xlabel("Share classified history-favoured\n($\\kappa$ = 0.25)")
    a.set_title("(a) Non-history truths", loc="left")
    split = ys[index[[k for k in order if k[0] == "T2"][0]]] + 0.5
    a.axhline(split, color=GREY, lw=0.6, ls=":")

    for arm, lam_ok in (("logit_normal", ("0.0", "0.5")), ("mixture", ("0.0", "0.5", "0.8"))):
        col, _, lab = ARM_STYLE[arm]
        for lam, mk in (("0.0", "o"), ("0.5", "s"), ("0.8", "^")):
            if lam not in lam_ok:
                continue
            xs, ps, los, his = [], [], [], []
            for r in rows:
                if r["heterogeneity_arm"] != arm or r["truth_class"] != "T3" or r["status"] != "FEASIBLE":
                    continue
                spec = dict(kv.split("=") for kv in r["truth"].split(","))
                if spec["lam"] != lam:
                    continue
                lo, hi = (float(v) for v in r["history_favoured_ci95"].split("-"))
                xs.append(float(spec["D1"]) + (0.35 if arm == "mixture" else -0.35))
                ps.append(float(r["history_favoured_rate"]))
                los.append(lo)
                his.append(hi)
            if not xs:
                continue
            b.errorbar(xs, ps, yerr=[[p - l for p, l in zip(ps, los)], [h - p for p, h in zip(ps, his)]],
                       fmt=mk, color=col, ms=4, lw=1.0, capsize=0,
                       mfc=col if lam == "0.0" else "white",
                       label=f"{lab}, $\\lambda$ = {float(lam):g}")
    b.axhline(0.8, color=GREY, lw=0.6, ls=":")
    b.set_ylim(-0.03, 1.0)
    b.set_xlim(0, 32)
    b.set_xticks([5, 10, 15, 20, 25, 30])
    b.set_xlabel("One-week history effect $\\Delta_1$\n(percentage points)")
    b.set_ylabel("Detection rate (history-favoured)")
    b.set_title("(b) History truths, maintained family", loc="left")
    handles, labels = [], []
    for ax in (a, b):
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, handlelength=1.2,
               columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=3.0, rect=(0, 0.13, 1, 1))
    save(fig, "fig1_known_truth_operating_characteristics.pdf")
    plt.close(fig)


# --- Figure 2: naive flag by mechanism class ------------------------------------------

ROWS2 = (
    ("A1", "No dynamics"),
    ("A2", "Latent need"),
    ("A7", "Latent need, history response removed"),
    ("A3", "Irrelevant history, resampled"),
    ("A4", "Irrelevant history, original order"),
    ("A5", "Latent need + resampled irrelevant history"),
    ("A6", "Latent need + irrelevant history, original order"),
    ("B4", r"Own history ($\theta$ = 1)"),
    ("B8", r"Own history + latent need ($\theta$ = 1)"),
)


def admissible_logit_normal(truth_table: str) -> dict[tuple[str, str], bool]:
    out = {}
    for line in truth_table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 14 or cells[0] not in POLS or cells[5] != "logit_normal":
            continue
        out[(cells[0], cells[1])] = cells[13] == "EVALUATED"
    return out


def fig2(plt, summary, adm):
    import numpy as np

    flags = summary["figB_naive_flag_logit_normal"]
    mat = np.array([[flags[f"{t}:{p}"] for p in POLS] for t, _ in ROWS2])
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.imshow(mat, cmap="Greys", vmin=0, vmax=1.25, aspect="auto")
    for i, (t, _) in enumerate(ROWS2):
        for j, p in enumerate(POLS):
            v = mat[i, j]
            star = "" if adm[(p, t)] else "*"
            ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([NAMES[p] for p in POLS])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(ROWS2)))
    ax.set_yticklabels([lab for _, lab in ROWS2], fontsize=8)
    ax.axhline(6.5, color=INK, lw=1.2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.text(2.62, 3.0, "no own-history\ndependence", rotation=90, va="center", ha="left", fontsize=7, color=MUTED)
    ax.text(2.62, 7.5, "own\nhistory", rotation=90, va="center", ha="left", fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "fig2_naive_flag_by_mechanism.pdf")
    plt.close(fig)


# --- Figure 3: in-family against out-of-family -----------------------------------------

def fig3(plt, rows, summary):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3), sharey=True)
    titles = {"logit_normal": "(a) Logit-normal heterogeneity", "mixture": "(b) Two-component mixture"}
    for ax, arm in zip(axes, ("logit_normal", "mixture")):
        first = True
        for r in rows:
            if r["heterogeneity_arm"] != arm or r["truth_class"] != "T3" or r["status"] != "FEASIBLE":
                continue
            spec = dict(kv.split("=") for kv in r["truth"].split(","))
            if spec["lam"] != "0.0":
                continue
            lo, hi = (float(v) for v in r["history_favoured_ci95"].split("-"))
            p = float(r["history_favoured_rate"])
            x = float(spec["D1"])
            ax.plot([x, x], [lo, hi], color=INK, lw=1.0)
            ax.plot(x, p, "D", color=INK, ms=4, label="maintained-family truths ($\\lambda$ = 0)" if first else None)
            first = False
        seen = set()
        for pt in summary["figC"]["p9007"]:
            if pt["arm"] != arm:
                continue
            col, mk = POL_STYLE[pt["policy"]]
            name = NAMES[pt["policy"]]
            if pt.get("evaluated", True) is False:
                key = "excl"
                ax.plot(pt["delta1_pp"], -0.06, mk, mfc="white", mec=col, ms=5, mew=1.0,
                        label="not evaluated (excluded by a pre-specified rule)" if key not in seen else None)
                seen.add(key)
                continue
            key = pt["policy"]
            ax.plot(pt["delta1_pp"], pt["det"], mk, color=col, ms=5, alpha=0.9,
                    label=f"{name} learned own-history truths" if key not in seen else None)
            seen.add(key)
        ax.set_title(titles[arm], loc="left")
        ax.set_xlabel(r"One-week history effect $\Delta_1$ (percentage points)")
        ax.set_xlim(0, 32)
        ax.set_ylim(-0.1, 1.0)
        ax.axhline(0, color=GREY, lw=0.4)
    axes[0].set_ylabel("Share classified history-favoured ($\\kappa$ = 0.25)")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig3_in_family_vs_out_of_family.pdf")
    plt.close(fig)


OUTPUTS = (
    "fig1_known_truth_operating_characteristics.pdf",
    "fig2_naive_flag_by_mechanism.pdf",
    "fig3_in_family_vs_out_of_family.pdf",
)


def render(plt, target: Path) -> dict[str, str]:
    """Draw the three figures into ``target`` and return their SHA-256."""
    global TARGET
    TARGET = target
    rows = list(csv.DictReader(INPUTS["t11"][0].open()))
    summary = json.loads(INPUTS["p9_final"][0].read_text())
    adm = admissible_logit_normal(INPUTS["p9_truths"][0].read_text())
    fig1(plt, rows)
    fig2(plt, summary, adm)
    fig3(plt, rows, summary)
    return {n: sha(target / n) for n in OUTPUTS}


def main() -> None:
    import tempfile

    check = "--check" in sys.argv
    out_dir = Path(sys.argv[sys.argv.index("--out") + 1]).resolve() if "--out" in sys.argv else None
    inputs = verify_inputs()
    plt = plt_setup()
    if check:
        committed = {n: sha(OUT / n) for n in OUTPUTS if (OUT / n).exists()}
        with tempfile.TemporaryDirectory(prefix="paperfigs-") as tmp:
            outputs = render(plt, Path(tmp))
        bad = [n for n in OUTPUTS if committed.get(n) != outputs[n]]
        if bad:
            raise SystemExit(f"--check failed; bytes differ for: {bad} (paper/figures/ untouched)")
        print("check OK: outputs byte-identical (paper/figures/ untouched)")
        return
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for n, h in render(plt, out_dir).items():
            print(n, h)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = render(plt, OUT)
    manifest = {
        "producer": "scripts/paper_publication_figures.py",
        "note": "publication rendering of frozen values; no recomputation; inputs hash-verified",
        "inputs_sha256": inputs,
        "outputs_sha256": outputs,
    }
    (OUT / "PUBLICATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    )
    for n, h in outputs.items():
        print(n, h)


if __name__ == "__main__":
    main()
