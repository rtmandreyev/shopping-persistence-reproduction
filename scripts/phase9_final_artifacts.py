#!/usr/bin/env python
"""Final Phase 9 publication artifact set (deterministic, identifier-free, hash-pinned).

    cd ~/complete-journey && uv run python scripts/phase9_final_artifacts.py

Rebuilt only from frozen inputs: the main local results (``963e65e9…``), the P9-007
results (``a38d9790…``) and the Phase 8 RQ2 results (read-only). It writes to
``docs/phases/PHASE_09/final/``:

- ``figA_controlled_effects.png``: the learned policies' controlled history and need
  effects (the instrument);
- ``figB_naive_decomposition.png``: the naive persistence flag by mechanism class
  (P9-007 anchored truths);
- ``figC_procedure_in_vs_out_of_family.png``: Phase 8 in-family detection against the
  P9-007 out-of-family truths, with verdict composition;
- ``tableA_controlled_effects.md``: the main controlled effects plus validity;
- ``tableB_out_of_family.md``: out-of-family admissibility, verdicts, FA / DET;
- ``CAPTIONS.md``: captions with the canonical P9-006 disclosure;
- ``final_summary.json`` and ``MANIFEST.json``.

No simulation, no network, no credentials.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "docs/phases/PHASE_09/final"
MAIN = ROOT / "outputs/diagnostics/phase9_local/results.json"
EXT = ROOT / "outputs/diagnostics/phase9_p9007/results.json"
P8 = ROOT / "docs/phases/PHASE_08_FINAL/rq2_results.json"
PINNED = {
    "main": "963e65e9172cd884361c39d49ec1b543dd529989d10c25b1ac6558b0854c7c69",
    "p9007": "a38d979057649663bda47561568b29ada5a8fd9ecc9d1ed847d43ae0208fa5ac",
}
POLS = ("P-J13", "P-J14", "P-M")
NAMES = {"P-J13": "Jev 1.13", "P-J14": "Jev 1.14", "P-M": "Mercury"}
ARMS = ("logit_normal", "mixture")
DISCLOSURE = (
    "Mercury (P-M) disclosure: production acquisition was stopped at sentinel time point 3 "
    "by the original pre-registered drift gate (P9-005). That gate was then shown, from "
    "pre-production smoke data and the frozen sentinel composition, to be statistically "
    "miscalibrated (estimated false-stop probability 0.37-0.49 under smoke-consistent "
    "models). After the stop, it was replaced (P9-006) by a stratified, 1% family-wise "
    "rule derived from the same pre-production information only. The first 1,708 "
    "responses were retained and are included in every Mercury estimate reported here, and "
    "acquisition resumed; P-M passed time point 4 under the "
    "amended rule. The amended rule needs a single BOTH-cell sentinel to move about 0.27 "
    "to stop, so its sensitivity to drift in that cell is weak; this remains a limitation."
)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def plt_setup():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, metadata={"Software": None}, bbox_inches="tight")


# --- Figure A / Table A ---------------------------------------------------------------

EFFECTS = (
    ("delta1_history", "Δ1\nlast week shopped"),
    ("delta4_history", "Δ4\nall 4 weeks shopped"),
    ("ahe_history", "AHE\naverage history effect"),
    ("delta_need_latent", "Need\nhigh vs low"),
)


def fig_a(main) -> dict:
    import numpy as np

    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    w = 0.26
    x = np.arange(len(EFFECTS))
    data = {}
    for i, p in enumerate(POLS):
        st = main["policies"][p]["static"]["primary"]
        v = np.array([st["estimands_pp"][k] for k, _ in EFFECTS])
        se = np.array([100 * st["estimand_se"][k] for k, _ in EFFECTS])
        data[p] = {
            k: {"pp": float(a), "se_pp": float(b)}
            for (k, _), a, b in zip(EFFECTS, v, se, strict=True)
        }
        ax.bar(x + (i - 1) * w, v, w, yerr=3 * se, capsize=2, label=NAMES[p], color=f"C{i}")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in EFFECTS], fontsize=8)
    ax.set_ylabel("change in policy P(shop), percentage points")
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    save(fig, "figA_controlled_effects.png")
    plt.close(fig)
    return data


def table_a(main) -> list[dict]:
    rows = []
    for p in POLS:
        st = main["policies"][p]["static"]["primary"]
        e, s, lab = st["estimands_pp"], st["estimand_se"], st["estimand_labels"]
        rows.append(
            {
                "policy": NAMES[p],
                **{k: (round(e[k], 2), round(100 * s[k], 2), lab[k]) for k, _ in EFFECTS},
                "delta_need_both": (
                    round(e["delta_need_both"], 2),
                    round(100 * s["delta_need_both"], 2),
                    lab["delta_need_both"],
                ),
                "validity": sorted({v["label"] for v in st["validity"].values()}),
                "sigma_rep": round(st["sigma_rep"], 4),
                "base_rate_mae": round(st["base_rate_fidelity"]["mean_abs_error"], 3),
            }
        )
    lines = [
        "# Table A — Controlled learned-policy effects and validity (primary convention)",
        "",
        "Entries: estimate ± SE in percentage points of P(shop); the label follows the frozen "
        "|estimate| > 3·SE rule. Validity is the frozen paraphrase + placebo + repeat rule "
        "over the four information components.",
        "",
        "| Policy | Δ1 history | Δ4 history | AHE history | Need (latent) | Need (both) | "
        "Validity (4 components) | σ_rep | Base-rate MAE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cell = lambda t: f"{t[0]:.2f} ± {t[1]:.2f} ({t[2]})"  # noqa: E731
        lines.append(
            f"| {r['policy']} | {cell(r['delta1_history'])} | {cell(r['delta4_history'])} | "
            f"{cell(r['ahe_history'])} | {cell(r['delta_need_latent'])} | {cell(r['delta_need_both'])} | "
            f"{' / '.join(r['validity'])} | {r['sigma_rep']} | {r['base_rate_mae']} |"
        )
    lines += ["", f"> {DISCLOSURE}"]
    (OUT / "tableA_controlled_effects.md").write_text("\n".join(lines) + "\n")
    return rows


# --- Figure B ---------------------------------------------------------------------------

CLASSES = (
    ("A1", "No dynamics"),
    ("A2", "Latent need"),
    ("A7", "Latent need (memory removed)"),
    ("A3", "Irrelevant history, time-scrambled"),
    ("A4", "Irrelevant history, serial"),
    ("A5", "Latent + scrambled history"),
    ("A6", "Latent + serial history"),
    ("B4", "Own history, θ = 1"),
    ("B8", "Own history + latent, θ = 1"),
)


def fig_b(ext) -> dict:
    import numpy as np

    plt = plt_setup()
    mat = np.zeros((len(CLASSES), len(POLS)))
    adm = {}
    for i, (t, _) in enumerate(CLASSES):
        for j, p in enumerate(POLS):
            rec = ext["policies"][p][t]["arms"]["logit_normal"]
            mat[i, j] = rec["level1"]["naive"]["union"]["rate"]
            adm[(t, p)] = rec["level2_status"] == "EVALUATED"
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.imshow(mat, vmin=0, vmax=1, cmap="Greys", aspect="auto")
    for i in range(len(CLASSES)):
        for j in range(len(POLS)):
            mark = "" if adm[(CLASSES[i][0], POLS[j])] else "*"
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}{mark}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if mat[i, j] > 0.5 else "black",
            )
    ax.set_xticks(range(len(POLS)))
    ax.set_xticklabels([NAMES[p] for p in POLS])
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels([lab for _, lab in CLASSES], fontsize=8)
    ax.axhline(6.5, color="C3", lw=1.2)
    ax.text(
        2.55, 3.0, "no own-history\ndependence", rotation=270, va="center", fontsize=7, color="C3"
    )
    ax.text(2.55, 7.5, "own history", rotation=270, va="center", fontsize=7, color="C3")
    save(fig, "figB_naive_decomposition.png")
    plt.close(fig)
    return {
        f"{t}:{p}": float(mat[i, j]) for i, (t, _) in enumerate(CLASSES) for j, p in enumerate(POLS)
    }


# --- Figure C / Table B -----------------------------------------------------------------


def verdict_mix(rec, arm) -> Counter:
    if rec.get("level2") and arm in rec["level2"]["arms"]:
        return Counter(rec["level2"]["arms"][arm]["comparative_counts"]["0.25"])
    return Counter()


def fig_c(ext, p8) -> dict:
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    colours = {"EQUIVALENT": "C0", "UNRESOLVED": "C7", "NEITHER": "C3", "LATENT_FAVOURED": "C2"}
    marks = {"P-J13": "o", "P-J14": "s", "P-M": "^"}
    data = {"phase8": {}, "p9007": []}
    for ax, arm in zip(axes, ARMS, strict=True):
        pts = sorted(
            (float(k.split("D1=")[1]), v["rate"])
            for k, v in p8[arm]["DET"]["0.25"]["DET_OVERALL"].items()
            if k.startswith("lam=0.0,D1=")
        )
        data["phase8"][arm] = pts
        ax.plot(
            *zip(*pts, strict=True),
            "k-",
            lw=1,
            marker=".",
            label="Phase 8 in-family history truths",
        )
        seen = set()
        for p in POLS:
            for t, v in ext["policies"][p].items():
                if v["truth"] != "HISTORY-INSTALLED":
                    continue
                rec = v["arms"][arm]
                d1 = 100 * rec["diagnostics_pop"]["delta1"]
                mix = verdict_mix(rec, arm)
                if not mix:
                    ax.scatter(
                        d1, -0.06, marker=marks[p], facecolors="none", edgecolors="0.5", s=22
                    )
                    data["p9007"].append(
                        {"policy": p, "truth": t, "arm": arm, "delta1_pp": d1, "evaluated": False}
                    )
                    continue
                dom = mix.most_common(1)[0][0]
                hf = mix.get("HISTORY_FAVOURED", 0) / sum(mix.values())
                lab = f"{NAMES[p]} — mostly {dom.replace('_', '-').lower()}"
                ax.scatter(
                    d1,
                    hf,
                    marker=marks[p],
                    color=colours.get(dom, "k"),
                    s=24,
                    label=lab if lab not in seen else None,
                )
                seen.add(lab)
                data["p9007"].append(
                    {
                        "policy": p,
                        "truth": t,
                        "arm": arm,
                        "delta1_pp": d1,
                        "det": hf,
                        "dominant": dom,
                    }
                )
        ax.set_title(
            {"logit_normal": "logit-normal heterogeneity", "mixture": "mixture heterogeneity"}[arm],
            fontsize=9,
        )
        ax.set_xlabel("one-week history effect Δ1 (pp)")
        ax.set_ylim(-0.1, 1.0)
    axes[0].set_ylabel("P(HISTORY-FAVOURED), κ = 0.25")
    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        for hh, ll in zip(h, lab, strict=True):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.16),
    )
    save(fig, "figC_procedure_in_vs_out_of_family.png")
    plt.close(fig)
    return data


def table_b(ext) -> list[dict]:
    rows = []
    for p in POLS:
        for cls, label in (
            ("NO-HISTORY", "no own-history (A1–A7)"),
            ("HISTORY-INSTALLED", "own history (B1–B8)"),
        ):
            recs = [
                (t, arm, v["arms"][arm])
                for t, v in ext["policies"][p].items()
                if v["truth"] == cls
                for arm in ARMS
            ]
            ev = [(t, arm, r) for t, arm, r in recs if r["level2_status"] == "EVALUATED"]
            mix = Counter()
            for _, arm, r in ev:
                mix.update(verdict_mix(r, arm))
            n = sum(mix.values()) or 1
            hf = max(
                (
                    r["level2"]["arms"][arm]["history_favoured"][k]["rate"]
                    for _, arm, r in ev
                    for k in ("0.15", "0.25", "0.35")
                ),
                default=None,
            )
            d1 = [100 * r["diagnostics_pop"]["delta1"] for _, _, r in ev]
            rows.append(
                {
                    "policy": NAMES[p],
                    "class": label,
                    "evaluated": f"{len(ev)}/{len(recs)}",
                    "max_hf_any_kappa": hf,
                    "verdicts_k025": {k: round(c / n, 3) for k, c in mix.most_common()},
                    "delta1_pp_range": (round(min(d1), 1), round(max(d1), 1)) if d1 else None,
                    "naive_flag_range": (
                        round(min(r["level1"]["naive"]["union"]["rate"] for _, _, r in recs), 2),
                        round(max(r["level1"]["naive"]["union"]["rate"] for _, _, r in recs), 2),
                    ),
                }
            )
    lines = [
        "# Table B — Out-of-family recovery on anchored learned-dynamics truths (P9-007)",
        "",
        "Truth-arm = truth × heterogeneity arm. 'Evaluated' = CI-1 admissible and not DEGENERATE "
        "(pre-registered rules). Max HISTORY-FAVOURED is over every evaluated truth-arm and "
        "κ ∈ {0.15, 0.25, 0.35}: it is false attribution for the no-own-history class and "
        "attribution to the maintained history family for the own-history class. 0/200 per "
        "truth-arm has an exact two-sided 95% Clopper–Pearson upper bound of 0.018. Mercury's "
        "excluded arms are those with the strongest learned response.",
        "",
        "| Policy | Truth class | Evaluated | Max HISTORY-FAVOURED | Verdicts at κ 0.25 | Δ1 range (pp) | "
        "Naive flag range (all truth-arms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mix = ", ".join(f"{k.replace('_', '-')} {v:.3f}" for k, v in r["verdicts_k025"].items())
        lines.append(
            f"| {r['policy']} | {r['class']} | {r['evaluated']} | {r['max_hf_any_kappa']} | {mix} | "
            f"{r['delta1_pp_range']} | {r['naive_flag_range']} |"
        )
    lines += [
        "",
        "Code-path check: the Phase 9 Level-2 wrapper reproduces Phase 8's stored in-family "
        "detection rates exactly (0.645, 0.285, 0.470), so HISTORY-FAVOURED is reachable.",
        "",
        f"> {DISCLOSURE}",
    ]
    (OUT / "tableB_out_of_family.md").write_text("\n".join(lines) + "\n")
    return rows


CAPTIONS = f"""# Final Phase 9 figure and table captions

**Figure A — Controlled history and need responses of three pinned learned decision
policies.**
- Bars give the change in the policy's stated probability of shopping (percentage points)
  when one information component is changed in otherwise identical synthetic states.
- Error bars are ±3 propagated repeat-noise SE.
- Every effect passes the frozen paraphrase, placebo and repeat validity rule.
- These are measurements of artificial decision policies, not of households.

**Figure B — The naive persistence flag by mechanism class.**
- The figure shows the share of 200 simulated panels flagged by the conventional naive
  persistence rule, on anchored known truths (Phase 8 heterogeneity layer fitted to the
  real CI-1, plus the learned dynamic response; logit-normal arm).
- Rows above the line contain no own-history dependence; rows below contain installed own
  history.
- \\* marks a truth that is not admissible (CI-1 infeasible).

**Figure C — The frozen Phase 8 model-comparison procedure, in-family and
out-of-family.**
- The line shows Phase 8's detection rate for history truths generated by its own history
  family.
- Markers show the P9-007 learned own-history truths, coloured by their dominant verdict.
  No out-of-family history truth was ever classified HISTORY-FAVOURED (κ = 0.25 shown;
  0/200 at every κ).
- Hollow markers are not evaluated (pre-registered INFEASIBLE or DEGENERATE).

**Table A** — Controlled effects and validity. **Table B** — Out-of-family admissibility,
verdicts and false attribution / detection.

**Canonical disclosure (Methods):** {DISCLOSURE}
"""


def main() -> int:
    """Write the final artifacts; with ``--check``, rebuild them in a temporary directory and
    compare with the committed ``docs/phases/PHASE_09/final/`` without writing there."""
    global OUT
    if "--check" in sys.argv:
        import tempfile

        committed = OUT
        with tempfile.TemporaryDirectory(prefix="p9final-") as tmp:
            OUT = Path(tmp)
            rc = build()
            if rc:
                return rc
            names = sorted(f.name for f in OUT.iterdir() if f.name != "MANIFEST.json")
            text = [n for n in names if not n.endswith(".png")]
            bad = [n for n in text if (OUT / n).read_bytes() != (committed / n).read_bytes()]
            png = [n for n in names if n.endswith(".png") and sha(OUT / n) != sha(committed / n)]
        print(json.dumps({"text_artifacts_identical": not bad, "differing": bad,
                          "png_bytes_differ": png}, indent=1))
        return 1 if bad else 0
    return build()


def build() -> int:
    main_res = json.loads(MAIN.read_text())
    ext = json.loads(EXT.read_text())
    p8 = json.loads(P8.read_text())
    if main_res["results_sha256"] != PINNED["main"] or ext["results_sha256"] != PINNED["p9007"]:
        print("REFUSED: input results differ from the pinned frozen results.")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "FINAL (Phase 9 accepted; see P9-008)",
        "inputs": {
            "main_results_sha256": PINNED["main"],
            "p9007_results_sha256": PINNED["p9007"],
            "phase8_rq2_results_file_sha256": sha(P8),
        },
        "figA": fig_a(main_res),
        "tableA": table_a(main_res),
        "figB_naive_flag_logit_normal": fig_b(ext),
        "figC": fig_c(ext, p8),
        "tableB": table_b(ext),
        "disclosure": DISCLOSURE,
    }
    (OUT / "CAPTIONS.md").write_text(CAPTIONS)
    (OUT / "final_summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n"
    )
    files = sorted(f for f in OUT.iterdir() if f.name != "MANIFEST.json")
    (OUT / "MANIFEST.json").write_text(
        json.dumps({f.name: sha(f) for f in files}, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps({"out": str(OUT), "files": [f.name for f in files]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
