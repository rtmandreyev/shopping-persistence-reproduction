"""Check of the numbers printed in the manuscript.

Every claim below names a phrase of ``paper/main.tex`` with a ``{}`` slot, the canonical source
of the number that fills the slot, and its printed format. The check formats the source value,
inserts it into the phrase, and requires the phrase to occur in the manuscript (whitespace
normalized). A changed source value or a mistyped manuscript number therefore fails.

Sources: the frozen aggregate records shipped with this package (the fixed-effects tables of
Section 2, the dynamic moments and prediction benchmark of Section 4, the known-truth and
real-panel records of Section 6 and the learned-model summary of Section 7) and
``expected/empirical_values.json``, the raw-data quantities that ``--standard`` recomputes from
the Complete Journey files.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Claim:
    cid: str
    section: str
    phrase: str
    value: float | str
    fmt: str = ""

    note: str = ""

    def text(self) -> str:
        if isinstance(self.value, str):
            return self.phrase.format(self.value)
        return self.phrase.format(fmt_half_up(self.value, self.fmt))


def fmt_half_up(x: float, fmt: str) -> str:
    """Format with round-half-up (as printed in the paper), not Python's round-half-even."""
    from decimal import ROUND_HALF_UP, Decimal

    m = re.fullmatch(r"(,?)\.(\d)f", fmt)
    if not m:
        return format(x, fmt)
    q = Decimal(1).scaleb(-int(m.group(2)))
    d = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    return f"{d:{m.group(1)}.{m.group(2)}f}"


class Sources:
    def __init__(self, root: Path):
        self.root = root
        self.emp = json.loads((root / "expected/empirical_values.json").read_text())
        self.t4 = {(r["outcome"], r["model"]): r for r in csv.DictReader(
            (root / "outputs/tables/panel/final/t4_core_panel_results.csv").open())}
        self.moments = {}
        for r in csv.DictReader((root / "docs/phases/PHASE_07_RUN_C/moment_register_real.csv").open()):
            if r["operator"] == "O_interior_window" and r["source"] == "real":
                self.moments[(r["moment_id"], r["component"])] = r
        self.bench = json.loads((root / "docs/phases/PHASE_07_RUN_C/benchmark_results.json").read_text())
        self.p8 = json.loads((root / "docs/phases/PHASE_08_FINAL/rq2_per_truth.json").read_text())
        self.real = json.loads((root / "docs/phases/PHASE_08_FINAL/real_rq1.json").read_text())
        self.gates = json.loads((root / "docs/phases/PHASE_08_FINAL/stage1_gates.json").read_text())
        self.p9 = json.loads((root / "docs/phases/PHASE_09/final/final_summary.json").read_text())

    def e(self, k):
        return self.emp[k]

    def coef(self, model, outcome="retailer_purchase", field="coefficient"):
        return float(self.t4[(outcome, model)][field])

    def mom(self, mid, comp, field="estimate"):
        return float(self.moments[(mid, comp)][field])

    def hf(self, arm, label, kappa="0.25"):
        return self.p8[arm][label]["hf"][kappa]["rate"]

    def count(self, arm, label, verdict, kappa="0.25"):
        return self.p8[arm][label]["comparative_counts"][kappa].get(verdict, 0)

    def fam(self, arm, f, kappa="0.25"):
        return self.real[arm]["families"][f]["kappas"][kappa]

    def ta(self, pol, key, i=0):
        row = next(r for r in self.p9["tableA"] if r["policy"] == pol)
        return row[key][i] if isinstance(row[key], list) else row[key]

    def tb(self, pol, own, key):
        row = next(r for r in self.p9["tableB"] if r["policy"] == pol and r["class"].startswith(own))
        return row["verdicts_k025"].get(key) if key in ("EQUIVALENT", "UNRESOLVED", "LATENT_FAVOURED", "NEITHER") else row[key]


def claims(S: Sources) -> list[Claim]:
    e = S.e
    LN, MX = "logit_normal", "mixture"
    c = [
        # Abstract and Introduction
        Claim("abs.P11", "Abstract", "retailer, {} percent of households", e("panel.P11_full") * 100, ".0f"),
        Claim("abs.P01", "Abstract", "compared with {} percent of those that had", e("panel.P01_full") * 100, ".0f"),
        Claim("intro.P11", "1", "given week, {} percent bought something", e("panel.P11_full") * 100, ".1f"),
        Claim("intro.P01", "1", "bought nothing, {} percent did", e("panel.P01_full") * 100, ".1f"),
        Claim("intro.gap", "1", "generates that {}-percentage-point gap", S.coef("M0") * 100, ".0f"),
        Claim("abs.detect", "Abstract", "assumes in at most {} percent", S.hf(LN, "lam=0.0,D1=30.0") * 100, ".0f"),
        # Section 2 and Table 2
        Claim("s2.lines", "2.1", "has {} receipt lines", e("data.lines"), ",.0f"),
        Claim("s2.baskets", "2.1", "lines in {} baskets", e("data.baskets"), ",.0f"),
        Claim("s2.stores", "2.1", "across {} stores", e("data.stores"), ".0f"),
        Claim("s2.products", "2.1", "stores and {} products", e("data.products"), ",.0f"),
        Claim("s2.demog", "2.1", "demographics for {} households", e("data.demographics"), ".0f"),
        Claim("s2.days", "2.1", "one retailer over {} days", e("data.days"), ".0f"),
        Claim("s2.part", "2.1", "household-weeks in all, {} percent contain", e("data.participation") * 100, ".1f"),
        Claim("s2.top", "2.1", "top tenth account for {} percent", e("desc.top_decile_revenue_share") * 100, ".1f"),
        Claim("s2.half", "2.1", "have a correlation of {}.", e("desc.half_window_corr"), ".2f"),
        Claim("s2.sp1", "2.1", "rank correlation of {} with total revenue", e("desc.spearman_active_revenue"), ".2f"),
        Claim("s2.sp2", "2.1", "and of only {} with revenue per trip", e("desc.spearman_active_revenue_per_basket"), ".2f"),
        Claim("t2.P11", "Table 2", "$P(y_t=1\\mid y_{{t-1}}=1)$ & {} &", e("panel.P11_full"), ".3f"),
        Claim("t2.P01", "Table 2", "$P(y_t=1\\mid y_{{t-1}}=0)$ & {} &", e("panel.P01_full"), ".3f"),
        Claim("t2.P11O", "Table 2", "=1)$ & 0.708 & {} \\\\", e("ruleO.P11"), ".3f"),
        Claim("t2.P01O", "Table 2", "=0)$ & 0.285 & {} \\\\", e("ruleO.P01"), ".3f"),
        Claim("t2.M0", "Table 2", "Pooled & {} & 0.379", S.coef("M0"), ".3f"),
        Claim("t2.M0se", "Table 2", "& ({}) & \\\\\n\\quad Week effects", S.coef("M0", field="paper_facing_se"), ".3f"),
        Claim("t2.M1", "Table 2", "Week effects & {} & 0.377", S.coef("M1"), ".3f"),
        Claim("t2.M2", "Table 2", "Household effects & {} & 0.113", S.coef("M2"), ".3f"),
        Claim("t2.M2se", "Table 2", "& ({}) & \\\\\n\\quad Household and week", S.coef("M2", field="paper_facing_se"), ".3f"),
        Claim("t2.M3", "Table 2", "Household and week effects & {} & 0.112", S.coef("M3"), ".3f"),
        Claim("t2.dk", "Table 2", "& [{}, 0.249] &", S.coef("M3", field="paper_facing_ci_low"), ".3f"),
        Claim("t2.dkhi", "Table 2", "& [0.100, {}] &", S.coef("M3", field="paper_facing_ci_high"), ".3f"),
        Claim("t2.O.M0", "Table 2", "Pooled & 0.424 & {}", S.mom("VI-2", "O-M0"), ".3f"),
        Claim("t2.O.M3", "Table 2", "& 0.174 & {} \\\\", S.mom("VI-2", "O-M3"), ".3f"),
        Claim("t2.O.lo", "Table 2", "[0.100, 0.249] & [{}, 0.120]", S.mom("VI-2", "O-M3", "uncertainty_low"), ".3f"),
        Claim("t2.O.hi", "Table 2", "[0.103, {}] \\\\", S.mom("VI-2", "O-M3", "uncertainty_high"), ".3f"),
        Claim("t2.21", "Table 2", "weeks 21--101 only & {} &", e("panel.M3_resp21_101"), ".3f"),
        Claim("s2.ruleOhh", "2.3", "inside its window\n({} of the 2,500 households)", e("ruleO.households"), ",.0f"),
        Claim("s2.before", "2.3", "On the full grid, {} percent of response cells", e("panel.share_cells_before_first") * 100, ".1f"),
        Claim("s2.wk20", "2.2", "({}\nof 2,500 have made", e("panel.first_purchase_by_week20"), ",.0f"),
        Claim("s2.P8", "2.3", "eight weeks earlier in place of last week's ({})", S.mom("VI-3", "beta_8"), ".3f"),
        Claim("s2.P3", "2.3", "when last week is also\nincluded ({})", S.mom("VI-3", "gamma_2"), ".3f"),
        Claim("s2.P5", "2.3", "last exactly one week ({})", S.mom("VI-4", "active_1"), ".3f"),
        # Section 3
        Claim("s3.listed", "3.1", "Of the 2,500 households, {} are listed", e("camp.listed_households"), ",.0f"),
        Claim("s3.median", "3.1", "the median listed household appears in {}",
              {3: "three", 4: "four", 5: "five"}[int(e("camp.median_campaigns_per_listed"))]),
        Claim("s3.overlap", "3.1", "for the median campaign {} percent", e("camp.median_concurrent_overlap") * 100, ".0f"),
        Claim("s3.ratio_lo", "3.1", "revenue before the campaign {} to 3.75", e("camp.prior_revenue_ratio_min"), ".1f"),
        Claim("s3.ratio_hi", "3.1", "1.6 to {} times", e("camp.prior_revenue_ratio_max"), ".2f"),
        Claim("s3.earliest", "3.1", "the earliest campaign, with {}\nrecipients", e("camp.earliest_recipients"), ".0f"),
        # Section 4 and Table 3
        Claim("t3.base", "Table 3", "None (overall shopping rate) & --- & {} &", e("pred.baseline_logloss"), ".4f"),
        Claim("t3.freq", "Table 3", "shopped so far & Logistic & {} &", e("pred.freq_logloss"), ".4f"),
        Claim("t3.freqauc", "Table 3", "& 0.5525 & {} \\\\", e("pred.freq_auc"), ".3f"),
        Claim("t3.rec", "Table 3", "were most recent & Logistic & {} &", e("pred.recency_logloss"), ".4f"),
        Claim("t3.recauc", "Table 3", "& 0.5210 & {} \\\\", e("pred.recency_auc"), ".3f"),
        Claim("t3.full", "Table 3", "prediction date & Logistic & {} &", S.bench["test"]["B1|tier2"]["log_loss"], ".4f"),
        Claim("t3.fullauc", "Table 3", "& 0.5043 & {} \\\\", S.bench["test"]["B1|tier2"]["roc_auc"], ".3f"),
        Claim("t3.gbm", "Table 3", "Gradient boosting & {} &", S.bench["test"]["B2|tier2"]["log_loss"], ".4f"),
        Claim("t3.gbmauc", "Table 3", "& 0.5017 & {} \\\\", S.bench["test"]["B2|tier2"]["roc_auc"], ".3f"),
        Claim("t3.gain", "Table 3", "\\multicolumn{{2}}{{c}}{{{} \\ [0.0219, 0.0276]}}", S.mom("VI-6", "delta_log_loss_recency"), ".4f"),
        Claim("t3.gainlo", "Table 3", "0.0248 \\ [{}, 0.0276]", S.mom("VI-6", "delta_log_loss_recency", "uncertainty_low"), ".4f"),
        Claim("t3.gainhi", "Table 3", "0.0248 \\ [0.0219, {}]", S.mom("VI-6", "delta_log_loss_recency", "uncertainty_high"), ".4f"),
        # Section 6 operating characteristics
        Claim("s6.d20", "6.1", "detected history in {}\npercent of panels", S.hf(LN, "lam=0.0,D1=20.0") * 100, ".1f"),
        Claim("s6.d30", "6.1", "and in {} percent at 30 points", S.hf(LN, "lam=0.0,D1=30.0") * 100, ".1f"),
        Claim("s6.m15", "6.1", "{}\npercent at 15 points", S.hf(MX, "lam=0.0,D1=15.0") * 100, ".0f"),
        Claim("s6.m20", "6.1", "against {} and 24.5 percent", S.hf(MX, "lam=0.0,D1=20.0") * 100, ".0f"),
        Claim("s6.m30", "6.1", "against 21 and {} percent", S.hf(MX, "lam=0.0,D1=30.0") * 100, ".1f"),
        Claim("s6.k35ln", "6.1", "it reaches {} under\nlogit-normal", max(S.hf(LN, "lam=0.0,D1=20.0", "0.35"), S.hf(LN, "lam=0.0,D1=30.0", "0.35")), ".2f"),
        Claim("s6.k35mx", "6.1", "heterogeneity and {} under the mixture", S.hf(MX, "lam=0.0,D1=30.0", "0.35"), ".2f"),
        Claim("s6.m15k15", "6.1", "detection falls from {} at $\\kappa=0.15$", S.hf(MX, "lam=0.0,D1=15.0", "0.15"), ".2f"),
        Claim("s6.m15k35", "6.1", "to {} at $\\kappa=0.35$", S.hf(MX, "lam=0.0,D1=15.0", "0.35"), ".3f"),
        Claim("s6.m30k15", "6.1", "\n{} at $\\kappa=0.15$ (Appendix", S.hf(MX, "lam=0.0,D1=30.0", "0.15"), ".3f"),
        Claim("s6.gate", "6.1", "accepted in {} to 100 percent", min(_gate_truth_in_acc(S)) * 100, ".1f"),
        # Real panel (Table D.1)
        Claim("d1.lnA1p", "Table D.1", "inconclusive & {} & one-week spells", S.fam(LN, "A1_PRIME")["kappa_star_F"], ".3f"),
        Claim("d1.lnA2", "Table D.1", "inadequate & {} & one-week spells", S.fam(LN, "A2")["kappa_star_F"], ".3f"),
        Claim("d1.mxA1p", "Table D.1", "adequate & {} & none", S.fam(MX, "A1_PRIME")["kappa_star_F"], ".3f"),
        Claim("d1.mxA2", "Table D.1", "adequate at 0.35 & {} & lag-8", S.fam(MX, "A2")["kappa_star_F"], ".3f"),
        Claim("d1.lnverd", "Table D.1", "\\textsc{{unr}} / \\textsc{{unr}} / \\textsc{{unr}} at", _comp_tag(S, LN)),
        Claim("d1.mxverd", "Table D.1", "\\textsc{{unr}} / \\textsc{{unr}} / \\textsc{{eq}} at", _comp_tag(S, MX)),
        # Table 4
        Claim("t4.j13", "Table 4", "Last week shopped & {} & (0.10) & 11.56", S.ta("Jev 1.13", "delta1_history"), ".2f"),
        Claim("t4.j14", "Table 4", "& 13.32 & (0.10) & {} & (0.10) & 28.55", S.ta("Jev 1.14", "delta1_history"), ".2f"),
        Claim("t4.m", "Table 4", "& (0.10) & {} & (0.20)", S.ta("Mercury", "delta1_history"), ".2f"),
        Claim("t4.ahe", "Table 4", "simulation & {} & (0.14) & 9.32", S.ta("Jev 1.13", "ahe_history"), ".2f"),
        Claim("t4.d4", "Table 4", "against none & {} & (0.29) & 28.65", S.ta("Jev 1.13", "delta4_history"), ".2f"),
        Claim("t4.d4m", "Table 4", "& (0.29) & {} & (0.55) \\\\", S.ta("Mercury", "delta4_history"), ".2f"),
        Claim("t4.nh", "Table 4", "History hidden & {} & (0.29) & 11.57", S.ta("Jev 1.13", "delta_need_latent"), ".2f"),
        Claim("t4.ns", "Table 4", "History shown & {} & (0.07) & 11.20", S.ta("Jev 1.13", "delta_need_both"), ".2f"),
        Claim("t4.mae", "Table 4", "no information & \\multicolumn{{2}}{{c}}{{{}}}", S.ta("Jev 1.13", "base_rate_mae"), ".3f"),
        # Table 5
        Claim("t5.j13a", "Table 5", "Jev 1.13 & 14&14 & 0.0&9.6 & {} &", S.tb("Jev 1.13", "no own", "EQUIVALENT"), ".3f"),
        Claim("t5.j14a", "Table 5", "Jev 1.14 & 14&14 & 0.0&8.6 & {} &", S.tb("Jev 1.14", "no own", "EQUIVALENT"), ".3f"),
        Claim("t5.ma", "Table 5", "Mercury  & 7&14  & 0.0&23.6 & {} &", S.tb("Mercury", "no own", "EQUIVALENT"), ".3f"),
        Claim("t5.j13b", "Table 5", "Jev 1.13 & 16&16 & 2.1&9.7 & {} &", S.tb("Jev 1.13", "own", "EQUIVALENT"), ".3f"),
        Claim("t5.j13bl", "Table 5", "& 0.176 & {} & ---", S.tb("Jev 1.13", "own", "LATENT_FAVOURED"), ".3f"),
        Claim("t5.j14b", "Table 5", "Jev 1.14 & 16&16 & 1.8&8.6 & {} &", S.tb("Jev 1.14", "own", "EQUIVALENT"), ".3f"),
        Claim("t5.mb", "Table 5", "& --- & 0.208 & --- & {} \\\\", S.tb("Mercury", "own", "NEITHER"), ".3f"),
        Claim("t5.zero", "Table 5", "History-favoured verdicts: {} of 200 panels in every",
              format(max(r["max_hf_any_kappa"] for r in S.p9["tableB"]) * 200, ".0f")),
        # Conclusion
        Claim("conc.P11", "9", "In this panel, {} percent", e("panel.P11_full") * 100, ".1f"),
        Claim("conc.P01", "9", "compared with {} percent of those that did not", e("panel.P01_full") * 100, ".1f"),
        Claim("conc.gap", "9", "lower that {}-point", S.coef("M0") * 100, ".0f"),
        Claim("conc.M2", "9", "association to {} points", S.coef("M2") * 100, ".0f"),
        Claim("conc.M3", "9", "lowers it to {}. Under", S.coef("M3") * 100, ".0f"),
        Claim("conc.O", "9", "two-way\ncoefficient is {} points", S.mom("VI-2", "O-M3") * 100, ".0f"),
        Claim("conc.detect", "9", "own form in at\nmost {} percent", S.hf(LN, "lam=0.0,D1=30.0") * 100, ".0f"),
    ]
    return c


def _gate_truth_in_acc(S: Sources) -> list[float]:
    """True-parameter inclusion (G1) at every tolerance, family and arm."""
    return [
        x
        for arm in ("logit_normal", "mixture")
        for fam in S.gates[arm].values()
        if isinstance(fam, dict) and "gate" in fam
        for pt in fam["gate"]["points"].values()
        for x in pt["G1_truth_inclusion"].values()
    ] or [
        pt["G1_truth_inclusion"]["0.25"]
        for arm in ("logit_normal", "mixture")
        for fam in S.gates[arm].values()
        if isinstance(fam, dict) and "gate" in fam
        for pt in fam["gate"]["points"].values()
    ]


def _comp_tag(S: Sources, arm: str) -> str:
    v = S.real[arm]["comparative"]["verdict"]
    tags = {"UNRESOLVED": "unr", "EQUIVALENT": "eq", "HISTORY_FAVOURED": "hf", "LATENT_FAVOURED": "lf", "NEITHER": "ne"}
    want = " / ".join(f"\\textsc{{{tags[v[k]]}}}" for k in ("0.15", "0.25", "0.35"))
    return want


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def check(root: Path) -> list[dict]:
    tex = normalize((root / "paper/main.tex").read_text())
    S = Sources(root)
    out = []
    for cl in claims(S):
        if cl.cid in ("d1.lnverd", "d1.mxverd"):
            phrase = cl.value + " at"
        else:
            phrase = cl.text()
        ok = normalize(phrase) in tex
        out.append({"id": cl.cid, "section": cl.section, "expected_phrase": normalize(phrase), "pass": ok,
                    "note": cl.note})
    return out
