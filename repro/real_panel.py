"""Independent recomputation of the real-panel comparison (paper Table D.1).

Uses the stored moments in ``artifacts/known_truth_panels.json.gz`` (real moments,
household-bootstrap SEs and every family point table) and the adequacy rule of
``repro.adequacy``. The recorded procedure evaluates each family on the coarse lattice plus one
refinement level, then adds the further replicates it simulated at the selected points
(``audit_rows``) and evaluates again. The binding moment is reported only when some
moment's kappa* is positive (no moment binds when kappa* = 0).
"""

from __future__ import annotations

import math

import numpy as np

from repro.adequacy import KAPPAS, Rule, comparative, parse, refine

NAMES = ("two-way lag", "lag-8", "lag-2 given lag 1", "gain", "one-week spells")


def pooled(e: dict, rows) -> dict:
    n1, m1, s1 = e["n"], np.asarray(e["mean"]), np.asarray(e["sd"])
    rows = np.asarray(rows, float).reshape(-1, 5)
    n = n1 + rows.shape[0]
    mean = (n1 * m1 + rows.sum(0)) / n
    var = ((n1 - 1) * s1**2 + n1 * m1**2 + (rows**2).sum(0) - n * mean**2) / (n - 1)
    return {"mean": mean.tolist(), "sd": np.sqrt(np.maximum(var, 0)).tolist(), "n": n,
            "feasible": e["feasible"]}


class RealFamily:
    def __init__(self, doc: dict, rule: Rule):
        self.rule = rule
        self.x = np.asarray(doc["real"])
        self.se = np.asarray(doc["real_se"])

    def point(self, e):
        st, ks, kj = self.rule.points(self.x, self.se, np.asarray([e["mean"]]), np.asarray([e["sd"]]),
                                      np.asarray([e["n"]], float), np.asarray([e["feasible"]]))
        return {k: st[k][0] for k in KAPPAS}, float(ks[0]), kj[0]

    def family(self, coarse: dict, refined: dict, levels: int = 1):
        outs = {parse(c): self.point(e) for c, e in coarse.items()}
        ref = {parse(c): e for c, e in refined.items()}
        step, added = 1.0, 0
        for _ in range(levels):
            ev = {c: {k: o[0][k] == "A" for k in KAPPAS} for c, o in outs.items()}
            new = refine(ev, {c: o[1] for c, o in outs.items()}, step, 144 - added)
            for c in new:
                if c not in ref:
                    raise KeyError(f"refinement point {c} missing")
                outs[c] = self.point(ref[c])
            added += len(new)
            step /= 2
        verdict = {}
        for k in KAPPAS:
            s = [o[0][k] for o in outs.values() if o[0][k] != "X"]
            verdict[k] = "ADEQUATE" if "A" in s else ("INADEQUATE" if s and all(x == "I" for x in s) else "INCONCLUSIVE")
        fin = [(o[1], c) for c, o in outs.items() if math.isfinite(o[1])]
        ks, am = min(fin)
        kj = outs[am][2]
        bind = None if max(kj) <= 0 else NAMES[int(np.argmax(kj))]
        return verdict, ks, bind


def recompute(doc: dict) -> dict:
    rule = Rule(doc["E"], doc["floor"])
    out = {}
    for arm, t in doc["tables"].items():
        rf = RealFamily(doc, rule)
        a1 = rf.point(t["A1"])[0]
        res = {"A1": {str(k): {"A": "ADEQUATE", "I": "INADEQUATE"}.get(a1[k], "INCONCLUSIVE") for k in KAPPAS}}
        for f in ("A1_PRIME", "A2"):
            refined = dict(t[f]["refined"])
            for c, rows in t[f]["audit_rows"].items():  # pool the real-panel audit replicates
                refined[c] = pooled(refined[c], rows)
            verdict, ks, bind = rf.family(t[f]["coarse"], refined)
            secondary = rf.family(t[f]["coarse"], {**refined, **t[f]["ref2"]}, levels=2)[0]
            res[f] = {"verdict": {str(k): v for k, v in verdict.items()}, "kappa_star": ks,
                      "binding": bind, "level2": {str(k): v for k, v in secondary.items()}}
        res["comparative"] = {str(k): comparative(res["A2"]["verdict"][str(k)], res["A1_PRIME"]["verdict"][str(k)])
                              for k in KAPPAS}
        res["comparative_level2"] = {str(k): comparative(res["A2"]["level2"][str(k)], res["A1_PRIME"]["level2"][str(k)])
                                     for k in KAPPAS}
        out[arm] = res
    sa = {}
    rf = RealFamily(doc, rule)
    for f in ("A1_PRIME", "A2"):
        sa[f] = rf.family(doc["sa_logit_normal"][f]["coarse"], doc["sa_logit_normal"][f]["refined"])[0]
    out["alternative_rule_logit_normal"] = {str(k): comparative(sa["A2"][k], sa["A1_PRIME"][k]) for k in KAPPAS}
    return out


def compare(rec: dict, stored: dict) -> list[str]:
    """Mismatches against the frozen real_rq1.json (no binding moment when kappa* = 0, as in
    Table D.1)."""
    bad = []
    pid = {"P1": "two-way lag", "P2": "lag-8", "P3": "lag-2 given lag 1", "P4": "gain", "P5": "one-week spells"}
    for arm in ("logit_normal", "mixture"):
        s = stored[arm]
        for k in ("0.15", "0.25", "0.35"):
            if rec[arm]["comparative"][k] != s["comparative"]["verdict"][k]:
                bad.append(f"{arm} comparative at {k}")
            if rec[arm]["A1"][k] != {"INADEQ": "INADEQUATE", "ADEQ": "ADEQUATE"}.get(s["A1"]["status"][k], "INCONCLUSIVE"):
                bad.append(f"{arm} A1 at {k}")
            for f in ("A1_PRIME", "A2"):
                if rec[arm][f]["verdict"][k] != s["families"][f]["kappas"][k]["verdict"]:
                    bad.append(f"{arm} {f} at {k}")
                if rec[arm][f]["level2"][k] != stored["secondary_level2"][arm][f][k]["verdict"]:
                    bad.append(f"{arm} {f} level-2 at {k}")
        for f in ("A1_PRIME", "A2"):
            fr = s["families"][f]["kappas"]["0.25"]
            if abs(rec[arm][f]["kappa_star"] - fr["kappa_star_F"]) > 1e-9:
                bad.append(f"{arm} {f} kappa*")
            want = None if fr["kappa_star_F"] == 0 else pid[fr["binding_moment"]]
            if rec[arm][f]["binding"] != want:
                bad.append(f"{arm} {f} binding moment")
    return bad
