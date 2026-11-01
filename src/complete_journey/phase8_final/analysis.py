"""Parent-side, deterministic analysis of the final run (no simulation here).

Builds the plug-in model tables from imported S1A images and final-run jobs, then the
Stage 1 gates, RQ2 operating characteristics, calibration-noise checks and, after the
guarded reveal, the real comparison and the S-a bracket.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from complete_journey.abm import config as abm_config
from complete_journey.abm import rq2
from complete_journey.abm import stage1_p8005 as s1p
from complete_journey.abm.adequacy import (
    ADEQ,
    HISTORY_FAVOURED,
    INADEQ,
    comparative_verdict,
    kappa_robustness,
)
from complete_journey.abm.config import require
from complete_journey.abm.emulator import LOGIT_NORMAL, MIXTURE
from complete_journey.abm.representability import (
    EXCLUDED,
    INDETERMINATE,
    INFEASIBLE_TRUTH,
    REPRESENTABLE,
    mde80_diag,
)
from complete_journey.phase8_final import settings as S
from complete_journey.phase8_final.evaluate import (
    Entry,
    FamilyResult,
    binding_moment,
    classify_direct,
    evaluate_family,
    gate_record_inputs,
    mde_bracket,
    outcome,
    pooled_stats,
    rate,
)
from complete_journey.phase8_final.jobs import (
    ARMS,
    FAMILIES,
    FEASIBLE,
    FJob,
    gate_indices,
    rq2_indices,
    truth_payload,
)
from complete_journey.phase8_final.store import legacy_load

LATTICE = tuple(float(i) for i in range(abm_config.GRID_SIZE))
PRIMARY = abm_config.PRIMARY_IDS


def vec(d: dict) -> np.ndarray:
    return np.array([d[j] for j in PRIMARY], float)


# --- job lists ----------------------------------------------------------------------


def coarse_feasible(arm: str, family: str) -> dict[tuple[float, float], bool]:
    return {
        (ip, ia): bool(legacy_load(s1p.Job("map", arm, family, ip, ia))["ci1_feasible"])
        for ip in LATTICE
        for ia in LATTICE
    }


def pruned_refinement_points(arm: str, family: str) -> list[tuple[float, float]]:
    feas = coarse_feasible(arm, family)
    pts = set()
    for i in range(abm_config.GRID_SIZE - 1):
        for j in range(abm_config.GRID_SIZE - 1):
            if any(feas[(float(i + di), float(j + dj))] for di in (0, 1) for dj in (0, 1)):
                for c in (
                    (i + 0.5, j),
                    (i, j + 0.5),
                    (i + 1, j + 0.5),
                    (i + 0.5, j + 1),
                    (i + 0.5, j + 0.5),
                ):
                    pts.add((float(c[0]), float(c[1])))
    return sorted(pts)


def ext_jobs() -> list[FJob]:
    jobs = [FJob("ext", MIXTURE, "A1")]
    for arm in ARMS:
        for f in FAMILIES:
            jobs += [FJob("ext", arm, f, ip, ia) for ip in LATTICE for ia in LATTICE]
    return jobs


def ref_jobs() -> list[FJob]:
    return [
        FJob("ref", arm, f, a, b)
        for arm in ARMS
        for f in FAMILIES
        for a, b in pruned_refinement_points(arm, f)
    ]


def gate_feasible(arm: str) -> list[int]:
    return [
        i
        for i in sorted(gate_indices())
        if legacy_load(s1p.Job("truth", arm, "TRUTH", index=i))["status"] == FEASIBLE
    ]


def theta_jobs() -> list[FJob]:
    pts = rq2.production_truth_points()
    return [
        FJob("theta", arm, index=i)
        for arm in ARMS
        for i in gate_feasible(arm)
        if pts[i].kind != "A1"
    ]


def gpanel_jobs() -> list[FJob]:
    return [FJob("gpanels", MIXTURE, index=i) for i in gate_feasible(MIXTURE)]


def truth_jobs(arm: str) -> list[FJob]:
    return [FJob("truth", arm, index=i) for i in rq2_indices() if i not in gate_indices()]


def feasible_rq2(arm: str, store) -> list[int]:
    return [i for i in rq2_indices() if truth_payload(arm, i, store)["status"] == FEASIBLE]


def panel_jobs(arm: str, store, block: str) -> list[FJob]:
    n_chunks = (S.Q if block == "rq2" else S.Q_POP) // S.PANEL_CHUNK
    return [
        FJob("panels", arm, index=i, sub=c, block=block)
        for i in feasible_rq2(arm, store)
        for c in range(n_chunks)
    ]


def successful_rq2_q(store, arm: str, index: int) -> list[int]:
    """Panel indices of the RQ2 block that produced moments, in q order."""
    out = []
    for c in range(S.Q // S.PANEL_CHUNK):
        rec = store.load(FJob("panels", arm, index=index, sub=c, block="rq2").key())
        require(rec is not None, "calibration-noise checks need the RQ2 panels")
        out += [p["q"] for p in rec["payload"]["panels"]]
    return sorted(out)


def calnoise_jobs(store) -> list[FJob]:
    """Predeclared checks on the first n SUCCESSFUL RQ2 panels (q order) of each truth."""
    labels = {p.label: i for i, p in enumerate(rq2.production_truth_points())}
    jobs = []
    for arm, family, truths, panels in S.CALNOISE_PLAN:
        for lab in truths:
            i = labels[lab]
            if truth_payload(arm, i, store)["status"] != FEASIBLE:
                continue
            for q in successful_rq2_q(store, arm, i)[:panels]:
                jobs += [
                    FJob("calnoise", arm, family, a, b, index=i, sub=q) for a, b in S.CALNOISE_THETA
                ]
    return jobs


# --- model tables -----------------------------------------------------------------


def _payload(store, job: FJob) -> dict:
    rec = store.load(job.key())
    require(rec is not None, f"missing final-run checkpoint {job}")
    return rec["payload"]


def coarse_entry(store, arm: str, family: str, ip: float, ia: float) -> Entry:
    m = legacy_load(s1p.Job("map", arm, family, ip, ia))
    ext = _payload(store, FJob("ext", arm, family, ip, ia))
    mean, sd, n = pooled_stats(
        m["replicates_ok"], vec(m["primary_mean"]), vec(m["primary_sd"]), ext["rows"]
    )
    return Entry(mean, sd, n, bool(m["ci1_feasible"]))


def a1_entry(store, arm: str) -> Entry:
    if arm == LOGIT_NORMAL:
        a1 = legacy_load(s1p.Job("a1", LOGIT_NORMAL, "A1"))
        return Entry(
            vec(a1["primary_mean"]),
            vec(a1["primary_sd"]),
            a1["replicates_ok"],
            bool(a1["ci1_feasible"]),
        )
    m = legacy_load(s1p.Job("map", MIXTURE, "A1"))
    ext = _payload(store, FJob("ext", MIXTURE, "A1"))
    mean, sd, n = pooled_stats(
        m["replicates_ok"], vec(m["primary_mean"]), vec(m["primary_sd"]), ext["rows"]
    )
    return Entry(mean, sd, n, bool(m["ci1_feasible"]))


def ref_entry(payload: dict) -> Entry:
    return Entry(
        vec(payload["primary_mean"]),
        vec(payload["primary_sd"]),
        payload["replicates_ok"],
        bool(payload["ci1_feasible"]),
    )


def build_tables(store, arm: str) -> dict:
    out = {"A1": a1_entry(store, arm)}
    for f in FAMILIES:
        coarse = {(ip, ia): coarse_entry(store, arm, f, ip, ia) for ip in LATTICE for ia in LATTICE}
        refined = {
            c: ref_entry(_payload(store, FJob("ref", arm, f, *c)))
            for c in pruned_refinement_points(arm, f)
        }
        out[f] = {"coarse": coarse, "refined": refined}
    return out


def theta_entry(store, arm: str, index: int) -> Entry:
    p = _payload(store, FJob("theta", arm, index=index))
    return Entry(vec(p["primary_mean"]), vec(p["primary_sd"]), p["replicates_ok"], True)


# --- panel evaluation ---------------------------------------------------------------


def panel_arrays(panels: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array([p["primary"] for p in panels], float)
    return arr, arr.std(axis=0, ddof=1)


def evaluate_panel(tables, data, se, tol):
    res = {
        f: evaluate_family(f, tables[f]["coarse"], tables[f]["refined"], data, se, tol)
        for f in FAMILIES
    }
    a1_out, _ = outcome((0.0, 0.0), tables["A1"], data, se, tol)
    return res, a1_out


def comparatives(res, tol) -> dict[float, str]:
    return {
        k: comparative_verdict(res["A2"].verdict(k, tol), res["A1_PRIME"].verdict(k, tol)).verdict
        for k in abm_config.KAPPAS
    }


# --- Stage 1 -------------------------------------------------------------------------


def gate_panels(store, arm: str, index: int) -> list[dict]:
    if arm == LOGIT_NORMAL:
        return legacy_load(s1p.Job("panels", arm, "TRUTH", index=index))["panels"]
    return _payload(store, FJob("gpanels", arm, index=index))["panels"]


def stage1_report(store, tol) -> dict:
    pts = rq2.production_truth_points()
    out, passed = {}, True
    for arm in ARMS:
        tables = build_tables(store, arm)
        fam_records = defaultdict(list)
        infeasible = defaultdict(list)
        for i in sorted(gate_indices()):
            p = pts[i]
            fam = {"T0": "A1", "T1": "A1_PRIME", "T3": "A2"}[p.truth_class]
            truth = legacy_load(s1p.Job("truth", arm, "TRUTH", index=i))
            if truth["status"] != FEASIBLE:
                infeasible[fam].append(p.label)
                continue
            panels = gate_panels(store, arm, i)
            arr, se = panel_arrays(panels)
            theta = tables["A1"] if fam == "A1" else theta_entry(store, arm, i)
            coord = (0.0, 0.0) if fam == "A1" else tuple(truth["coord"])
            for row in arr:
                res, a1_out = evaluate_panel(tables, row, se, tol)
                t_out, _ = outcome(coord, theta, row, se, tol)
                true_res = res[fam] if fam != "A1" else FamilyResult("A1", (a1_out,), 0)
                g = gate_record_inputs(true_res, t_out, res["A1_PRIME"], res["A2"], tol)
                fam_records[fam].append(
                    rq2.GateRecord(
                        p.label,
                        p.truth_class,
                        g["truth_in_acc"],
                        g["family_verdict"],
                        g["truth_in_nr"],
                        g["comparative"],
                        g["recovery_distance"],
                    )
                )
        arm_out = {}
        for fam in ("A1", "A1_PRIME", "A2"):
            recs = fam_records[fam]
            gate = rq2.perfect_model_gate(recs) if recs else {"points": {}, "pass": False}
            ok = bool(gate["pass"] and recs)
            arm_out[fam] = {
                "gate": gate,
                "infeasible_truths": infeasible[fam],
                "assessable_panels": len(recs),
                "pass": ok,
            }
            passed = passed and ok
        out[arm] = arm_out
    out["pass"] = passed
    return out


# --- RQ2 ----------------------------------------------------------------------------


def block_panels(store, arm: str, index: int, block: str) -> list[dict]:
    n_chunks = (S.Q if block == "rq2" else S.Q_POP) // S.PANEL_CHUNK
    out = []
    for c in range(n_chunks):
        out += _payload(store, FJob("panels", arm, index=index, sub=c, block=block))["panels"]
    return out


def naive_flags(row: np.ndarray) -> tuple[bool, bool]:
    m3 = row[0] > abm_config.NAIVE_Z * abm_config.NAIVE_M3_SE
    rec = row[3] > abm_config.NAIVE_Z * abm_config.NAIVE_REC_SE
    return bool(m3), bool(rec)


def rq2_truth_eval(store, arm: str, index: int, tables, tol) -> dict:
    p = rq2.production_truth_points()[index]
    rows, se = panel_arrays(block_panels(store, arm, index, "rq2"))
    pop, pop_sd = panel_arrays(block_panels(store, arm, index, "pop"))
    competitor = "A2" if p.truth_class in rq2.NON_HISTORY_CLASSES else "A1_PRIME"
    image = list(tables[competitor]["coarse"].values()) + list(
        tables[competitor]["refined"].values()
    )
    states = {
        k: classify_direct(pop.mean(0), pop_sd, len(pop), image, tol, k) for k in abm_config.KAPPAS
    }
    hf = {k: [] for k in abm_config.KAPPAS}
    comp_counts = {k: Counter() for k in abm_config.KAPPAS}
    fam_counts = {f: {k: Counter() for k in abm_config.KAPPAS} for f in ("A1", *FAMILIES)}
    naive = []
    for row in rows:
        res, a1_out = evaluate_panel(tables, row, se, tol)
        comp = comparatives(res, tol)
        for k in abm_config.KAPPAS:
            hf[k].append(comp[k] == HISTORY_FAVOURED)
            comp_counts[k][comp[k]] += 1
            for f in FAMILIES:
                fam_counts[f][k][res[f].verdict(k, tol).verdict] += 1
            a1v = (
                "ADEQUATE"
                if (a1_out.feasible and a1_out.status[k] == ADEQ)
                else (
                    "INADEQUATE"
                    if (not a1_out.feasible or a1_out.status[k] == INADEQ)
                    else "INCONCLUSIVE"
                )
            )
            fam_counts["A1"][k][a1v] += 1
        naive.append(naive_flags(row))
    return {
        "label": p.label,
        "truth_class": p.truth_class,
        "kind": p.kind,
        "lam": dict(p.params).get("lam"),
        "delta1_pp": dict(p.params).get("delta1_pp"),
        "q": len(rows),
        "q_pop": len(pop),
        "competitor": competitor,
        "states": states,
        "hf": {str(k): rate(v) for k, v in hf.items()},
        "comparative_counts": {str(k): dict(v) for k, v in comp_counts.items()},
        "family_verdict_counts": {
            f: {str(k): dict(c) for k, c in v.items()} for f, v in fam_counts.items()
        },
        "naive": {
            "union": rate([a or b for a, b in naive]),
            "m3": rate([a for a, _ in naive]),
            "rec": rate([b for _, b in naive]),
        },
    }


def rq2_report(store, tol, per_truth: dict) -> dict:
    """Aggregate FA / DET / MDE / representability per arm from per-truth evaluations."""
    ref = abm_config.KAPPA_REFERENCE
    out = {}
    for arm in ARMS:
        rows = per_truth[arm]
        infeasible = [t for t in rows if rows[t].get("status") == INFEASIBLE_TRUTH]
        ev = {t: r for t, r in rows.items() if r.get("status") != INFEASIBLE_TRUTH}
        nh = {t: r for t, r in ev.items() if r["truth_class"] in rq2.NON_HISTORY_CLASSES}
        fa = {}
        for k in abm_config.KAPPAS:
            ks = str(k)
            overall = {t: r["hf"][ks] for t, r in nh.items()}
            rep = {t: v for t, v in overall.items() if nh[t]["states"][k] == REPRESENTABLE}
            t2 = [v["rate"] for t, v in overall.items() if nh[t]["truth_class"] == "T2"]
            t2r = [v["rate"] for t, v in rep.items() if nh[t]["truth_class"] == "T2"]
            shares = Counter(nh[t]["states"][k] for t in nh)
            fa[ks] = {
                "FA_PROC_OVERALL": {
                    "per_point": overall,
                    "max_t2": max(t2) if t2 else None,
                    "max_non_history": (
                        max(v["rate"] for v in overall.values()) if overall else None
                    ),
                },
                "FA_PROC_REP": {
                    "per_point": rep,
                    "max_t2": max(t2r) if t2r else None,
                    "max_non_history": (max(v["rate"] for v in rep.values()) if rep else None),
                },
                "shares": {s: shares.get(s, 0) for s in (REPRESENTABLE, EXCLUDED, INDETERMINATE)},
                "infeasible_truths": [
                    t for t in infeasible if rows[t]["truth_class"] in rq2.NON_HISTORY_CLASSES
                ],
            }
        fa_naive = {t: r["naive"] for t, r in nh.items()}
        hist = {t: r for t, r in ev.items() if r["truth_class"] in ("T3", "T4")}
        det = {}
        for k in abm_config.KAPPAS:
            ks = str(k)
            det[ks] = {
                "DET_OVERALL": {t: r["hf"][ks] for t, r in hist.items()},
                "DET_REP": {
                    t: r["hf"][ks] for t, r in hist.items() if r["states"][k] == REPRESENTABLE
                },
                "shares": dict(Counter(r["states"][k] for r in hist.values())),
            }
        max_fa = fa[str(ref)]["FA_PROC_OVERALL"]["max_non_history"]
        max_fa = 1.0 if max_fa is None else max_fa  # no assessable FA truth: not admissible
        mde = {}
        for lam in abm_config.LAMBDA_TRUTH_GRID:
            t3 = {
                r["delta1_pp"]: r
                for t, r in rows.items()
                if r.get("truth_class") == "T3" and r.get("lam") == lam
            }
            grid = list(abm_config.DELTA1_GRID_PP[1:])
            reach = [d for d in grid if d in t3 and t3[d].get("status") != INFEASIBLE_TRUTH]
            infeas = frozenset(d for d in grid if d not in reach)
            detk = {(lam, d): t3[d]["hf"][str(ref)]["rate"] for d in reach}
            counts = {d: (t3[d]["hf"][str(ref)]["k"], t3[d]["hf"][str(ref)]["n"]) for d in reach}
            try:
                m80 = rq2.mde80(detk, lam, max_fa, infeas)
            except Exception as exc:  # reachability not monotone: report, never extrapolate
                m80 = f"UNDEFINED ({exc})"
            repr_set = frozenset(d for d in reach if t3[d]["states"][ref] == REPRESENTABLE)
            mde[str(lam)] = {
                "MDE_80": m80,
                "MDE_80_bracket_lower_bound_rule": (
                    rq2.UNDEFINED_MDE
                    if max_fa > abm_config.FA_ADMISSIBILITY
                    else mde_bracket(counts, reach)
                ),
                "MDE_80_DIAG": mde80_diag(detk, lam, max_fa, repr_set),
                "reachable": reach,
                "infeasible": sorted(infeas),
            }
        cross = {}
        for t, r in ev.items():
            cross[t] = {"state": r["states"][ref], "comparative": r["comparative_counts"][str(ref)]}
        out[arm] = {
            "FA": fa,
            "FA_NAIVE": fa_naive,
            "DET": det,
            "MDE": mde,
            "state_x_verdict": cross,
            "infeasible_truths": infeasible,
            "admissible": max_fa <= abm_config.FA_ADMISSIBILITY,
        }
    a, b = out[LOGIT_NORMAL], out[MIXTURE]
    same_adm = a["admissible"] == b["admissible"]
    same_mde = all(a["MDE"][lam]["MDE_80"] == b["MDE"][lam]["MDE_80"] for lam in a["MDE"])
    out["heterogeneity"] = {
        "rule": "ROBUST iff FA admissibility and MDE_80 at every λ agree across arms "
        "(P8-006, predeclared)",
        "admissibility_agrees": same_adm,
        "mde80_agrees": same_mde,
        "label": S.RQ2_LABELS["robust"] if (same_adm and same_mde) else S.RQ2_LABELS["dependent"],
    }
    return out


# --- calibration-noise checks -----------------------------------------------------


def calnoise_report(store, tol, per_panel_se: dict) -> dict:
    tau = tol.tau(abm_config.KAPPA_REFERENCE)
    out = {}
    for arm, family, _truths, _panels in S.CALNOISE_PLAN:
        key = f"{arm}:{family}"
        shifts, agree, flips, pairs, feas_mismatch = [], {k: [] for k in abm_config.KAPPAS}, 0, 0, 0
        for job in [j for j in calnoise_jobs(store) if j.arm == arm and j.family == family]:
            p = _payload(store, job)
            m = legacy_load(s1p.Job("map", arm, family, job.a, job.b))
            plug = Entry(
                vec(m["primary_mean"]),
                vec(m["primary_sd"]),
                m["replicates_ok"],
                bool(m["ci1_feasible"]),
            )
            chunk = _payload(
                store,
                FJob("panels", arm, index=job.index, sub=job.sub // S.PANEL_CHUNK, block="rq2"),
            )
            panel = next(x for x in chunk["panels"] if x["q"] == job.sub)
            data = np.array(panel["primary"], float)
            se = per_panel_se[(arm, job.index)]
            pairs += 1
            if p["ci1_feasible"]:
                recal = Entry(
                    vec(p["primary_mean"]), vec(p["primary_sd"]), p["replicates_ok"], True
                )
            else:
                recal = Entry(plug.mean, plug.sd, plug.replicates, False)
            if recal.feasible and plug.feasible:
                shifts.append(np.abs(recal.mean - plug.mean) / tau)
            if recal.feasible != plug.feasible:
                feas_mismatch += 1
            o1, _ = outcome((job.a, job.b), recal, data, se, tol)
            o2, _ = outcome((job.a, job.b), plug, data, se, tol)
            if o1.feasible and o2.feasible:  # statuses exist only for feasible points
                for k in abm_config.KAPPAS:
                    agree[k].append(o1.status[k] == o2.status[k])
                    if {o1.status[k], o2.status[k]} == {ADEQ, INADEQ}:
                        flips += 1
        sh = np.array(shifts) if shifts else np.zeros((0, 5))
        p90 = np.quantile(sh, 0.9, axis=0).tolist() if len(sh) else None
        disagree = {str(k): 1.0 - float(np.mean(v)) for k, v in agree.items() if v}
        mismatch_rate = feas_mismatch / pairs if pairs else 0.0
        sensitive = bool(
            (p90 is not None and max(p90) > S.CALNOISE_SHIFT_P90_MAX)
            or any(d > S.CALNOISE_DISAGREE_MAX for d in disagree.values())
            or mismatch_rate > S.CALNOISE_DISAGREE_MAX
        )

        out[key] = {
            "pairs": pairs,
            "feasible_pairs": int(len(sh)),
            "feasibility_mismatch": feas_mismatch,
            "feasibility_mismatch_rate": mismatch_rate,
            "shift_over_tau_median": np.median(sh, axis=0).tolist() if len(sh) else None,
            "shift_over_tau_p90": p90,
            "shift_over_tau_max": sh.max(axis=0).tolist() if len(sh) else None,
            "status_disagreement": disagree,
            "adeq_inadeq_flips": flips,
            "qualifier": S.CALNOISE_LABEL if sensitive else None,
        }
    return out


def h2_evidence(tol) -> dict:
    """Recorded A1′ logit-normal evidence: S1-P8005 H2 per-panel recalibration (direct,
    R = 25) against the plug-in image (S1A, R = 50) at the same θ and panel."""
    tau = tol.tau(abm_config.KAPPA_REFERENCE)
    shifts, agree, trans = [], {k: [] for k in abm_config.KAPPAS}, Counter()
    panels_cache = {}
    for job in [j for j in s1p.h2_jobs(LOGIT_NORMAL) if j.family == "A1_PRIME"]:
        rec = s1p.store().load(job.key())
        if rec is None:
            continue
        p = rec["payload"]
        if not p["ci1_feasible"]:
            continue
        ip, ia = p["coord"]
        m = legacy_load(s1p.Job("map", LOGIT_NORMAL, "A1_PRIME", ip, ia))
        if p["truth_index"] not in panels_cache:
            panels_cache[p["truth_index"]] = legacy_load(
                s1p.Job("panels", LOGIT_NORMAL, "TRUTH", index=p["truth_index"])
            )["panels"]
        arr, se = panel_arrays(panels_cache[p["truth_index"]])
        data = arr[p["position"]]
        a = Entry(vec(p["primary_mean"]), vec(p["primary_sd"]), 25, True)
        b = Entry(vec(m["primary_mean"]), vec(m["primary_sd"]), 50, True)
        shifts.append(np.abs(a.mean - b.mean) / tau)
        oa, _ = outcome((ip, ia), a, data, se, tol)
        ob, _ = outcome((ip, ia), b, data, se, tol)
        for k in abm_config.KAPPAS:
            agree[k].append(oa.status[k] == ob.status[k])
        trans[(oa.status[abm_config.KAPPA_REFERENCE], ob.status[abm_config.KAPPA_REFERENCE])] += 1
    sh = np.array(shifts)
    flips = sum(v for (x, y), v in trans.items() if {x, y} == {ADEQ, INADEQ})
    return {
        "pairs": int(len(sh)),
        "shift_over_tau_median": np.median(sh, 0).tolist(),
        "shift_over_tau_p90": np.quantile(sh, 0.9, 0).tolist(),
        "shift_over_tau_max": sh.max(0).tolist(),
        "status_agreement": {str(k): float(np.mean(v)) for k, v in agree.items()},
        "transitions_kappa_0.25": {f"{x}->{y}": v for (x, y), v in trans.items()},
        "adeq_inadeq_flips": int(flips),
        "note": "recalibrated side R = 25, plug-in side R = 50 (CRN streams 0..24 shared)",
    }


# --- real comparison (post-reveal only; callers pass the guarded real targets) ------


def real_family(f: str, tables, data, se, tol, entries_override=None, levels=1) -> dict:
    coarse = dict(tables[f]["coarse"])
    refined = dict(tables[f]["refined"])
    if entries_override:
        refined.update({c: e for c, e in entries_override.items() if c in refined})
        coarse.update({c: e for c, e in entries_override.items() if c in coarse})
    res = evaluate_family(f, coarse, refined, data, se, tol, levels=levels)
    out = {"refinement_points": res.refinement_points, "kappas": {}}
    allpts = {**coarse, **refined}
    for k in abm_config.KAPPAS:
        v = res.verdict(k, tol)
        acc = [o.coord for o in res.outcomes if o.feasible and o.status[k] == ADEQ]
        nr = [o.coord for o in res.outcomes if o.feasible and o.status[k] != INADEQ]
        out["kappas"][str(k)] = {
            "verdict": v.verdict,
            "label": v.label,
            "boundary_dependent": v.boundary_dependent,
            "kappa_star_F": v.kappa_star,
            "argmin": v.argmin,
            "binding_moment": (
                binding_moment(allpts[v.argmin], data, se, tol) if v.argmin is not None else None
            ),
            "resolution_limited": list(v.resolution_limited),
            "acceptance_set": acc,
            "not_clearly_inadequate": nr,
            "ci1_feasible_points": sum(o.feasible for o in res.outcomes),
        }
    out["images"] = {
        f"{c[0]:g},{c[1]:g}": {
            "mean": e.mean.tolist(),
            "sd": e.sd.tolist(),
            "R": e.replicates,
            "feasible": e.feasible,
        }
        for c, e in sorted(allpts.items())
        if c in {o.coord for o in res.outcomes}
    }
    out["ci1_feasible_set"] = [o.coord for o in res.outcomes if o.feasible]
    out["_result"] = res
    return out


def audit_points(fam_out: dict, k_ref: str) -> list[tuple[float, float]]:
    """κ* argmin plus up to 10 accepted points (smallest κ* first), frozen §AC rule."""
    res = fam_out["_result"]
    acc = sorted(
        [o for o in res.outcomes if o.feasible and o.status[float(k_ref)] == ADEQ],
        key=lambda o: (o.kappa_star, o.coord),
    )[: S.AUDIT_MAX_ACCEPTED]
    pts = [o.coord for o in acc]
    am = fam_out["kappas"][k_ref]["argmin"]
    if am is not None and tuple(am) not in pts:
        pts.insert(0, tuple(am))
    return pts


def comparative_block(fams: dict, tol) -> dict:
    comps = {}
    for k in abm_config.KAPPAS:
        c = comparative_verdict(
            fams["A2"]["_result"].verdict(k, tol), fams["A1_PRIME"]["_result"].verdict(k, tol)
        )
        comps[k] = c
    return {
        "verdict": {str(k): c.verdict for k, c in comps.items()},
        "label": {str(k): c.label for k, c in comps.items()},
        "kappa_robustness": kappa_robustness(comps),
    }


def strip(d):
    if isinstance(d, dict):
        return {
            (k if isinstance(k, str) else str(k)): strip(v)
            for k, v in d.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }
    if isinstance(d, (list, tuple)):
        return [strip(v) for v in d]
    if isinstance(d, float) and not math.isfinite(d):
        return str(d)
    return d
