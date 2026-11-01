"""P8-005 implementation preflight (Run C, Part I): tiny synthetic fixtures C1–C23.

SYNTHETIC NON-EVIDENCE. Every fixture uses stage-0 synthetic targets or toy records,
tiny N and R, and returns a dict with a ``pass`` flag. None is a scientific result,
and none touches real dynamic targets or the S1-P8005 namespace except C23, which
checks that the namespaces cannot collide.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager

import numpy as np

from complete_journey.abm import architectures as arch
from complete_journey.abm import config, frontier, mixture, representability, rq2, span, stage1
from complete_journey.abm import emulator as em
from complete_journey.abm import stage1_p8005 as s1p
from complete_journey.abm.adequacy import HISTORY_FAVOURED, LATENT_FAVOURED, Tolerances
from complete_journey.abm.calibration import calibrate, simulated_ci1
from complete_journey.abm.config import Phase8ContractError
from complete_journey.abm.guards import TargetVector
from complete_journey.abm.seeds import Role, SeedKey, Stream, truth_key

STAGE0 = 0


def _key(rep: int = 0, role: Role = Role.DATA) -> SeedKey:
    return SeedKey(stage=STAGE0, role=role, replicate=rep)


def _target(spec, het, n: int, reps: int) -> TargetVector:
    sim = simulated_ci1(spec, het, STAGE0, n, reps)
    return TargetVector("synthetic", dict(zip(config.CI1_COMPONENTS, sim, strict=True)))


def toy_tolerances() -> Tolerances:
    return Tolerances(
        {"P1": 0.10, "P2": 0.08, "P3": 0.10, "P4": 0.03, "P5": 0.05},
        {"P1": 0.01, "P2": 0.01, "P3": 0.01, "P4": 0.004, "P5": 0.01},
    )


def c1_mixture_simulation(n: int = 40_000) -> dict:
    het = arch.MixtureHeterogeneity(0.3, -1.0, 2.5, 0.6)
    world = arch.simulate(arch.a1(), het, _key(), n)
    comps = _key().generator(Stream.MIXTURE).random(n)
    b = comps < 1 - het.pi
    a = world.alpha
    ok = {
        "component_share": abs(b.mean() - (1 - het.pi)) < 0.01,
        "mean": abs(a.mean() - het.mean) < 0.03,
        "variance": abs(a.var() - het.variance) < 0.06,
        "component_means_ordered": a[~b].mean() < a[b].mean(),
        "within_sd": abs(a[b].std() - het.s) < 0.02 and abs(a[~b].std() - het.s) < 0.02,
    }
    ln = arch.simulate(arch.a1(), arch.Heterogeneity(0.2, 1.1), _key(), 500)
    ln2 = arch.simulate(arch.a1(), arch.Heterogeneity(0.2, 1.1), _key(), 500)
    ok["logit_normal_unchanged"] = np.array_equal(ln.behavior, ln2.behavior)
    return {"checks": ok, "pass": all(ok.values())}


def c2_recovery(n: int = 1500, r_cal: int = 4) -> dict:
    truth = arch.MixtureHeterogeneity(0.3, -1.0, 2.5, 0.6)
    target = _target(arch.a1(), truth, n, 40)
    prim = calibrate(arch.a1(), target, STAGE0, n, r_cal)
    cal = mixture.calibrate_mixture(arch.a1(), target, STAGE0, prim.heterogeneity, n, r_cal)
    h = cal.heterogeneity
    err = {
        "pi": abs(h.pi - truth.pi),
        "mu1": abs(h.mu1 - truth.mu1),
        "d": abs(h.d - truth.d),
        "s": abs(h.s - truth.s),
    }
    ok = cal.feasible and err["pi"] < 0.08 and err["d"] < 0.4 and err["s"] < 0.2
    return {
        "truth": truth.as_dict(),
        "fit": h.as_dict(),
        "abs_error": err,
        "feasible": cal.feasible,
        "primary_feasible": prim.feasible,
        "evaluations": cal.evaluations,
        "pass": bool(ok),
    }


def c3_label_switching(samples: int = 2000) -> dict:
    rng = np.random.default_rng(3)
    hets = [mixture.from_unconstrained(rng.normal(0, 4, 4)) for _ in range(samples)]
    ordered = all(
        h.mu1 < h.mu2 and h.d >= mixture.D_LO and h.mu2 <= mixture.MU_HI + 1e-12 for h in hets
    )
    starts = mixture.starts(arch.Heterogeneity(0.2, 1.5))
    starts_ok = all(s.mu1 < s.mu2 for s in starts) and len({s.pi for s in starts}) == 3
    round_trip = all(
        np.allclose(mixture.to_unconstrained(mixture.from_unconstrained(u)), u, atol=1e-6)
        for u in rng.normal(0, 2, (50, 4))
    )
    return {
        "ordered": ordered,
        "starts_ordered": starts_ok,
        "round_trip": round_trip,
        "pass": bool(ordered and starts_ok and round_trip),
    }


def c4_weak_separation(n: int = 1500, r_cal: int = 4) -> dict:
    truth = arch.MixtureHeterogeneity(0.5, -0.2, 0.4, 1.2)  # d <= 2s: unimodal
    target = _target(arch.a1(), truth, n, 40)
    prim = calibrate(arch.a1(), target, STAGE0, n, r_cal)
    cal = mixture.calibrate_mixture(arch.a1(), target, STAGE0, prim.heterogeneity, n, r_cal)
    flagged = mixture.UNIMODAL in cal.flags or mixture.MIXTURE_BOUNDARY in cal.flags
    return {
        "flags": list(cal.flags),
        "feasible": cal.feasible,
        "fit": cal.heterogeneity.as_dict(),
        "pass": bool(flagged and cal.feasible),
    }


def c5_a1_mixture_gate(n: int = 1200, r_cal: int = 3) -> dict:
    good = _target(arch.a1(), arch.Heterogeneity(0.3, 1.2), n, 20)
    prim = calibrate(arch.a1(), good, STAGE0, n, r_cal)
    cal = mixture.calibrate_mixture(arch.a1(), good, STAGE0, prim.heterogeneity, n, r_cal)
    impossible = TargetVector(
        "synthetic", {"mean": 0.5, "q10": 0.6, "q25": 0.2, "q50": 0.5, "q75": 0.4, "q90": 0.95}
    )  # non-monotone quantiles
    prim_bad = calibrate(arch.a1(), impossible, STAGE0, n, r_cal)
    bad = mixture.calibrate_mixture(arch.a1(), impossible, STAGE0, prim_bad.heterogeneity, n, r_cal)
    lab_good = representability.heterogeneity_label("X", "X", cal.feasible)
    lab_bad = representability.heterogeneity_label("X", None, bad.feasible)
    return {
        "available_case_feasible": cal.feasible,
        "impossible_case_feasible": bad.feasible,
        "labels": [lab_good, lab_bad],
        "pass": bool(
            cal.feasible
            and not bad.feasible
            and mixture.INFEASIBLE in bad.flags
            and lab_bad == representability.BRACKET_UNAVAILABLE
        ),
    }


def _shared_alpha(spec_a, spec_b, het, n=800):
    wa = arch.simulate(spec_a, het, _key(2), n)
    wb = arch.simulate(spec_b, het, _key(2), n)
    return wa, wb


def c6_a1_prime_mixture(n: int = 800) -> dict:
    het = arch.MixtureHeterogeneity(0.4, -1.2, 2.0, 0.7)
    w, w_a1 = _shared_alpha(arch.a1_prime(0.9, 1.0), arch.a1(), het, n)
    w2 = arch.simulate(arch.a1_prime(0.9, 1.0), het, _key(2), n)
    wl = arch.simulate(arch.a1_prime(0.9, 1.0), arch.Heterogeneity(0.0, 1.0), _key(2), n)
    ok = {
        "alpha_shared_with_a1": np.array_equal(w.alpha, w_a1.alpha),
        "deterministic": np.array_equal(w.behavior, w2.behavior),
        "latent_path_unchanged_by_layer": np.array_equal(w.latent, wl.latent),
        "rate_in_unit_interval": bool(0 < w.behavior.mean() < 1),
    }
    return {"checks": ok, "pass": all(ok.values())}


def c7_a2_mixture(n: int = 800) -> dict:
    het = arch.MixtureHeterogeneity(0.4, -1.2, 2.0, 0.7)
    w, w_lat = _shared_alpha(arch.a2(2.0, 0.6), arch.a1_prime(0.5, 0.5), het, n)
    h = w.history
    expected = np.zeros_like(h)
    expected[:, 1:] = 0.6 * h[:, :-1] + 0.4 * w.behavior[:, :-1]
    ok = {
        "alpha_shared_with_a1_prime": np.array_equal(w.alpha, w_lat.alpha),
        "stock_recursion": np.allclose(h[:, 1:], expected[:, 1:]),
        "stock_in_unit_interval": bool((h >= 0).all() and (h <= 1).all()),
    }
    return {"checks": ok, "pass": all(ok.values())}


def _truth(mean, sd=None, q=200):
    sd = np.full(5, 0.01) if sd is None else sd
    return representability.TruthMoments(np.asarray(mean, float), sd, q)


def _classify(image_means, kappa=0.25, image_sd=0.01):
    tol = toy_tolerances()
    truth = _truth([0.2, 0.1, 0.15, 0.02, 0.3])
    m = np.asarray(image_means, float).reshape(-1, 5)
    image = representability.CompetitorImage(m, np.full_like(m, image_sd))
    e = 0.1 * tol.tau(0.15)
    return representability.classify(truth, image, tol, e, kappa), tol, truth


def c8_representable() -> dict:
    st, _, _ = _classify([[0.2, 0.1, 0.15, 0.02, 0.3], [0.9, 0.9, 0.9, 0.9, 0.9]])
    return {"state": st, "pass": st == representability.REPRESENTABLE}


def c9_excluded() -> dict:
    st, _, _ = _classify([[0.6, 0.5, 0.6, 0.3, 0.8]])
    return {"state": st, "pass": st == representability.EXCLUDED}


def c10_indeterminate() -> dict:
    tol = toy_tolerances()
    tau = tol.tau(0.25)
    base = np.array([0.2, 0.1, 0.15, 0.02, 0.3])
    shifted = base.copy()
    shifted[0] += tau[0]  # sits on the tolerance edge: overlap of the margins
    st, _, _ = _classify([shifted])
    return {"state": st, "pass": st == representability.INDETERMINATE}


def c11_empty() -> dict:
    tol = toy_tolerances()
    image = representability.CompetitorImage(np.zeros((0, 5)), np.zeros((0, 5)))
    st = representability.classify(
        _truth([0.2, 0.1, 0.15, 0.02, 0.3]), image, tol, 0.1 * tol.tau(0.15), 0.25
    )
    return {"state": st, "pass": st == representability.EXCLUDED}


def _records(cls_label_rate):
    recs = []
    for (cls, label, lam, d1), rate in cls_label_rate.items():
        for q in range(20):
            verdict = HISTORY_FAVOURED if q < round(rate * 20) else LATENT_FAVOURED
            recs.append(
                rq2.VerdictRecord(
                    cls, label, q, dict.fromkeys(config.KAPPAS, verdict), 0.0, 0.0, lam, d1
                )
            )
    return recs


def c12_conditional_fa() -> dict:
    rates = {
        ("T1", "a", None, None): 0.05,
        ("T1", "b", None, None): 0.30,
        ("T2", "c", None, None): 0.50,
        ("T0", "A1", None, None): 0.0,
    }
    recs = _records(rates)
    rep, exc, ind = (
        representability.REPRESENTABLE,
        representability.EXCLUDED,
        representability.INDETERMINATE,
    )
    states = {
        ("T1", "a"): dict.fromkeys(config.KAPPAS, rep),
        ("T1", "b"): dict.fromkeys(config.KAPPAS, ind),
        ("T2", "c"): dict.fromkeys(config.KAPPAS, exc),
        ("T0", "A1"): dict.fromkeys(config.KAPPAS, rep),
    }
    out = representability.fa_proc_conditional(recs, states)
    overall = rq2.fa_proc(recs)
    ok = (
        out["FA_PROC_OVERALL"]["per_point"] == overall["per_point"]
        and set(out["FA_PROC_REP"]["per_point"]) == {("T1", "a"), ("T0", "A1")}
        and out["shares"]["counts"] == {rep: 2, exc: 1, ind: 1}
        and out["FA_PROC_REP"]["max_non_history"] == 0.05
    )
    return {
        "fa_rep": out["FA_PROC_REP"]["max_non_history"],
        "fa_overall": out["FA_PROC_OVERALL"]["max_non_history"],
        "shares": out["shares"],
        "pass": bool(ok),
    }


def c13_conditional_det() -> dict:
    rates = {("T3", "x", 0.5, 5.0): 0.4, ("T3", "y", 0.5, 10.0): 0.9, ("T3", "z", 0.5, 15.0): 0.95}
    recs = _records(rates)
    states = {
        ("T3", "x"): dict.fromkeys(config.KAPPAS, representability.EXCLUDED),
        ("T3", "y"): dict.fromkeys(config.KAPPAS, representability.REPRESENTABLE),
        ("T3", "z"): dict.fromkeys(config.KAPPAS, representability.INDETERMINATE),
    }
    out = representability.det_conditional(recs, states)
    ok = (
        out["DET_OVERALL"] == rq2.detection(recs)
        and list(out["DET_REP"]) == [(0.5, 10.0)]
        and out["shares"]["counts"][representability.INDETERMINATE] == 1
    )
    return {"det_rep": out["DET_REP"], "pass": bool(ok)}


def c14_mde_diag() -> dict:
    det = {
        (0.5, d): v
        for d, v in zip(config.DELTA1_GRID_PP[1:], (0.1, 0.3, 0.5, 0.85, 0.9, 0.95), strict=True)
    }
    full = rq2.mde80(det, 0.5, 0.05)
    diag = representability.mde80_diag(det, 0.5, 0.05, frozenset({2.5, 20.0, 30.0}))
    none = representability.mde80_diag(det, 0.5, 0.05, frozenset())
    inadmissible = representability.mde80_diag(det, 0.5, 0.2, frozenset({10.0}))
    ok = (
        full == 15.0
        and diag == 20.0
        and none == representability.NO_REPRESENTABLE
        and inadmissible == rq2.UNDEFINED_MDE
    )
    return {"mde80": full, "mde80_diag": diag, "none": none, "pass": bool(ok)}


def c15_frontier() -> dict:
    rows = [
        {
            "ip": 0,
            "ia": 0,
            "ci1_residuals": [0.005] * 6,
            "ci1_feasible": True,
            "images": {"P1": 0.1, "P2": 0.05, "P3": 0.1},
        },
        {
            "ip": 0,
            "ia": 1,
            "ci1_residuals": [0.03, 0, 0, 0, 0, 0],
            "ci1_feasible": False,
            "images": {"P1": 0.3, "P2": 0.2, "P3": 0.25},
        },
    ]
    snapshot = [dict(r) for r in rows]
    out = frontier.frontier(rows)
    ok = (
        out["points_total"] == 2
        and out["points_ci1_infeasible"] == 1
        and out["frontier"]["0.020"]["points"] == 1
        and out["frontier"]["0.030"]["P1"]["max"] == 0.3
        and out["is_gate"] is False
        and rows == snapshot
        and [r["ci1_feasible"] for r in rows] == [True, False]
    )
    return {"frontier_0.030": out["frontier"]["0.030"], "pass": bool(ok)}


def c16_span_map() -> dict:
    a = span.SimulatedImage("A", np.array([[0.1, 0.1, 0.1]]), np.full((1, 3), 0.001))
    b = span.SimulatedImage("B", np.array([[0.9, 0.9, 0.9]]), np.full((1, 3), 0.001))
    m = span.descriptive_map(a, b)
    gate = span.span_check(a, b)
    ok = (
        m["status"] == span.DESCRIPTIVE_MAP
        and m["coverage_a_in_b"] == 0.0
        and "keep_a" not in m
        and gate.status == span.STOP
    )
    return {"map_status": m["status"], "historical_gate_status": gate.status, "pass": bool(ok)}


def _toy_table(noise: np.ndarray) -> tuple[em.ReferenceTable, callable]:
    ips = ias = (0.0, 1.0, 2.0)

    def f(ip, ia, eta):
        mu, s = eta
        base = 0.1 + 0.02 * ip + 0.03 * ia + 0.01 * ip * ia + 0.004 * ip**2 + 0.05 * mu - 0.02 * s
        return np.full(5, base)

    k = 2
    design = np.zeros((3, 3, k + 1, k))
    mean, sd, ci1 = np.zeros((3, 3, 3, 5)), np.full((3, 3, 3, 5), 0.01), np.zeros((3, 3, 3, 6))
    for a, ip in enumerate(ips):
        for b, ia in enumerate(ias):
            nodes = em.design_nodes(em.LOGIT_NORMAL, np.array([0.0, 1.0]))
            design[a, b] = nodes
            for n_, node in enumerate(nodes):
                mean[a, b, n_] = f(ip, ia, node) + noise
                ci1[a, b, n_] = 0.5 + 0.1 * node[0] - 0.05 * node[1] + np.zeros(6)
    return em.ReferenceTable("A2", em.LOGIT_NORMAL, (ips, ias), design, mean, sd, ci1), f


def c17_c18_crn_toy(seed: int = 17) -> dict:
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.002, 5)  # Monte Carlo error shared by table and CRN reference
    table, f = _toy_table(common)
    tol = toy_tolerances()
    recs = []
    for p in ((0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (1.5, 1.5)):
        for k, off in enumerate(em.validation_offsets(em.LOGIT_NORMAL)):
            eta = np.array([0.0, 1.0]) + off
            mu, sd = em.predict(table, *p, eta)
            exact = f(*p, eta)
            recs.append(
                em.ValidationRecord(
                    "A2",
                    p,
                    k,
                    eta,
                    mu,
                    sd,
                    exact + common,
                    np.full(5, 0.01),
                    exact + rng.normal(0, 0.002, 5),
                    np.full(5, 0.01),
                )
            )
    crn = em.crn_gate(recs, tol)
    ind = em.independent_diagnostic(recs, tol)
    known = max(abs(0.004 * 0.25) for _ in recs)  # bilinear error of 0.004·ip² at cell centre
    crn_matches = np.allclose(crn["e"], known, atol=1e-12)
    return {
        "crn_max_error": crn["e"].tolist(),
        "known_interpolation_error": known,
        "independent_max_error": ind["e"].tolist(),
        "c17_pass": bool(crn_matches),
        "c18_pass": bool((ind["e"] > crn["e"]).all()),
        "pass": bool(crn_matches and (ind["e"] > crn["e"]).all()),
    }


def c19_stream_separation() -> dict:
    model = SeedKey(stage=config.P8005_STAGE_KEY, role=Role.MODEL, replicate=0)
    truth = truth_key(config.P8005_STAGE_KEY, "T1", 3, 0)
    rq2_truth = truth_key(3, "T1", 3, 0)
    streams = list(Stream)
    disjoint = all(model.spawn_key(s) != truth.spawn_key(s) for s in streams)
    stage_sep = all(truth.spawn_key(s) != rq2_truth.spawn_key(s) for s in streams)
    u_model = model.generator(Stream.BERNOULLI).random(5)
    u_truth = truth.generator(Stream.BERNOULLI).random(5)
    validation_roles = {Role.MODEL}  # P8-004 references: MODEL streams only
    ok = (
        disjoint
        and stage_sep
        and not np.allclose(u_model, u_truth)
        and Role.DATA not in (validation_roles)
    )
    return {
        "model_vs_truth_disjoint": disjoint,
        "gate_vs_rq2_disjoint": stage_sep,
        "pass": bool(ok),
    }


def c20_infeasible_truth() -> dict:
    states = {
        ("T3", "lam=0.8,D1=30.0"): representability.INFEASIBLE_TRUTH,
        ("T3", "lam=0.8,D1=5.0"): dict.fromkeys(config.KAPPAS, representability.REPRESENTABLE),
    }
    sh = representability.shares(states, sorted(states), 0.25)
    labels = {(0.8, 30.0): ("T3", "lam=0.8,D1=30.0"), (0.8, 5.0): ("T3", "lam=0.8,D1=5.0")}
    ann = representability.annotate_mde_grid(states, labels, 0.25)
    ok = (
        sh["infeasible_truths"] == 1
        and sh["assessed_points"] == 1
        and ann[(0.8, 30.0)] == representability.INFEASIBLE_TRUTH
        and len(ann) == 2
    )
    return {"shares": sh, "annotation": {str(k): v for k, v in ann.items()}, "pass": bool(ok)}


def c21_heterogeneity_dependent() -> dict:
    lab = representability.heterogeneity_label("HISTORY_FAVOURED", "NEITHER", True)
    robust = representability.heterogeneity_label("NEITHER", "NEITHER", True)
    return {
        "labels": [lab, robust],
        "pass": lab == representability.HET_DEPENDENT and robust == representability.ROBUST,
    }


def c22_bracket_unavailable() -> dict:
    lab = representability.heterogeneity_label("NEITHER", "NEITHER", False)
    return {"label": lab, "pass": lab == representability.BRACKET_UNAVAILABLE}


def c23_namespace() -> dict:
    refused = []
    for path in (
        f"{stage1.ARCHIVE}/x.json",
        f"{stage1.WORK}/checkpoints/x.json",
        stage1.SNAPSHOT,
        "docs/elsewhere.json",
    ):
        try:
            s1p.require_new_namespace(path)
            refused.append(False)
        except Phase8ContractError:
            refused.append(True)
    try:
        s1p.write_json(f"{stage1.ARCHIVE}/s1_span.json", {"x": 1})
        overwrite_refused = False
    except Phase8ContractError:
        overwrite_refused = True
    s1p.require_new_namespace(f"{s1p.ARCHIVE}/x.json")
    s1p.require_new_namespace(f"{s1p.WORK}/checkpoints/x.json")
    archived = s1p.verify_archived_stage1()
    ok = (
        all(refused)
        and overwrite_refused
        and s1p.STAGE != stage1.STAGE
        and s1p.ARCHIVE != stage1.ARCHIVE
        and len(archived) == 7
    )
    return {
        "old_paths_refused": refused,
        "overwrite_refused": overwrite_refused,
        "stage_keys": [stage1.STAGE, s1p.STAGE],
        "pass": bool(ok),
    }


FIXTURES = {
    "C1": c1_mixture_simulation,
    "C2": c2_recovery,
    "C3": c3_label_switching,
    "C4": c4_weak_separation,
    "C5": c5_a1_mixture_gate,
    "C6": c6_a1_prime_mixture,
    "C7": c7_a2_mixture,
    "C8": c8_representable,
    "C9": c9_excluded,
    "C10": c10_indeterminate,
    "C11": c11_empty,
    "C12": c12_conditional_fa,
    "C13": c13_conditional_det,
    "C14": c14_mde_diag,
    "C15": c15_frontier,
    "C16": c16_span_map,
    "C17-C18": c17_c18_crn_toy,
    "C19": c19_stream_separation,
    "C20": c20_infeasible_truth,
    "C21": c21_heterogeneity_dependent,
    "C22": c22_bracket_unavailable,
    "C23": c23_namespace,
}


def run_all() -> dict:
    out = {name: fn() for name, fn in FIXTURES.items()}
    out["all_pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


@contextmanager
def dry_run_patches():
    """In-process patches shrinking S1-P8005 to plumbing size (see ``pipeline_dry_run``)."""
    from unittest import mock

    from complete_journey.abm import procedure as proc
    from complete_journey.abm.checkpoint import CheckpointStore
    from complete_journey.paths import PROJECT_ROOT

    work = f"{s1p.WORK}/dry_run"
    root_dir = PROJECT_ROOT / work
    tmp_store = CheckpointStore(root_dir / "checkpoints", "dry-run", "dry-run")
    patches = [
        mock.patch.object(config, "P8005_STAGE1_AUTHORIZED", True),
        mock.patch.object(config, "N_HOUSEHOLDS", 600),
        mock.patch.object(config, "R_CAL", 2),
        mock.patch.object(config, "R_DIRECT", 3),
        mock.patch.object(config, "Q_GATE", 4),
        mock.patch.object(config, "REFINEMENT_MAX_POINTS", 4),
        mock.patch.object(s1p, "LATTICE", (0.0, 5.5, 11.0)),
        mock.patch.object(s1p, "VALIDATION_POINTS", ((2.75, 2.75), (5.5, 11.0))),
        mock.patch.object(s1p, "MAP_R", 3),
        mock.patch.object(s1p, "H2_PANELS_PER_FAMILY", 2),
        mock.patch.object(s1p, "h2_subgrid", lambda: ((0.0, 0.0), (5.5, 11.0))),
        mock.patch.object(s1p, "ARCHIVE", f"{work}/archive"),
        mock.patch.object(s1p, "store", lambda root=None: tmp_store),
        mock.patch.object(em, "TABLE_R", 3),
        mock.patch.object(em, "INDEPENDENT_R", 3),
        mock.patch.object(em, "INDEPENDENT_FIRST_REPLICATE", 3),
        mock.patch.object(em, "REFINED_R", 3),
        mock.patch.object(proc, "H2_R", 3),
        mock.patch.object(mixture, "PI_PROFILE_GRID", (0.3, 0.7)),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield root_dir


def pipeline_dry_run(keep: bool = False) -> dict:
    """Plumbing dry run of the whole S1-P8005 ``execute`` pipeline at tiny counts.

    ENGINEERING ONLY, never evidence: N = 600, R <= 3, a 3 x 3 lattice, Q = 4, a
    throwaway checkpoint store and output directory under the ignored work path. The
    S1-P8005 authorization is patched in-process only; nothing is committed. After the
    (expected, tiny-R) S1B stop it still exercises the S1C, S1D and representability
    steps, which the real run never does after a stop.
    """
    import shutil

    from complete_journey.paths import PROJECT_ROOT

    shutil.rmtree(PROJECT_ROOT / f"{s1p.WORK}/dry_run", ignore_errors=True)
    out = {}
    with dry_run_patches() as root_dir:
        try:
            out["log"] = s1p.execute(workers=1)
            out["stopped"] = None
        except s1p.ResearcherStop as exc:
            out["stopped"] = str(exc)
            tol = s1p.tolerances()
            arms = s1p.available_arms()
            for arm in arms:
                s1p.run_jobs(s1p.panel_jobs(arm), 1)
                s1p.run_jobs(s1p.gate_eval_jobs(arm), 1)
            out["perfect_model_pass"] = {a: s1p.perfect_model_report(a)["pass"] for a in arms}
            out["frontier_points"] = {
                a: {f: v["points_total"] for f, v in s1p.frontier_report(arms)[a].items()}
                for a in arms
            }
            out["representability"] = {a: s1p.representability_preflight(a, tol) for a in arms}
        out["artifacts"] = sorted(p.name for p in (root_dir / "archive").glob("*.json"))
        out["checkpoints"] = len(list((root_dir / "checkpoints").glob("*.json")))
    if not keep:
        shutil.rmtree(root_dir, ignore_errors=True)
    return out


def cost_probe(points_per_family: int = 2) -> dict:
    """Engineering cost probe at production N (CPU seconds). NEVER used scientifically.

    Stage-0 seeds (not the S1-P8005 key 11), the real CI-1 only as the calibration
    target (an authorized use), and every fitted value discarded: only timings and
    evaluation counts are returned.
    """
    import time

    from complete_journey.abm import procedure as proc
    from complete_journey.abm.mapping import delta12, solve_delta
    from complete_journey.abm.moments import moment_bundle, primary_moments
    from complete_journey.abm.operators import world_panel

    def cpu(fn):
        t = time.process_time()
        value = fn()
        return value, time.process_time() - t

    target = stage1.real_ci1_target()
    out = {"note": "ENGINEERING COST PROBE: timings only, stage-0 seeds, outputs discarded"}
    a1, out["a1_primary_cal_s"] = cpu(lambda: calibrate(arch.a1(), target, STAGE0))
    a1m, out["a1_mixture_cal_s"] = cpu(
        lambda: mixture.calibrate_mixture(arch.a1(), target, STAGE0, a1.heterogeneity)
    )
    probe_points = ((3.0, 4.0), (9.0, 9.0), (6.0, 1.0))[:points_per_family]
    rows = []
    for f in ("A1_PRIME", "A2"):
        for ip, ia in probe_points:
            spec = s1p.spec_for(f, ip, ia)
            prim, tp = cpu(
                lambda spec=spec: calibrate(spec, target, STAGE0, start=a1.heterogeneity)
            )
            mcal, tm = cpu(
                lambda spec=spec, prim=prim: mixture.calibrate_mixture(
                    spec, target, STAGE0, prim.heterogeneity
                )
            )
            _, td = cpu(lambda mcal=mcal: mixture.diagnostics(mcal, STAGE0))
            rows.append(
                {
                    "family": f,
                    "point": [ip, ia],
                    "primary_cal_s": tp,
                    "primary_evals": prim.evaluations,
                    "mixture_cal_s": tm,
                    "mixture_evals": mcal.evaluations,
                    "mixture_diag_s": td,
                }
            )
    out["calibrations"] = rows
    worlds = []
    for spec in (arch.a1_prime(0.9, 1.0), arch.a2(2.0, 0.8)):
        for r in range(4):
            w = arch.simulate(spec, a1.heterogeneity, _key(r, Role.MODEL), config.N_HOUSEHOLDS)
            _, tb = cpu(lambda w=w: moment_bundle(world_panel(w)))
            _, tpm = cpu(lambda w=w: primary_moments(world_panel(w)))
            worlds.append({"bundle_s": tb, "primary_s": tpm})
    out["world_bundle_s"] = float(np.mean([w["bundle_s"] for w in worlds]))
    out["world_primary_s"] = float(np.mean([w["primary_s"] for w in worlds]))
    _, out["delta12_s"] = cpu(
        lambda: delta12(
            arch.a2(2.0, 0.6), a1.heterogeneity, _key(0, Role.MODEL), config.N_HOUSEHOLDS
        )
    )
    _, out["solve_delta_s"] = cpu(
        lambda: solve_delta(5.0, 0.5, a1.heterogeneity, _key(0, Role.MODEL), config.N_HOUSEHOLDS)
    )
    # Emulator procedure per panel on a production-size synthetic table.
    w = np.array([0.1, 0.05])

    def fn(ip, ia, eta):
        m = 0.1 + 0.002 * ip + 0.001 * ia + eta @ w
        return np.full(5, m), np.array(
            [0.5 + 0.1 * eta[0], 0.16, 0.3, 0.54, 0.77, 0.9 - 0.05 * eta[1]]
        )

    lat = tuple(float(i) for i in range(12))
    k = 2
    design = np.zeros((12, 12, k + 1, k))
    mean, sd = np.zeros((12, 12, 3, 5)), np.full((12, 12, 3, 5), 0.01)
    ci1 = np.zeros((12, 12, 3, 6))
    for a, ip in enumerate(lat):
        for b, ia in enumerate(lat):
            nodes = em.design_nodes(em.LOGIT_NORMAL, np.array([0.2, 1.5]))
            design[a, b] = nodes
            for n_, node in enumerate(nodes):
                mean[a, b, n_], ci1[a, b, n_] = fn(ip, ia, node)
    table = em.ReferenceTable("A2", em.LOGIT_NORMAL, (lat, lat), design, mean, sd, ci1)
    panel = proc.Panel(
        np.array([0.53, 0.16, 0.3, 0.54, 0.77, 0.91]),
        dict.fromkeys(config.PRIMARY_IDS, 0.12),
        dict.fromkeys(config.PRIMARY_IDS, 0.01),
    )
    _, out["emulator_family_eval_s"] = cpu(
        lambda: proc.evaluate_family(table, panel, toy_tolerances(), (3.3, 4.4))
    )
    return out


def identification_fixtures(n: int = 1500, r_cal: int = 4) -> dict:
    """P8-005.14 item 1: descriptive pi profile and Jacobian condition at a small-d and a
    large-d synthetic mixture (SYNTHETIC NON-EVIDENCE; sets no threshold)."""
    out = {}
    for name, truth in (
        ("large_d", arch.MixtureHeterogeneity(0.3, -1.0, 2.5, 0.6)),
        ("small_d", arch.MixtureHeterogeneity(0.5, -0.2, 0.4, 1.2)),
    ):
        target = _target(arch.a1(), truth, n, 40)
        prim = calibrate(arch.a1(), target, STAGE0, n, r_cal)
        cal = mixture.calibrate_mixture(arch.a1(), target, STAGE0, prim.heterogeneity, n, r_cal)
        diag = mixture.diagnostics(cal, STAGE0, n, r_cal)
        profile = mixture.pi_profile(arch.a1(), target, STAGE0, cal.heterogeneity, n, r_cal)
        objs = [p["objective"] for p in profile]
        out[name] = {
            "truth": truth.as_dict(),
            "fit": cal.heterogeneity.as_dict(),
            "flags": list(cal.flags),
            "jacobian_condition": diag["jacobian_condition"],
            "separation": diag["separation"],
            "pi_profile": profile,
            "profile_objective_range": max(objs) - min(objs),
            "profile_gate_passes": sum(p["gate_pass"] for p in profile),
        }
    return out
