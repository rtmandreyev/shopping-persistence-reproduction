"""Full Phase 9 pipeline on mock providers, with no network and no credentials:
manifest, requests, mock responses, raw archive, policy table, static estimands and
validity labels, closed loop (raw and intercept-calibrated, shuffled control and
logistic surrogate), analytic-vs-Monte-Carlo checks, Level 1 and Level 2 recovery,
cross-policy reporting and figures.

The production run uses the same functions with archived provider answers in place
of mock answers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from complete_journey.systemone import closedloop as C
from complete_journey.systemone import estimands as E
from complete_journey.systemone import lattice as L
from complete_journey.systemone import recovery as R
from complete_journey.systemone import reporting as Rep
from complete_journey.systemone import settings as S
from complete_journey.systemone.budget import Budget
from complete_journey.systemone.mocks import MockProvider
from complete_journey.systemone.policy import PolicyData
from complete_journey.systemone.request import manifest, ordered, production_requests
from complete_journey.systemone.runner import run


def dgps(lams=S.NEED_LAMBDAS) -> list[C.DGP]:
    out = [C.DGP(S.NEITHER), C.DGP(S.HISTORY), C.DGP(S.HISTORY, shuffled=True)]
    out += [C.DGP(S.LATENT, lam) for lam in lams]
    out += [C.DGP(S.BOTH, lam) for lam in lams]
    out += [C.DGP(S.BOTH, lam, shuffled=True) for lam in lams]
    return out


def truth_of(dgp: C.DGP) -> str:
    """Structural known truth, as in Phase 8 RQ2. The DGP installs own-history dependence
    exactly when the policy sees the household's own history (a history-visible,
    unshuffled cell). The installed size (AHE) is reported separately for each variant."""
    if dgp.shuffled or dgp.cell in (S.NEITHER, S.LATENT):
        return R.NO_HISTORY_TRUTH
    return R.HISTORY_TRUTH


def run_policy_archive(policy: str, mock, root: Path, k: int, noise_sd: float = 0.0) -> dict:
    reqs = ordered(production_requests(policy, k), S.POLICIES.index(policy))
    m = manifest(reqs, "production", k, extra={"status": "MOCK"})
    budget = Budget(root / "budget.json")
    t0 = time.time()
    res = run(
        reqs,
        MockProvider(mock, noise_sd=noise_sd),
        root / policy,
        budget,
        "production",
        m["manifest_sha256"],
        sleep=lambda s: None,
    )
    return {
        "manifest_sha256": m["manifest_sha256"],
        "requests": len(reqs),
        "done": res.done,
        "skipped": res.skipped,
        "attempts": res.attempts,
        "seconds": time.time() - t0,
        "budget": budget.summary(),
    }


def policy_pipeline(
    pd: PolicyData,
    k: int,
    panels: int,
    q_pop: int,
    n: int,
    workers: int,
    dgp_list: list[C.DGP],
    calibrate: bool,
    policy_index: int,
) -> dict:
    out = {"policy": pd.policy, "static": {}, "recovery": {}, "analytic_check": {}}
    for conv in ("primary", "normalized"):
        out["static"][conv] = E.static_report(pd, conv)
    out["convention_dependence"] = E.convention_dependence(
        out["static"]["primary"], out["static"]["normalized"]
    )
    real_mean = None
    if calibrate:
        from complete_journey.abm.stage1 import real_ci1_target

        real_mean = real_ci1_target().values["mean"]
    for di, dgp in enumerate(dgp_list):
        truth = truth_of(dgp)
        tab = pd.table(dgp.cell, "primary")
        variants = {"raw": tab}
        if calibrate:
            delta = C.calibrate_shift(
                lambda d, c=dgp.cell: pd.table(c, "primary", shift=d),
                dgp,
                k,
                real_mean,
                (policy_index, di, 9),
                n=n,
                r=2,
                iters=18,
            )
            variants["calibrated"] = pd.table(dgp.cell, "primary", shift=delta)
        if not dgp.shuffled and dgp.cell != S.NEITHER:
            variants["surrogate"] = C.fit_surrogate(tab, dgp.cell, k)["table"]
        for vi, (vname, vtab) in enumerate(variants.items()):
            rows = R.run_block(
                vtab, dgp, k, (policy_index, di, vi, 0), panels, n=n, workers=workers
            )
            pop = R.run_block(vtab, dgp, k, (policy_index, di, vi, 1), q_pop, n=n, workers=workers)
            hv, _ = S.CELL_VISIBILITY[dgp.cell]
            # Own-history effect: structurally zero for history-hidden and donor-shuffled
            # DGPs. The table's history sensitivity is reported for the shuffled control.
            installed = (
                0.0 if (dgp.shuffled or not hv) else E.ahe_from_table(vtab, k, dgp.cell, dgp.lam)
            )
            sens = E.ahe_from_table(vtab, k, dgp.cell, dgp.lam) if (dgp.shuffled and hv) else None
            out["recovery"][f"{dgp.name}:{vname}"] = {
                "truth": truth,
                "installed_ahe": installed,
                "installed_ahe_pp": 100 * installed,
                "installed_basis": "structural zero"
                if (dgp.shuffled or not hv)
                else "measured on this variant's table",
                "policy_history_sensitivity_ahe": sens,
                "level1": R.level1(rows),
                "level2": R.level2(rows, pop, truth),
            }
        if not dgp.shuffled:
            lv = len(L.grid(k)) // 2
            an = C.analytic(tab, lv, L.grid(k)[lv], dgp)
            sims = [C.simulate(tab, dgp, k, (policy_index, di, 7, j), n=n) for j in range(4)]
            mc = np.concatenate([s["behavior"][s["profile_index"] == lv] for s in sims])
            out["analytic_check"][dgp.name] = {
                "level": lv,
                "mc_rate": float(mc.mean()),
                "analytic_rate": float(an["rate_by_week"].mean()),
                "mc_joint11": float((mc[:, :-1] * mc[:, 1:]).mean()),
                "analytic_joint11": float(an["joint11_by_week"].mean()),
                "households": int(mc.shape[0]),
                "tv_after_burn_in": an["tv_after_burn_in"],
            }
    return out


def figures(results: dict, pds: dict, k: int, outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, axes = plt.subplots(1, len(pds), figsize=(4.2 * len(pds), 3.4), squeeze=False)
    for ax, (pol, pd) in zip(axes[0], pds.items(), strict=True):
        for f in (L.grid(k)[1], L.grid(k)[4], L.grid(k)[8]):
            ys = [
                np.mean(
                    [
                        pd.pi(L.SemanticState(S.HISTORY, f, h, None))
                        for h in S.HISTORY_PATTERNS
                        if sum(h) == c
                    ]
                )
                for c in range(5)
            ]
            ax.plot(range(5), ys, marker="o", label=f"stated {f:.0%}")
        ax.set_title(f"{pol} (MOCK)")
        ax.set_xlabel("shopping weeks in last 4")
        ax.set_ylabel("policy probability")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = outdir / "mock_policy_surfaces.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))
    return paths


def run_e2e(
    root: Path,
    mocks: dict,
    k: int = 10,
    panels: int = 6,
    q_pop: int = 6,
    n: int = 600,
    workers: int = 1,
    lams=(0.8,),
    calibrate: bool = True,
    noise: dict | None = None,
) -> dict:
    root = Path(root)
    noise = noise or {}
    summary = {"k": k, "panels": panels, "q_pop": q_pop, "n": n, "archives": {}, "policies": {}}
    pds = {}
    for policy, mock in mocks.items():
        summary["archives"][policy] = run_policy_archive(
            policy, mock, root, k, noise_sd=noise.get(policy, 0.0)
        )
        pds[policy] = PolicyData.from_archive(root / policy, policy, k)
    for i, (policy, pd) in enumerate(pds.items()):
        summary["policies"][policy] = policy_pipeline(
            pd, k, panels, q_pop, n, workers, dgps(lams), calibrate, i
        )
    labels = {p: Rep.flat_labels(r) for p, r in summary["policies"].items()}
    if S.P_J13 in labels and S.P_J14 in labels:
        jev = Rep.jev_generic(labels[S.P_J13], labels[S.P_J14])
        summary["jev_generic"] = jev
        if S.P_M in labels:
            summary["cross_vendor"] = Rep.cross_vendor(jev, labels[S.P_M])
    if S.P_J13 in pds and S.P_M in pds:
        summary["exploratory"] = Rep.exploratory_disagreement(pds[S.P_J13], pds[S.P_M])
    summary["figures"] = figures(summary, pds, k, root / "figures")
    (root / "e2e_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary
