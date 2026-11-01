"""Independent recomputation of the paper's empirical numbers from the raw CSV files.

Sections 2-4 and Appendices A-C: data facts, transition rates, the fixed-effects ladder with
clustered and Driscoll-Kraay standard errors, rule-O quantities (the five dynamic moments and
the calibration target), the prediction benchmark's frequency and recency tiers, campaign facts
and the dry-spaghetti composition table. Written from the paper's definitions with numpy,
scipy, pandas, duckdb and scikit-learn; it does not call the project library, so it is a second
implementation of the published numbers. Runtime: a few minutes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

T = 102


def _csv(raw: Path, name: str) -> str:
    return str(raw / f"{name}.csv").replace("'", "''")


def load(raw: Path):
    import duckdb

    con = duckdb.connect()
    con.execute(f"create table t as select * from read_csv_auto('{_csv(raw, 'transaction_data')}', header=true)")
    return con


def demean(Z, groups, tol=1e-13):
    Z = Z.astype(float).copy()
    if not groups:
        return Z - Z.mean(0)
    for _ in range(100000):
        prev = Z.copy()
        for c in groups:
            cnt = np.bincount(c)
            with np.errstate(invalid="ignore", divide="ignore"):  # unused codes have zero count
                means = np.stack([np.bincount(c, weights=Z[:, j]) for j in range(Z.shape[1])], 1) / cnt[:, None]
            Z -= means[c]
        if len(groups) == 1 or np.abs(Z - prev).max() < tol:
            return Z
    raise RuntimeError("alternating projections did not converge")


def fe_fit(y, X, hh, wk, ent, tim):
    groups = ([hh] if ent else []) + ([wk] if tim else [])
    Z = demean(np.column_stack([y, X]), groups)
    b = np.linalg.lstsq(Z[:, 1:], Z[:, 0], rcond=None)[0]
    return b, Z[:, 0] - Z[:, 1:] @ b, Z[:, 1:]


def _meat(X, e, codes):
    _, c = np.unique(codes, return_inverse=True, axis=0) if codes.ndim > 1 else np.unique(codes, return_inverse=True)
    tot = np.zeros((c.max() + 1, X.shape[1]))
    np.add.at(tot, c, X * e[:, None])
    G, n = c.max() + 1, len(X)
    return G / (G - 1) * (n - 1) / n * tot.T @ tot


def ses(X, e, hh, wk, dfr):
    """Household-clustered, two-way clustered and Driscoll-Kraay (Bartlett 4 and 8) SEs."""
    n = len(X)
    bread = np.linalg.inv(X.T @ X)
    f = n / dfr
    mh, mt, mht = _meat(X, e, hh), _meat(X, e, wk), _meat(X, e, np.column_stack([hh, wk]))
    out = {"hh": np.sqrt(np.diag(f * bread @ mh @ bread))[0],
           "twoway": np.sqrt(np.diag(f * bread @ (mh + mt - mht) @ bread))[0]}
    _, c = np.unique(wk, return_inverse=True)
    tot = np.zeros((c.max() + 1, X.shape[1]))
    np.add.at(tot, c, X * e[:, None])
    for bw in (4, 8):
        m = tot.T @ tot
        for lag in range(1, bw + 1):
            cr = tot[lag:].T @ tot[:-lag]
            m += (1 - lag / (bw + 1)) * (cr + cr.T)
        out[f"dk{bw}"] = np.sqrt(np.diag(f * bread @ m @ bread))[0]
    return out


def compute(raw: Path) -> dict[str, float]:
    from scipy.stats import spearmanr

    v: dict[str, float] = {}
    con = load(raw)
    r = con.execute("""select count(*), count(distinct household_key), count(distinct BASKET_ID),
        count(distinct STORE_ID), count(distinct PRODUCT_ID), max(DAY), count(distinct WEEK_NO) from t""").fetchone()
    for k, x in zip(("lines", "households", "baskets", "stores", "products", "days", "weeks"), r, strict=True):
        v[f"data.{k}"] = float(x)
    v["data.demographics"] = float(con.execute(
        f"select count(distinct household_key) from read_csv_auto('{_csv(raw, 'hh_demographic')}')").fetchone()[0])
    v["data.campaigns"] = float(con.execute(
        f"select count(distinct CAMPAIGN) from read_csv_auto('{_csv(raw, 'campaign_desc')}')").fetchone()[0])
    hw = con.execute("""select household_key, WEEK_NO, sum(SALES_VALUE) rev, count(distinct BASKET_ID) b
        from t group by 1, 2""").df()
    hhs = np.sort(hw.household_key.unique())
    H = len(hhs)
    idx = {h: i for i, h in enumerate(hhs)}
    Y = np.zeros((H, T), int)
    REV = np.zeros((H, T))
    BAS = np.zeros((H, T))
    hi = hw.household_key.map(idx).to_numpy()
    wi = hw.WEEK_NO.to_numpy() - 1
    Y[hi, wi] = (hw.b.to_numpy() > 0).astype(int)
    REV[hi, wi] = hw.rev.to_numpy()
    BAS[hi, wi] = hw.b.to_numpy()
    v["data.participation"] = Y.mean()
    tot = REV.sum(1)
    v["desc.top_decile_revenue_share"] = np.sort(tot)[::-1][: H // 10].sum() / tot.sum()
    v["desc.half_window_corr"] = np.corrcoef(REV[:, :51].sum(1), REV[:, 51:].sum(1))[0, 1]
    act = Y.sum(1)
    v["desc.spearman_active_revenue"] = spearmanr(act, tot).statistic
    v["desc.spearman_active_revenue_per_basket"] = spearmanr(act, tot / BAS.sum(1)).statistic

    # --- full grid, responses 2-102 -------------------------------------------------
    rr = np.arange(2, 103) - 1
    yy, xl = Y[:, rr].ravel(), Y[:, rr - 1].ravel()
    v["panel.P11_full"], v["panel.P01_full"] = yy[xl == 1].mean(), yy[xl == 0].mean()
    hh, wk = np.repeat(np.arange(H), len(rr)), np.tile(rr, H)
    for m, (e, t) in {"M0": (0, 0), "M1": (0, 1), "M2": (1, 0), "M3": (1, 1)}.items():
        b, res, X = fe_fit(yy, xl[:, None], hh, wk, e, t)
        v[f"panel.{m}_full"] = b[0]
        dfr = len(yy) - 2 - ((H - 1) if e else 0) - ((len(rr) - 1) if t else 0)
        s = ses(X, res, hh, wk, dfr)
        if m == "M3":
            for k, x in s.items():
                v[f"panel.M3_full_se_{k}"] = x
        else:
            v[f"panel.{m}_full_se_twoway"] = s["twoway"]
    rr21 = np.arange(21, 102) - 1
    b, _, _ = fe_fit(Y[:, rr21].ravel(), Y[:, rr21 - 1].ravel()[:, None],
                     np.repeat(np.arange(H), len(rr21)), np.tile(rr21, H), 1, 1)
    v["panel.M3_resp21_101"] = b[0]
    f = np.argmax(Y == 1, axis=1) + 1
    lw = T - np.argmax(Y[:, ::-1] == 1, axis=1)
    v["panel.first_purchase_by_week20"] = float((f <= 20).sum())
    v["panel.share_cells_before_first"] = float(np.mean(np.arange(2, 103)[None, :] < f[:, None]))

    # --- rule O --------------------------------------------------------------------
    ws, we = f + 1, np.minimum(lw - 1, 101)
    elig = np.clip(we - ws + 1, 0, None) >= 8
    v["ruleO.households"] = float(elig.sum())

    def rows(maxlag):
        H_, R_ = [], []
        for i in np.where(elig)[0]:
            r_ = np.arange(max(2, ws[i] + maxlag), we[i] + 1)
            H_.append(np.full(len(r_), i))
            R_.append(r_)
        return np.concatenate(H_), np.concatenate(R_)

    h1, r1 = rows(1)
    y1, x1 = Y[h1, r1 - 1], Y[h1, r1 - 2]
    v["ruleO.P11"], v["ruleO.P01"] = y1[x1 == 1].mean(), y1[x1 == 0].mean()
    hc, wc = pd.factorize(h1)[0], pd.factorize(r1)[0]
    for m, (e, t) in {"M0": (0, 0), "M1": (0, 1), "M2": (1, 0), "M3": (1, 1)}.items():
        v[f"ruleO.{m}"] = fe_fit(y1, x1[:, None], hc, wc, e, t)[0][0]
    h8, r8 = rows(8)
    y8 = Y[h8, r8 - 1]
    hc, wc = pd.factorize(h8)[0], pd.factorize(r8)[0]
    for k in (1, 2, 4, 8):
        v[f"ruleO.common_beta{k}"] = fe_fit(y8, Y[h8, r8 - 1 - k][:, None], hc, wc, 1, 1)[0][0]
    v["ruleO.gamma2"] = fe_fit(y8, np.column_stack([Y[h8, r8 - 2], Y[h8, r8 - 3]]), hc, wc, 1, 1)[0][1]
    rates = np.array([Y[i, ws[i] - 1: we[i]].mean() for i in np.where(elig)[0]])
    v["ruleO.ci1_mean"] = rates.mean()
    for q, x in zip((10, 25, 50, 75, 90), np.quantile(rates, [0.1, 0.25, 0.5, 0.75, 0.9]), strict=True):
        v[f"ruleO.ci1_q{q}"] = x
    ones = total = 0
    for i in np.where(elig)[0]:
        seq = Y[i, ws[i] - 1: we[i]]
        s = 0
        for j in range(1, len(seq) + 1):
            if j == len(seq) or seq[j] != seq[s]:
                if seq[s] == 1 and s != 0 and j != len(seq):  # interior spells only
                    total += 1
                    ones += (j - s) == 1
                s = j
    v["ruleO.one_week_spell_share"] = ones / total

    # --- prediction benchmark (frequency and recency tiers; gain moment) ----------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cs = np.cumsum(Y, 1)

    def feats(hh_, t_):
        n = (t_ - f[hh_] + 1).astype(float)
        k = cs[hh_, t_ - 1].astype(float)
        rate = (k + 1) / (n + 2)
        lg = np.log(rate / (1 - rate))
        t0 = np.column_stack([np.log1p(n), lg, lg * np.log1p(n)])
        st = np.maximum(f[hh_], t_ - 3)
        before = np.where(st >= 2, cs[hh_, np.maximum(st - 2, 0)], 0)
        r4 = (cs[hh_, t_ - 1] - before) / (t_ - st + 1)
        last = np.array([np.max(np.where(Y[h, :tt] == 1)[0]) + 1 for h, tt in zip(hh_, t_, strict=True)])
        return t0, np.column_stack([t0, Y[hh_, t_ - 1].astype(float), r4,
                                    np.log1p(np.minimum(t_ - last, 12).astype(float))])

    def ll(y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def logit():
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, tol=1e-8, max_iter=5000))

    hh_ = np.repeat(np.arange(H), 100)
    r_ = np.tile(np.arange(2, 102), H)
    keep = f[hh_] <= r_ - 1
    hh_, r_ = hh_[keep], r_[keep]
    y = Y[hh_, r_ - 1].astype(float)
    X0, X1 = feats(hh_, r_ - 1)
    tr, te = r_ <= 84, r_ >= 85
    v["pred.baseline_logloss"] = ll(y[te], np.full(te.sum(), y[tr].mean()))
    for name, X in (("freq", X0), ("recency", X1)):
        p = logit().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        v[f"pred.{name}_logloss"], v[f"pred.{name}_auc"] = ll(y[te], p), float(roc_auc_score(y[te], p))
    hh_ = np.repeat(np.arange(H), 100)
    r_ = np.tile(np.arange(2, 102), H)
    keep = elig[hh_] & (r_ <= we[hh_]) & (r_ - 1 >= ws[hh_])
    hh_, r_ = hh_[keep], r_[keep]
    y = Y[hh_, r_ - 1].astype(float)
    X0, X1 = feats(hh_, r_ - 1)
    tr, ev = r_ <= 68, r_ >= 69
    l0 = ll(y[ev], logit().fit(X0[tr], y[tr]).predict_proba(X0[ev])[:, 1])
    l1 = ll(y[ev], logit().fit(X1[tr], y[tr]).predict_proba(X1[ev])[:, 1])
    v["pred.gain_moment"] = l0 - l1

    # --- campaigns ---------------------------------------------------------------------
    ct = con.execute(f"select distinct household_key, CAMPAIGN from read_csv_auto('{_csv(raw, 'campaign_table')}')").df()
    cd = con.execute(f"select * from read_csv_auto('{_csv(raw, 'campaign_desc')}')").df().set_index("CAMPAIGN")
    v["camp.listed_households"] = float(ct.household_key.nunique())
    v["camp.median_campaigns_per_listed"] = float(ct.groupby("household_key").size().median())
    ov = []
    for c in cd.index:
        s_, e_ = cd.loc[c, "START_DAY"], cd.loc[c, "END_DAY"]
        others = [o for o in cd.index if o != c and cd.loc[o, "START_DAY"] <= e_ and cd.loc[o, "END_DAY"] >= s_]
        hc_ = set(ct[ct.CAMPAIGN == c].household_key)
        ov.append(len(hc_ & set(ct[ct.CAMPAIGN.isin(others)].household_key)) / len(hc_))
    v["camp.median_concurrent_overlap"] = float(np.median(ov))
    dw = con.execute("select distinct DAY, WEEK_NO from t").df().set_index("DAY").WEEK_NO
    revw = pd.DataFrame(REV, index=hhs)
    ratios = []
    for c in cd.index:
        sw = dw.loc[cd.loc[c, "START_DAY"]]
        prior = revw.iloc[:, : sw - 1].sum(1)
        m = np.isin(hhs, ct[ct.CAMPAIGN == c].household_key.unique())
        ratios.append(prior[m].mean() / prior[~m].mean())
    v["camp.prior_revenue_ratio_min"], v["camp.prior_revenue_ratio_max"] = min(ratios), max(ratios)
    first = cd.START_DAY.idxmin()
    v["camp.earliest_recipients"] = float((ct.CAMPAIGN == first).sum())
    early = cd[(cd.START_DAY >= 224) & (cd.START_DAY <= 323)].index
    v["camp.early_list_sum"] = float(ct[ct.CAMPAIGN.isin(early)].groupby("CAMPAIGN").size().sum())
    v["camp.early_distinct"] = float(ct[ct.CAMPAIGN.isin(early)].household_key.nunique())
    v["camp.later_start_min_day"] = float(cd.drop(early).START_DAY.min())
    firstday = ct.merge(cd.reset_index()[["CAMPAIGN", "START_DAY"]], on="CAMPAIGN").groupby("household_key").START_DAY.min()
    v["camp.first_listed_day346_or_later"] = float((firstday >= 346).sum())

    # --- dry-spaghetti composition (Appendix B) -----------------------------------------
    ev = con.execute(f"""with l as (select t.household_key, t.BASKET_ID, t.WEEK_NO, t.STORE_ID, p.BRAND
          from t join read_csv_auto('{_csv(raw, 'product')}') p using (PRODUCT_ID)
          where p.SUB_COMMODITY_DESC = 'SPAGHETTI DRY' and t.QUANTITY > 0)
        select household_key, BASKET_ID, WEEK_NO, STORE_ID, sum((BRAND='National')::int) nat,
               sum((BRAND='Private')::int) prv from l group by 1, 2, 3, 4""").df()
    ev["comp"] = np.where(ev.prv == 0, "N", np.where(ev.nat == 0, "P", "M"))
    v["comp.mixed_baskets"] = float((ev.comp == "M").sum())
    ev = ev.sort_values(["household_key", "WEEK_NO"])
    g = ev.groupby(["household_key", "WEEK_NO"]).agg(n=("comp", lambda s: (s == "N").sum()),
                                                     p=("comp", lambda s: (s == "P").sum())).reset_index()
    for c in "np":
        g["prior_" + c] = g.groupby("household_key")[c].cumsum() - g[c]
    ev = ev.merge(g[["household_key", "WEEK_NO", "prior_n", "prior_p"]], on=["household_key", "WEEK_NO"])
    pure = ev[ev.comp != "M"].copy()
    pure["y"] = (pure.comp == "P").astype(int)
    pp = pure.prior_n + pure.prior_p
    pure["H"] = np.where(pp == 0, "H0", np.where(pure.prior_p == 0, "HN", np.where(pure.prior_n == 0, "HP", "HB")))
    v["comp.pure_events"] = float(len(pure))
    v["comp.households"] = float(pure.household_key.nunique())
    v["comp.stores"] = float(pure.STORE_ID.nunique())
    for h, grp in pure.groupby("H"):
        v[f"comp.{h}_events"] = float(len(grp))
        v[f"comp.{h}_private_share"] = grp.y.mean()
    hs = pure.groupby("household_key").y.agg(["min", "max"])
    v["comp.only_private_households"] = float((hs["min"] == 1).sum())
    v["comp.only_national_households"] = float((hs["max"] == 0).sum())
    return {k: float(x) for k, x in v.items()}
