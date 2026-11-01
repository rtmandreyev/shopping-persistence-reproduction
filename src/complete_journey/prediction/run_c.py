"""Phase 7 Run C: one-time real-data reveal and its audit reproduction.

Authority:
- P7-003 (frozen design);
- the accepted Run B synthetic preflight;
- the committed design snapshot ``docs/phases/PHASE_07_RUN_C/design_snapshot.json``.

``run_real`` executes the frozen pipeline once:
1. benchmark selection on validation only;
2. a ledger entry naming the selections is written **before** the test period is
   touched;
3. a single test evaluation of the selected models, after which the ledger records
   the test as consumed;
4. VI-6 and the empirical moment register;
5. the frozen household bootstrap.

A second ``run_real`` refuses once the ledger exists. ``audit_real`` reproduces the
archived numbers from frozen inputs using the ledger's recorded selections. It never
re-runs model selection and never reopens the test period. No function deletes or
resets the ledger.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from complete_journey.paths import PROJECT_ROOT
from complete_journey.prediction import config, protocol
from complete_journey.prediction import features as feat
from complete_journey.prediction import modeling as mdl
from complete_journey.prediction import moments as mom
from complete_journey.prediction import panel as panel_module
from complete_journey.prediction.operators import realtime_rows
from complete_journey.prediction.panel import CanonicalPanel, load_real_panel, require

RUN_C_DIR = "docs/phases/PHASE_07_RUN_C"
SNAPSHOT = f"{RUN_C_DIR}/design_snapshot.json"
SNAPSHOT_SHA = f"{RUN_C_DIR}/design_snapshot.sha256"
LEDGER = f"{RUN_C_DIR}/test_ledger.json"
BENCHMARK = f"{RUN_C_DIR}/benchmark_results.json"
MOMENTS = f"{RUN_C_DIR}/moment_register_real.csv"
BOOTSTRAP_SUMMARY = f"{RUN_C_DIR}/bootstrap_summary.json"
PROVENANCE = f"{RUN_C_DIR}/run_c_provenance.json"
#: Local, Git-ignored (outputs/tables/**): observation-level predictions and replicates.
LOCAL_DIR = "outputs/tables/predictive/run_c_local"
LOCAL_PREDICTIONS = f"{LOCAL_DIR}/test_predictions.parquet"
LOCAL_REPLICATES = f"{LOCAL_DIR}/bootstrap_replicates.csv"

DATA_INPUTS = (
    "data/processed/econometrics_household_week.parquet",
    "data/processed/predictive_household_week_features.parquet",
)
OPERATOR_PLAN = {
    config.OPERATOR_OFFLINE: ("CI-1", "K-1", "K-2", "VI-1", "VI-2", "VI-3", "VI-4", "VI-5", "VI-6"),
    config.OPERATOR_SENSITIVITY: ("CI-1", "VI-1", "VI-2", "VI-3", "VI-4", "VI-6"),
    config.OPERATOR_DIAGNOSTIC: ("CI-1", "VI-1", "VI-2", "VI-3", "VI-4", "VI-6"),
}
BENCHMARK_PLAN = (
    (
        "B0",
        "base_rate",
        "train-validation prevalence; validation log loss of the train-only base rate reported",
    ),
    ("B1", "tier0", "fixed C=1.0 fitted on train+validation; no tuning"),
    ("B1", "tier1", "fixed C=1.0 fitted on train+validation; no tuning"),
    ("B1", "tier2", "C grid selected on validation log loss; refit on train+validation"),
    ("B2", "tier2", "12-point grid selected on validation log loss; refit on train+validation"),
)
BOOTSTRAP_WORKERS = 3
BENCHMARK_THREADS = 4
INTERVAL = "percentile [2.5%, 97.5%] over successful replicates; SE = sd(ddof=1)"
FAILURE_POLICY = "any failed replicate stops the run after all replicates; nothing is dropped"
AUDIT_ATOL = 1e-9


def _path(relative: str, root: Path) -> Path:
    return root / relative


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, default=str) + "\n").encode()


def _source_files(root: Path) -> list[Path]:
    files = sorted((root / "src/complete_journey/prediction").glob("*.py"))
    return [*files, root / "scripts/run_predictive.py"]


def build_snapshot(root: Path = PROJECT_ROOT) -> dict:
    """Everything that defines the reveal, recomputable without timestamps."""
    import duckdb
    import linearmodels
    import pyarrow
    import scipy
    import sklearn
    import statsmodels

    inventory = feat.inventory()
    return {
        "authority": "P7-003 design; Run B synthetic preflight accepted",
        "target": {
            "variable": config.TARGET_SOURCE,
            "definition": "Y_{i,r}=1 if any recorded retailer line in response week r=t+1",
            "responses": config.RESPONSE_WEEKS,
        },
        "predictive_population": {
            "name": config.POPULATION_REALTIME,
            "rule": "a recorded purchase at or before origin t (point-in-time); response <= 101",
        },
        "splits": dict(config.SPLITS),
        "recency_windows": {"train": config.RECENCY_TRAIN, "evaluate": config.RECENCY_EVAL},
        "operators": {
            config.OPERATOR_OFFLINE: "interior window [f+1, min(l-1,101)], >= 8 weeks; lag k needs r-k in window",
            config.OPERATOR_SENSITIVITY: "responses 20-101, full-grid lags (week 19 lags response 20)",
            config.OPERATOR_DIAGNOSTIC: "responses 2-101, full grid",
        },
        "min_interior_weeks": config.MIN_INTERIOR_WEEKS,
        "tiers": {
            "tier0": list(config.TIER0),
            "tier1": list(config.TIER1),
            "tier2": list(feat.tier2_columns()),
        },
        "inventory_sha256": _sha(root / config.FEATURE_INVENTORY),
        "inventory_rows": len(inventory),
        "models": {
            "B0": "training prevalence (PrevalenceModel)",
            "B1": f"Pipeline(SimpleImputer(median, add_indicator) -> StandardScaler -> LogisticRegression(l1_ratio=0, solver={config.B1_SOLVER}, tol={config.B1_TOL}, max_iter={config.B1_MAX_ITER}))",
            "B1_fixed_C": config.B1_FIXED_C,
            "B1_tier2_grid": list(config.B1_TIER2_GRID),
            "B2": f"HistGradientBoostingClassifier(max_iter={config.B2_MAX_ITER}, early_stopping=False, random_state={config.SEED})",
            "B2_grid": {k: list(v) for k, v in config.B2_GRID.items()},
        },
        "benchmark_plan": [list(p) for p in BENCHMARK_PLAN],
        "metrics": {
            "primary": config.PRIMARY_METRIC,
            "secondary": config.SECONDARY_METRIC,
            "descriptive": list(config.DESCRIPTIVE_METRICS),
            "calibration": "intercept (logit offset GLM) and slope (GLM on logit p); 10 equal-frequency reliability bins",
            "probability_clip": config.PROBABILITY_CLIP,
        },
        "recency_gain": {
            "primary": f"{config.RECENCY_GAIN} = LL(B1|Tier0) - LL(B1|Tier1); positive = recency helps",
            "secondary": f"{config.RECENCY_GAIN_SECONDARY} = Brier(Tier0) - Brier(Tier1)",
            "population": config.OPERATOR_OFFLINE,
            "also_computed_for": [config.OPERATOR_SENSITIVITY, config.OPERATOR_DIAGNOSTIC],
            "s_g_rows": "S/G VI-6 rows restricted to origins at or after the first recorded "
            "purchase (Tier 0 undefined before it; point-in-time rule)",
            "ordering": "computed after the single benchmark test evaluation",
        },
        "moments": {
            "operator_plan": {k: list(v) for k, v in OPERATOR_PLAN.items()},
            "environment": "E-1 weekly path under operator O (point estimates)",
            "shuffle_check": f"D-4 on operator O, {config.SHUFFLE_REPLICATIONS} shuffles, atol {config.SHUFFLE_ATOL}",
            "ci1_quantiles": list(config.CI1_QUANTILES),
            "lags": list(config.LAG_SET),
            "spell_bins": [list(b) for b in config.SPELL_BINS],
        },
        "bootstrap": {
            "method": "household cluster bootstrap: whole trajectories with replacement, duplicates relabelled",
            "replications": config.BOOTSTRAP_PRODUCTION_B,
            "seeds": f"numpy SeedSequence({config.SEED}).spawn({config.BOOTSTRAP_PRODUCTION_B})",
            "content": "all operator-plan moments, including VI-6 with refit, jointly per replicate",
            "interval": INTERVAL,
            "failure_policy": FAILURE_POLICY,
            "workers": BOOTSTRAP_WORKERS,
            "threads_per_worker": 1,
        },
        "benchmark_threads": BENCHMARK_THREADS,
        "seed": config.SEED,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "linearmodels": linearmodels.__version__,
            "duckdb": duckdb.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "source_sha256": {str(p.relative_to(root)): _sha(p) for p in _source_files(root)},
        "data_sha256": {p: _sha(root / p) for p in DATA_INPUTS},
        "test_policy": "test period 85-101 touched exactly once by the five selected models; "
        "the ledger records selections before scoring",
    }


def write_snapshot(root: Path = PROJECT_ROOT) -> str:
    """Create the pre-reveal snapshot (Commit A). Refuses to overwrite an existing one."""
    path = _path(SNAPSHOT, root)
    require(not path.exists(), "design snapshot already exists; it is never rewritten")
    snapshot = build_snapshot(root)
    snapshot["source_commit_at_snapshot"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    payload = _canonical(snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    _path(SNAPSHOT_SHA, root).write_text(f"{digest}  design_snapshot.json\n")
    return digest


def verify_snapshot(root: Path = PROJECT_ROOT) -> str:
    path = _path(SNAPSHOT, root)
    require(path.is_file(), "committed design snapshot missing; Run C cannot start")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    recorded = _path(SNAPSHOT_SHA, root).read_text().split()[0]
    require(digest == recorded, "design snapshot bytes do not match their recorded hash")
    stored = json.loads(payload)
    stored.pop("source_commit_at_snapshot", None)
    current = json.loads(_canonical(build_snapshot(root)))
    differing = sorted(k for k in set(stored) | set(current) if stored.get(k) != current.get(k))
    require(not differing, f"current design differs from the committed snapshot: {differing}")
    return digest


def ledger_state(root: Path = PROJECT_ROOT) -> str | None:
    path = _path(LEDGER, root)
    return json.loads(path.read_bytes())["status"] if path.is_file() else None


@contextlib.contextmanager
def _context(mode: str):
    require(mode in ("reveal", "audit"), "unknown Run C mode")
    previous = panel_module._RUN_C_MODE
    panel_module._RUN_C_MODE = mode
    try:
        yield
    finally:
        panel_module._RUN_C_MODE = previous


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _benchmark_frames(panel: CanonicalPanel, root: Path):
    store = pd.read_parquet(root / DATA_INPUTS[1])
    rows = feat.attach_tier2(feat.build_features(panel, realtime_rows(panel).rows, tier=1), store)
    return mdl.temporal_split(rows)


def _fit_plan(panel, parts, selections=None):
    """Fit the five frozen benchmark models; selection on validation unless given."""
    train, val = parts["train"], parts["validation"]
    train_val = pd.concat([train, val])
    tier_cols = {"tier0": config.TIER0, "tier1": config.TIER1, "tier2": feat.tier2_columns()}
    fitted, validation = {}, {}
    b0 = mdl.fit_selected(panel, "B0", train, val, config.TIER0)
    fitted["B0|base_rate"] = b0
    validation["B0|base_rate"] = b0.validation_log_loss
    for tier in ("tier0", "tier1"):
        train_only = mdl.fit_fixed_b1(panel, train, tier_cols[tier])
        validation[f"B1|{tier}"] = {
            "train_only_validation_log_loss": mdl.log_loss(val["y"], train_only.predict(val))
        }
        fitted[f"B1|{tier}"] = mdl.fit_fixed_b1(panel, train_val, tier_cols[tier])
    for name in ("B1", "B2"):
        key = f"{name}|tier2"
        if selections is None:
            model = mdl.fit_selected(panel, name, train, val, tier_cols["tier2"])
            validation[key] = model.validation_log_loss
        else:  # audit: recorded selection, no search
            params = selections[key]
            estimator = mdl.make_b1(params["C"]) if name == "B1" else mdl.make_b2(params)
            estimator = estimator.fit(
                train_val[list(tier_cols["tier2"])].to_numpy(float), train_val["y"].to_numpy(float)
            )
            model = mdl.FittedModel(name, params, estimator, tuple(tier_cols["tier2"]), {})
        fitted[key] = model
    return fitted, validation


def _moment_records(panel: CanonicalPanel) -> list[dict]:
    """Operator-plan moments, including VI-6 (refit inside this panel)."""
    fns = {
        "CI-1": mom.ci1,
        "K-1": mom.k1_between_share,
        "K-2": mom.k2_concentration,
        "VI-1": mom.vi1_transitions,
        "VI-2": mom.vi2_ladder,
        "VI-3": mom.vi3_lag_profile,
        "VI-4": mom.vi4_spells,
        "VI-5": mom.vi5_half_to_half,
        "VI-6": protocol.vi6_records,
    }
    records = []
    for operator, ids in OPERATOR_PLAN.items():
        for moment_id in ids:
            records.extend(fns[moment_id](panel, operator))
    return mom.validate_records(records)


def _replicate(panel: CanonicalPanel, index: int, mode: str):
    """One bootstrap replicate (worker process; enters the Run C context locally)."""
    from threadpoolctl import threadpool_limits

    panel_module._RUN_C_MODE = mode
    child = np.random.SeedSequence(config.SEED).spawn(config.BOOTSTRAP_PRODUCTION_B)[index]
    try:
        with threadpool_limits(1):
            sample = mom.household_bootstrap(panel, np.random.default_rng(child))
            records = _moment_records(sample)
        return (
            index,
            [(r["operator"], r["moment_id"], r["component"], r["estimate"]) for r in records],
            None,
        )
    except Exception as error:  # recorded, never silently dropped
        return index, None, f"{type(error).__name__}: {error}"


def _bootstrap(panel: CanonicalPanel, indices, mode: str):
    from joblib import Parallel, delayed

    return Parallel(n_jobs=BOOTSTRAP_WORKERS, backend="loky")(
        delayed(_replicate)(panel, i, mode) for i in indices
    )


def _key(r) -> tuple:
    return (r["operator"], r["moment_id"], r["component"])


def replicate_frame(results) -> pd.DataFrame:
    rows = [
        {"replicate": i, "operator": o, "moment_id": m, "component": c, "estimate": e}
        for i, recs, _ in results
        if recs
        for (o, m, c, e) in recs
    ]
    return pd.DataFrame(
        rows, columns=["replicate", "operator", "moment_id", "component", "estimate"]
    )


def assemble_moments(
    point: list[dict], environment: list[dict], replicates: pd.DataFrame
) -> pd.DataFrame:
    """Point estimates with bootstrap SE and percentile interval (environment: point only)."""
    summary = replicates.groupby(["operator", "moment_id", "component"])["estimate"].agg(
        se=lambda x: x.std(ddof=1),
        lo=lambda x: x.quantile(0.025),
        hi=lambda x: x.quantile(0.975),
        n="count",
    )
    point = [dict(r) for r in point]
    for r in point:
        require(_key(r) in summary.index, f"no bootstrap distribution for {_key(r)}")
        row = summary.loc[_key(r)]
        r.update(
            uncertainty_low=float(row["lo"]),
            uncertainty_high=float(row["hi"]),
            uncertainty_method=f"household bootstrap B={int(row['n'])}; {INTERVAL}",
        )
    records = mom.validate_records(point + [dict(r) for r in environment])
    frame = pd.DataFrame(records, columns=mom.MOMENT_FIELDS)
    frame["bootstrap_se"] = [
        float(summary.loc[_key(r)]["se"]) if _key(r) in summary.index else None for r in records
    ]
    return frame


def run_real(root: Path = PROJECT_ROOT) -> dict:
    """The one-time reveal. Refuses if the ledger exists in any state."""
    from threadpoolctl import threadpool_limits

    snapshot_sha = verify_snapshot(root)
    require(ledger_state(root) is None, "Run C test ledger exists: the reveal is not repeatable")
    timings, started = {}, time.time()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    data = {p: _sha(root / p) for p in DATA_INPUTS}
    with _context("reveal"):
        panel = load_real_panel(root)
        parts = _benchmark_frames(panel, root)
        t = time.time()
        with threadpool_limits(BENCHMARK_THREADS):
            fitted, validation = _fit_plan(panel, parts)
        timings["benchmark_training_and_tuning_s"] = round(time.time() - t, 1)
        selections = {k: m.params for k, m in fitted.items()}
        ledger = {
            "status": "TEST_EVALUATION_STARTED",
            "test_weeks": config.SPLITS["test"],
            "design_snapshot_sha256": snapshot_sha,
            "source_commit": commit,
            "data_sha256": data,
            "selected_before_test": selections,
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        _write_json(_path(LEDGER, root), ledger)  # written BEFORE the test is touched
        t = time.time()
        test, predictions = {}, []
        for key, model in fitted.items():
            test[key] = mdl.score(model, parts["test"])
            predictions.append(
                parts["test"][["household_key", "origin_week", "response_week", "y"]].assign(
                    model=key, p=model.predict(parts["test"])
                )
            )
        timings["test_evaluation_s"] = round(time.time() - t, 1)
        ledger.update(
            status="TEST_CONSUMED",
            consumed_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            models_evaluated_once=sorted(test),
        )
        _write_json(_path(LEDGER, root), ledger)
        local = _path(LOCAL_PREDICTIONS, root)
        local.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(predictions, ignore_index=True).to_parquet(local, index=False)

        t = time.time()
        point = _moment_records(panel)
        environment = mom.e1_records(panel, config.OPERATOR_OFFLINE)
        shuffle = mom.shuffle_check(panel, seed=config.SEED)
        timings["point_moments_s"] = round(time.time() - t, 1)
        t = time.time()
        results = _bootstrap(panel, range(config.BOOTSTRAP_PRODUCTION_B), "reveal")
        timings["bootstrap_s"] = round(time.time() - t, 1)
    failures = [(i, err) for i, _, err in results if err]
    replicates = replicate_frame(results)
    replicates.to_csv(_path(LOCAL_REPLICATES, root), index=False)
    require(not failures, f"bootstrap replicate failures (not dropped): {failures[:5]}")
    rows = assemble_moments(point, environment, replicates)
    rows.to_csv(_path(MOMENTS, root), index=False)
    prevalence = {s: float(parts[s]["y"].mean()) for s in parts}
    benchmark = {
        "population": config.POPULATION_REALTIME,
        "split_support": {
            s: {
                "rows": int(len(f)),
                "households": int(f["household_key"].nunique()),
                "prevalence": prevalence[s],
            }
            for s, f in parts.items()
        },
        "validation": validation,
        "selected": selections,
        "test": test,
        "descriptive_tier_differences_on_test": {
            "B1_tier0_minus_tier1_log_loss": test["B1|tier0"]["log_loss"]
            - test["B1|tier1"]["log_loss"],
            "B1_tier1_minus_B1_tier2_log_loss": test["B1|tier1"]["log_loss"]
            - test["B1|tier2"]["log_loss"],
            "B1_tier1_minus_B2_tier2_log_loss": test["B1|tier1"]["log_loss"]
            - test["B2|tier2"]["log_loss"],
        },
    }
    _write_json(_path(BENCHMARK, root), benchmark)
    _write_json(
        _path(BOOTSTRAP_SUMMARY, root),
        {
            "replications": config.BOOTSTRAP_PRODUCTION_B,
            "successful": int(replicates["replicate"].nunique()),
            "failures": len(failures),
            "interval": INTERVAL,
            "workers": BOOTSTRAP_WORKERS,
            "local_replicates_sha256": _sha(_path(LOCAL_REPLICATES, root)),
            "shuffle_check_operator_O": shuffle,
        },
    )
    timings["total_s"] = round(time.time() - started, 1)
    archive = [BENCHMARK, MOMENTS, BOOTSTRAP_SUMMARY, LEDGER]
    _write_json(
        _path(PROVENANCE, root),
        {
            "design_snapshot_sha256": snapshot_sha,
            "source_commit": commit,
            "data_sha256": data,
            "timings": timings,
            "archive_sha256": {p: _sha(_path(p, root)) for p in archive},
            "local_predictions_sha256": _sha(_path(LOCAL_PREDICTIONS, root)),
        },
    )
    return {"timings": timings, "ledger": ledger_state(root)}


def audit_real(root: Path = PROJECT_ROOT, replicate_checks: int = 5) -> dict:
    """Reproduce archived numbers from frozen inputs and recorded selections (no search)."""
    from threadpoolctl import threadpool_limits

    verify_snapshot(root)
    require(ledger_state(root) == "TEST_CONSUMED", "audit needs a consumed Run C test ledger")
    ledger = json.loads(_path(LEDGER, root).read_bytes())
    benchmark = json.loads(_path(BENCHMARK, root).read_bytes())
    archived = pd.read_csv(_path(MOMENTS, root))
    with _context("audit"):
        panel = load_real_panel(root)
        parts = _benchmark_frames(panel, root)
        with threadpool_limits(BENCHMARK_THREADS):
            fitted, _ = _fit_plan(panel, parts, selections=ledger["selected_before_test"])
        for key, model in fitted.items():
            require(model.params == ledger["selected_before_test"][key], f"selection drift: {key}")
            again = mdl.score(model, parts["test"])
            for metric in ("log_loss", "brier"):
                require(
                    abs(again[metric] - benchmark["test"][key][metric]) <= AUDIT_ATOL,
                    f"test metric not reproduced: {key} {metric}",
                )
        point = {_key(r): r["estimate"] for r in _moment_records(panel)}
        checks = []
        if _path(LOCAL_REPLICATES, root).is_file() and replicate_checks:
            local = pd.read_csv(_path(LOCAL_REPLICATES, root))
            for index, recs, err in _bootstrap(panel, range(replicate_checks), "audit"):
                require(err is None, f"audit replicate failed: {err}")
                ref = local.loc[local["replicate"] == index].set_index(
                    ["operator", "moment_id", "component"]
                )["estimate"]
                for o, m, c, e in recs:
                    require(
                        abs(e - ref.loc[(o, m, c)]) <= AUDIT_ATOL,
                        f"replicate {index} drift {m}:{c}",
                    )
                checks.append(index)
    for _, row in archived.loc[archived["moment_id"] != "E-1"].iterrows():
        require(
            abs(point[(row["operator"], row["moment_id"], row["component"])] - row["estimate"])
            <= AUDIT_ATOL,
            f"moment not reproduced: {row['operator']} {row['moment_id']} {row['component']}",
        )
    return {
        "benchmark_models_reproduced": sorted(fitted),
        "moments_reproduced": int(len(archived)),
        "bootstrap_replicates_reproduced": checks,
        "ledger": ledger["status"],
    }
