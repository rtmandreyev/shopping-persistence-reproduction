"""The checks behind each mode of ``reproduce.py``.

Directory names under ``docs/phases/`` keep the study's stage labels: ``PHASE_07_RUN_C`` holds
the prediction benchmark and the real panel's dynamic moments (Section 4), the ``PHASE_08_*``
folders the in-family known truths and the real-panel comparison (Section 6, Appendix D), and
``PHASE_09`` the learned-model truths (Section 7, Appendices E and F).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from repro.report import FAIL, PASS, SKIP, WARN, Report

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RAW_FILES = (
    "campaign_desc.csv", "campaign_table.csv", "causal_data.csv", "coupon.csv",
    "coupon_redempt.csv", "hh_demographic.csv", "product.csv", "transaction_data.csv",
)
EXPECTED_PAGES = 40
LEARNED_MODEL_PINS = {  # results_sha256 of the frozen learned-model simulation runs
    "phase9_local": "963e65e9172cd884361c39d49ec1b543dd529989d10c25b1ac6558b0854c7c69",
    "phase9_p9007": "a38d979057649663bda47561568b29ada5a8fd9ecc9d1ed847d43ae0208fa5ac",
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --- shared checks ------------------------------------------------------------------------


def checksums():
    lines = (ROOT / "SHA256SUMS").read_text().splitlines()
    bad = [rel for h, rel in (ln.split("  ", 1) for ln in lines if ln.strip())
           if not (ROOT / rel).is_file() or sha(ROOT / rel) != h]
    return (FAIL, f"{len(bad)} of {len(lines)} files differ: {bad[:5]}") if bad else (
        PASS, f"{len(lines)} files match SHA256SUMS")


def numbers():
    from repro import numbers as N

    res = N.check(ROOT)
    bad = [r for r in res if not r["pass"]]
    notes = [f"{r['id']}: {r['note']}" for r in res if r["note"]]
    detail = f"{len(res) - len(bad)} of {len(res)} manuscript numbers match their canonical sources"
    if bad:
        return FAIL, detail + "; mismatches: " + "; ".join(f"{r['id']} expected '{r['expected_phrase']}'" for r in bad[:5]), {"results": res}
    if notes:
        detail += "; documented notes: " + " | ".join(notes)
    return PASS, detail, {"results": res, "notes": notes}


def known_truth_verdicts(workers):
    from repro import known_truth as K

    rec = K.recompute(ART, workers=workers)
    bad = K.compare(ART, rec)
    n = sum(len(v) for v in rec.values())
    if bad:
        return FAIL, f"{len(bad)} mismatches: {bad[:5]}"
    return PASS, f"all {n} known truths (200 panels each) reproduce the frozen verdict, family and naive-rule counts"


def real_panel_comparison():
    from repro import known_truth as K
    from repro import real_panel as RP

    rec = RP.recompute(K.load(ART))
    stored = json.loads((ROOT / "docs/phases/PHASE_08_FINAL/real_rq1.json").read_text())
    bad = RP.compare(rec, stored)
    alt = set(rec["alternative_rule_logit_normal"].values())
    if alt != {"UNRESOLVED"}:
        bad.append(f"alternative observation rule gave {alt}")
    return (FAIL, f"mismatches: {bad}") if bad else (
        PASS, "Table D.1 verdicts, kappa*, binding moments, finer-grid analysis and alternative rule reproduced",
        {"recomputed": rec})


def grid_diagnostic(workers):
    """The all-points diagnostic of Appendix D.7, which is not part of the recorded design."""
    from repro import known_truth as K

    rec = K.recompute(ART, all_points=True, workers=workers)
    exp = json.loads((ROOT / "expected/grid_diagnostic.json").read_text())
    got = {f"{a}|{lab}": rec[a][lab]["comparative"]["0.25"].get("HISTORY_FAVOURED", 0) for a, lab in
           [tuple(k.split("|")) for k in exp["history_favoured_k025"]]}
    fa = sum(rec[a][lab]["comparative"][k].get("HISTORY_FAVOURED", 0) for a in rec for lab in rec[a]
             for k in ("0.15", "0.25", "0.35") if lab in exp["history_free_truths"][a])
    ok = got == exp["history_favoured_k025"] and fa == 0
    return (PASS if ok else FAIL), f"all-points detection {got}; history-free history-favoured = {fa}"


def learned_model_consistency():
    """Cross-checks of the shipped learned-model aggregates (the responses are not shipped)."""
    fs = json.loads((ROOT / "docs/phases/PHASE_09/final/final_summary.json").read_text())
    l2 = json.loads((ROOT / "docs/phases/PHASE_09/results_p9007/level2_code_path_check.json").read_text())
    p8 = json.loads((ROOT / "docs/phases/PHASE_08_FINAL/rq2_results.json").read_text())
    bad = []
    for key, v in l2.items():
        arm, label = key.split(":", 1)
        if v["phase9_wrapper"] != p8[arm]["DET"]["0.25"]["DET_OVERALL"][label]["rate"]:
            bad.append(f"code-path check {key}")
    if any(r["max_hf_any_kappa"] != 0 for r in fs["tableB"]):
        bad.append("a learned truth was classified history-favoured")
    for r in fs["tableB"]:
        if abs(sum(r["verdicts_k025"].values()) - 1) > 0.002:
            bad.append(f"verdict shares of {r['policy']} {r['class']} do not sum to 1")
    return (FAIL, "; ".join(bad)) if bad else (
        PASS, "the Section 7 code path reproduces the in-family detection rates; no learned truth is "
              "history-favoured; verdict shares sum to 1")


def clopper_pearson():
    from scipy.stats import beta

    hi = float(beta.ppf(0.975, 1, 200))
    return (PASS, f"0 of 200: exact upper bound {hi:.4f}") if f"{hi:.3f}" == "0.018" else (FAIL, f"{hi}")


def figures(build: Path):
    out = build / "figures"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/paper_publication_figures.py"), "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if r.returncode != 0:
        return FAIL, (r.stdout + r.stderr).strip()[-400:]
    names = ("fig1_known_truth_operating_characteristics.pdf", "fig2_naive_flag_by_mechanism.pdf",
             "fig3_in_family_vs_out_of_family.pdf")
    differ = [n for n in names if sha(out / n) != sha(ROOT / "paper/figures" / n)]
    if differ:
        import matplotlib

        return WARN, (f"inputs hash-verified and figures rendered to {out}; bytes differ from the published "
                      f"PDFs for {differ} (matplotlib {matplotlib.__version__}; byte identity is expected only on "
                      "the original platform). Compare visually.")
    return PASS, f"all three figures re-rendered byte-identically into {out}"


def paper(build: Path):
    tool = shutil.which("tectonic")
    if not tool:
        return SKIP, "tectonic is not installed; the shipped paper/main.pdf is verified by checksum instead"
    work = build / "paper"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(ROOT / "paper", work)
    # -C (--only-cached): never download TeX resources during reproduction.
    r = subprocess.run([tool, "-C", "-k", "main.tex"], cwd=work, capture_output=True, text=True)
    if r.returncode != 0:
        log = (r.stdout + r.stderr).lower()
        if "cache" in log or "bundle" in log or "network" in log or "download" in log:
            return SKIP, ("tectonic has no cached TeX bundle, and reproduction never downloads one; "
                          "see README.md (Software) for the one-time setup")
        return FAIL, (r.stdout + r.stderr)[-400:]
    pages = None
    if shutil.which("pdfinfo"):
        info = subprocess.run(["pdfinfo", str(work / "main.pdf")], capture_output=True, text=True).stdout
        pages = int(next(x.split()[-1] for x in info.splitlines() if x.startswith("Pages:")))
    if pages is not None and pages != EXPECTED_PAGES:
        return FAIL, f"compiled PDF has {pages} pages, expected {EXPECTED_PAGES}"
    return PASS, f"paper compiled into {work / 'main.pdf'}" + (f" ({pages} pages)" if pages else "")


# --- modes -------------------------------------------------------------------------------


def quick(rep: Report, build: Path, workers: int, verify_only: bool = False, standard: bool = False):
    rep.run("checksums of shipped files", checksums)
    rep.run("manuscript numbers vs canonical sources", numbers)
    rep.run("known-truth verdicts recomputed from stored moments (Section 6)", known_truth_verdicts, workers)
    rep.run("real-panel comparison recomputed (Table D.1)", real_panel_comparison)
    rep.run("grid-resolution diagnostic recomputed (Appendix D.7)", grid_diagnostic, workers)
    rep.run("learned-model aggregates mutually consistent (Section 7)", learned_model_consistency)
    rep.run("exact binomial interval for 0 of 200", clopper_pearson)
    if verify_only:
        rep.add("figures and paper compilation", SKIP, "not run in --check (verification only)")
        return
    rep.run("publication figures re-rendered from hash-pinned inputs", figures, build)
    rep.run("paper compiles", paper, build)
    if not standard:
        rep.add("raw-data reproduction", SKIP, "needs the Complete Journey files: run --standard (data/README.md)")
    rep.add("learned-model simulations", SKIP,
            "need an authorized copy of the stored responses: run --learned-model-simulations")


DATASET = "dunnhumby_The-Complete-Journey"
CSV_DIR = "dunnhumby_The-Complete-Journey CSV"


def find_dataset(data_dir: Path) -> Path | None:
    """The folder that holds the user guide PDF and the CSV subfolder (see data/README.md)."""
    for d in (data_dir, data_dir / DATASET, data_dir / "raw" / DATASET):
        if all((d / CSV_DIR / f).is_file() for f in RAW_FILES):
            return d
    return None


def raw_manifest(dataset: Path):
    import csv

    rows = list(csv.DictReader((ROOT / "data/raw/manifest_raw.csv").open(newline="")))
    bad = []
    for r in rows:
        rel = r["relative_path"].strip()
        f = dataset / Path(rel).relative_to(DATASET)
        if not f.is_file():
            bad.append(f"missing {rel}")
        elif f.stat().st_size != int(r["size_bytes"]) or sha(f) != r["sha256"].strip():
            bad.append(f"differs {rel}")
    return (FAIL, "; ".join(bad)) if bad else (
        PASS, f"all {len(rows)} raw files (8 CSV files and the user guide) match data/raw/manifest_raw.csv")


def empirical(raw: Path):
    from repro import empirical as E

    got = E.compute(raw)
    exp = json.loads((ROOT / "expected/empirical_values.json").read_text())
    tol = json.loads((ROOT / "expected/tolerances.json").read_text())
    bad = []
    for k, v in exp.items():
        t = next((x for p, x in tol.items() if k.startswith(p)), tol["default"])
        if k not in got or abs(got[k] - v) > t:
            bad.append(f"{k}: got {got.get(k)} expected {v} (tol {t})")
    return (FAIL, "; ".join(bad[:8])) if bad else (
        PASS, f"{len(exp)} empirical quantities (Sections 2-4, Appendices A-C) reproduced within tolerance",
        {"values": got})


def _ignore(src, names):
    """Skip build outputs, environments and caches; skip the top-level data/ folder only."""
    skip = {"build", ".venv", ".git", "__pycache__", ".pytest_cache"}
    if Path(src).resolve() == ROOT.resolve():
        skip |= {"data"}
    return [n for n in names if n in skip]


def workspace(build: Path) -> Path:
    """A disposable copy of the package for commands that write into the tree."""
    ws = build / "workspace"
    if ws.exists():
        shutil.rmtree(ws)
    shutil.copytree(ROOT, ws, ignore=_ignore)
    (ws / "data/raw").mkdir(parents=True)
    shutil.copy2(ROOT / "data/raw/manifest_raw.csv", ws / "data/raw/manifest_raw.csv")
    _git_init(ws)
    return ws


def _git_init(ws: Path) -> None:
    """Some frozen code records ``git rev-parse HEAD``; give the copy a local, remote-less repo."""
    git = shutil.which("git")
    if git is None:
        return
    env = {**os.environ, "GIT_AUTHOR_NAME": "reproduction", "GIT_AUTHOR_EMAIL": "reproduction@example.invalid",
           "GIT_COMMITTER_NAME": "reproduction", "GIT_COMMITTER_EMAIL": "reproduction@example.invalid"}
    subprocess.run([git, "init", "-q"], cwd=ws, check=True, env=env)
    subprocess.run([git, "commit", "-q", "--allow-empty", "-m", "reproduction workspace"], cwd=ws, check=True, env=env)


def canonical_layer(build: Path, dataset: Path):
    ws = workspace(build)
    (ws / "data/raw" / DATASET).symlink_to(dataset.resolve(), target_is_directory=True)
    t = time.time()
    r = subprocess.run([sys.executable, "scripts/build_canonical_layer.py"], cwd=ws, capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if r.returncode != 0:
        return FAIL, f"processed data build failed (exit {r.returncode}): {(r.stdout + r.stderr)[-600:]}"
    ledger = json.loads((ROOT / "docs/phases/PHASE_07_RUN_C/test_ledger.json").read_text())
    pins = {Path(k).name: v for k, v in ledger["data_sha256"].items()}
    got = {p.name: sha(p) for p in (ws / "data/processed").glob("*.parquet")}
    matched = [n for n, h in pins.items() if got.get(n) == h]
    missing = [n for n in pins if n not in got]
    differ = [n for n in pins if n in got and got[n] != pins[n]]
    if differ or missing:
        return FAIL, f"processed tables differ from their recorded hashes: differ {differ}, missing {missing}"
    return PASS, (f"processed data rebuilt in {time.time() - t:.0f} s; the {len(matched)} tables whose "
                  f"hashes were recorded when the study ran are byte-identical ({', '.join(matched)}); "
                  f"{len(got)} tables built in {ws / 'data/processed'} (household-level derived data: keep private)")


def standard(rep: Report, build: Path, data_dir: Path, workers: int):
    dataset = find_dataset(data_dir)
    if dataset is None:  # report missing data before the slower checks
        rep.add("raw Complete Journey files", FAIL,
                f"not found under {data_dir}; see data/README.md for the required files and layout")
        return
    quick(rep, build, workers, standard=True)
    rep.run("raw files match the manifest", raw_manifest, dataset)
    rep.run("empirical results recomputed from raw data (independent implementation)", empirical, dataset / CSV_DIR)
    rep.run("processed data rebuilt with the study's code", canonical_layer, build, dataset)


def learned_model_simulations(rep: Report, build: Path, archive: Path | None, workers: int, small: bool,
                              stores: Path | None = None):
    """Rerun the Section 7 simulations from an authorized copy of the stored responses.

    The direct run rebuilds the model-driven truths; the anchored run (Section 7.5) also needs
    the in-family checkpoint stores, because its model comparison reads the frozen family tables.
    Both are checkpointed, so an interrupted run resumes in the same workspace.
    """
    if small:
        print(f"Learned-model simulations at small scale (minutes); {workers} workers.", flush=True)
    else:
        print("Learned-model simulations: the original runs took 2.27 h (direct) and 7.52 h (anchored) of wall "
              f"time on 8 workers; this run uses {workers}.", flush=True)
    if archive is None:
        rep.add("learned-model simulations", SKIP,
                "no response archive supplied (--archive DIR); the stored responses are not redistributed "
                "(NOTICE.md)")
        return
    if not archive.is_dir():
        rep.add("learned-model simulations", FAIL, f"--archive {archive} does not exist")
        return
    if not all((archive / "production" / p).is_dir() for p in ("P-J13", "P-J14", "P-M")):
        rep.add("learned-model simulations", FAIL,
                f"{archive} is not a response archive: production/P-J13, production/P-J14 and production/P-M "
                "were not found in it")
        return
    if stores is not None and not all((stores / n).is_dir() for n in ("phase8_final", "phase8_stage1_p8005")):
        rep.add("learned-model simulations", FAIL,
                f"{stores} does not contain the checkpoint stores phase8_final/ and phase8_stage1_p8005/")
        return
    ws = build / "learned_model_workspace"
    if not ws.exists():  # an existing workspace is kept so that checkpoints resume
        shutil.copytree(ROOT, ws, ignore=_ignore)
        _git_init(ws)
        (ws / "outputs/decision_logs").mkdir(parents=True, exist_ok=True)
        (ws / "outputs/decision_logs/phase9").symlink_to(archive.resolve(), target_is_directory=True)
        if stores is not None:
            (ws / "outputs/diagnostics").mkdir(parents=True, exist_ok=True)
            for name in ("phase8_final", "phase8_stage1_p8005"):
                (ws / "outputs/diagnostics" / name).symlink_to((stores / name).resolve(), target_is_directory=True)
    have_stores = (ws / "outputs/diagnostics/phase8_final/checkpoints").exists()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    scale = "small scale" if small else "full scale"
    for anchored, label in ((False, "phase9_local"), (True, "phase9_p9007")):
        name = f"{'anchored' if anchored else 'direct'} learned-model run ({scale})"
        if anchored and not have_stores:
            rep.add(name, SKIP, "needs the in-family checkpoint stores (--checkpoint-stores DIR)")
            continue
        cmd = [sys.executable, "scripts/systemone_local_run.py", "--engineering" if small else "--execute",
               "--workers", str(workers)] + (["--p9007"] if anchored else [])
        t = time.time()
        r = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            rep.add(name, FAIL, (r.stdout + r.stderr)[-600:], time.time() - t)
            continue
        suffix = "_eng" if small else ""
        res = json.loads((ws / f"outputs/diagnostics/{label}{suffix}/results.json").read_text())
        if small:
            rep.add(name, PASS, "completed at small scale (not compared with the frozen results)", time.time() - t)
        elif res["results_sha256"] == LEARNED_MODEL_PINS[label]:
            rep.add(name, PASS, f"results digest equals the frozen value {LEARNED_MODEL_PINS[label][:16]}",
                    time.time() - t)
        else:
            rep.add(name, FAIL, f"results digest {res['results_sha256'][:16]} differs from the frozen value",
                    time.time() - t)
    if not small:
        r = subprocess.run([sys.executable, "scripts/phase9_final_artifacts.py", "--check"], cwd=ws,
                           capture_output=True, text=True, env=env)
        rep.add("Section 7 tables rebuilt from the regenerated results", PASS if r.returncode == 0 else FAIL,
                (r.stdout + r.stderr).strip()[-300:])


KNOWN_TRUTH_TEXT = """The in-family known-truth simulations (Section 6) are not rerun by this package.

The original production run consisted of 3,948 jobs and took 37.4 hours of wall time on six worker
processes (macOS on Apple silicon); the jobs used 199.7 hours of processor time in total. Its runner (scripts/final_run.py) checkpoints every job and draws from
seeded streams, but it is bound to the frozen design snapshot and refuses to start a stage that
has already been run. A new run would therefore be a new experiment.

What is verified instead: --quick recomputes every verdict of Section 6 and Appendix D (53
known truths of 200 panels each, and the real-panel comparison) from the stored per-panel
moments in artifacts/known_truth_panels.json.gz, with an implementation written independently
of the study's code, and requires exact agreement with the frozen records.
"""

MODEL_QUERIES_TEXT = """The original learned-model queries (Section 7) are not repeated. No call is made.

The responses were collected in October 2026 from pinned model versions: jev-1.13.0 and
jev-1.14.0 through TypeSafe's interface, and inception/mercury-decide-20260930 through
OpenRouter with Inception as the only provider. Each model answered the same 3,368 requests.
Hosted versions can change or be retired, so exact regeneration may no longer be possible.
Every simulation in the paper is built from the stored responses, which are not redistributed
while the providers' redistribution terms are unconfirmed (NOTICE.md). The file
docs/phases/PHASE_09/production_archive_freeze.json records their tree hashes and digests, so
that an authorized copy can be verified exactly (--learned-model-simulations --archive DIR).
"""
