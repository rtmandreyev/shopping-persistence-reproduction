#!/usr/bin/env python
"""Export a compact, release-safe extract of the stored Phase 8 results (development side).

    uv run python scripts/export_phase8_compact.py OUT.json.gz

Reads the private Phase 8 checkpoint stores (``outputs/diagnostics/phase8_final/`` and
``outputs/diagnostics/phase8_stage1_p8005/``) and the real moment register, with its own
JSON loader (no project imports), and writes one gzipped JSON with:

- the real moments, their household-bootstrap SEs, E_j, the floors and the constant c;
- every family point table (coarse lattice, precomputed level-1 refinement points, the
  real-panel audit rows, level-2 points and the alternative-observation-rule tables): the
  per-point mean and SD of the five moments, the replicate count and calibration feasibility;
- the five moments of every stored RQ2 panel (200 per feasible truth and heterogeneity arm).

No household data, paths, hosts or credentials are included: these are aggregate moments of
the project's own simulations. The extract is what the reproduction package's independent
re-implementation of the adequacy rule reads (added after an independent audit, 2026-10-30).
"""

from __future__ import annotations

import csv
import glob
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs/diagnostics/phase8_final/checkpoints"
LEGACY = ROOT / "outputs/diagnostics/phase8_stage1_p8005/checkpoints"
P = ("P1", "P2", "P3", "P4", "P5")
REAL_IDS = {"P1": ("VI-2", "O-M3"), "P2": ("VI-3", "beta_8"), "P3": ("VI-3", "gamma_2"),
            "P4": ("VI-6", "delta_log_loss_recency"), "P5": ("VI-4", "active_1")}
ARMS = ("logit_normal", "mixture")
FAMS = ("A1_PRIME", "A2")


def vec(d):
    return [float(d[j]) for j in P]


def load_store(pattern):
    out = {}
    for f in glob.glob(str(pattern)):
        d = json.loads(Path(f).read_text())
        key = (d["family"], d["point"])
        if key in out:
            raise SystemExit(f"duplicate checkpoint {key}")
        out[key] = d["payload"]
    return out


def pooled(n1, m1, s1, rows):
    m1, s1 = np.asarray(m1), np.asarray(s1)
    rows = np.asarray(rows, float).reshape(-1, 5)
    n = n1 + rows.shape[0]
    mean = (n1 * m1 + rows.sum(0)) / n
    ss = (n1 - 1) * s1**2 + n1 * m1**2 + (rows**2).sum(0)
    var = (ss - n * mean**2) / (n - 1)
    return mean.tolist(), np.sqrt(np.maximum(var, 0)).tolist(), int(n)


def entry(m, s, n, f):
    return {"mean": list(map(float, m)), "sd": list(map(float, s)), "n": int(n), "feasible": bool(f)}


def key(c):
    return f"{c[0]:g},{c[1]:g}"


def main() -> int:
    out_path = Path(sys.argv[1])
    fin = load_store(FINAL / "*/*/*.json")
    leg = load_store(LEGACY / "*.json")
    vals, ses = {}, {}
    with (ROOT / "docs/phases/PHASE_07_RUN_C/moment_register_real.csv").open(newline="") as h:
        for row in csv.DictReader(h):
            if row["operator"] == "O_interior_window" and row["source"] == "real":
                for j, (m, c) in REAL_IDS.items():
                    if row["moment_id"] == m and row["component"] == c:
                        vals[j], ses[j] = float(row["estimate"]), float(row["bootstrap_se"])
    real, se = vec(vals), vec(ses)
    c = (norm.ppf(0.95) + norm.ppf((1 + 0.8 ** (1 / 5)) / 2)) * math.sqrt(1 + (1.25 * 1.2) ** 2 / 50) / 0.9
    a1ln = leg[("A1", "a1:logit_normal:-1,-1:0:0")]
    E = [r - m for r, m in zip(real, vec(a1ln["primary_mean"]), strict=True)]
    tables = {}
    for arm in ARMS:
        t = {}
        if arm == "logit_normal":
            t["A1"] = entry(vec(a1ln["primary_mean"]), vec(a1ln["primary_sd"]),
                            a1ln["replicates_ok"], a1ln["ci1_feasible"])
        else:
            m = leg[("A1", "map:mixture:-1,-1:0:0")]
            ext = fin[("A1", "FINAL:ext:mixture:-1,-1:0:0:")]
            t["A1"] = entry(*pooled(int(m["replicates_ok"]), vec(m["primary_mean"]),
                                    vec(m["primary_sd"]), ext["rows"]), m["ci1_feasible"])
        for f in FAMS:
            coarse, refined, ref2, audit = {}, {}, {}, {}
            for ip in range(12):
                for ia in range(12):
                    m = leg[(f, f"map:{arm}:{ip},{ia}:0:0")]
                    ext = fin[(f, f"FINAL:ext:{arm}:{ip},{ia}:0:0:")]
                    coarse[key((ip, ia))] = entry(*pooled(int(m["replicates_ok"]), vec(m["primary_mean"]),
                                                          vec(m["primary_sd"]), ext["rows"]),
                                                   m["ci1_feasible"])
            for (fam, pt), pay in fin.items():
                if fam != f:
                    continue
                parts = pt.split(":")
                if parts[1] in ("ref", "ref2", "audit") and parts[2] == arm:
                    a, b = parts[3].split(",")
                    k = key((float(a), float(b)))
                    if parts[1] == "audit":
                        audit[k] = [list(map(float, r)) for r in pay["rows"]]
                    else:
                        (refined if parts[1] == "ref" else ref2)[k] = entry(
                            vec(pay["primary_mean"]), vec(pay["primary_sd"]),
                            pay["replicates_ok"], pay["ci1_feasible"])
            t[f] = {"coarse": coarse, "refined": refined, "ref2": ref2, "audit_rows": audit}
        tables[arm] = t
    sa = {f: {"coarse": {}, "refined": {}} for f in FAMS}
    for (fam, pt), pay in fin.items():
        parts = pt.split(":")
        if parts[1] in ("sa", "saref") and fam in FAMS:
            a, b = parts[3].split(",")
            sa[fam]["coarse" if parts[1] == "sa" else "refined"][key((float(a), float(b)))] = entry(
                vec(pay["primary_mean"]), vec(pay["primary_sd"]), pay["replicates_ok"],
                pay["ci1_feasible"])
    chunks = {}
    for (_fam, pt), pay in fin.items():
        parts = pt.split(":")
        if parts[1] == "panels" and parts[6] == "rq2":
            arm, sub = parts[2], int(parts[5])
            chunks.setdefault((arm, pay["label"]), {})[sub] = [p["primary"] for p in pay["panels"]]
    panels = {arm: {} for arm in ARMS}
    for (arm, label), subs in sorted(chunks.items()):
        panels[arm][label] = [list(map(float, r)) for s in sorted(subs) for r in subs[s]]
    doc = {
        "schema": "phase8-compact/1",
        "moments": list(P),
        "real": real, "real_se": se, "E": E, "c": c, "floor": [c * s for s in se],
        "tables": tables, "sa_logit_normal": sa, "rq2_panels": panels,
    }
    data = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    out_path.write_bytes(gzip.compress(data, mtime=0))
    print(out_path, len(data), "bytes uncompressed;",
          sum(len(v) for v in panels.values()), "truth cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
