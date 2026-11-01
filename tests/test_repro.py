"""Tests of the reproduction checks (run with: uv run --frozen --extra dev pytest tests)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repro import adequacy as A  # noqa: E402
from repro import modes as M  # noqa: E402
from repro import numbers as N  # noqa: E402
from repro.report import FAIL, PASS, SKIP, Report  # noqa: E402

# --- adequacy rule ------------------------------------------------------------------------


def _rule():
    return A.Rule(E=[0.1] * 5, floor=[0.01] * 5)


def test_point_status_adequate_inadequate_neither():
    rule = _rule()
    M_ = np.array([[0.0] * 5, [0.2, 0, 0, 0, 0], [0.026, 0, 0, 0, 0]])
    st, ks, kj = rule.points(np.zeros(5), np.full(5, 0.001), M_, np.full((3, 5), 0.001),
                             np.full(3, 200.0), np.array([True, True, True]))
    assert [st[0.25][i] for i in range(3)] == ["A", "I", "N"]
    assert ks[0] == 0.0 and np.isfinite(ks[1])


def test_infeasible_points_never_count():
    rule = _rule()
    st, ks, _ = rule.points(np.zeros(5), np.full(5, 0.001), np.zeros((1, 5)), np.full((1, 5), 0.001),
                            np.array([200.0]), np.array([False]))
    assert st[0.25][0] == "X" and ks[0] == np.inf


def test_comparative_table_and_unresolved_default():
    assert A.comparative("ADEQUATE", "INADEQUATE") == "HISTORY_FAVOURED"
    assert A.comparative("INADEQUATE", "ADEQUATE") == "LATENT_FAVOURED"
    assert A.comparative("ADEQUATE", "ADEQUATE") == "EQUIVALENT"
    assert A.comparative("INADEQUATE", "INADEQUATE") == "NEITHER"
    assert A.comparative("ADEQUATE", "INCONCLUSIVE") == "UNRESOLVED"
    assert A.comparative("INCONCLUSIVE", "INADEQUATE") == "UNRESOLVED"


# --- manuscript number checker ------------------------------------------------------------


def _shadow(tmp_path: Path, tex_edit=None) -> Path:
    for name in ("docs", "outputs", "expected", "artifacts"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    (tmp_path / "paper").mkdir()
    tex = (ROOT / "paper/main.tex").read_text()
    if tex_edit:
        old, new = tex_edit
        assert old in tex
        tex = tex.replace(old, new, 1)
    (tmp_path / "paper/main.tex").write_text(tex)
    return tmp_path


def test_number_checker_passes_on_the_manuscript():
    res = N.check(ROOT)
    assert all(r["pass"] for r in res), [r["id"] for r in res if not r["pass"]]
    assert len(res) >= 90


@pytest.mark.parametrize("edit,cid", [
    (("& 0.174 & 0.112", "& 0.175 & 0.112"), "t2.M3"),
    (("13.32 & (0.10)", "13.31 & (0.10)"), "t4.j13"),
    (("& adequate & 0.000 & none", "& adequate & 0.000 & two-way lag"), "d1.mxA1p"),
])
def test_number_checker_catches_a_changed_number(tmp_path, edit, cid):
    res = {r["id"]: r for r in N.check(_shadow(tmp_path, edit))}
    assert res[cid]["pass"] is False


def test_round_half_up():
    assert N.fmt_half_up(64.5, ".0f") == "65"
    assert N.fmt_half_up(0.2845, ".3f") == "0.285"
    assert N.fmt_half_up(2595732, ",.0f") == "2,595,732"


# --- checksums, statuses and skips ----------------------------------------------------------


def test_checksums_detect_a_modified_file(tmp_path, monkeypatch):
    f = tmp_path / "a.txt"
    f.write_text("x")
    (tmp_path / "SHA256SUMS").write_text(f"{'0' * 64}  a.txt\n")
    monkeypatch.setattr(M, "ROOT", tmp_path)
    status, detail = M.checksums()
    assert status == FAIL
    (tmp_path / "SHA256SUMS").write_text(f"{M.sha(f)}  a.txt\n")
    assert M.checksums()[0] == PASS


def test_learned_model_simulations_without_archive_are_skipped_by_design(tmp_path):
    rep = Report("learned-model-simulations", tmp_path)
    M.learned_model_simulations(rep, tmp_path, None, 1, False)
    assert [r.status for r in rep.results] == [SKIP]


def test_standard_without_raw_data_fails_clearly(tmp_path, monkeypatch):
    rep = Report("standard", tmp_path)
    monkeypatch.setattr(M, "quick", lambda *a, **k: None)
    M.standard(rep, tmp_path, tmp_path / "nowhere", 1)
    assert rep.results[-1].status == FAIL and "data/README.md" in rep.results[-1].detail


def test_a_crashing_check_is_a_failure_not_a_skip(tmp_path):
    rep = Report("x", tmp_path)

    def boom():
        raise ValueError("broken")

    rep.run("crashes", boom)
    assert rep.failed and rep.results[0].status == FAIL


def test_reproduce_blocks_sockets_in_process(monkeypatch, tmp_path):
    import socket

    import reproduce

    monkeypatch.setattr(M, "quick", lambda *a, **k: None)
    assert reproduce.main(["--check", "--build-dir", str(tmp_path)]) == 0
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.create_connection(("example.org", 80))


def test_stored_panels_cover_every_frozen_known_truth():
    import json

    from repro import known_truth as K

    doc = K.load(ROOT / "artifacts")
    stored = json.loads((ROOT / "docs/phases/PHASE_08_FINAL/rq2_per_truth.json").read_text())
    want = {(a, lab) for a in stored for lab, t in stored[a].items() if t.get("comparative_counts")}
    got = {(a, lab) for a in doc["rq2_panels"] for lab in doc["rq2_panels"][a]}
    assert got == want and all(len(v) == 200 for a in doc["rq2_panels"] for v in doc["rq2_panels"][a].values())


def test_no_absolute_home_paths_in_package_text():
    bad = []
    for p in ROOT.rglob("*"):
        if p.is_file() and "build" not in p.parts and ".venv" not in p.parts and p.suffix in (
                ".py", ".md", ".json", ".csv", ".toml", ".sh", ".txt", ".tex", ".cff"):
            t = p.read_text(errors="ignore")
            if any(s in t for s in ("/Us" + "ers/", "/ho" + "me/")):
                bad.append(str(p.relative_to(ROOT)))
    assert bad == []


def test_workspace_copy_leaves_sha256sums_unchanged(tmp_path):
    before = (ROOT / "SHA256SUMS").read_text()
    ws = M.workspace(tmp_path)
    assert (ws / "reproduce.py").is_file() and not (ws / "build").exists()
    shutil.rmtree(ws)
    assert (ROOT / "SHA256SUMS").read_text() == before


def test_learned_model_simulations_with_a_wrong_archive_path_fail(tmp_path):
    rep = Report("learned-model-simulations", tmp_path)
    M.learned_model_simulations(rep, tmp_path, tmp_path / "does-not-exist", 1, False)
    assert [r.status for r in rep.results] == [FAIL]


@pytest.mark.parametrize("args,flag", [
    (["--data-dir", "x"], "--data-dir"),
    (["--check", "--data-dir", "x"], "--data-dir"),
    (["--standard", "--archive", "x"], "--archive"),
    (["--quick", "--checkpoint-stores", "x"], "--checkpoint-stores"),
    (["--standard", "--small-scale"], "--small-scale"),
    (["--model-queries", "--workers", "2"], "--workers"),
    (["--known-truth-production", "--data-dir", "x"], "--data-dir"),
])
def test_an_option_the_mode_does_not_use_is_rejected(args, flag, tmp_path, capsys):
    import reproduce

    with pytest.raises(SystemExit) as e:
        reproduce.main([*args, "--build-dir", str(tmp_path)])
    assert e.value.code == 2 and flag in capsys.readouterr().err
    assert not (tmp_path / "reproduction_report_quick.json").exists()


def test_options_the_mode_uses_are_accepted(monkeypatch, tmp_path):
    import reproduce

    seen = {}
    monkeypatch.setattr(M, "quick", lambda *a, **k: None)
    monkeypatch.setattr(M, "standard", lambda rep, b, d, w: seen.update(data=d, workers=w))
    monkeypatch.setattr(M, "learned_model_simulations",
                        lambda rep, b, arc, w, small, stores: seen.update(archive=arc, small=small, stores=stores))
    assert reproduce.main(["--check", "--workers", "2", "--build-dir", str(tmp_path)]) == 0
    assert reproduce.main(["--standard", "--data-dir", "d", "--workers", "3", "--build-dir", str(tmp_path)]) == 0
    assert seen["data"] == Path("d") and seen["workers"] == 3
    assert reproduce.main(["--learned-model-simulations", "--archive", "a", "--checkpoint-stores", "s",
                           "--small-scale", "--build-dir", str(tmp_path)]) == 0
    assert seen["archive"] == Path("a") and seen["small"] and seen["stores"] == Path("s")


def test_workers_must_be_positive(tmp_path):
    import reproduce

    with pytest.raises(SystemExit) as e:
        reproduce.main(["--check", "--workers", "0", "--build-dir", str(tmp_path)])
    assert e.value.code == 2


def _fake_archive(path: Path) -> Path:
    for p in ("P-J13", "P-J14", "P-M"):
        (path / "production" / p).mkdir(parents=True)
    return path


def test_learned_model_workspace_follows_the_options_given(tmp_path, monkeypatch):
    import subprocess as sp

    monkeypatch.setattr(M.subprocess, "run", lambda *a, **k: sp.CompletedProcess(a, 1, "", "stub"))
    monkeypatch.setattr(M, "_git_init", lambda ws: None)
    build, a1, a2 = tmp_path / "b", _fake_archive(tmp_path / "a1"), _fake_archive(tmp_path / "a2")
    stores = tmp_path / "s"
    for n in ("phase8_final/checkpoints", "phase8_stage1_p8005"):
        (stores / n).mkdir(parents=True)
    rep = Report("learned-model-simulations", build)
    M.learned_model_simulations(rep, build, a1, 1, True)
    assert [r.status for r in rep.results] == [FAIL, SKIP]  # direct run fails (stub); anchored needs stores
    rep = Report("learned-model-simulations", build)
    M.learned_model_simulations(rep, build, a1, 1, True, stores)  # stores added on a second run are used
    assert [r.status for r in rep.results] == [FAIL, FAIL]
    assert (build / "learned_model_workspace/outputs/diagnostics/phase8_final").resolve() == (stores / "phase8_final").resolve()
    rep = Report("learned-model-simulations", build)
    M.learned_model_simulations(rep, build, a2, 1, True)  # a different archive is refused, not ignored
    assert [r.status for r in rep.results] == [FAIL] and "different --archive" in rep.results[0].detail


def test_section7_rebuild_is_skipped_unless_both_full_runs_reproduce(tmp_path, monkeypatch):
    import subprocess as sp

    monkeypatch.setattr(M.subprocess, "run", lambda *a, **k: sp.CompletedProcess(a, 1, "", "stub"))
    monkeypatch.setattr(M, "_git_init", lambda ws: None)
    rep = Report("learned-model-simulations", tmp_path / "b")
    M.learned_model_simulations(rep, tmp_path / "b", _fake_archive(tmp_path / "a"), 1, False)
    assert rep.results[-1].check.startswith("Section 7 tables") and rep.results[-1].status == SKIP


def test_expected_empirical_values_equal_the_frozen_fixed_effects_table():
    import csv
    import json

    exp = json.loads((ROOT / "expected/empirical_values.json").read_text())
    t4 = {r["model"]: r for r in csv.DictReader((ROOT / "outputs/tables/panel/final/t4_core_panel_results.csv").open())
          if r["outcome"] == "retailer_purchase"}
    for m in ("M0", "M1", "M2", "M3"):
        assert abs(exp[f"panel.{m}_full"] - float(t4[m]["coefficient"])) < 1e-9
    assert abs(exp["panel.M3_full_se_dk8"] - float(t4["M3"]["paper_facing_se"])) < 1e-9
    assert abs(exp["panel.M3_full_se_twoway"] - float(t4["M3"]["original_two_way_se"])) < 1e-9
