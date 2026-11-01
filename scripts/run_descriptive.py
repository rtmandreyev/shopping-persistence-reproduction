"""Phase 3 descriptive-analysis entry point.

Run 1 supports **infrastructure validation only**. There is deliberately no
subcommand that produces a Phase 3 figure, table or finding: later runs extend this
entry point with their own bounded subcommands.

Usage:

    PYTHONPATH= .venv/bin/python scripts/run_descriptive.py --check
    PYTHONPATH= .venv/bin/python scripts/run_descriptive.py --check-department

``--question-a-final`` rebuilds the weekly series behind the two approved figures
and publishes them into ``outputs/figures/descriptive/final/``: each figure is built
twice and published only if the two PNG byte streams are identical, its text and
bytes are checked for customer identifiers, and the publication policy is re-verified
(the approved figures allowlisted by name, every exploratory artifact still ignored).

``--check`` runs five gates and exits without creating anything:
  1. every accepted Phase 2 Parquet artifact still matches its contract;
  2. the reconciliation gate — accepted headline totals recomputed by independent
     routes (these are accepted Phase 2 anchors, not Phase 3 findings);
  3. the output policy — exploratory artifacts untracked, no final artifact
     automatically trackable, journey mapping local-only;
  4. the challenge register — schema valid and empty (no finding exists);
  5. no substantive Phase 3 output exists anywhere.

``--check-department`` builds the neutral department derivation in a throwaway
connection and validates its contract (grain, 1:1 join, fan-out, key uniqueness,
blank-label preservation, totality reconciliation). It publishes nothing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from complete_journey.descriptive import (
    artifacts,
    challenges,
    department,
    final_figures,
    question_a,
    question_b,
    question_c,
    question_d,
    question_e,
    reconcile,
    synthesis,
)
from complete_journey.paths import PROJECT_ROOT

#: Deliberately UNAPPROVED artifact names, used only to probe the policy. They must
#: stay untrackable even now that two approved figures are allowlisted by name.
_PROBE_FIGURE = "probe_unapproved_figure.png"
_PROBE_TABLE = "probe_unapproved_table.csv"

#: Phase 3 descriptive output scopes, in exploratory/final pairs.
_DESCRIPTIVE_DIRS = (
    "outputs/figures/descriptive/exploratory",
    "outputs/figures/descriptive/final",
    "outputs/tables/descriptive/exploratory",
    "outputs/tables/descriptive/final",
)


def _step(number: int, total: int, title: str) -> None:
    print(f"\n[{number}/{total}] {title}")


def check_accepted_inputs() -> None:
    checks = reconcile.verify_accepted_artifacts()
    print(f"  accepted artifacts verified: {len(checks)}/{len(checks)} match their contracts")
    for check in checks:
        print(f"    {check.table:<36} rows={check.rows}")


def check_reconciliation_gate() -> None:
    checks = reconcile.run_reconciliation_gate()
    print(reconcile.format_gate_report(checks))
    print("  gate PASSED -- inputs reconcile; no descriptive result may be published otherwise")


def check_output_policy() -> None:
    probes: tuple[tuple[artifacts.ArtifactKind, str], ...] = (
        ("figure", _PROBE_FIGURE),
        ("table", _PROBE_TABLE),
    )
    for kind, probe in probes:
        artifacts.assert_exploratory_untracked(kind, probe)
        artifacts.assert_final_not_auto_trackable(kind, probe)
        print(f"  {kind}: exploratory probe ignored, final probe not auto-trackable")
    artifacts.assert_local_only(artifacts.journey_mapping_path())
    print(
        f"  journey mapping stays local: {artifacts.journey_mapping_path().relative_to(PROJECT_ROOT)}"
    )
    approved = {
        f"outputs/figures/descriptive/final/{name}" for name in final_figures.FINAL_FIGURE_FILENAMES
    } | {f"outputs/tables/descriptive/final/{name}" for name in final_figures.FINAL_TABLE_FILENAMES}
    tracked = _tracked_descriptive_files()
    unexpected = [
        path for path in tracked if not path.endswith(".gitkeep") and path not in approved
    ]
    if unexpected:
        raise SystemExit(f"STOP: unexpected tracked descriptive artifact(s): {unexpected}")
    print(
        f"  tracked files under outputs/*/descriptive: {len(tracked)} "
        f"(.gitkeep placeholders + {len([p for p in tracked if p in approved])} approved figure(s))"
    )


def check_challenge_register() -> None:
    """The register must match the researcher-approved promotion set exactly.

    Promotion is a researcher decision: a run may not promote its own candidate, and it
    may not quietly drop or downgrade an approved one either.
    """
    records = challenges.read_register()
    promoted = tuple(record.candidate_id for record in challenges.promoted_records(records))
    if promoted != challenges.APPROVED_PROMOTED_IDS:
        raise SystemExit(
            f"STOP: promoted records are {list(promoted)}, but the researcher-approved set is "
            f"{list(challenges.APPROVED_PROMOTED_IDS)}; promotion and demotion are researcher "
            "decisions, not run outputs"
        )
    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    print(f"  register schema valid, records={len(records)}, statuses={counts}")
    print(f"  promoted (researcher-approved): {list(promoted)}")
    print(f"  register path: {challenges.REGISTER_PATH.relative_to(PROJECT_ROOT)}")


def check_publication_state() -> None:
    """Exploratory stays ignored; the final scope holds only allowlisted approvals.

    From Run 3 two researcher-approved figures are published deliberately, so the
    invariant is no longer "the final scope is empty" but "nothing reaches the final
    scope without an explicit per-path allowlist entry, and an allowlisted artifact is
    one the researcher approved by name".
    """
    approved_figures = set(final_figures.FINAL_FIGURE_FILENAMES)
    approved_tables = set(final_figures.FINAL_TABLE_FILENAMES)
    problems: list[str] = []
    counts: dict[str, int] = {}
    for relative in _DESCRIPTIVE_DIRS:
        counts[relative] = 0
        for path in sorted((PROJECT_ROOT / relative).iterdir()):
            if path.name == ".gitkeep":
                continue
            counts[relative] += 1
            ignored, _ = artifacts.git_ignore_status(path)
            if relative.endswith("exploratory"):
                if not ignored:
                    problems.append(f"exploratory artifact is trackable: {path.name}")
            elif "/figures/" in relative:
                if ignored:
                    problems.append(f"final figure is not allowlisted: {path.name}")
                if path.name not in approved_figures:
                    problems.append(f"unapproved artifact in the final figure scope: {path.name}")
            else:
                if ignored:
                    problems.append(f"final table is not allowlisted: {path.name}")
                if path.name not in approved_tables:
                    problems.append(f"unapproved artifact in the final table scope: {path.name}")
    if problems:
        raise SystemExit(f"STOP: publication policy violated: {problems}")
    exploratory_counts = {
        relative: count for relative, count in counts.items() if relative.endswith("exploratory")
    }
    final_counts = {
        relative: count for relative, count in counts.items() if relative.endswith("final")
    }
    print(f"  exploratory artifacts all Git-ignored: {exploratory_counts}")
    print(
        f"  final scope holds {sum(final_counts.values())} artifact(s), each allowlisted and "
        f"approved: figures {sorted(approved_figures)}, tables {sorted(approved_tables)}"
    )


def _tracked_descriptive_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "outputs/tables/descriptive", "outputs/figures/descriptive"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def run_check() -> int:
    total = 5
    _step(1, total, "accepted Phase 2 inputs")
    check_accepted_inputs()
    _step(2, total, "reconciliation gate")
    check_reconciliation_gate()
    _step(3, total, "output and publication policy")
    check_output_policy()
    _step(4, total, "challenge-record infrastructure")
    check_challenge_register()
    _step(5, total, "publication state")
    check_publication_state()
    print(
        "\nPhase 3 infrastructure check PASSED. This check produced no descriptive "
        "statistic, figure, table or finding of its own."
    )
    return 0


def run_check_department() -> int:
    _step(1, 1, "department derivation (validation only; nothing is published)")
    con = reconcile.open_connection()
    try:
        department.register_sources(con)
        report = department.validate(con)
        print(department.format_report(report))
    finally:
        con.close()
    print(
        "\nDepartment derivation contract validated in a throwaway connection. "
        "No department result was interpreted, ranked or published."
    )
    return 0


def run_question_a() -> int:
    """Question A: retailer activity over time (descriptive, no promotion)."""
    total = 3
    _step(0, total, "reconciliation gate (must pass before analysis)")
    con = reconcile.open_connection()
    try:
        reconcile.register_accepted_views(con)
        checks = reconcile.run_reconciliation_gate(con)
        print(f"  gate PASSED ({len(checks)} accepted anchors, all routes agree)")

        _step(1, total, "department derivation (temporary view for the composition check)")
        department.register_sources(con)
        department.build_view(con)
        totality = department.totality_residuals(con)
        print(
            f"  view rows={totality.rows} department labels={totality.distinct_departments} "
            f"units residual={totality.units_residual} "
            f"retailer SALES_VALUE residual={totality.sales_value_residual!r}"
        )

        _step(2, total, "Question A weekly series, summaries and exploratory outputs")
        result = question_a.build_question_a(con)
        print(question_a.format_report(result))
    finally:
        con.close()
    print(
        "\nQuestion A exploratory pass complete. These outputs are untracked local working "
        "files: re-running this pass republishes and promotes nothing. The accepted findings, "
        "their published figures and the allowlist are maintained separately "
        "(docs/phases/PHASE_03_DESCRIPTIVE_ANALYTICS.md, section 16)."
    )
    return 0


def run_question_b_final() -> int:
    """Publish the two researcher-approved Question B figures (F3, F4)."""
    total = 3
    _step(1, total, "reconciliation gate (must pass before publication)")
    con = reconcile.open_connection()
    try:
        reconcile.register_accepted_views(con)
        checks = reconcile.run_reconciliation_gate(con)
        print(f"  gate PASSED ({len(checks)} accepted anchors, all routes agree)")

        _step(2, total, "household-level object behind the approved figures")
        household = question_b.load_household_level(con)
        question_b.reconcile_household_object(household)
        print(
            f"  households={len(household)} "
            f"total retailer SALES_VALUE={household['total_sales_value'].sum():.2f} "
            f"(object reconciles to the accepted anchors)"
        )

        _step(3, total, "publish F3 and F4 (deterministic, identifier-checked)")
        published = final_figures.publish_question_b_figures(household)
        for figure in published:
            print(
                f"  {figure.name}: sha256={figure.sha256[:16]} "
                f"reproducible={figure.reproducible} identifiers={figure.no_identifiers} "
                f"path={figure.path.relative_to(PROJECT_ROOT)}"
            )
    finally:
        con.close()
    print(
        "\nApproved Question B figures published. No other Phase 3 artifact was promoted, "
        "no exploratory output became trackable, and no persistence figure was added."
    )
    return 0


def run_question_c_final() -> int:
    """Publish the approved Question C artifacts: F5, F6, F7 and the T2 handoff table."""
    total = 3
    _step(1, total, "reconciliation gate (must pass before publication)")
    con = reconcile.open_connection()
    try:
        reconcile.register_accepted_views(con)
        checks = reconcile.run_reconciliation_gate(con)
        print(f"  gate PASSED ({len(checks)} accepted anchors, all routes agree)")

        _step(2, total, "commodity and department objects behind the approved artifacts")
        question_c.register_question_c_views(con)
        question_c.load_department_view(con)
        department_level = question_c._frame(con, "13_department_level.sql")
        commodity = question_c._frame(con, "11_commodity_level.sql")
        matrix = question_c.build_t2_matrix(con)
        print(
            f"  commodity labels={len(commodity)} department labels={len(department_level)} "
            f"T2 rows={len(matrix)} total retailer SALES_VALUE={commodity['retailer_sales_value'].sum():.2f}"
        )

        _step(3, total, "publish F5, F6, F7 and T2 (deterministic, identifier-checked)")
        figures = final_figures.publish_question_c_figures(commodity, department_level)
        for figure in figures:
            print(
                f"  {figure.name}: sha256={figure.sha256[:16]} reproducible={figure.reproducible} "
                f"identifiers={figure.no_identifiers} path={figure.path.relative_to(PROJECT_ROOT)}"
            )
        table = final_figures.publish_t2(matrix)
        print(
            f"  {table.name}: sha256={table.sha256[:16]} reproducible={table.reproducible} "
            f"identifiers={table.no_identifiers} rows={len(matrix)} "
            f"path={table.path.relative_to(PROJECT_ROOT)}"
        )
    finally:
        con.close()
    print(
        "\nApproved Question C artifacts published. No commodity was selected, no other Phase 3 "
        "artifact was promoted, and no exploratory output became trackable."
    )
    return 0


def run_question_d_final() -> int:
    """Publish the approved Question D artifacts: F8 and the T3 handoff table."""
    total = 3
    _step(1, total, "reconciliation gate (must pass before publication)")
    con = reconcile.open_connection()
    try:
        reconcile.register_accepted_views(con)
        checks = reconcile.run_reconciliation_gate(con)
        print(f"  gate PASSED ({len(checks)} accepted anchors, all routes agree)")

        _step(2, total, "campaign objects behind the approved artifacts")
        question_d.register_question_d_views(con)
        result = question_d.build_question_d(con)
        t3 = question_d.build_t3(result.matrix)
        print(
            f"  campaigns={len(result.campaign_level)} T3 rows={len(t3)} "
            f"T3 columns={len(t3.columns)} "
            f"listed pairs={int(result.campaign_level['listed_households'].sum())} "
            f"recorded redemption rows={int(result.campaign_level['recorded_redemption_rows'].sum())}"
        )

        _step(3, total, "publish F8 and T3 (deterministic, identifier-checked)")
        for figure in final_figures.publish_question_d_figures(result.campaign_level):
            print(
                f"  {figure.name}: sha256={figure.sha256[:16]} reproducible={figure.reproducible} "
                f"identifiers={figure.no_identifiers} path={figure.path.relative_to(PROJECT_ROOT)}"
            )
        table = final_figures.publish_t3(t3)
        print(
            f"  {table.name}: sha256={table.sha256[:16]} reproducible={table.reproducible} "
            f"identifiers={table.no_identifiers} rows={len(t3)} "
            f"path={table.path.relative_to(PROJECT_ROOT)}"
        )
    finally:
        con.close()
    print(
        "\nApproved Question D artifacts published. No campaign was selected, no assigned-unit or "
        "comparison-group variable exists, no campaign effect was estimated and no exploratory "
        "output became trackable."
    )
    return 0


def run_question_e_final() -> int:
    """Publish the approved Question E artifact: F9."""
    total = 3
    _step(1, total, "reconciliation gate (must pass before publication)")
    con = reconcile.open_connection()
    try:
        reconcile.register_accepted_views(con)
        checks = reconcile.run_reconciliation_gate(con)
        print(f"  gate PASSED ({len(checks)} accepted anchors, all routes agree)")

        _step(2, total, "frozen selection and the journey object behind the approved figure")
        result = question_e.build_question_e(con)
        print(
            f"  journeys={len(result.journeys)} rows labels={sorted(result.journeys['household_label'].unique())} "
            f"weeks={result.journeys['week_no'].nunique()}"
        )

        _step(3, total, "publish F9 (deterministic, identifier-checked)")
        for figure in final_figures.publish_question_e_figures(result.journeys):
            print(
                f"  {figure.name}: sha256={figure.sha256[:16]} reproducible={figure.reproducible} "
                f"identifiers={figure.no_identifiers} path={figure.path.relative_to(PROJECT_ROOT)}"
            )
    finally:
        con.close()
    print(
        "\nApproved Question E artifact published. No household identifier reaches a published or "
        "tracked artifact, no campaign response is estimated, and Phase 3 synthesis is separate."
    )
    return 0


def run_synthesis() -> int:
    """Publish the paper-facing core KPI table (T1) and report the accepted-evidence inventory."""
    total = 3
    _step(1, total, "reconciliation gate (must pass before publication)")
    con = reconcile.open_connection()
    try:
        reconcile.register_accepted_views(con)
        checks = reconcile.run_reconciliation_gate(con)
        print(f"  gate PASSED ({len(checks)} accepted anchors, all routes agree)")

        _step(2, total, "T1 assembled from accepted evidence only, then verified against it")
        kpis = synthesis.build_t1(con)
        for line in synthesis.check_t1(kpis):
            print(f"  {line}")
        print(f"  {synthesis.accepted_sources_note()}")
        print(f"  T1 rows={len(kpis)} domains={sorted(set(kpis['domain']))}")

        _step(3, total, "publish T1 and report the accepted-evidence inventory")
        table = final_figures.publish_t1(kpis)
        print(
            f"  {table.name}: sha256={table.sha256[:16]} reproducible={table.reproducible} "
            f"identifiers={table.no_identifiers} rows={len(kpis)} "
            f"path={table.path.relative_to(PROJECT_ROOT)}"
        )
        inventory = synthesis.load_register_inventory()
        print(f"  promoted ({inventory.promoted_count}): {inventory.promoted}")
        print(f"  challenged ({len(inventory.challenged)}): {inventory.challenged}")
        print(f"  failed ({len(inventory.failed)}): {inventory.failed}")
    finally:
        con.close()
    print(
        "\nT1 published. No new substantive statistic was introduced, no commodity or campaign was "
        "selected and Phase 3 remains open for researcher closure."
    )
    return 0


def run_question_e() -> int:
    """Question E: representative customer journeys (explanatory; exploratory run)."""
    total = 4
    _step(1, total, "reconciliation gate (must pass before analysis)")
    con = reconcile.open_connection()
    try:
        reconcile.register_accepted_views(con)
        checks = reconcile.run_reconciliation_gate(con)
        print(f"  gate PASSED ({len(checks)} accepted anchors, all routes agree)")

        _step(2, total, "frozen selection, journeys and descriptions")
        result = question_e.build_question_e(con)
        print(question_e.format_report(result))

        _step(3, total, "local outputs (Git-ignored: label mapping and journey summary)")
        local = question_e.write_local_outputs(
            result.selection, result.journeys, result.descriptors
        )
        for name, path in sorted(local.items()):
            ignored, _rule = artifacts.git_ignore_status(path)
            print(f"  {name} -> {path.relative_to(PROJECT_ROOT)} (ignored={ignored})")

        _step(4, total, "candidate F9 (ignored, not published)")
        figure = question_e.write_candidate_figure(result)
        ignored, _rule = artifacts.git_ignore_status(figure)
        print(f"  candidate F9 -> {figure.relative_to(PROJECT_ROOT)} (ignored={ignored})")
    finally:
        con.close()
    print(
        "\nQuestion E pass complete. The four journeys are selected by the frozen rule, no household "
        "was hand-picked or substituted, no campaign effect was estimated and F9 stays Git-ignored. "
        "Phase 3 synthesis has not begun."
    )
    return 0


def run_question_d() -> int:
    """Question D: campaign structure and Phase 6 identification reconnaissance (exploratory)."""
    total = 4
    _step(1, total, "reconciliation gate (must pass before analysis)")
    con = reconcile.open_connection()
    try:
        reconcile.register_accepted_views(con)
        checks = reconcile.run_reconciliation_gate(con)
        print(f"  gate PASSED ({len(checks)} accepted anchors, all routes agree)")

        _step(2, total, "campaign, calendar, support, selection and trajectory objects")
        question_d.register_question_d_views(con)
        result = question_d.build_question_d(con)
        print(question_d.format_report(result))

        _step(3, total, "exploratory outputs (candidate F8, trajectory diagnostic, T3 precursor)")
        written = question_d.write_exploratory_outputs(result)
        for name, path in sorted(written.items()):
            print(f"  {name} -> {path.relative_to(PROJECT_ROOT)}")

        _step(4, total, "campaigns outside the declared reconnaissance support")
        screened = result.screened
        truncated = screened[screened["observed_weeks_in_post_window"] == 0]
        print(
            f"  campaigns with a truncated post window: {sorted(int(c) for c in truncated['campaign'])}"
        )
        print(
            "  campaigns extending beyond observed transaction support: "
            f"{sorted(int(c) for c in screened.loc[screened['campaign_extends_beyond_transaction_support'] == 1, 'campaign'])}"
        )
    finally:
        con.close()
    print(
        "\nQuestion D reconnaissance pass complete. Nothing is promoted, no campaign is "
        "selected, no assigned-unit or comparison-group variable exists, no campaign effect is "
        "estimated and no Phase 6 identification step is taken. Exploratory F8/T3 candidates stay "
        "Git-ignored."
    )
    return 0


def run_question_c() -> int:
    """Question C: product/commodity structure and Phase 5 shortlist reconnaissance."""
    total = 3
    _step(0, total, "reconciliation gate (must pass before analysis)")
    con = reconcile.open_connection()
    try:
        reconcile.register_accepted_views(con)
        checks = reconcile.run_reconciliation_gate(con)
        print(f"  gate PASSED ({len(checks)} accepted anchors, all routes agree)")

        _step(1, total, "department derivation + accepted choice-layer views")
        question_c.register_question_c_views(con)
        view = question_c.load_department_view(con)
        print(f"  department derivation view ready: {view}")

        _step(2, total, "commodity, department, occasion, price and shortlist diagnostics")
        result = question_c.build_question_c(con)
        print(question_c.format_report(result))
    finally:
        con.close()
    print(
        "\nQuestion C reconnaissance pass complete. This pass promotes nothing of its own and "
        "selects no commodity: the promotion decisions recorded in the challenge register were "
        "taken by the researcher, and the approved artifacts (F5, F6, F7 and T2) are published "
        "by --question-c-final, not here. Exploratory outputs from this pass stay Git-ignored."
    )
    return 0


def run_question_b() -> int:
    """Question B: household heterogeneity, concentration and persistence (exploratory)."""
    total = 3
    _step(0, total, "reconciliation gate (must pass before analysis)")
    con = reconcile.open_connection()
    try:
        reconcile.register_accepted_views(con)
        checks = reconcile.run_reconciliation_gate(con)
        print(f"  gate PASSED ({len(checks)} accepted anchors, all routes agree)")

        _step(1, total, "household-level object, household-week measures, anomaly trace")
        result = question_b.build_question_b(con)

        _step(2, total, "distributions, association, concentration, persistence, variance")
        print(question_b.format_report(result))
    finally:
        con.close()
    print(
        "\nQuestion B exploratory pass complete. Candidates await a researcher promotion "
        "decision: nothing was promoted, F3/F4 were not published and no allowlist entry "
        "was added."
    )
    return 0


def run_question_a_final() -> int:
    """Publish the two researcher-approved Question A figures (F1, F2)."""
    total = 4
    _step(1, total, "reconciliation gate (must pass before publication)")
    con = reconcile.open_connection()
    try:
        reconcile.register_accepted_views(con)
        checks = reconcile.run_reconciliation_gate(con)
        print(f"  gate PASSED ({len(checks)} accepted anchors, all routes agree)")

        _step(2, total, "department derivation (temporary view, as in the analysis pass)")
        department.register_sources(con)
        department.build_view(con)
        totality = department.totality_residuals(con)
        print(
            f"  view rows={totality.rows} department labels={totality.distinct_departments} "
            f"units residual={totality.units_residual}"
        )

        _step(3, total, "weekly series behind the approved figures")
        frame, _, _ = question_a.build_weekly_frame(con)
        if len(frame) != 102:
            raise SystemExit(f"STOP: expected 102 weeks, built {len(frame)}")
        print(
            f"  weeks={len(frame)} partial weeks={int(frame['is_partial_week'].sum())} "
            f"total retailer SALES_VALUE={frame['total_retailer_sales_value'].sum():.2f}"
        )

        _step(4, total, "publish F1 and F2 (deterministic, identifier-checked)")
        published = final_figures.publish_question_a_figures(frame)
        for figure in published:
            print(
                f"  {figure.name}: sha256={figure.sha256[:16]} "
                f"reproducible={figure.reproducible} identifiers={figure.no_identifiers} "
                f"path={figure.path.relative_to(PROJECT_ROOT)}"
            )
    finally:
        con.close()
    print(
        "\nApproved Question A figures published. No other Phase 3 artifact was "
        "promoted, and no exploratory output became trackable."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 descriptive analytics (Question A: exploratory pass and "
        "the two approved published figures)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check",
        action="store_true",
        help="verify accepted inputs, reconciliation gate, output policy, challenge register",
    )
    group.add_argument(
        "--check-department",
        action="store_true",
        help="build and validate the department derivation (publishes nothing)",
    )
    group.add_argument(
        "--question-a",
        action="store_true",
        help="run the Question A descriptive pass (exploratory outputs only)",
    )
    group.add_argument(
        "--question-b",
        action="store_true",
        help="run the Question B exploratory pass (exploratory outputs only)",
    )
    group.add_argument(
        "--question-c",
        action="store_true",
        help="run the Question C product/commodity reconnaissance pass (exploratory only)",
    )
    group.add_argument(
        "--question-d",
        action="store_true",
        help="run the Question D campaign/Phase 6 reconnaissance pass (exploratory only)",
    )
    group.add_argument(
        "--question-e-final",
        action="store_true",
        help="publish the approved Question E artifact (F9)",
    )
    group.add_argument(
        "--synthesis-final",
        action="store_true",
        help="publish the paper-facing core KPI table (T1) and report the evidence inventory",
    )
    group.add_argument(
        "--question-e",
        action="store_true",
        help="run the Question E representative-customer-journeys pass (exploratory only)",
    )
    group.add_argument(
        "--question-d-final",
        action="store_true",
        help="publish the approved Question D artifacts (F8, T3)",
    )
    group.add_argument(
        "--question-c-final",
        action="store_true",
        help="publish the approved Question C artifacts (F5, F6, F7, T2)",
    )
    group.add_argument(
        "--question-b-final",
        action="store_true",
        help="publish the two researcher-approved Question B figures (F3, F4)",
    )
    group.add_argument(
        "--question-a-final",
        action="store_true",
        help="publish the two researcher-approved Question A figures (F1, F2)",
    )
    args = parser.parse_args(argv)

    if args.check:
        return run_check()
    if args.check_department:
        return run_check_department()
    if args.question_a:
        return run_question_a()
    if args.question_a_final:
        return run_question_a_final()
    if args.question_b:
        return run_question_b()
    if args.question_c:
        return run_question_c()
    if args.question_c_final:
        return run_question_c_final()
    if args.question_d:
        return run_question_d()
    if args.question_e:
        return run_question_e()
    if args.question_e_final:
        return run_question_e_final()
    if args.synthesis_final:
        return run_synthesis()
    if args.question_d_final:
        return run_question_d_final()
    return run_question_b_final()


if __name__ == "__main__":
    sys.exit(main())
