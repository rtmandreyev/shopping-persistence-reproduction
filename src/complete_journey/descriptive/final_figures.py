"""Researcher-approved final Phase 3 figures for Question A.

Two figures, and only these two, are approved for the final scope:

* F1 - weekly retailer activity and observed purchase support
  (retailer SALES_VALUE, active households, trips, in-support households)
* F2 - participation and activity intensity
  (active households with the in-support reference, retailer SALES_VALUE per
  active household-week, trips per active household, retailer SALES_VALUE per trip)

Rules encoded here, not left to prose:

* native units on every panel; no z-scores, no standardization;
* a shared week 1-102 x-axis, with weeks 1 and 102 visibly marked as partially
  observed and never rescaled per day;
* ``SALES_VALUE`` is labelled as retailer SALES_VALUE / retailer revenue;
* the support construct is named *in-support households* and carries the explicit
  statement that it is an observed purchase-support envelope and not entry, exit,
  acquisition, churn or verified retailer membership;
* no campaign timing and no unit totals on either figure;
* publication is deterministic (the figure is built twice and the two PNG byte
  streams must match) and identifier-free (checked before the file is written).
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from complete_journey.descriptive import artifacts  # noqa: E402
from complete_journey.descriptive.contracts import (  # noqa: E402
    FIGURE_FORBIDDEN_VOCABULARY,
)

FINAL_FIGURE_DIR: Final[Path] = artifacts.output_dir("figure", "final")
EXPLORATORY_FIGURE_DIRS: Final[tuple[Path, ...]] = (
    artifacts.output_dir("figure", "exploratory"),
    artifacts.output_dir("table", "exploratory"),
)

F1_FILENAME: Final[str] = "f1_weekly_retailer_activity.png"
F2_FILENAME: Final[str] = "f2_participation_and_intensity.png"
F3_FILENAME: Final[str] = "f3_household_heterogeneity.png"
F4_FILENAME: Final[str] = "f4_household_activity_concentration.png"
F5_FILENAME: Final[str] = "f5_product_structure.png"
F6_FILENAME: Final[str] = "f6_commodity_penetration_intensity.png"
F7_FILENAME: Final[str] = "f7_national_private_composition.png"
F8_FILENAME: Final[str] = "f8_campaign_landscape.png"
F9_FILENAME: Final[str] = "f9_representative_customer_journeys.png"

#: The Phase 5 handoff table. It is a final artifact like the figures, so it is
#: allowlisted per path and written deterministically, but it lives in the tables scope.
T2_FILENAME: Final[str] = "t2_commodity_diagnostics.csv"

#: The Phase 6 research handoff: one row per campaign, all 30 retained.
T3_FILENAME: Final[str] = "t3_campaign_reconnaissance.csv"

#: The compact paper-facing core KPI table: one row per already-accepted descriptive statistic.
T1_FILENAME: Final[str] = "t1_core_kpis.csv"

#: Figures approved per question, and the full approved set. Every name here has one
#: explicit allowlist entry in .gitignore; nothing else in the final scope may exist.
QUESTION_A_FIGURE_FILENAMES: Final[tuple[str, ...]] = (F1_FILENAME, F2_FILENAME)
QUESTION_B_FIGURE_FILENAMES: Final[tuple[str, ...]] = (F3_FILENAME, F4_FILENAME)
QUESTION_C_FIGURE_FILENAMES: Final[tuple[str, ...]] = (F5_FILENAME, F6_FILENAME, F7_FILENAME)
QUESTION_D_FIGURE_FILENAMES: Final[tuple[str, ...]] = (F8_FILENAME,)
QUESTION_E_FIGURE_FILENAMES: Final[tuple[str, ...]] = (F9_FILENAME,)
FINAL_FIGURE_FILENAMES: Final[tuple[str, ...]] = (
    *QUESTION_A_FIGURE_FILENAMES,
    *QUESTION_B_FIGURE_FILENAMES,
    *QUESTION_C_FIGURE_FILENAMES,
    *QUESTION_D_FIGURE_FILENAMES,
    *QUESTION_E_FIGURE_FILENAMES,
)
FINAL_TABLE_FILENAMES: Final[tuple[str, ...]] = (T1_FILENAME, T2_FILENAME, T3_FILENAME)
FINAL_TABLE_DIR: Final[Path] = artifacts.output_dir("table", "final")

#: Rows each final table must carry: one per unit of the artifact's own grain.
T2_EXPECTED_ROWS: Final[int] = 308
T3_EXPECTED_ROWS: Final[int] = 30
#: T1 row count (inside the declared 15-25 band) so a silently truncated KPI table cannot ship.
T1_EXPECTED_ROWS: Final[int] = 24
FINAL_TABLE_EXPECTED_ROWS: Final[dict[str, int]] = {
    T1_FILENAME: T1_EXPECTED_ROWS,
    T2_FILENAME: T2_EXPECTED_ROWS,
    T3_FILENAME: T3_EXPECTED_ROWS,
}

#: The vendor's coupon/miscellaneous label. It is the largest raw commodity label but it is
#: not a conventional purchasable commodity, so it is excluded from the *presentational*
#: rankings in F5 and F7 only. It stays in the data, in T2 and in every reconciliation.
NON_CONVENTIONAL_LABEL: Final[str] = "COUPON/MISC ITEMS"

FIGURE_DPI: Final[int] = 110
PARTIAL_WEEKS: Final[tuple[int, int]] = (1, 102)

SUPPORT_DEFINITION: Final[str] = (
    "in-support households: first_purchase_week <= week <= last_purchase_week"
)
SUPPORT_CAVEAT: Final[str] = (
    "Observed purchase-support envelope, not entry/exit, customer acquisition or "
    "churn, and not verified retailer membership. Its decline late in the sample is "
    "the observation window thinning, not households leaving."
)
PARTIAL_NOTE: Final[str] = (
    "Grey bands mark weeks 1 and 102, which are only partially observed (5 and 6 of 7 "
    "trading days): they are kept in the series as recorded and never rescaled per day."
)
SALES_VALUE_NOTE: Final[str] = (
    "Retailer SALES_VALUE is retailer revenue: the sales value the retailer recorded "
    "on these transactions. It is not a measure of household spending and not an "
    "item-level amount."
)
F1_CAPTION: Final[str] = (
    "Panel D is the observed purchase-support envelope. "
    f"{SUPPORT_CAVEAT} No campaign timing and no unit totals are shown."
)
F1_FULL_CAPTION: Final[str] = f"{PARTIAL_NOTE}\n{SALES_VALUE_NOTE}\n{F1_CAPTION}"
F2_CAPTION: Final[str] = (
    "Descriptive arithmetic relationships: the weekly retailer total equals active "
    "households times retailer SALES_VALUE per active household-week, and trips times "
    "retailer SALES_VALUE per trip. This is not a causal decomposition. "
    f"{SUPPORT_CAVEAT} Every panel is in native units and every rate names its "
    "denominator; no unit totals, no standardized series and no campaign timing are shown."
)
F2_FULL_CAPTION: Final[str] = f"{PARTIAL_NOTE}\n{F2_CAPTION}"


#: F3/F4 vocabulary. The log1p note and the concentration limits are part of the
#: approved design, so they live here rather than in the builder.
LOG1P_NOTE: Final[str] = (
    "Panels B and C use a log1p x-axis so that both ends of a very skewed distribution "
    "are visible. The log1p scale is VISUAL ONLY: every reported quantity is in raw units "
    "(see the Question B exploratory tables)."
)
F3_CAPTION: Final[str] = (
    "Empirical CDFs, population S0, all 2,500 households, no filtering to S1/S2. "
    f"{LOG1P_NOTE} {SALES_VALUE_NOTE} No unit totals, no campaign information and no "
    "customer segments are shown, and no household identifier appears. Dashed markers are "
    "the median, p90 and p99 of each measure."
)
F4_CAPTION: Final[str] = (
    "Shares of RECORDED retailer activity across households at this retailer. This is not "
    "income inequality, not welfare inequality and not a normative statement about customer "
    "value, and it is not a customer-value ranking. Shares select exactly ceil(n x fraction) "
    f"households: 25 / 125 / 250 / 500 of 2,500. {SALES_VALUE_NOTE} The Gini coefficients "
    "(supplementary, not headline) are 0.5028 for retailer SALES_VALUE, 0.4760 for trips and "
    "0.2971 for active weeks. No household identifier appears."
)


def _ecdf_values(values, log_x: bool):
    series = values.dropna().sort_values()
    raw = series.to_numpy(dtype=float)
    x = log1p_scale(raw) if log_x else raw
    y = np.arange(1, len(raw) + 1) / len(raw)
    return x, y


def log1p_scale(values):
    """Log1p scale for figure axes only; reported quantities stay in raw units."""
    return np.log1p(np.clip(np.asarray(values, dtype=float), 0, None))


@dataclass(frozen=True)
class PublishedFigure:
    """One published final figure and the evidence that it was safe to publish."""

    name: str
    path: Path
    sha256: str
    reproducible: bool
    no_identifiers: bool


def _mark_partial_weeks(axes: Sequence[plt.Axes], frame) -> None:
    """Shade weeks 1 and 102 as partially observed.

    The shading is explained in the caption rather than annotated inside the axes,
    so no text can sit over the plotted series.
    """
    observed_days = frame["observed_days_in_week"]
    if len(PARTIAL_WEEKS) != int((observed_days < 7).sum()):
        raise SystemExit("STOP: the partial-week set does not match the observed-day counts")
    for ax in axes:
        for week_no in PARTIAL_WEEKS:
            ax.axvspan(week_no - 0.5, week_no + 0.5, color="0.85", zorder=0)
        ax.set_xlim(0.5, 102.5)
        ax.set_xticks([1, 20, 40, 60, 80, 100])


def _finish(fig: Figure, title: str, caption: str) -> None:
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.text(0.008, 0.005, caption, fontsize=6.4, va="bottom", wrap=True)


def build_f1(frame) -> Figure:
    """F1 - weekly retailer activity and observed purchase support."""
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 10.4), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.955, bottom=0.145, hspace=0.32)
    weeks = frame["week_no"]

    axes[0].plot(weeks, frame["total_retailer_sales_value"], color="#1f4e79", lw=1.3)
    axes[0].set_ylabel("weekly retailer SALES_VALUE\n(retailer revenue)", fontsize=8)
    axes[0].set_title("A. Weekly retailer SALES_VALUE", fontsize=9, loc="left")

    axes[1].plot(weeks, frame["active_households"], color="#2e7d32", lw=1.3)
    axes[1].set_ylabel("households with a recorded\npurchase in the week", fontsize=8)
    axes[1].set_title("B. Active households", fontsize=9, loc="left")

    axes[2].plot(weeks, frame["trips"], color="#6a1b9a", lw=1.3)
    axes[2].set_ylabel("baskets (trips)", fontsize=8)
    axes[2].set_title("C. Trips", fontsize=9, loc="left")

    axes[3].plot(weeks, frame["in_support_households"], color="#b26500", lw=1.3)
    axes[3].set_ylabel("households inside their own\nobserved purchase-support window", fontsize=8)
    axes[3].set_title(
        "D. In-support households (observed purchase support)", fontsize=9, loc="left"
    )
    axes[3].set_xlabel("week of the 102-week observation window", fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7.5)
    _mark_partial_weeks(axes, frame)
    _finish(fig, "F1 - weekly retailer activity and observed purchase support", F1_FULL_CAPTION)
    return fig


def build_f2(frame) -> Figure:
    """F2 - participation and activity intensity."""
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 10.4), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.955, bottom=0.145, hspace=0.32)
    weeks = frame["week_no"]

    axes[0].plot(
        weeks, frame["active_households"], color="#2e7d32", lw=1.4, label="active households"
    )
    axes[0].plot(
        weeks,
        frame["in_support_households"],
        color="#b26500",
        lw=1.1,
        ls="--",
        label="in-support reference (first_purchase_week <= week <= last_purchase_week)",
    )
    axes[0].set_ylabel("households", fontsize=8)
    axes[0].set_title("A. Active households, with in-support reference", fontsize=9, loc="left")
    axes[0].legend(fontsize=6.6, loc="lower right", framealpha=0.9)

    axes[1].plot(
        weeks, frame["retailer_sales_value_per_active_household_week"], color="#1f4e79", lw=1.3
    )
    axes[1].set_ylabel("retailer SALES_VALUE\nper active household-week", fontsize=8)
    axes[1].set_title("B. Retailer SALES_VALUE per active household-week", fontsize=9, loc="left")

    axes[2].plot(weeks, frame["trips_per_active_household"], color="#6a1b9a", lw=1.3)
    axes[2].set_ylabel("trips per active household", fontsize=8)
    axes[2].set_title("C. Trips per active household", fontsize=9, loc="left")

    axes[3].plot(weeks, frame["retailer_sales_value_per_trip"], color="#b26500", lw=1.3)
    axes[3].set_ylabel("retailer SALES_VALUE\nper trip", fontsize=8)
    axes[3].set_title("D. Retailer SALES_VALUE per trip", fontsize=9, loc="left")
    axes[3].set_xlabel("week of the 102-week observation window", fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7.5)
    _mark_partial_weeks(axes, frame)
    _finish(fig, "F2 - participation and activity intensity", F2_FULL_CAPTION)
    return fig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def figure_text_blocks(fig: Figure) -> list[str]:
    """Every string a reader can see in the figure, for the identifier check.

    Panel titles are collected for all three title positions: matplotlib keeps separate
    left/centre/right title artists, and ``get_title()`` alone returns only the centre one,
    which would silently skip a left-aligned panel title.
    """
    blocks: list[str] = [fig.get_suptitle()]
    blocks.extend(text.get_text() for text in fig.texts)
    for ax in fig.axes:
        blocks.extend(ax.get_title(loc=loc) for loc in ("left", "center", "right"))
        blocks.extend(
            (
                ax.get_ylabel(),
                ax.get_xlabel(),
                ax.get_legend().get_texts()[0].get_text() if ax.get_legend() else "",
            )
        )
        blocks.extend(text.get_text() for text in ax.texts)
        for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            blocks.append(tick.get_text())
    return [block for block in blocks if block]


def assert_no_identifiers(fig: Figure, plotted_columns, png_bytes: bytes) -> None:
    """Refuse to publish a figure that draws, or embeds, a customer identifier.

    The check is applied to the columns the builder actually draws, because the Question B
    figures are computed from the household-level object, which legitimately carries
    ``household_key`` as its row key even though no plotted panel uses it. Passing the
    plotted column set keeps the guard honest: a figure that started drawing a household
    identifier would fail here.
    """
    artifacts.assert_no_identifier_columns(plotted_columns)
    for text in figure_text_blocks(fig):
        lowered = text.lower()
        for term in sorted(FIGURE_FORBIDDEN_VOCABULARY):
            if term in lowered:
                raise SystemExit(f"STOP: figure text contains forbidden term {term!r}: {text!r}")
    lowered_bytes = png_bytes.lower()
    for term in ("household_key", b"tEXt".lower()):
        needle = term if isinstance(term, bytes) else term.encode()
        if needle == b"text":
            continue
        if needle in lowered_bytes:
            raise SystemExit(f"STOP: figure PNG bytes contain {term!r}")


def write_deterministic_figure(
    builder: Callable[[object], Figure],
    frame,
    path: Path,
    *,
    plotted_columns: Sequence[str] | None = None,
) -> PublishedFigure:
    """Build the figure twice and publish only if the two PNG streams are identical.

    ``plotted_columns`` names the columns the builder draws; it defaults to the whole
    input frame, which is the strict choice for frames that carry no row key.
    """
    columns = tuple(plotted_columns) if plotted_columns is not None else tuple(frame.columns)
    hashes: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for attempt in (1, 2):
            fig = builder(frame)
            probe = Path(tmpdir) / f"{path.stem}_{attempt}.png"
            fig.savefig(probe, dpi=FIGURE_DPI, facecolor="white", metadata={"Software": None})
            plt.close(fig)
            hashes.append(_sha256(probe))
        if hashes[0] != hashes[1]:
            raise SystemExit(
                f"STOP: {path.name} is not reproducible; two builds gave {hashes[0][:16]} and "
                f"{hashes[1][:16]}"
            )
        reproducible = True

        # Validate the render before anything reaches the final scope: a figure that fails the
        # identifier or forbidden-text guard must never be written, not even transiently.
        fig = builder(frame)
        probe = Path(tmpdir) / f"{path.stem}_publish.png"
        fig.savefig(probe, dpi=FIGURE_DPI, facecolor="white", metadata={"Software": None})
        png_bytes = probe.read_bytes()
        try:
            assert_no_identifiers(fig, columns, png_bytes)
        finally:
            plt.close(fig)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png_bytes)

    published = PublishedFigure(
        name=path.name,
        path=path,
        sha256=_sha256(path),
        reproducible=reproducible,
        no_identifiers=True,
    )
    if published.sha256 != hashes[0]:
        raise SystemExit(f"STOP: published {path.name} differs from the reproducible build")
    return published


def publish_question_a_figures(frame) -> list[PublishedFigure]:
    """Write F1 and F2 into the final scope and verify the publication policy."""
    published = [
        write_deterministic_figure(build_f1, frame, FINAL_FIGURE_DIR / F1_FILENAME),
        write_deterministic_figure(build_f2, frame, FINAL_FIGURE_DIR / F2_FILENAME),
    ]
    verify_publication_policy(published)
    return published


def publish_question_b_figures(household) -> list[PublishedFigure]:
    """Write F3 and F4 into the final scope and verify the publication policy."""
    published = [
        write_deterministic_figure(
            build_f3,
            household,
            FINAL_FIGURE_DIR / F3_FILENAME,
            plotted_columns=F3_PLOTTED_COLUMNS,
        ),
        write_deterministic_figure(
            build_f4,
            household,
            FINAL_FIGURE_DIR / F4_FILENAME,
            plotted_columns=F4_PLOTTED_COLUMNS,
        ),
    ]
    verify_publication_policy(published)
    return published


def verify_publication_policy(
    published: Sequence[PublishedFigure], published_tables: Sequence[str] = ()
) -> None:
    """Only the approved artifacts may exist or be allowlisted; everything else stays ignored.

    Covers both final scopes: the figures directory and the tables directory. A file in
    either final scope that is not an approved, allowlisted artifact is refused.
    """
    unknown_tables = set(published_tables) - set(FINAL_TABLE_FILENAMES)
    if unknown_tables:
        raise SystemExit(f"STOP: table(s) outside the approved set: {sorted(unknown_tables)}")
    unexpected_tables = [
        path.name
        for path in sorted(FINAL_TABLE_DIR.iterdir())
        if path.name != ".gitkeep" and path.name not in FINAL_TABLE_FILENAMES
    ]
    if unexpected_tables:
        raise SystemExit(
            f"STOP: unapproved artifact(s) in the final table scope: {unexpected_tables}"
        )
    for name in sorted(set(FINAL_TABLE_FILENAMES)):
        path = FINAL_TABLE_DIR / name
        if not path.is_file():
            continue
        ignored, rule = artifacts.git_ignore_status(path)
        if ignored:
            raise SystemExit(
                f"STOP: {name} must be allowlisted explicitly, but Git still ignores it "
                f"(rule {rule!r})"
            )

    approved = set(FINAL_FIGURE_FILENAMES)
    names = {figure.name for figure in published}
    unknown = names - approved
    if unknown:
        raise SystemExit(f"STOP: figure(s) outside the approved set: {sorted(unknown)}")

    unexpected: list[str] = []
    for path in sorted(FINAL_FIGURE_DIR.iterdir()):
        if path.name == ".gitkeep" or path.name in approved:
            continue
        unexpected.append(path.name)
    if unexpected:
        raise SystemExit(f"STOP: unapproved artifact(s) in the final scope: {unexpected}")

    # Every approved figure present must carry its explicit allowlist entry, and every
    # figure just published must be present.
    for name in sorted(approved):
        path = FINAL_FIGURE_DIR / name
        if not path.is_file():
            continue
        ignored, rule = artifacts.git_ignore_status(path)
        if ignored:
            raise SystemExit(
                f"STOP: {name} must be allowlisted explicitly, but Git still ignores it "
                f"(rule {rule!r})"
            )
    missing = [name for name in names if not (FINAL_FIGURE_DIR / name).is_file()]
    if missing:
        raise SystemExit(f"STOP: published figure(s) missing on disk: {missing}")
    exploratory_ignored = 0
    for directory in EXPLORATORY_FIGURE_DIRS:
        for path in directory.iterdir():
            if path.name == ".gitkeep":
                continue
            ignored, _ = artifacts.git_ignore_status(path)
            if not ignored:
                raise SystemExit(f"STOP: exploratory artifact is trackable: {path}")
            exploratory_ignored += 1
    print(
        f"  publication policy OK: final figures allowlisted={sorted(names)}, "
        f"exploratory artifacts still ignored={exploratory_ignored}"
    )


def _mark_quantiles(ax, values, *, log_x: bool) -> None:
    """Draw median / p90 / p99 markers with their values labelled in raw units."""
    quantiles = values.dropna().quantile([0.5, 0.9, 0.99])
    for label, quantile in zip(("median", "p90", "p99"), quantiles, strict=True):
        position = float(log1p_scale([quantile])[0]) if log_x else float(quantile)
        ax.axvline(position, color="0.45", lw=0.7, ls=":")
        ax.annotate(
            f"{label} {quantile:,.0f}",
            xy=(position, 0.06),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=6.4,
            color="0.3",
            rotation=90,
            va="bottom",
        )


#: Columns the approved Question B figures draw. Listed explicitly so the identifier
#: guard checks what is plotted rather than the row key the object carries.
F3_PLOTTED_COLUMNS: Final[tuple[str, ...]] = (
    "active_weeks",
    "total_sales_value",
    "total_baskets",
    "sales_value_per_trip",
)
F4_PLOTTED_COLUMNS: Final[tuple[str, ...]] = ("total_sales_value", "total_baskets", "active_weeks")


def build_f3(household) -> Figure:
    """F3 - household heterogeneity: four ECDF panels, population S0."""
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.165, hspace=0.4, wspace=0.22)
    panels = (
        (axes[0, 0], "active_weeks", "A. Active weeks (of the 102-week grid)", False),
        (axes[0, 1], "total_sales_value", "B. Total retailer SALES_VALUE", True),
        (axes[1, 0], "total_baskets", "C. Trips (baskets)", True),
        (axes[1, 1], "sales_value_per_trip", "D. Retailer SALES_VALUE per trip", False),
    )
    for ax, column, title, log_x in panels:
        x, y = _ecdf_values(household[column], log_x)
        ax.plot(x, y, lw=1.6, color="#1f4e79")
        _mark_quantiles(ax, household[column], log_x=log_x)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel("cumulative share of households", fontsize=7.5)
        ax.set_xlabel(
            ("log1p scale (visual only)" if log_x else "raw units") + " - values in the tables",
            fontsize=7,
        )
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, 1)
    _finish(fig, "F3 - household heterogeneity (population S0)", F3_CAPTION)
    return fig


def build_f4(household) -> Figure:
    """F4 - concentration of recorded retailer activity, with the top-1% overlap inset."""
    from complete_journey.descriptive import question_b

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.87, bottom=0.2, wspace=0.2)

    colours = {
        "total_sales_value": "#1f4e79",
        "total_baskets": "#2e7d32",
        "active_weeks": "#b26500",
    }
    fractions = question_b.CONCENTRATION_FRACTIONS
    table = question_b.concentration_table(household)
    for column, label, _total in question_b.CONCENTRATION_TARGETS:
        points = question_b.lorenz_table(household, column)
        share = float(table.loc[table["target"] == column, "top_10pct_share"].iloc[0])
        axes[0].plot(
            points["population_share"] * 100,
            points["activity_share"] * 100,
            lw=1.6,
            color=colours[column],
            label=f"{label} (top 10% = {share * 100:.2f}%)",
        )
    axes[0].plot([0, 100], [0, 100], color="0.5", lw=0.9, ls="--", label="equal distribution")
    axes[0].set_xlabel("cumulative share of households (%), smallest to largest", fontsize=8)
    axes[0].set_ylabel("cumulative share of recorded activity (%)", fontsize=8)
    axes[0].set_title("A. Lorenz-style concentration curves", fontsize=9, loc="left")
    axes[0].legend(fontsize=6.8, loc="upper left")
    axes[0].grid(alpha=0.25, lw=0.5)

    components = question_b.concentration_component_table(household)
    overlap = dict(zip(components["statistic"], components["value"], strict=True))
    axes[0].text(
        0.98,
        0.03,
        "TOP-1% OVERLAP (25 households each)\n"
        "retailer SALES_VALUE vs trips: "
        f"{overlap['top_value_households_also_top_trips'] * 100:.0f}%\n"
        "retailer SALES_VALUE vs active weeks: "
        f"{overlap['top_value_households_also_top_active_weeks'] * 100:.0f}%",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.9},
    )

    positions = np.arange(len(fractions))
    width = 0.26
    labels = [f"top {int(fraction * 100)}%" for fraction in fractions]
    for offset, (column, label, _total) in enumerate(question_b.CONCENTRATION_TARGETS):
        row = table.loc[table["target"] == column].iloc[0]
        axes[1].bar(
            positions + (offset - 1) * width,
            [row[f"top_{int(fraction * 100)}pct_share"] * 100 for fraction in fractions],
            width=width,
            color=colours[column],
            label=label,
        )
    axes[1].set_xticks(positions, labels, fontsize=8)
    axes[1].set_ylabel("share of that measure's recorded total (%)", fontsize=8)
    axes[1].set_title("B. Top-x% shares of recorded activity", fontsize=9, loc="left")
    axes[1].legend(fontsize=6.8)
    axes[1].grid(alpha=0.25, lw=0.5, axis="y")
    for ax in axes:
        ax.tick_params(labelsize=7.5)
    _finish(fig, "F4 - concentration of recorded retailer activity across households", F4_CAPTION)
    return fig


# --- Question C figures and the T2 handoff table -----------------------------

F5_CAPTION: Final[str] = (
    "Panel A: the 12 largest departments by retailer SALES_VALUE (retailer revenue), with "
    "observed-household penetration labelled (penetration = distinct buying households / "
    "2,500, never market penetration). Panel B: the 15 largest CONVENTIONAL commodity labels "
    "by retailer SALES_VALUE. COUPON/MISC ITEMS is the largest commodity label in the data at "
    "approximately 639,879 of retailer SALES_VALUE, and it is EXCLUDED FROM THIS "
    "CONVENTIONAL-PRODUCT DISPLAY ONLY because it is the vendor's coupon/miscellaneous label "
    "rather than a purchasable commodity: it remains in the data, in the complete descriptive "
    "tables (T2) and in every reconciliation, and the exclusion is presentational, not "
    "analytical. The unlabelled sentinel department carries 0.00 of recorded value and is not "
    "shown. No raw-unit ranking is used anywhere (Question A finding QA-03)."
)
F6_CAPTION: Final[str] = (
    "One point per conventional commodity label; x is observed-household penetration "
    "(distinct buying households / 2,500, never market penetration) and y is retailer "
    "SALES_VALUE (retailer revenue) per buying household-week, so the denominator is that "
    "commodity's own buying household-weeks. Point size encodes the number of distinct "
    "purchased products. The descriptive rank relationship is Spearman(value rank, "
    "value-per-buying-household-week rank) = -0.1024, i.e. how large a commodity is in "
    "recorded retailer SALES_VALUE is NOT tracking how much value each buying household-week "
    "records; large commodities are generally broad and repeatedly purchased rather than "
    "unusually high-intensity. This is a descriptive association, not a causal statement, "
    "not a decomposition and not evidence about preferences. COUPON/MISC ITEMS is excluded "
    "from this display only and remains in T2."
)
F7_CAPTION: Final[str] = (
    "Top 20 CONVENTIONAL commodity labels by retailer SALES_VALUE, in the same order in both "
    "panels. Panel A numerator = recorded private-label retailer SALES_VALUE in the commodity, "
    "denominator = that commodity's national plus private retailer SALES_VALUE. Panel B "
    "numerator = recorded private-label units, denominator = national plus private units. "
    "The two are different metrics and NEITHER IS 'the' private-label share; the unit-based "
    "metric carries the Question A QA-03 caveat (recorded units are dominated by very "
    "large-quantity lines in two labels, and unit measures are not physically comparable "
    "across products merely because they lack such lines). COUPON/MISC ITEMS is excluded from "
    "this display only and remains in the complete table T2, which retains all 308 labels."
)


#: Columns the approved Question C artifacts draw, listed explicitly so the identifier
#: guard checks what is plotted rather than every column of the loaded objects.
F5_PLOTTED_COLUMNS: Final[tuple[str, ...]] = (
    "department_desc",
    "department_is_unlabelled",
    "observed_household_penetration",
    "retailer_sales_value",
    "commodity_desc",
)
F6_PLOTTED_COLUMNS: Final[tuple[str, ...]] = (
    "commodity_desc",
    "observed_household_penetration",
    "sales_value_per_buying_household_week",
    "distinct_products",
    "retailer_sales_value",
)
F7_PLOTTED_COLUMNS: Final[tuple[str, ...]] = (
    "commodity_desc",
    "private_share_of_sales_value",
    "private_share_of_units",
    "retailer_sales_value",
)


CAMPAIGN_TYPE_COLOURS: Final[dict[str, str]] = {
    "TypeA": "#1f4e79",
    "TypeB": "#b26500",
    "TypeC": "#2e7d32",
}

#: Columns the approved F8 draws, listed explicitly so the identifier guard checks what is drawn.
F8_PLOTTED_COLUMNS: Final[tuple[str, ...]] = (
    "campaign",
    "campaign_type",
    "start_week",
    "end_week",
    "listed_households",
    "accepted_phase1_recipient_overlap_pct",
    "campaign_extends_beyond_transaction_support",
)

F8_CAPTION: Final[str] = (
    "Campaign STRUCTURE only: no outcome variable is shown on any axis, recorded retailer "
    "SALES_VALUE is deliberately not overlaid, and there is no campaign-relative or event-time "
    "axis, so no panel can be read as evidence about how any campaign changed recorded behaviour. "
    "Panel A: every campaign's calendar span "
    "across the 102 observed weeks, one row per campaign ordered by start week, colour encoding "
    "campaign type (legend); the week mapping is the accepted Phase 1 convention and campaign "
    "start/end days are preserved in T3, so a campaign touching a week does NOT mean the entire "
    "week lay inside the exact day-level campaign window. Campaign 24 is marked '>' because its "
    "day-level end lies beyond the observed transaction window; campaigns 15 and 24 also have no "
    "observed weeks after their span in the reconnaissance windows. Panel B: listed households per "
    "campaign (factual database membership, never an assigned unit). Panel C: the accepted Phase 1 "
    "listed-campaign overlap of each campaign's own listed recipients - listed-campaign overlap, "
    "not realized exposure or contact. Timing, size and overlap are three separate dimensions and "
    "no composite score is implied."
)


def _conventional(commodity: pd.DataFrame) -> pd.DataFrame:
    """Drop the vendor's coupon/miscellaneous ledger label from a presentational ranking."""
    return commodity[commodity["commodity_desc"] != NON_CONVENTIONAL_LABEL]


def build_f5(commodity: pd.DataFrame, department_level: pd.DataFrame) -> Figure:
    """F5 - department and commodity structure (conventional display, sentinel stated)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2))
    fig.subplots_adjust(left=0.16, right=0.985, top=0.9, bottom=0.24, wspace=0.55)

    departments = department_level[
        ~department_level["department_is_unlabelled"].fillna(False)
    ].nlargest(12, "retailer_sales_value")
    axes[0].barh(
        departments["department_desc"][::-1],
        departments["retailer_sales_value"][::-1],
        color="#1f4e79",
    )
    for position, penetration in enumerate(departments["observed_household_penetration"][::-1]):
        axes[0].text(
            departments["retailer_sales_value"][::-1].iloc[position],
            position,
            f"  pen {penetration:.3f}",
            va="center",
            fontsize=6.4,
        )
    axes[0].set_xlabel("retailer SALES_VALUE (retailer revenue)", fontsize=8)
    axes[0].set_title("A. Largest departments (12)", fontsize=9, loc="left")
    axes[0].tick_params(labelsize=7)
    axes[0].grid(alpha=0.25, lw=0.5, axis="x")

    commodities = _conventional(commodity).nlargest(15, "retailer_sales_value")
    axes[1].barh(
        commodities["commodity_desc"][::-1],
        commodities["retailer_sales_value"][::-1],
        color="#2e7d32",
    )
    for position, penetration in enumerate(commodities["observed_household_penetration"][::-1]):
        axes[1].text(
            commodities["retailer_sales_value"][::-1].iloc[position],
            position,
            f"  pen {penetration:.3f}",
            va="center",
            fontsize=6.4,
        )
    axes[1].set_xlabel("retailer SALES_VALUE (retailer revenue)", fontsize=8)
    axes[1].set_title("B. Largest conventional commodities (15)", fontsize=9, loc="left")
    axes[1].tick_params(labelsize=6.6)
    axes[1].grid(alpha=0.25, lw=0.5, axis="x")
    _finish(fig, "F5 - department and commodity structure (population S0)", F5_CAPTION)
    return fig


def build_f6(commodity: pd.DataFrame) -> Figure:
    """F6 - observed-household penetration against purchase-conditional intensity."""
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.26)
    frame = _conventional(commodity)
    sizes = 8 + 70 * (frame["distinct_products"] / frame["distinct_products"].max())
    ax.scatter(
        frame["observed_household_penetration"] * 100,
        frame["sales_value_per_buying_household_week"],
        s=sizes,
        alpha=0.5,
        color="#1f4e79",
        edgecolors="none",
    )
    # Label the largest commodities, skipping any label that would land on top of one already
    # placed, so the callouts stay legible without shrinking the figure.
    placed: list[tuple[float, float]] = []
    for record in frame.nlargest(12, "retailer_sales_value").itertuples(index=False):
        x = record.observed_household_penetration * 100
        y = record.sales_value_per_buying_household_week
        if any(abs(x - other_x) < 6.0 and abs(y - other_y) < 1.8 for other_x, other_y in placed):
            continue
        placed.append((x, y))
        ax.annotate(
            record.commodity_desc,
            (x, y),
            fontsize=6.2,
            xytext=(5, 4 if len(placed) % 2 else -9),
            textcoords="offset points",
        )
        if len(placed) == 8:
            break
    ax.set_xlabel("observed-household penetration (%, denominator 2,500 households)", fontsize=8)
    ax.set_ylabel("retailer SALES_VALUE per buying household-week", fontsize=8)
    ax.text(
        0.03,
        0.96,
        "Spearman(value rank, intensity rank) = -0.1024",
        transform=ax.transAxes,
        fontsize=7.6,
        va="top",
    )
    ax.set_xlim(-4, 112)  # headroom so right-edge callouts are not clipped
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7.5)
    _finish(
        fig,
        "F6 - commodity penetration against purchase-conditional intensity",
        F6_CAPTION,
    )
    return fig


def build_f7(commodity: pd.DataFrame, top_n: int = 20) -> Figure:
    """F7 - national/private composition in both metrics, one shared commodity ordering."""
    frame = _conventional(commodity).nlargest(top_n, "retailer_sales_value")
    order = frame["commodity_desc"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.2))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.9, bottom=0.2, wspace=0.62)
    panels = (
        (
            axes[0],
            "private_share_of_sales_value",
            "A. Private-label share of retailer SALES_VALUE",
            "private-label retailer SALES_VALUE",
            "national + private retailer SALES_VALUE",
        ),
        (
            axes[1],
            "private_share_of_units",
            "B. Private-label share of recorded units (QA-03 caveat)",
            "private-label units",
            "national + private units",
        ),
    )
    for ax, column, title, numerator_text, denominator_text in panels:
        values = frame.set_index("commodity_desc").loc[order, column] * 100
        ax.barh(order[::-1], values[::-1], color="#b26500" if ax is axes[0] else "#1f4e79")
        ax.set_xlim(0, 100)
        ax.set_xlabel(
            f"private-label share (%)\nnumerator: {numerator_text} / denominator: {denominator_text}",
            fontsize=6.4,
        )
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=6.6)
        ax.grid(alpha=0.25, lw=0.5, axis="x")
    _finish(
        fig, "F7 - national/private-label composition (top 20 conventional commodities)", F7_CAPTION
    )
    return fig


def write_deterministic_table(frame: pd.DataFrame, path: Path) -> PublishedFigure:
    """Write the handoff table twice and publish only if the two byte streams are identical.

    The identifier guard runs on the table's own columns, and the row count is checked
    against the declared expectation so a silently truncated handoff table cannot ship.
    """
    artifacts.assert_no_identifier_columns(frame.columns)
    expected = FINAL_TABLE_EXPECTED_ROWS.get(path.name, len(frame))
    if len(frame) != expected:
        raise SystemExit(
            f"STOP: {path.name} must carry exactly {expected} row(s); got {len(frame)}"
        )
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if payload != frame.to_csv(index=False, lineterminator="\n").encode("utf-8"):
        raise SystemExit(f"STOP: {path.name} is not reproducible across two writes")
    # Validate before publishing, so a rejected table never lands in the final scope.
    for term in sorted(FIGURE_FORBIDDEN_VOCABULARY):
        if term == "price":
            continue  # column names such as candidate-price usability are explicit labels
        if term.encode() in payload.lower():
            raise SystemExit(f"STOP: {path.name} contains forbidden term {term!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    if path.read_bytes() != payload:
        raise SystemExit(f"STOP: {path.name} differs from the reproducible build")
    return PublishedFigure(
        name=path.name,
        path=path,
        sha256=_sha256(path),
        reproducible=True,
        no_identifiers=True,
    )


def publish_t3(matrix: pd.DataFrame) -> PublishedFigure:
    """Publish the Phase 6 campaign reconnaissance handoff table (one row per campaign)."""
    published = write_deterministic_table(matrix, FINAL_TABLE_DIR / T3_FILENAME)
    verify_publication_policy([], published_tables=[published.name])
    return published


def publish_t2(matrix: pd.DataFrame) -> PublishedFigure:
    """Publish the Phase 5 handoff table (one row per commodity, all 308 labels)."""
    published = write_deterministic_table(matrix, FINAL_TABLE_DIR / T2_FILENAME)
    verify_publication_policy([], published_tables=[published.name])
    return published


def build_f8(campaign_level: pd.DataFrame) -> Figure:
    """F8 - campaign landscape: calendar spans, listing breadth and listed-recipient overlap.

    Campaign STRUCTURE only. No outcome variable appears on any axis, no campaign-relative or
    event-time axis exists, and recorded retailer SALES_VALUE is deliberately not overlaid, so the
    figure cannot be read as evidence about how a campaign changed recorded behaviour.
    """
    ordered = campaign_level.sort_values(["start_week", "campaign"]).reset_index(drop=True)
    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 9.0), gridspec_kw={"width_ratios": [2.6, 1.0, 1.0]}
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.26, wspace=0.75)

    axis = axes[0]
    for position, record in enumerate(ordered.itertuples(index=False)):
        colour = CAMPAIGN_TYPE_COLOURS.get(str(record.campaign_type), "#555555")
        axis.barh(
            position,
            record.end_week - record.start_week + 1,
            left=record.start_week,
            height=0.72,
            color=colour,
            alpha=0.9,
        )
        if record.campaign_extends_beyond_transaction_support == 1:
            # The day-level end lies beyond the observed data: mark the truncated right edge.
            axis.annotate(
                ">",
                (record.end_week + 0.6, position),
                fontsize=10.0,
                va="center",
                ha="left",
                color=colour,
            )
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels(
        [
            f"{int(record.campaign)}  ({record.campaign_type[-1]})"
            for record in ordered.itertuples(index=False)
        ],
        fontsize=6.4,
    )
    axis.invert_yaxis()
    axis.set_xlim(0.5, 104.5)
    axis.set_xlabel("observed week (accepted Phase 1 week mapping)", fontsize=8)
    axis.set_title(
        "A. Campaign calendar spans (30 campaigns, by start week)",
        fontsize=8.5,
        loc="left",
    )
    axis.grid(alpha=0.25, lw=0.5, axis="x")
    handles = [
        plt.Line2D([0], [0], color=colour, lw=6, label=campaign_type)
        for campaign_type, colour in sorted(CAMPAIGN_TYPE_COLOURS.items())
    ]
    axis.legend(handles=handles, fontsize=6.4, loc="lower right", framealpha=0.9)

    listing_axis = axes[1]
    listing_axis.barh(range(len(ordered)), ordered["listed_households"], color="#1f4e79")
    listing_axis.set_yticks(range(len(ordered)))
    listing_axis.set_yticklabels([])
    listing_axis.invert_yaxis()
    listing_axis.set_xlabel("listed households", fontsize=8)
    listing_axis.set_title("B. Listed households", fontsize=8.5, loc="left")
    listing_axis.grid(alpha=0.25, lw=0.5, axis="x")

    overlap_axis = axes[2]
    overlap_axis.barh(
        range(len(ordered)), ordered["accepted_phase1_recipient_overlap_pct"], color="#b26500"
    )
    overlap_axis.set_yticks(range(len(ordered)))
    overlap_axis.set_yticklabels([])
    overlap_axis.invert_yaxis()
    overlap_axis.set_xlim(0, 100)
    overlap_axis.set_xlabel("listed-recipient overlap of campaign span (%)", fontsize=7.4)
    overlap_axis.set_title("C. Listed-campaign overlap", fontsize=8.5, loc="left")
    overlap_axis.grid(alpha=0.25, lw=0.5, axis="x")

    fig.suptitle("F8 - campaign landscape (campaign structure only)", fontsize=11)
    fig.text(0.07, 0.005, F8_CAPTION, fontsize=6.2, va="bottom", wrap=True)
    return fig


F9_PLOTTED_COLUMNS: Final[tuple[str, ...]] = (
    "household_label",
    "week_no",
    "retailer_purchase",
    "weekly_sales_value",
    "listed_campaign_active_count",
    "recorded_redemption_rows",
    "partial_week",
)

F9_CAPTION: Final[str] = (
    "The four households were selected by the FROZEN deterministic rule of the Phase 3 plan "
    "(high-frequency, median-frequency, intermittent, and coupon/campaign-active strata, each time the "
    "eligible household whose whole-sample retailer SALES_VALUE is closest to its stratum median, with "
    "the rule's documented deterministic tie-break). They are representative ONLY under that "
    "nearest-stratum-median rule and are NOT customer archetypes. Weekly retailer SALES_VALUE is "
    "retailer revenue (retailer SALES_VALUE) and is not a household grocery bill; a week with no bar "
    "means no recorded purchase from this retailer that week, not zero grocery consumption. Grey background bands mark weeks in which the "
    "household was listed for at least one campaign and red markers are recorded coupon redemptions: "
    "both are factual context only - listing is not an assigned unit or an observed offer contact, "
    "recorded redemption is not proof of offer receipt, and no campaign response is measured or "
    "estimated anywhere in this figure. Dotted lines mark weeks 1 and 102, which are partial observed "
    "weeks (5 and 6 days)."
)


def build_f9(journeys: pd.DataFrame) -> Figure:
    """F9 - representative customer journeys: four vertically aligned panels, shared week axis.

    Weekly retailer SALES_VALUE dominates; campaign listing is light background shading and recorded
    redemption small markers, so the campaign context cannot be read as an annotation of the series.
    """
    ordered = ("A", "B", "C", "D")
    panel_titles = {
        "A": "A - High-frequency journey",
        "B": "B - Median-frequency journey",
        "C": "C - Intermittent journey",
        "D": "D - Coupon/campaign-active journey",
    }
    colours = {"A": "#1f4e79", "B": "#2e7d32", "C": "#8e24aa", "D": "#b26500"}
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 10.4), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.175, hspace=0.26)
    for axis, label in zip(axes, ordered, strict=True):
        frame = journeys[journeys["household_label"] == label].sort_values("week_no")
        for week in frame.loc[frame["listed_campaign_active_count"] > 0, "week_no"]:
            axis.axvspan(week - 0.5, week + 0.5, color="#dddddd", alpha=0.35, lw=0, zorder=0)
        axis.bar(
            frame["week_no"],
            frame["weekly_sales_value"],
            width=0.8,
            color=colours[label],
            alpha=0.9,
            zorder=2,
        )
        active = frame[frame["retailer_purchase"] == 1]
        axis.scatter(
            active["week_no"],
            active["weekly_sales_value"],
            s=5,
            color="#111111",
            marker=".",
            zorder=3,
            label="active week (recorded purchase)",
        )
        redemption = frame[frame["recorded_redemption_rows"] > 0]
        if not redemption.empty:
            axis.scatter(
                redemption["week_no"],
                np.full(len(redemption), axis.get_ylim()[1] * 0.03),
                s=20,
                color="#c62828",
                marker="v",
                zorder=4,
                label="recorded coupon redemption (factual context)",
            )
        for week in (1, 102):
            axis.axvline(week, color="#888888", lw=0.8, ls=":", zorder=1)
        axis.set_title(panel_titles[label], fontsize=9, loc="left")
        axis.set_ylabel("weekly retailer\nSALES_VALUE", fontsize=7.5)
        axis.grid(alpha=0.2, lw=0.4, axis="y")
        axis.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    axes[-1].set_xlabel("observed week (weeks 1 and 102 are partial observed weeks)", fontsize=8)
    fig.suptitle("F9 - representative customer journeys (four frozen-rule households)", fontsize=11)
    fig.text(0.075, 0.008, F9_CAPTION, fontsize=6.2, va="bottom", wrap=True)
    return fig


def publish_question_e_figures(journeys: pd.DataFrame) -> list[PublishedFigure]:
    """Write F9 into the final scope and verify the publication policy."""
    published = [
        write_deterministic_figure(
            build_f9,
            journeys,
            FINAL_FIGURE_DIR / F9_FILENAME,
            plotted_columns=F9_PLOTTED_COLUMNS,
        )
    ]
    verify_publication_policy(published)
    return published


def publish_t1(kpis: pd.DataFrame) -> PublishedFigure:
    """Publish the compact paper-facing core KPI table (one row per accepted statistic)."""
    published = write_deterministic_table(kpis, FINAL_TABLE_DIR / T1_FILENAME)
    verify_publication_policy([], published_tables=[published.name])
    return published


def publish_question_d_figures(campaign_level: pd.DataFrame) -> list[PublishedFigure]:
    """Write F8 into the final scope and verify the publication policy."""
    published = [
        write_deterministic_figure(
            build_f8,
            campaign_level,
            FINAL_FIGURE_DIR / F8_FILENAME,
            plotted_columns=F8_PLOTTED_COLUMNS,
        )
    ]
    verify_publication_policy(published)
    return published


def publish_question_c_figures(
    commodity: pd.DataFrame, department_level: pd.DataFrame
) -> list[PublishedFigure]:
    """Write F5, F6 and F7 into the final scope and verify the publication policy."""
    published = [
        write_deterministic_figure(
            lambda pair: build_f5(pair[0], pair[1]),
            (commodity, department_level),
            FINAL_FIGURE_DIR / F5_FILENAME,
            plotted_columns=F5_PLOTTED_COLUMNS,
        ),
        write_deterministic_figure(
            build_f6,
            commodity,
            FINAL_FIGURE_DIR / F6_FILENAME,
            plotted_columns=F6_PLOTTED_COLUMNS,
        ),
        write_deterministic_figure(
            build_f7,
            commodity,
            FINAL_FIGURE_DIR / F7_FILENAME,
            plotted_columns=F7_PLOTTED_COLUMNS,
        ),
    ]
    verify_publication_policy(published)
    return published
