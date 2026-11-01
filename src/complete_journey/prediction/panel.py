"""Canonical household-week participation panel and the Run B real-data fitting guard.

A ``CanonicalPanel`` is the single input type for every Phase 7 operator, feature,
moment and model. It carries its origin ("real" or "synthetic"). Model fitting and
scoring call ``require_fit_allowed`` and refuse real outcomes until Run C sets the
authorization constant in ``config`` (no flag, no environment variable).

Behaviour and observation are separate objects. ``recorded`` is what the analyst
observes. Synthetic panels may also carry the latent ``behavior`` panel that generated
the record, so tests can prove that observation was applied after behaviour.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from complete_journey.prediction import config

COLUMNS = ("household_key", "week_no", config.TARGET_SOURCE)
#: Key-label-invariant SHA-256 of the accepted real 2,500 x 102 participation grid.
REAL_PANEL_FINGERPRINT = "020680f61eb5b03b2776e7624da11a87738aaa1ff2eefcec6f5edeb35256c841"


class Phase7ContractError(ValueError):
    """A Phase 7 contract, timing, leakage or authorization rule was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase7ContractError(message)


def outcome_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted household keys, n_households x 102 int8 matrix) of a complete grid."""
    wide = frame.pivot(index="household_key", columns="week_no", values=config.TARGET_SOURCE)
    return wide.index.to_numpy(), wide.to_numpy(dtype=np.int8)


def label_invariant_fingerprint(frame: pd.DataFrame) -> str:
    """Hash of the outcome grid with households sorted by their own outcome bytes."""
    _, matrix = outcome_matrix(frame)
    rows = sorted(bytes(row) for row in matrix)
    return hashlib.sha256(b"".join(rows)).hexdigest()


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    require(tuple(frame.columns[:3]) == COLUMNS, f"panel columns must start with {COLUMNS}")
    require(not frame.duplicated(["household_key", "week_no"]).any(), "duplicate panel key")
    weeks = np.arange(config.FIRST_WEEK, config.LAST_WEEK + 1)
    require(set(frame["week_no"].unique()) == set(weeks), "panel must cover weeks 1-102 exactly")
    require(
        len(frame) == frame["household_key"].nunique() * len(weeks),
        "panel must be a complete household x week grid",
    )
    require(frame[config.TARGET_SOURCE].isin([0, 1]).all(), "participation must be binary")
    return frame.sort_values(["household_key", "week_no"], ignore_index=True)


@dataclass(frozen=True)
class CanonicalPanel:
    """Recorded participation grid plus origin; optionally the generating behaviour."""

    recorded: pd.DataFrame
    origin: str
    behavior: pd.DataFrame | None = field(default=None, compare=False)
    label: str = ""

    def __post_init__(self):
        require(self.origin in ("real", "synthetic"), "panel origin must be real or synthetic")
        object.__setattr__(self, "recorded", validate_frame(self.recorded))
        if self.origin == "synthetic":
            require(
                (self.recorded["household_key"] >= config.SYNTHETIC_KEY_OFFSET).all(),
                "synthetic household keys must lie in the synthetic key space",
            )
            require(
                label_invariant_fingerprint(self.recorded) != REAL_PANEL_FINGERPRINT,
                "panel labelled synthetic reproduces the real outcome grid",
            )
        if self.behavior is not None:
            require(self.origin == "synthetic", "only synthetic panels carry behaviour")
            object.__setattr__(self, "behavior", validate_frame(self.behavior))


ORIGIN_COLUMN = "panel_origin"


#: Process-local Run C context, set only by ``prediction.run_c`` for the one-time reveal
#: ("reveal") or its audit reproduction ("audit"). Real fits need BOTH the committed
#: authorization token and an active Run C context; nothing else can enable them.
_RUN_C_MODE: str | None = None


def _authorized() -> bool:
    return config.REAL_FIT_AUTHORIZATION == config.REAL_FIT_TOKEN and _RUN_C_MODE in (
        "reveal",
        "audit",
    )


def require_fit_allowed(panel: CanonicalPanel) -> None:
    """Refuse any predictive fit or score on real outcomes during Run B."""
    if panel.origin == "real":
        require(
            _authorized(),
            "Phase 7: fitting or scoring a predictive model on real outcomes is not "
            "authorized outside the Run C reveal/audit context (prediction.run_c)",
        )


def require_frames_allowed(*frames, panel: CanonicalPanel | None = None) -> None:
    """Data-level guard: every row carries its panel origin; real rows need authorization.

    Feature builders stamp ``panel_origin`` on every row, so the guard follows the data
    rather than a separately passed panel object.
    """
    for frame in frames:
        require(ORIGIN_COLUMN in frame.columns, "feature rows lack their panel origin stamp")
        origins = set(frame[ORIGIN_COLUMN].unique())
        require(origins <= {"real", "synthetic"}, f"unknown row origins: {origins}")
        if panel is not None:
            require(origins <= {panel.origin}, "feature rows do not belong to the given panel")
        if "real" in origins:
            require(
                _authorized(),
                "Phase 7: fitting or scoring a predictive model on real outcomes is not "
                "authorized outside the Run C reveal/audit context (prediction.run_c)",
            )


def household_support(frame: pd.DataFrame) -> pd.DataFrame:
    """First/last *recorded* purchase week per household (NaN if never recorded)."""
    active = frame.loc[frame[config.TARGET_SOURCE] == 1]
    support = active.groupby("household_key")["week_no"].agg(f="min", l="max")
    keys = pd.Index(frame["household_key"].unique(), name="household_key")
    return support.reindex(keys)


def load_real_panel(root=None) -> CanonicalPanel:
    """Read the accepted participation grid (read-only)."""
    from complete_journey.paths import PROJECT_ROOT

    root = PROJECT_ROOT if root is None else root
    path = root / "data/processed/econometrics_household_week.parquet"
    require(path.is_file(), f"accepted Phase 2 table missing: {path}")
    frame = pd.read_parquet(path, columns=list(COLUMNS))
    frame[config.TARGET_SOURCE] = frame[config.TARGET_SOURCE].astype(np.int8)
    return CanonicalPanel(frame, origin="real", label="accepted Phase 2 grid")
