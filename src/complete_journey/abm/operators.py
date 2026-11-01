"""Phase 8 purchase-envelope operators (design §W, §X).

Both operators hand the Phase 7 analysis operator O, used verbatim, a recorded panel
that is **identical to generated behaviour**. Nothing is forced, masked away or
assigned after the fact:

- **Primary (§W.2).** Behaviour exists for every household in every week 1–102. O
  derives each household's own purchase envelope [f_i, l_i] from its first and last
  *simulated* purchases, then applies the interior, lag and |W_i| ≥ 8 rules.
- **S-a (§X).** Behaviour is generated only inside an imposed availability window,
  and is 0 outside because the household is unavailable. O again derives the purchase
  envelope from realized purchases, so the envelope may be shorter than the
  availability window.

The Phase 7 endpoint-anchored rule (``prediction.operators.record_observation``)
forces recorded purchases at assigned window ends. It is superseded for Phase 8 and
appears here only in ``historical_forced_endpoint_panel``. That function serves the
diagnostic showing the rule's defects; its panels are never accepted as Phase 8
worlds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from complete_journey.abm import config
from complete_journey.abm.architectures import World
from complete_journey.abm.config import require
from complete_journey.abm.guards import require_stage
from complete_journey.abm.seeds import SeedKey, Stream
from complete_journey.prediction import config as p7
from complete_journey.prediction.operators import (
    draw_empirical_windows,
    interior_windows,
    record_observation,
)
from complete_journey.prediction.panel import CanonicalPanel, household_support

PHASE8_PANEL_TAG = "phase8-world"


def household_keys(n: int) -> np.ndarray:
    return config.SYNTHETIC_KEY_OFFSET + np.arange(1, n + 1)


def long_frame(matrix: np.ndarray) -> pd.DataFrame:
    n, weeks = matrix.shape
    require(weeks == config.T_WEEKS, "a world records exactly weeks 1-102")
    return pd.DataFrame(
        {
            "household_key": np.repeat(household_keys(n), weeks),
            "week_no": np.tile(np.arange(p7.FIRST_WEEK, p7.LAST_WEEK + 1), n),
            p7.TARGET_SOURCE: matrix.ravel().astype(np.int8),
        }
    )


def world_panel(world: World) -> CanonicalPanel:
    """The analyst's record of a Phase 8 world: exactly the generated behaviour."""
    require(world.operator in config.PHASE8_OPERATORS, "unknown Phase 8 operator")
    if world.operator == config.OPERATOR_PRIMARY:
        require(world.availability is None, "the primary operator assigns no windows")
    else:
        require(world.availability is not None, "S-a needs availability windows")
    frame = long_frame(world.behavior)
    return CanonicalPanel(
        frame,
        origin="synthetic",
        behavior=frame.copy(),
        label=f"{PHASE8_PANEL_TAG}:{world.operator}:{world.spec.family.name}",
    )


def require_phase8_panel(panel: CanonicalPanel) -> None:
    """Refuse panels that are not unaltered Phase 8 worlds (e.g. forced endpoints)."""
    require(panel.origin == "synthetic", "Phase 8 Run B evaluates synthetic worlds only")
    require(panel.label.startswith(PHASE8_PANEL_TAG), "panel is not a Phase 8 world record")
    require(panel.behavior is not None, "Phase 8 worlds carry their behaviour")
    rec = panel.recorded[p7.TARGET_SOURCE].to_numpy()
    beh = panel.behavior[p7.TARGET_SOURCE].to_numpy()
    require(np.array_equal(rec, beh), "recorded panel differs from behaviour (forced/masked)")


def envelopes(panel: CanonicalPanel) -> pd.DataFrame:
    """Per-household purchase envelope and interior via the Phase 7 operator code.

    Columns: f, l (first/last recorded purchase; NaN if none), w_start, w_end, w_len,
    eligible, status ∈ {"no_purchase", "single_purchase", "short", "eligible"}.
    """
    support = household_support(panel.recorded)
    windows = interior_windows(panel)
    out = support.join(windows)
    status = np.where(
        out["f"].isna(),
        "no_purchase",
        np.where(
            out["f"] == out["l"],
            "single_purchase",
            np.where(out["eligible"], "eligible", "short"),
        ),
    )
    out["status"] = status
    return out


def assign_availability(pairs: pd.DataFrame, n: int, key: SeedKey, origin: str) -> np.ndarray:
    """S-a: assign (f, l) pairs as availability windows with the declared Phase 7 rule
    (without replacement, independent of household type). Real E-2 pairs need Stage 4."""
    require(origin in ("synthetic", "real"), "availability origin must be synthetic or real")
    if origin == "real":
        require_stage("availability")
    rng = key.generator(Stream.AVAILABILITY)
    drawn = draw_empirical_windows(pairs, household_keys(n), rng)
    return drawn[["f", "l"]].to_numpy(dtype=int)


def historical_forced_endpoint_panel(world: World, windows: np.ndarray) -> CanonicalPanel:
    """DIAGNOSTIC ONLY: the superseded Phase 7 Run B rule (record-only forced endpoints)."""
    frame = long_frame(world.behavior)
    w = pd.DataFrame(
        {"household_key": household_keys(world.n), "f": windows[:, 0], "l": windows[:, 1]}
    )
    panel = record_observation(frame, w, label="historical-forced-endpoint-diagnostic")
    return panel
