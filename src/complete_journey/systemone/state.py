"""Canonical synthetic state and question text (provider-neutral; P9-001, P9-002).

A ``SemanticState`` holds the scientific variables. Template A and its paraphrase B
render them into the ``state`` object and three Noul questions that every policy
receives verbatim. Every history summary is derived in code from the 4-week vector
``h``, so no logically inconsistent state can be rendered.

The wording is a *candidate* that the researcher must freeze before any external call.
"""

from __future__ import annotations

from dataclasses import dataclass

from complete_journey.systemone import settings as S
from complete_journey.systemone.settings import require

NOT_PROVIDED = {"A": "Not provided.", "B": "This information is not available."}


@dataclass(frozen=True, order=True)
class SemanticState:
    """One synthetic household-week state. ``None`` = the information is not provided."""

    cell: str
    profile: float | None  # stated long-run frequency (grid level) or None (unavailable)
    history: tuple[int, ...] | None  # (h1, h2, h3, h4), h1 = last week
    need: str | None  # "low" / "usual" / "high"
    placebo: str = S.PLACEBO_DEFAULT

    def __post_init__(self):
        require(self.cell in S.CELLS, f"unknown cell {self.cell}")
        hv, nv = S.CELL_VISIBILITY[self.cell]
        require((self.history is not None) == hv, "history presence must match the cell")
        require((self.need is not None) == nv, "need presence must match the cell")
        if self.history is not None:
            require(
                len(self.history) == S.HISTORY_WEEKS and set(self.history) <= {0, 1},
                "history must be a binary 4-week vector",
            )
        if self.need is not None:
            require(self.need in S.NEED_LEVELS, f"unknown need level {self.need}")
        if self.profile is not None:
            require(0.0 < self.profile < 1.0, "profile must be a frequency in (0, 1)")
        require(self.placebo in S.PLACEBO_VALUES, "unknown placebo value")

    @property
    def count(self) -> int | None:
        return None if self.history is None else int(sum(self.history))

    @property
    def weeks_since_last(self) -> int | None:
        """Weeks since the last shopping week within the 4-week window (None if none)."""
        if self.history is None:
            return None
        for j, v in enumerate(self.history):
            if v:
                return j + 1
        return None

    def key(self) -> str:
        prof = "U" if self.profile is None else f"{self.profile:.2f}"
        hist = "-" if self.history is None else "".join(map(str, self.history))
        return f"{self.cell}|{prof}|{hist}|{self.need or '-'}|{self.placebo}"

    def with_placebo(self, value: str) -> SemanticState:
        return SemanticState(self.cell, self.profile, self.history, self.need, value)


def percent(profile: float) -> str:
    """Generic grid level as a whole percent (grid levels are multiples of 5%)."""
    pct = round(profile * 100)
    require(abs(pct - profile * 100) < 1e-9 and pct % 5 == 0, "profile must be on the 5% grid")
    return f"{pct}%"


# --- Template A ---------------------------------------------------------------------
_HIST_A = {1: "shopped at this grocery store", 0: "did not shop at this grocery store"}
_AGO_A = {1: "Last week", 2: "2 weeks ago", 3: "3 weeks ago", 4: "4 weeks ago"}
_NEED_A = {
    "low": "In the coming week, the household needs fewer groceries than usual.",
    "usual": "In the coming week, the household needs about the usual amount of groceries.",
    "high": "In the coming week, the household needs more groceries than usual.",
}

# --- Template B (paraphrase: same information, same numeric format) -----------------------
_HIST_B = {1: "made a purchase at this grocery store", 0: "made no purchase at this grocery store"}
_AGO_B = {1: "The previous week", 2: "2 weeks ago", 3: "3 weeks ago", 4: "4 weeks ago"}
_NEED_B = {
    "low": "For the next week, the household's grocery needs are below their normal level.",
    "usual": "For the next week, the household's grocery needs are at their normal level.",
    "high": "For the next week, the household's grocery needs are above their normal level.",
}


def render_state(s: SemanticState, template: str) -> dict[str, str]:
    """The ``state`` object sent to every policy (keys and values are both model input)."""
    require(template in S.TEMPLATES, "unknown template")
    na = NOT_PROVIDED[template]
    if template == "A":
        profile = (
            na
            if s.profile is None
            else f"Over the long run, this household shops at this grocery store in about {percent(s.profile)} of weeks."
        )
        if s.history is None:
            recent, count = na, na
        else:
            recent = " ".join(f"{_AGO_A[j + 1]}: {_HIST_A[v]}." for j, v in enumerate(s.history))
            count = f"Shopped at this grocery store in {s.count} of the last 4 weeks."
        need = na if s.need is None else _NEED_A[s.need]
        return {
            "long_run_shopping_frequency": profile,
            "recent_shopping_by_week": recent,
            "weeks_with_shopping_in_last_4": count,
            "grocery_need_in_coming_week": need,
            S.PLACEBO_CANDIDATES[S.PLACEBO_CANDIDATE]["A"][0]: S.PLACEBO_CANDIDATES[
                S.PLACEBO_CANDIDATE
            ]["A"][1].format(v=s.placebo),
        }
    profile = (
        na
        if s.profile is None
        else f"In the long term, this household makes a purchase at this grocery store in roughly {percent(s.profile)} of all weeks."
    )
    if s.history is None:
        recent, count = na, na
    else:
        recent = " ".join(f"{_AGO_B[j + 1]}: {_HIST_B[v]}." for j, v in enumerate(s.history))
        count = f"Purchases at this grocery store in {s.count} of the previous 4 weeks."
    need = na if s.need is None else _NEED_B[s.need]
    return {
        "typical_purchase_frequency": profile,
        "purchases_in_recent_weeks": recent,
        "purchase_weeks_out_of_previous_4": count,
        "grocery_needs_next_week": need,
        S.PLACEBO_CANDIDATES[S.PLACEBO_CANDIDATE]["B"][0]: S.PLACEBO_CANDIDATES[
            S.PLACEBO_CANDIDATE
        ]["B"][1].format(v=s.placebo),
    }


_QUESTIONS = {
    "A": {
        S.Q_SHOP: {
            "type": "noul",
            "instructions": "Will this household shop at this grocery store in the coming week?",
            "criteria": {
                "true": "The household makes at least one purchase at this grocery store in the coming week.",
                "false": "The household makes no purchase at this grocery store in the coming week.",
            },
        },
        S.Q_SKIP: {
            "type": "noul",
            "instructions": "Will this household make no purchase at this grocery store in the coming week?",
            "criteria": {
                "true": "The household makes no purchase at this grocery store in the coming week.",
                "false": "The household makes at least one purchase at this grocery store in the coming week.",
            },
        },
        S.Q_TYPE: {
            "type": "noul",
            "instructions": "Is this a household that shops at this grocery store in most weeks over the long run?",
            "criteria": {
                "true": "Over the long run, the household shops at this grocery store in more than half of all weeks.",
                "false": "Over the long run, the household shops at this grocery store in half of all weeks or fewer.",
            },
        },
    },
    "B": {
        S.Q_SHOP: {
            "type": "noul",
            "instructions": "During the next week, will this household make a purchase at this grocery store?",
            "criteria": {
                "true": "During the next week, the household buys something at this grocery store at least once.",
                "false": "During the next week, the household buys nothing at this grocery store.",
            },
        },
        S.Q_SKIP: {
            "type": "noul",
            "instructions": "During the next week, will this household buy nothing at this grocery store?",
            "criteria": {
                "true": "During the next week, the household buys nothing at this grocery store.",
                "false": "During the next week, the household buys something at this grocery store at least once.",
            },
        },
        S.Q_TYPE: {
            "type": "noul",
            "instructions": "In the long term, does this household make a purchase at this grocery store in the majority of weeks?",
            "criteria": {
                "true": "In the long term, the household buys at this grocery store in more than half of all weeks.",
                "false": "In the long term, the household buys at this grocery store in half of all weeks or fewer.",
            },
        },
    },
}


def questions(template: str, which: tuple[str, ...] = (S.Q_SHOP, S.Q_SKIP, S.Q_TYPE)) -> dict:
    require(template in S.TEMPLATES, "unknown template")
    require(set(which) <= set(S.QUESTION_ROLES) and which, "unknown question id")
    return {q: _deepcopy(_QUESTIONS[template][q]) for q in which}


def _deepcopy(d):
    if isinstance(d, dict):
        return {k: _deepcopy(v) for k, v in d.items()}
    return d
