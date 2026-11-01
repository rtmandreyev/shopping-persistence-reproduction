"""Policy table: archived (or mock) answers into per-state policy values.

Conventions (P9-001):
- ``primary``: pi = the ``shop`` noul;
- ``normalized``: pi = shop / (shop + skip), computed per repeat and then averaged.
  If shop + skip <= 0 for a state, its normalized value is undefined (NaN). Any DGP
  that reaches such a state reports the convention as UNDEFINED and never imputes it.

The closed loop uses only template-A, default-placebo, ``production`` answers averaged
over repeats. The paraphrase (template B), placebo and sentinel answers serve the
validity rule and drift checks.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from complete_journey.systemone import lattice as L
from complete_journey.systemone import settings as S
from complete_journey.systemone.archive import Archive
from complete_journey.systemone.request import production_requests, request_from_identity
from complete_journey.systemone.settings import require
from complete_journey.systemone.state import SemanticState

CONVENTIONS = ("primary", "normalized")


def convention_value(shop: float, skip: float | None, convention: str) -> float:
    require(convention in CONVENTIONS, "unknown convention")
    if convention == "primary":
        return float(shop)
    if skip is None:
        return float("nan")
    denom = shop + skip
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(shop / denom)


@dataclass
class PolicyData:
    policy: str
    k: int
    # (purpose, template, state_key, timepoint) -> repeat -> {question: value}
    cells: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(dict)))
    states: dict = field(default_factory=dict)  # state_key -> SemanticState

    def add(
        self,
        purpose: str,
        template: str,
        state: SemanticState,
        timepoint: int,
        repeat: int,
        answers: dict,
    ):
        slot = self.cells[(purpose, template, state.key(), timepoint)][repeat]
        for q, v in answers.items():
            require(q not in slot, f"duplicate answer for {q} at {state.key()}")
            slot[q] = float(v)
        self.states[state.key()] = state

    # --- builders ------------------------------------------------------------------
    @classmethod
    def from_archive(cls, root, policy: str, k: int) -> PolicyData:
        pd = cls(policy, k)
        for rec in Archive(root).successes():
            req = request_from_identity({**rec["identity"], "request_hash": rec["request_hash"]})
            require(req.policy == policy, "archive holds another policy")
            pd.add(req.purpose, req.template, req.state, req.timepoint, req.repeat, rec["answers"])
        return pd

    @classmethod
    def from_mock(
        cls, mock, policy: str, k: int, noise_sd: float = 0.0, batched: bool = True
    ) -> PolicyData:
        """The same table a mock-provider run would archive, built without I/O."""
        pd = cls(policy, k)
        sd = noise_sd or mock.params.get("noise_sd", 0.0)
        for req in production_requests(policy, k, batched=batched):
            vals = dict(
                zip(
                    (S.Q_SHOP, S.Q_SKIP, S.Q_TYPE),
                    mock.answer(req.state, req.template),
                    strict=True,
                )
            )
            if sd:
                seed = int(req.request_hash[:16], 16)
                g = np.random.default_rng(seed)
                vals = {q: float(np.clip(v + g.normal(0, sd), 0, 1)) for q, v in vals.items()}
            pd.add(
                req.purpose,
                req.template,
                req.state,
                req.timepoint,
                req.repeat,
                {q: vals[q] for q in req.question_ids},
            )
        return pd

    # --- accessors ----------------------------------------------------------------------
    def _reps(
        self, purpose: str, template: str, s: SemanticState, timepoint: int = 0
    ) -> list[dict]:
        slot = self.cells.get((purpose, template, s.key(), timepoint))
        require(
            slot is not None and len(slot) > 0,
            f"unqueried state {s.key()} ({purpose}/{template}/{timepoint})",
        )
        return [slot[r] for r in sorted(slot)]

    def values(
        self,
        s: SemanticState,
        convention: str,
        purpose: str = "production",
        template: str = "A",
        timepoint: int = 0,
    ) -> np.ndarray:
        reps = self._reps(purpose, template, s, timepoint)
        return np.array(
            [convention_value(r[S.Q_SHOP], r.get(S.Q_SKIP), convention) for r in reps], float
        )

    def pi(self, s: SemanticState, convention: str = "primary") -> float:
        return float(np.mean(self.values(s, convention)))

    def paraphrase(self, s: SemanticState, convention: str = "primary") -> float:
        return float(np.mean(self.values(s, convention, purpose="paraphrase", template="B")))

    def placebo(self, s: SemanticState, convention: str = "primary") -> float | None:
        flipped = s.with_placebo(S.PLACEBO_VALUES[1])
        if (("placebo", "A", flipped.key(), 0)) not in self.cells:
            return None
        return float(np.mean(self.values(flipped, convention, purpose="placebo")))

    def sentinel(self, s: SemanticState, t: int, convention: str = "primary") -> float:
        return float(np.mean(self.values(s, convention, purpose="sentinel", timepoint=t)))

    def type_probe(self, s: SemanticState) -> float:
        return float(np.mean([r[S.Q_TYPE] for r in self._reps("production", "A", s)]))

    def coherence_gap(self, s: SemanticState) -> float:
        return float(
            np.mean([r[S.Q_SHOP] + r[S.Q_SKIP] - 1.0 for r in self._reps("production", "A", s)])
        )

    def repeat_sd(self, convention: str = "primary", question: str = S.Q_SHOP) -> float:
        """Pooled single-response SD over template-A production states with >= 2 repeats.
        For ``question=Q_SHOP`` the values follow the convention. Other questions use
        their raw noul. States whose values are undefined under the convention are
        excluded and counted in ``self.repeat_sd_dropped``."""
        var, dropped = [], 0
        for (purpose, template, key, _), slot in self.cells.items():
            if purpose == "production" and template == "A" and len(slot) >= 2:
                if question == S.Q_SHOP:
                    v = self.values(self.states[key], convention)
                else:
                    v = np.array([slot[r][question] for r in sorted(slot)], float)
                if np.all(np.isfinite(v)):
                    var.append(np.var(v, ddof=1))
                else:
                    dropped += 1
        self.repeat_sd_dropped = dropped
        require(len(var) > 0, "repeat SD needs repeated states")
        return float(np.sqrt(np.mean(var)))

    def table(self, cell: str, convention: str = "primary", shift: float = 0.0) -> np.ndarray:
        """Dense lookup array pi[profile, h_index, need_index] for a closed-loop DGP.
        Entries the DGP can never reach stay NaN. Reaching one is a loud error."""
        g = L.grid(self.k)
        hv, nv = S.CELL_VISIBILITY[cell]
        arr = np.full((len(g), 16, 3), np.nan)
        for i, f in enumerate(g):
            for hi, h in enumerate(S.HISTORY_PATTERNS if hv else [None]):
                for ni, n in enumerate(S.NEED_LEVELS if nv else [None]):
                    p = self.pi(SemanticState(cell, f, h, n), convention)
                    if shift and np.isfinite(p):
                        p = shifted(p, shift)
                    hs = [hi] if hv else range(16)
                    ns = [ni] if nv else range(3)
                    for a in hs:
                        for b in ns:
                            arr[i, a, b] = p
        return arr


def shifted(p: float | np.ndarray, delta: float):
    """Global logit-intercept shift (the secondary calibration). Strictly monotone."""
    q = np.clip(p, 1e-9, 1 - 1e-9)
    return 1.0 / (1.0 + np.exp(-(np.log(q / (1 - q)) + delta)))
