"""Pure descriptive helpers: concentration curves and top-activity shares.

Deliberately pure: no database access, no file I/O, no plotting, no inference, and
no interpretation. Every function takes a sequence of nonnegative activity amounts
(retailer SALES_VALUE, trips, active weeks, ...) and returns numbers.

**These helpers describe recorded retailer activity only.** They answer "how much of
a recorded retailer total sits with how many households" and nothing else; names and
docstrings describe that arithmetic, not a distributional verdict.

Denominators are explicit: population shares always divide by the number of
observations supplied, and activity shares by the sum supplied. Callers choose the
population; the helper never assumes S0/S1/S2.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

#: Headline top-activity shares for household concentration (plan section 6, F4).
#: On the accepted 2,500-household universe these select exactly 25 / 125 / 250
#: households.
HEADLINE_TOP_SHARE_FRACTIONS: Final[tuple[float, float, float]] = (0.01, 0.05, 0.10)


def _validated(values: Sequence[float]) -> list[float]:
    """Return the values as a nonnegative float list, or raise.

    Raises:
        ValueError: on an empty input (there is no population to describe) or on a
            negative amount, because a negative activity total makes the
            concentration shares meaningless rather than merely surprising.
    """
    amounts = [float(value) for value in values]
    if not amounts:
        raise ValueError("no observations supplied; concentration is undefined for an empty set")
    negative = [amount for amount in amounts if amount < 0]
    if negative:
        raise ValueError(
            "activity amounts must be nonnegative; got "
            f"{len(negative)} negative value(s), e.g. {negative[0]!r}"
        )
    return amounts


def _total(values: Sequence[float]) -> float:
    return math.fsum(values)


def cumulative_population_share(values: Sequence[float]) -> tuple[float, ...]:
    """Cumulative share of the population: ``i / n`` for ``i = 1 .. n``."""
    amounts = _validated(values)
    n = len(amounts)
    return tuple(index / n for index in range(1, n + 1))


def cumulative_activity_share(values: Sequence[float]) -> tuple[float, ...]:
    """Cumulative share of total activity after sorting ascending.

    The curve is a function of the multiset of amounts, so ties need no tie-break:
    permuting equal values cannot change it. A zero total returns all zeros rather
    than ``nan`` — there is no activity to concentrate, and a NaN would propagate
    silently into a table.
    """
    amounts = sorted(_validated(values))
    total = _total(amounts)
    if total == 0.0:
        return tuple(0.0 for _ in amounts)
    running = 0.0
    shares: list[float] = []
    for amount in amounts:
        running += amount
        shares.append(running / total)
    return tuple(shares)


def lorenz_coordinates(values: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ``(population_share, activity_share)`` including the ``(0, 0)`` origin."""
    amounts = _validated(values)
    population = (0.0, *cumulative_population_share(amounts))
    activity = (0.0, *cumulative_activity_share(amounts))
    return population, activity


def top_activity_share(
    values: Sequence[float], fraction: float | None = None, *, households: int | None = None
) -> float:
    """Share of total activity held by the largest households.

    Exactly one selection rule applies, and it is stated rather than implied: the
    number of selected households is ``ceil(n * fraction)`` with a minimum of one.
    On the accepted 2,500-household universe, ``0.01 / 0.05 / 0.10`` therefore
    select exactly 25 / 125 / 250 households.

    Because the statistic depends only on the *k largest values*, ties cannot
    change it: selection is by count, not by identity, so the result is exactly
    deterministic. Fractions of the population that are not whole numbers are
    rounded up (a "top 1%" of 2,501 observations selects 26, never 25).

    A zero total returns ``0.0`` (no activity to concentrate).

    Args:
        values: Nonnegative activity amounts, one per household.
        fraction: Population fraction to select, e.g. ``0.05`` for the top 5%.
        households: Absolute count of households to select (alternative to
            ``fraction``; must be a positive integer, clipped to ``n``).

    Raises:
        ValueError: if neither or both selectors are given, if the fraction is not
            in ``(0, 1]``, or if ``households`` is not positive.
    """
    amounts = _validated(values)
    n = len(amounts)
    if (fraction is None) == (households is None):
        raise ValueError("provide exactly one of 'fraction' or 'households'")
    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must lie in (0, 1]; got {fraction!r}")
        k = max(1, math.ceil(n * fraction))
    else:
        assert households is not None  # narrows for type checkers; guarded above
        if households < 1:
            raise ValueError(f"households must be >= 1; got {households!r}")
        k = min(households, n)
    total = _total(amounts)
    if total == 0.0:
        return 0.0
    selected = sorted(amounts, reverse=True)[:k]
    return math.fsum(selected) / total


def gini(values: Sequence[float]) -> float:
    """Concentration summary statistic (secondary to the curves).

    Reported only as a supplementary number; the Lorenz-style curves and top-x%
    shares are the primary household-concentration evidence. This is a concentration
    summary of recorded retailer activity, never a distributional verdict, and it is
    never interpreted as one. A zero total returns ``0.0``.
    """
    amounts = sorted(_validated(values))
    n = len(amounts)
    total = _total(amounts)
    if total == 0.0:
        return 0.0
    weighted = math.fsum((index + 1) * amount for index, amount in enumerate(amounts))
    return (2.0 * weighted) / (n * total) - (n + 1) / n
