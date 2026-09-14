"""Recurrence detection.

The dataset generator emits two kinds of repeating series and nothing else:

* monthly on a fixed day-of-month (rent, utilities, salary, subscriptions), and
* a fixed day interval of 5, 7, 10, 14 or 21 days (groceries, transport, dining).

`detect` recovers those shapes from history so the forecast can project them
forward. It never invents a series from a single observation.
"""

from __future__ import annotations

import calendar
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Callable, Iterable, Sequence

from .config import (
    FIXED_INTERVAL_AGREEMENT,
    FIXED_INTERVALS,
    MIN_OCCURRENCES,
    MONTHLY_DOM_COVERAGE,
    MONTHLY_DOM_MAX_GROUPS,
)
from .money import money

MONTHLY = "monthly"
FIXED = "fixed"


@dataclass(frozen=True)
class Observation:
    """One historical occurrence of a series, already converted to home currency."""

    when: date
    amount: Decimal
    event_id: str
    category: str
    flexibility: str
    minimum_allowed_amount: Decimal | None
    description: str = ""


@dataclass(frozen=True)
class Series:
    kind: str                 # MONTHLY or FIXED
    parameter: int            # day-of-month, or interval in days
    direction: str            # credit or debit
    category: str
    event_type: str
    observations: tuple[Observation, ...]

    def key(self, user_id: str) -> tuple[str, str, str, str, str, int]:
        """Stable identity of this recurring group, independent of any one
        request's history cutoff - the same physical income or expense stream
        always yields the same key regardless of which request built it.
        Used to attach an income-stability classification (see
        income_classifier.py) to a series across every request that touches
        this user.
        """
        return (user_id, self.category, self.direction, self.event_type, self.kind, self.parameter)

    @property
    def anchor(self) -> Observation:
        return self.observations[-1]

    @property
    def event_id(self) -> str:
        return self.anchor.event_id

    def occurrences(self, start: date, end: date) -> list[date]:
        """Projected dates strictly after the anchor and inside [start, end]."""
        dates: list[date] = []
        cursor = self._advance(self.anchor.when)
        guard = 0
        while cursor <= end and guard < 400:
            if cursor >= start:
                dates.append(cursor)
            cursor = self._advance(cursor)
            guard += 1
        return dates

    def _advance(self, from_date: date) -> date:
        if self.kind == FIXED:
            return from_date + timedelta(days=self.parameter)
        return _add_month(from_date, self.parameter)


def _add_month(from_date: date, day_of_month: int) -> date:
    year, month = from_date.year, from_date.month + 1
    if month > 12:
        month, year = 1, year + 1
    return date(year, month, min(day_of_month, calendar.monthrange(year, month)[1]))


ESTIMATORS: dict[str, Callable[[Sequence[Decimal]], Decimal]] = {
    "last": lambda values: values[-1],
    "mean": lambda values: sum(values) / len(values),
    "mean3": lambda values: sum(values[-3:]) / len(values[-3:]),
    "mean6": lambda values: sum(values[-6:]) / len(values[-6:]),
    "median": lambda values: Decimal(str(statistics.median([float(v) for v in values]))),
    "max": max,
    "max3": lambda values: max(values[-3:]),
    "min": min,
    "min3": lambda values: min(values[-3:]),
}


# A single occurrence this many times the group's median is treated as a
# one-off (a bulk purchase, a catch-up payment) rather than representative of
# the recurring amount - "distinguish recurring expenses from ... unusual
# events" per the problem statement. It still counts as a real past debit
# (already reflected in the opening balance); it is only excluded from the
# figure projected forward.
OUTLIER_MEDIAN_MULTIPLE = Decimal("3")


def _drop_amount_outliers(values: Sequence[Decimal]) -> list[Decimal]:
    if len(values) < 4:
        return list(values)
    ordered = sorted(values)
    median = ordered[len(ordered) // 2]
    if median <= 0:
        return list(values)
    kept = [v for v in values if v <= median * OUTLIER_MEDIAN_MULTIPLE]
    return kept or list(values)


def estimate_amount(series: Series, estimator: str) -> Decimal:
    """Project the next amount of a series using the configured estimator."""
    values = _drop_amount_outliers([observation.amount for observation in series.observations])
    return money(ESTIMATORS[estimator](values))


def detect(observations: Iterable[Observation], direction: str, category: str, event_type: str) -> list[Series]:
    """Split one (category, direction, event_type) group into recurring series."""
    ordered = sorted(observations, key=lambda o: (o.when, o.event_id))
    if len(ordered) < MIN_OCCURRENCES:
        return []
    dates = [o.when for o in ordered]

    if len(ordered) >= 3:
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        counts = Counter(gaps)
        top_gap, top_count = counts.most_common(1)[0]
        if (
            not _looks_monthly(dates)
            and top_gap in FIXED_INTERVALS
            and top_count / len(gaps) >= FIXED_INTERVAL_AGREEMENT
        ):
            return [Series(FIXED, top_gap, direction, category, event_type, tuple(ordered))]

    buckets: dict[int, list[Observation]] = defaultdict(list)
    for observation in ordered:
        buckets[observation.when.day].append(observation)
    return [
        Series(MONTHLY, day, direction, category, event_type, tuple(members))
        for day, members in sorted(buckets.items())
        if len(members) >= MIN_OCCURRENCES
    ]


def _looks_monthly(dates: Sequence[date]) -> bool:
    """True when most events repeat on the same few days of the month."""
    day_counts = Counter(d.day for d in dates)
    repeated = sum(count for count in day_counts.values() if count >= MIN_OCCURRENCES)
    return (
        repeated / len(dates) >= MONTHLY_DOM_COVERAGE
        and len(day_counts) <= MONTHLY_DOM_MAX_GROUPS
    )
