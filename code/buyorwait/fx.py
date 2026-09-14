"""Dated currency conversion.

Rule from the problem statement: a foreign-currency cash event is converted with
the row matching its settlement date and its stated from/to direction. We accept
the inverse row as well, because a rate is symmetric, and we never fall back to a
different date silently: an unresolvable pair raises so the run fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from .money import money


class MissingRate(LookupError):
    """Raised when no dated rate covers a conversion the forecast needs."""


@dataclass(frozen=True)
class RateTable:
    rates: dict[tuple[date, str, str], Decimal]

    def convert(self, amount: Decimal, from_currency: str, to_currency: str, on: date) -> Decimal:
        if from_currency == to_currency:
            return money(amount)
        direct = self.rates.get((on, from_currency, to_currency))
        if direct is not None:
            return money(amount * direct)
        inverse = self.rates.get((on, to_currency, from_currency))
        if inverse is not None and inverse != 0:
            return money(amount / inverse)
        pivot = self._via_pivot(from_currency, to_currency, on)
        if pivot is not None:
            return money(amount * pivot)
        raise MissingRate(
            "no rate for " + from_currency + "->" + to_currency + " on " + on.isoformat()
        )

    def _via_pivot(self, from_currency: str, to_currency: str, on: date) -> Decimal | None:
        """One-hop bridge, e.g. ZAR->EUR->USD, using only rows dated `on`."""
        same_day = {
            (a, b): rate for (day, a, b), rate in self.rates.items() if day == on
        }
        legs: dict[tuple[str, str], Decimal] = {}
        for (a, b), rate in same_day.items():
            legs[(a, b)] = rate
            if rate != 0:
                legs.setdefault((b, a), Decimal(1) / rate)
        for (a, b), first in legs.items():
            if a != from_currency:
                continue
            second = legs.get((b, to_currency))
            if second is not None:
                return first * second
        return None
