"""The 90-day safety check.

A plan is safe when the projected balance never drops below the user's
`minimum_balance_to_keep` at any point in the forecast window. Everything the
planner asks - how much is safe today, when a full payment first becomes safe,
whether a candidate plan survives - is answered by the two primitives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Iterable, Mapping, Sequence

from .ledger import LedgerState, PROJECTED
from .money import ZERO, floor_money, money


@dataclass(frozen=True)
class Payment:
    when: date
    amount: Decimal


@dataclass(frozen=True)
class SpendingChange:
    """`stop:<event_id>` when new_amount is None, otherwise `reduce_to:<event_id>:<amount>`."""

    event_id: str
    new_amount: Decimal | None

    def render(self) -> str:
        from .money import format_plan_amount

        if self.new_amount is None:
            return "stop:" + self.event_id
        return "reduce_to:" + self.event_id + ":" + format_plan_amount(self.new_amount)


class Forecast:
    """Balance projection for one user, from one request date."""

    def __init__(self, state: LedgerState) -> None:
        self.state = state
        self.start = state.request_date
        self.end = state.horizon_end
        self.opening = money(state.profile.balance)
        self.minimum = money(state.profile.minimum_balance)

    # -- core ----------------------------------------------------------------
    def _daily_net(self, changes: Sequence[SpendingChange] = ()) -> dict[date, Decimal]:
        overrides: Mapping[str, Decimal | None] = {c.event_id: c.new_amount for c in changes}
        net: dict[date, Decimal] = {}
        for flow in self.state.flows:
            amount = flow.amount
            if flow.event_id in overrides and flow.source == PROJECTED and flow.is_debit:
                replacement = overrides[flow.event_id]
                amount = ZERO if replacement is None else -money(replacement)
            net[flow.when] = net.get(flow.when, ZERO) + amount
        return net

    def balance_path(
        self, payments: Iterable[Payment] = (), changes: Sequence[SpendingChange] = ()
    ) -> list[tuple[date, Decimal]]:
        net = self._daily_net(changes)
        for payment in payments:
            net[payment.when] = net.get(payment.when, ZERO) - money(payment.amount)
        balance = self.opening
        path = [(self.start, balance)]
        for when in sorted(net):
            if when < self.start or when > self.end:
                continue
            balance += net[when]
            path.append((when, balance))
        return path

    def min_balance(
        self, payments: Iterable[Payment] = (), changes: Sequence[SpendingChange] = ()
    ) -> Decimal:
        return min(balance for _, balance in self.balance_path(payments, changes))

    def is_safe(
        self, payments: Iterable[Payment] = (), changes: Sequence[SpendingChange] = ()
    ) -> bool:
        return self.min_balance(payments, changes) >= self.minimum

    # -- the two questions the planner asks ----------------------------------
    def amount_safe_today(
        self, requested_amount: Decimal, changes: Sequence[SpendingChange] = ()
    ) -> Decimal:
        """Largest amount payable on the request date without ever breaching the minimum.

        The balance is linear in a payment made on day one, so the answer is the
        forecast trough minus the minimum, with no search required.
        """
        headroom = self.min_balance(changes=changes) - self.minimum
        return max(ZERO, min(money(requested_amount), floor_money(headroom)))

    def earliest_full_payment_date(
        self, amount: Decimal, changes: Sequence[SpendingChange] = ()
    ) -> date | None:
        """First date in the window on which the whole amount clears the safety check.

        A payment on day d only depresses the balance from d onwards, so we walk
        the suffix minimum of the untouched path.
        """
        path = self.balance_path(changes=changes)
        milestones = [when for when, _ in path]
        balances = [balance for _, balance in path]
        count = len(balances)
        suffix_min = list(balances)
        for index in range(count - 2, -1, -1):
            suffix_min[index] = min(suffix_min[index], suffix_min[index + 1])

        cursor = self.start
        index = 0                      # index of the last milestone at or before cursor
        while cursor <= self.end:
            while index + 1 < count and milestones[index + 1] <= cursor:
                index += 1
            # The untouched balance is flat between flow dates, so its minimum
            # from `cursor` onward is the lower of the value carried into
            # `cursor` and the minimum of every later milestone. A payment on
            # `cursor` then shifts that whole trough down by `amount`.
            carried = balances[index]
            later = suffix_min[index + 1] if index + 1 < count else carried
            trough = min(carried, later)
            if trough - money(amount) >= self.minimum:
                return cursor
            cursor += timedelta(days=1)
        return None
