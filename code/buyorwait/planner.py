"""Plan generation and the tie-break ladder.

Every candidate is built explicitly, checked against the 90-day safety rule, and
then ranked by the six criteria the problem statement fixes, in that order. No
model is involved: given the same ledger the same plan comes out every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from itertools import combinations
from typing import Sequence

from .config import MAX_SPENDING_CHANGES
from .dataset import PaymentOption, Profile, Request
from .forecast import Forecast, Payment, SpendingChange
from .ledger import FlexibleExpense, LedgerState
from .money import ZERO, money

STATUS_NOW = "affordable_now"
STATUS_PLAN = "affordable_with_plan"
STATUS_LATER = "affordable_later"
STATUS_NONE = "not_affordable"

METHOD_FULL = "full_payment"
METHOD_PARTIAL = "partial_payment"
METHOD_INSTALMENTS = "installments"
METHOD_WAIT = "wait"
METHOD_NONE = "not_recommended"


@dataclass(frozen=True)
class Candidate:
    method: str
    payments: tuple[Payment, ...]
    changes: tuple[SpendingChange, ...]
    option: PaymentOption | None
    total_paid: Decimal

    @property
    def rank(self) -> tuple:
        """The tie-break ladder, most significant first."""
        option_key = self.option.sort_key if self.option else (0, "")
        return (
            0 if not self.changes else 1,        # 2. require no spending changes
            len(self.changes),                   #    then as few as possible
            self.total_paid,                     # 3. minimise the total amount paid
            self.payments[0].when,               # 4. start payment earlier
            len(self.payments),                  # 5. use fewer payments
            option_key,                          # 6. lowest payment_option_id
        )


@dataclass(frozen=True)
class Decision:
    request: Request
    amount_safe_to_pay: Decimal
    status: str
    method: str
    payments: tuple[Payment, ...]
    changes: tuple[SpendingChange, ...]
    earliest_full_payment: date | None
    option: PaymentOption | None
    min_balance: Decimal


def option_schedule(option: PaymentOption) -> tuple[Payment, ...]:
    """Expand a supplied option into the exact payments it commits the user to."""
    step = option.payment_frequency_days or 0
    return tuple(
        Payment(option.first_payment_date + timedelta(days=step * index), option.payment_amount)
        for index in range(option.number_of_payments)
    )


def _change_sets(flexible: Sequence[FlexibleExpense]) -> list[tuple[SpendingChange, ...]]:
    """Every allowed combination of at most three changes, each on a distinct event.

    Stopping and reducing the same event are mutually exclusive, which falls out
    of building one option per event before taking combinations.
    """
    per_event: list[list[SpendingChange]] = []
    for item in flexible:
        options: list[SpendingChange] = []
        if item.can_stop:
            options.append(SpendingChange(item.event_id, None))
        if item.can_reduce and item.minimum_allowed_amount is not None:
            options.append(SpendingChange(item.event_id, item.minimum_allowed_amount))
        if options:
            per_event.append(options)
    result: list[tuple[SpendingChange, ...]] = []
    for size in range(1, min(MAX_SPENDING_CHANGES, len(per_event)) + 1):
        for chosen in combinations(range(len(per_event)), size):
            variants: list[tuple[SpendingChange, ...]] = [()]
            for index in chosen:
                variants = [
                    existing + (option,)
                    for existing in variants
                    for option in per_event[index]
                ]
            result.extend(variants)
    return result


def plan(
    request: Request,
    profile: Profile,
    state: LedgerState,
    options: Sequence[PaymentOption],
) -> Decision:
    forecast = Forecast(state)
    requested = money(request.requested_amount)
    deadline = request.desired_completion_date

    # Reported before any optional spending change, as the output contract requires.
    amount_safe = forecast.amount_safe_today(requested)
    earliest = forecast.earliest_full_payment_date(requested)

    change_sets: list[tuple[SpendingChange, ...]] = [()]
    if amount_safe < requested and state.flexible:
        change_sets.extend(_change_sets(state.flexible))

    candidates: list[Candidate] = []
    for changes in change_sets:
        candidates.extend(
            _candidates_under(request, profile, forecast, options, changes, requested, deadline)
        )

    if candidates:
        best = min(candidates, key=lambda c: c.rank)
        status = _status_for(best, request, profile, forecast, requested)
        return Decision(
            request=request,
            amount_safe_to_pay=amount_safe,
            status=status,
            method=best.method,
            payments=best.payments,
            changes=tuple(sorted(best.changes, key=_change_sort_key)),
            earliest_full_payment=earliest,
            option=best.option,
            min_balance=forecast.min_balance(best.payments, best.changes),
        )

    # Nothing completes the request by the deadline.
    if earliest is not None and METHOD_FULL in profile.payment_methods:
        return Decision(
            request=request,
            amount_safe_to_pay=amount_safe,
            status=STATUS_LATER,
            method=METHOD_WAIT,
            payments=(Payment(earliest, requested),),
            changes=(),
            earliest_full_payment=earliest,
            option=None,
            min_balance=forecast.min_balance([Payment(earliest, requested)]),
        )
    return Decision(
        request=request,
        amount_safe_to_pay=amount_safe,
        status=STATUS_LATER if earliest is not None else STATUS_NONE,
        method=METHOD_NONE,
        payments=(),
        changes=(),
        earliest_full_payment=earliest,
        option=None,
        min_balance=forecast.min_balance(),
    )


def _candidates_under(
    request: Request,
    profile: Profile,
    forecast: Forecast,
    options: Sequence[PaymentOption],
    changes: tuple[SpendingChange, ...],
    requested: Decimal,
    deadline: date,
) -> list[Candidate]:
    found: list[Candidate] = []
    request_date = request.request_date

    if METHOD_FULL in profile.payment_methods:
        payments = (Payment(request_date, requested),)
        if request_date <= deadline and forecast.is_safe(payments, changes):
            found.append(Candidate(METHOD_FULL, payments, changes, _full_option(options), requested))

    if METHOD_PARTIAL in profile.payment_methods and request.allows_partial_payment:
        safe_now = forecast.amount_safe_today(requested, changes)
        second_date = forecast.earliest_full_payment_date(requested, changes)
        if ZERO < safe_now < requested and second_date is not None and second_date <= deadline:
            remainder = requested - safe_now
            payments = (Payment(request_date, safe_now), Payment(second_date, remainder))
            if forecast.is_safe(payments, changes):
                found.append(Candidate(METHOD_PARTIAL, payments, changes, None, requested))

    if METHOD_INSTALMENTS in profile.payment_methods:
        for option in options:
            if option.payment_method != METHOD_INSTALMENTS:
                continue
            if not _installments_allowed(option, profile):
                continue
            payments = option_schedule(option)
            if payments[-1].when > deadline:
                continue
            if forecast.is_safe(payments, changes):
                total = sum((p.amount for p in payments), ZERO)
                found.append(Candidate(METHOD_INSTALMENTS, payments, changes, option, total))
    return found


def _installments_allowed(option: PaymentOption, profile: Profile) -> bool:
    """A blank `max_installment_months` means the user will not consider instalments."""
    if profile.max_installment_months is None:
        return False
    return option.number_of_payments <= profile.max_installment_months


def _full_option(options: Sequence[PaymentOption]) -> PaymentOption | None:
    for option in options:
        if option.payment_method == "full_payment":
            return option
    return None


def _status_for(
    best: Candidate, request: Request, profile: Profile, forecast: Forecast, requested: Decimal
) -> str:
    if (
        best.method == METHOD_FULL
        and not best.changes
        and best.payments[0].when == request.request_date
    ):
        return STATUS_NOW
    return STATUS_PLAN


def _change_sort_key(change: SpendingChange) -> tuple[int, str]:
    digits = "".join(ch for ch in change.event_id if ch.isdigit())
    return (int(digits) if digits else 0, change.event_id)
