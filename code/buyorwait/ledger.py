"""Financial-state reconstruction.

Turns raw event rows into the projected cash movements of one user over the
forecast window. Everything that decides whether a row moves cash - status,
direction, linkage, currency, evidence from messages and images - is resolved
here, so the forecaster downstream only sees signed amounts on dates.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from dataclasses import replace as dataclasses_replace
from datetime import date, timedelta
from decimal import Decimal
from typing import Iterable, Sequence

from .config import EXPENSE_ESTIMATOR, FORECAST_DAYS, INCOME_ESTIMATOR
from .dataset import DEAD_STATUSES, Dataset, Event, Profile
from .evidence import Amendment, EvidenceSet
from .fx import MissingRate, RateTable
from .money import money
from .recurrence import Observation, Series, _add_month, detect, estimate_amount

# Only a salary series repeats. Refunds, windfalls and investment sales are
# one-off credits and must never be projected forward.
RECURRING_CREDIT = {("income", "salary")}
# Debit families that repeat. investment_purchase is a discrete decision, not a bill.
RECURRING_DEBIT_TYPES = {"expense", "subscription", "debt_payment"}

EXPLICIT = "explicit"
PROJECTED = "projected"


def _subtract_month(from_date: date) -> date:
    year, month = from_date.year, from_date.month - 1
    if month < 1:
        month, year = 12, year - 1
    return date(year, month, min(from_date.day, calendar.monthrange(year, month)[1]))


@dataclass(frozen=True)
class IncomeChange:
    effective_date: date | None
    amount: Decimal
    scope: str                    # "ongoing" or "next_payment"


@dataclass(frozen=True)
class IncomePolicy:
    """How payroll messages rewrite the projected salary track.

    Applied to the finished list of salary flows rather than to one flow at a
    time, because "your next salary is reduced" is a statement about position in
    the sequence, not about a date.
    """

    changes: tuple[IncomeChange, ...] = ()
    stop_after: date | None = None
    moved_to: date | None = None

    @classmethod
    def from_amendments(cls, amendments) -> "IncomePolicy":
        ordered = sorted(amendments, key=lambda a: (a.sent_at, a.source))
        changes: list[IncomeChange] = []
        stop_after: date | None = None
        moved_to: date | None = None
        for amendment in ordered:
            if amendment.kind == "income_amount_change" and amendment.amount is not None:
                changes.append(
                    IncomeChange(amendment.effective_date, amendment.amount, amendment.scope)
                )
            elif amendment.kind == "income_end":
                stop_after = amendment.effective_date
            elif amendment.kind == "income_date_change" and amendment.effective_date is not None:
                moved_to = amendment.effective_date
        return cls(tuple(changes), stop_after, moved_to)

    @property
    def is_empty(self) -> bool:
        return not self.changes and self.stop_after is None and self.moved_to is None

    def apply(self, flows: list["Flow"]) -> list["Flow"]:
        if self.is_empty:
            return flows
        salary = sorted(
            [f for f in flows if _is_salary_flow(f)], key=lambda f: f.when
        )
        others = [f for f in flows if not _is_salary_flow(f)]
        rebuilt: list[Flow] = []
        for flow in salary:
            when = self._move(flow.when)
            if self.stop_after is not None and when > self.stop_after:
                continue              # evidence says no confirmed income after this
            rebuilt.append(dataclasses_replace(flow, when=when))
        rebuilt.sort(key=lambda f: f.when)
        for change in self.changes:
            self._apply_change(rebuilt, change)
        return others + rebuilt

    def _apply_change(self, salary: list["Flow"], change: IncomeChange) -> None:
        if change.scope == "next_payment":
            for index, flow in enumerate(salary):
                if change.effective_date is None or flow.when >= change.effective_date:
                    salary[index] = dataclasses_replace(flow, amount=change.amount)
                    return
            return
        for index, flow in enumerate(salary):
            if change.effective_date is None or flow.when >= change.effective_date:
                salary[index] = dataclasses_replace(flow, amount=change.amount)

    def _move(self, when: date) -> date:
        """A payroll date change moves that month's credit and every later cycle."""
        if self.moved_to is None:
            return when
        target = self.moved_to
        if (when.year, when.month) == (target.year, target.month):
            return target
        if when < target:
            return when
        shifted = target
        guard = 0
        while (shifted.year, shifted.month) != (when.year, when.month) and guard < 400:
            shifted = _add_month(shifted, target.day)
            guard += 1
        return shifted if guard < 400 else when


def _is_salary_flow(flow: "Flow") -> bool:
    return flow.category == "salary" and flow.amount > 0


@dataclass(frozen=True)
class Flow:
    """One projected cash movement in the user's home currency."""

    when: date
    amount: Decimal          # negative for money out, positive for money in
    event_id: str
    category: str
    source: str              # EXPLICIT (a supplied row) or PROJECTED (a recurrence)
    flexibility: str = "fixed"
    minimum_allowed_amount: Decimal | None = None
    description: str = ""

    @property
    def is_debit(self) -> bool:
        return self.amount < 0


@dataclass(frozen=True)
class FlexibleExpense:
    """A recurring debit the user has authorised us to stop or reduce."""

    event_id: str
    category: str
    amount: Decimal          # positive magnitude of one occurrence
    minimum_allowed_amount: Decimal | None
    can_stop: bool
    can_reduce: bool
    description: str = ""


@dataclass(frozen=True)
class LedgerState:
    profile: Profile
    request_date: date
    horizon_end: date
    flows: tuple[Flow, ...]
    flexible: tuple[FlexibleExpense, ...]
    notes: tuple[str, ...]


class LedgerBuilder:
    """Builds a `LedgerState` for one (user, request_date) pair."""

    def __init__(
        self,
        data: Dataset,
        evidence: EvidenceSet | None = None,
        income_classification: dict[tuple, bool] | None = None,
    ) -> None:
        self.data = data
        self.rates = RateTable(data.rates)
        self.evidence = evidence or EvidenceSet.empty()
        # Maps a recurring credit series' key (see Series.key) to whether it
        # should be trusted to recur. Absent from the dict (a series the
        # classifier never saw, or a failed/skipped call) defaults to True -
        # fail-open, so a missing classification never silently drops income
        # the engine would otherwise have counted. See income_classifier.py.
        self.income_classification = income_classification or {}

    # -- entry point ---------------------------------------------------------
    def build(self, user_id: str, request_date: date, request_id: str = "") -> LedgerState:
        profile = self.data.profiles[user_id]
        horizon_end = request_date + timedelta(days=FORECAST_DAYS)
        events = self.data.events.get(user_id, [])
        amendments = self.evidence.for_request(user_id, request_id, request_date)

        usable = [e for e in events if self._counts_as_cash(e, amendments)]
        explicit = self._explicit_flows(usable, profile, request_date, horizon_end, amendments)
        series = self._build_series(usable, profile, request_date, amendments, user_id)
        projected = self._projected_flows(series, request_date, horizon_end, amendments)
        flexible = self._flexible_expenses(series, profile, amendments)

        policy = IncomePolicy.from_amendments(amendments)
        merged = policy.apply(explicit + projected)
        flows = tuple(sorted(merged, key=lambda f: (f.when, f.event_id)))
        notes = tuple(a.note for a in amendments if a.note)
        return LedgerState(profile, request_date, horizon_end, flows, tuple(flexible), notes)

    # -- rule: which rows move cash at all -----------------------------------
    def _counts_as_cash(self, event: Event, amendments: Sequence[Amendment]) -> bool:
        if event.status in DEAD_STATUSES:
            return False
        if event.direction == "non_cash":
            return False                      # unrealized investment value is not cash
        if event.event_type == "investment_valuation":
            return False
        for amendment in amendments:
            if amendment.kind == "event_cancelled" and amendment.related_event_id == event.event_id:
                return False
            if amendment.kind == "internal_transfer" and event.event_id in amendment.event_ids:
                return False                  # matching debit/credit between the user's own accounts
        return True

    # -- rule: resolve the amount of one row ---------------------------------
    def _resolve_amount(self, event: Event, amendments: Sequence[Amendment]) -> Decimal | None:
        for amendment in amendments:
            if amendment.kind == "event_amount" and amendment.related_event_id == event.event_id:
                if amendment.amount is not None:
                    return amendment.amount
        if event.amount is not None:
            return event.amount
        return None                           # blank and no evidence: cannot be invented

    def _resolve_date(self, event: Event, amendments: Sequence[Amendment]) -> date:
        for amendment in amendments:
            if amendment.kind == "event_delayed" and amendment.related_event_id == event.event_id:
                if amendment.effective_date is not None:
                    return amendment.effective_date
        return event.cash_date

    def _to_home(self, event: Event, amount: Decimal, when: date, home: str) -> Decimal | None:
        try:
            return self.rates.convert(amount, event.currency, home, when)
        except MissingRate:
            return None

    # -- explicit future rows ------------------------------------------------
    def _explicit_flows(
        self,
        events: Iterable[Event],
        profile: Profile,
        request_date: date,
        horizon_end: date,
        amendments: Sequence[Amendment],
    ) -> list[Flow]:
        flows: list[Flow] = []
        for event in events:
            amount = self._resolve_amount(event, amendments)
            if amount is None:
                continue
            when = self._resolve_date(event, amendments)
            converted = self._to_home(event, amount, when, profile.home_currency)
            if converted is None:
                continue
            if event.status == "pending":
                # Reserve pending debits. Never bank a pending credit.
                if event.is_credit:
                    continue
                when = max(when, request_date)
            elif event.status == "scheduled":
                pass
            else:                              # settled rows are already inside the balance
                if when <= request_date:
                    continue
            if not (request_date <= when <= horizon_end):
                continue
            signed = converted if event.is_credit else -converted
            flows.append(
                Flow(
                    when=when,
                    amount=signed,
                    event_id=event.event_id,
                    category=event.category,
                    source=EXPLICIT,
                    flexibility=event.flexibility,
                    minimum_allowed_amount=event.minimum_allowed_amount,
                    description=event.description,
                )
            )
        return flows

    # -- recurring series ----------------------------------------------------
    def _build_series(
        self,
        events: Iterable[Event],
        profile: Profile,
        request_date: date,
        amendments: Sequence[Amendment],
        user_id: str,
    ) -> list[Series]:
        groups: dict[tuple[str, str, str], list[Observation]] = {}
        for event in events:
            if not self._may_recur(event):
                continue
            amount = self._resolve_amount(event, amendments)
            if amount is None:
                continue
            when = self._resolve_date(event, amendments)
            # History up to the request, plus any confirmed future row, which is
            # the anchor the next cycle should be measured from.
            if event.status == "settled" and when >= request_date:
                continue
            if event.status == "pending":
                continue
            converted = self._to_home(event, amount, when, profile.home_currency)
            if converted is None:
                continue
            key = (event.category, event.direction, event.event_type)
            groups.setdefault(key, []).append(
                Observation(
                    when=when,
                    amount=converted,
                    event_id=event.event_id,
                    category=event.category,
                    flexibility=event.flexibility,
                    minimum_allowed_amount=event.minimum_allowed_amount,
                    description=event.description,
                )
            )
        series: list[Series] = []
        for (category, direction, event_type), observations in groups.items():
            series.extend(detect(observations, direction, category, event_type))
        series = [
            item
            for item in series
            if item.direction != "credit"
            or self.income_classification.get(item.key(user_id), True)
        ]
        series.extend(self._series_from_evidence(amendments, profile, request_date, series))
        return series

    def _may_recur(self, event: Event) -> bool:
        if event.is_credit:
            return (event.event_type, event.category) in RECURRING_CREDIT
        return event.event_type in RECURRING_DEBIT_TYPES

    def _series_from_evidence(
        self,
        amendments: Sequence[Amendment],
        profile: Profile,
        request_date: date,
        detected_series: Sequence[Series],
    ) -> list[Series]:
        """A message may announce income that has no history at all (a first salary).

        Many "your first salary will be X" messages turn out to *confirm* a
        salary that already has real settled history (the message reflects
        what the user believes, not what the data shows) rather than
        announcing a genuinely new one. Synthesizing a series for those too
        would double the salary every cycle - once from real history, once
        from the message - so this only fires when no already-detected
        monthly salary series lands on the same day-of-month. A message that
        merely restates known income is then correctly treated as
        informational: the real, already-detected series covers it.
        """
        existing_salary_days = {
            item.parameter
            for item in detected_series
            if item.category == "salary" and item.direction == "credit" and item.kind == "monthly"
        }
        created: list[Series] = []
        for amendment in amendments:
            if amendment.kind != "income_start":
                continue
            if amendment.amount is None or amendment.effective_date is None:
                continue
            amount = amendment.amount
            if amendment.currency and amendment.currency != profile.home_currency:
                continue                       # stated in another currency: not convertible safely
            start = amendment.effective_date
            if start.day in existing_salary_days:
                continue                       # confirms known income; do not double it
            # Anchor one cycle earlier so the announced date is itself projected.
            observation = Observation(
                when=_subtract_month(start),
                amount=amount,
                event_id="evidence:" + amendment.source,
                category="salary",
                flexibility="fixed",
                minimum_allowed_amount=None,
            )
            created.append(
                Series("monthly", start.day, "credit", "salary", "income", (observation,))
            )
        return created

    # -- projection ----------------------------------------------------------
    def _projected_flows(
        self,
        series: Sequence[Series],
        request_date: date,
        horizon_end: date,
        amendments: Sequence[Amendment],
    ) -> list[Flow]:
        flows: list[Flow] = []
        for item in series:
            estimator = INCOME_ESTIMATOR if item.direction == "credit" else EXPENSE_ESTIMATOR
            base = estimate_amount(item, estimator)
            anchor = item.anchor
            for when in item.occurrences(request_date, horizon_end):
                if item.direction == "credit":
                    amount = base
                else:
                    amount = self._apply_expense_amendments(item, base, when, amendments)
                if amount is None:
                    continue                   # evidence says this cycle does not happen
                if not (request_date <= when <= horizon_end):
                    continue
                signed = amount if item.direction == "credit" else -amount
                flows.append(
                    Flow(
                        when=when,
                        amount=signed,
                        event_id=item.event_id,
                        category=item.category,
                        source=PROJECTED,
                        flexibility=anchor.flexibility,
                        minimum_allowed_amount=anchor.minimum_allowed_amount,
                        description=anchor.description,
                    )
                )
        return flows

    def _is_recurring_income(self, event: Event) -> bool:
        return event.is_credit and (event.event_type, event.category) in RECURRING_CREDIT

    def _apply_expense_amendments(
        self, item: Series, amount: Decimal | None, when: date, amendments: Sequence[Amendment]
    ) -> Decimal | None:
        if amount is None or item.direction == "credit":
            return amount
        for amendment in sorted(amendments, key=lambda a: a.sent_at):
            if amendment.kind != "recurring_expense_change":
                continue
            if amendment.category and amendment.category != item.category:
                continue
            if amendment.effective_date is not None and when < amendment.effective_date:
                continue
            if amendment.amount is not None:
                amount = amendment.amount
            elif amendment.percent_change is not None:
                amount = money(amount * (Decimal(1) + amendment.percent_change / Decimal(100)))
        return amount

    # -- spending-change candidates ------------------------------------------
    def _flexible_expenses(
        self, series: Sequence[Series], profile: Profile, amendments: Sequence[Amendment]
    ) -> list[FlexibleExpense]:
        found: dict[str, FlexibleExpense] = {}
        for item in series:
            if item.direction == "credit":
                continue
            anchor = item.anchor
            if item.category in profile.protected_categories:
                continue                       # the user protected this category
            can_stop = (
                anchor.flexibility in {"stoppable", "reducible_or_stoppable"}
                and item.category in profile.stoppable_categories
            )
            can_reduce = (
                anchor.flexibility in {"reducible", "reducible_or_stoppable"}
                and item.category in profile.reducible_categories
            )
            if not (can_stop or can_reduce):
                continue
            estimator = EXPENSE_ESTIMATOR
            found[item.event_id] = FlexibleExpense(
                event_id=item.event_id,
                category=item.category,
                amount=estimate_amount(item, estimator),
                minimum_allowed_amount=anchor.minimum_allowed_amount,
                can_stop=can_stop,
                can_reduce=can_reduce,
                description=anchor.description,
            )
        return sorted(found.values(), key=lambda f: (-f.amount, f.event_id))
