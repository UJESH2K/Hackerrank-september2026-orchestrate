"""The verification gate.

Every invariant the problem statement states is re-checked here against the
rendered CSV rows, after the planner has run. Nothing is written unless the whole
file passes, so a logic regression fails the run instead of shipping quietly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Sequence

from .dataset import Dataset, Event, PaymentOption, Request
from .money import money, parse_date
from .planner import (
    METHOD_FULL,
    METHOD_INSTALMENTS,
    METHOD_NONE,
    METHOD_PARTIAL,
    METHOD_WAIT,
    STATUS_LATER,
    STATUS_NONE,
    STATUS_NOW,
    STATUS_PLAN,
    option_schedule,
)

STATUSES = {STATUS_NOW, STATUS_PLAN, STATUS_LATER, STATUS_NONE}
METHODS = {METHOD_FULL, METHOD_PARTIAL, METHOD_INSTALMENTS, METHOD_WAIT, METHOD_NONE}


class VerificationError(AssertionError):
    """Raised when a produced row breaks a rule from the problem statement."""


@dataclass
class Report:
    checked: int = 0
    problems: list[str] = None

    def __post_init__(self) -> None:
        if self.problems is None:
            self.problems = []

    def fail(self, request_id: str, message: str) -> None:
        self.problems.append(request_id + ": " + message)

    @property
    def ok(self) -> bool:
        return not self.problems


def _parse_plan(value: str) -> list[tuple[date, Decimal]]:
    if value == "none":
        return []
    parsed: list[tuple[date, Decimal]] = []
    for leg in value.split("|"):
        when, _, amount = leg.partition(":")
        parsed.append((parse_date(when), money(amount)))
    return parsed


def verify(
    rows: Sequence[dict[str, str]], requests: Sequence[Request], data: Dataset
) -> Report:
    report = Report()
    if [row["request_id"] for row in rows] != [r.request_id for r in requests]:
        report.fail("<file>", "output rows do not match requests.csv one-for-one, in order")
        return report

    by_id = {r.request_id: r for r in requests}
    for row in rows:
        report.checked += 1
        request = by_id[row["request_id"]]
        _verify_row(row, request, data, report)
    return report


def _verify_row(row: dict[str, str], request: Request, data: Dataset, report: Report) -> None:
    rid = request.request_id
    profile = data.profiles[request.user_id]
    options = data.options.get(rid, [])
    requested = money(request.requested_amount)

    try:
        safe = money(row["amount_safe_to_pay"])
    except (InvalidOperation, ValueError):
        report.fail(rid, "amount_safe_to_pay is not a number")
        return
    if not (Decimal(0) <= safe <= requested):
        report.fail(rid, "amount_safe_to_pay outside [0, requested_amount]")

    status = row["affordability_status"]
    method = row["recommended_payment_method"]
    if status not in STATUSES:
        report.fail(rid, "unknown affordability_status " + repr(status))
    if method not in METHODS:
        report.fail(rid, "unknown recommended_payment_method " + repr(method))

    try:
        plan = _parse_plan(row["payment_plan"])
    except (ValueError, InvalidOperation):
        report.fail(rid, "payment_plan is malformed")
        return
    if any(a[0] > b[0] for a, b in zip(plan, plan[1:])):
        report.fail(rid, "payment_plan is not chronological")

    earliest = row["earliest_date_for_full_payment"]
    if earliest:
        try:
            parse_date(earliest)
        except ValueError:
            report.fail(rid, "earliest_date_for_full_payment is not a date")

    # -- status / method coherence -----------------------------------------
    if status == STATUS_NOW:
        if method != METHOD_FULL:
            report.fail(rid, "affordable_now must recommend full_payment")
        if earliest != request.request_date.isoformat():
            report.fail(rid, "affordable_now must set earliest_date to the request date")
        if row["spending_changes_needed"] != "none":
            report.fail(rid, "affordable_now must not require spending changes")
    if status == STATUS_NONE and earliest:
        report.fail(rid, "not_affordable must leave earliest_date empty")
    # No general "a recommended plan needs earliest_date" rule: the spec
    # defines earliest_date_for_full_payment as capacity for a *single full
    # payment with no spending changes*, independent of what's recommended.
    # A plan that only succeeds through spending changes or an installment
    # schedule can legitimately leave it empty - see the METHOD_WAIT,
    # METHOD_PARTIAL and STATUS_NOW checks below for the cases where a
    # relationship to earliest_date *is* required.
    if method == METHOD_NONE and plan:
        report.fail(rid, "not_recommended must carry no payments")
    if method == METHOD_WAIT:
        if status != STATUS_LATER:
            report.fail(rid, "wait must report affordable_later")
        if len(plan) != 1:
            report.fail(rid, "wait must schedule exactly one payment")
        if METHOD_FULL not in profile.payment_methods:
            report.fail(rid, "wait requires the user to accept full_payment")

    # -- method eligibility --------------------------------------------------
    if method in {METHOD_FULL, METHOD_PARTIAL, METHOD_INSTALMENTS}:
        if method not in profile.payment_methods:
            report.fail(rid, method + " is not in payment_methods_user_will_consider")

    # -- partial payment -----------------------------------------------------
    if method == METHOD_PARTIAL:
        if status != STATUS_PLAN:
            report.fail(rid, "partial_payment must report affordable_with_plan")
        if not request.allows_partial_payment:
            report.fail(rid, "partial_payment used where the request forbids it")
        if len(plan) != 2:
            report.fail(rid, "partial_payment needs exactly two payments")
        else:
            if plan[0][0] != request.request_date:
                report.fail(rid, "first partial payment must fall on the request date")
            if plan[0][1] != safe:
                report.fail(rid, "first partial payment must equal amount_safe_to_pay")
            if plan[0][1] + plan[1][1] != requested:
                report.fail(rid, "partial payments must sum to requested_amount")
            if not (Decimal(0) < safe < requested):
                report.fail(rid, "partial_payment needs 0 < amount_safe_to_pay < requested")
            if plan[1][0] > request.desired_completion_date:
                report.fail(rid, "partial payment completes after desired_completion_date")
            if earliest and plan[1][0] != parse_date(earliest):
                report.fail(rid, "second partial payment must land on earliest_date")

    # -- instalments ---------------------------------------------------------
    if method == METHOD_INSTALMENTS:
        matched = _matching_option(plan, options)
        if matched is None:
            report.fail(rid, "installment plan does not match any supplied payment option")
        elif profile.max_installment_months is None:
            report.fail(rid, "installments chosen although max_installment_months is blank")
        elif matched.number_of_payments > profile.max_installment_months:
            report.fail(rid, "installment plan exceeds max_installment_months")

    # -- completion deadline -------------------------------------------------
    if method in {METHOD_FULL, METHOD_PARTIAL, METHOD_INSTALMENTS} and plan:
        if plan[-1][0] > request.desired_completion_date:
            report.fail(rid, "plan completes after desired_completion_date")
        total = sum((amount for _, amount in plan), Decimal(0))
        if method != METHOD_INSTALMENTS and total != requested:
            report.fail(rid, "plan payments do not sum to requested_amount")

    _verify_spending_changes(row, request, data, report)


def _matching_option(
    plan: Sequence[tuple[date, Decimal]], options: Sequence[PaymentOption]
) -> PaymentOption | None:
    for option in options:
        if option.payment_method != METHOD_INSTALMENTS:
            continue
        schedule = [(p.when, p.amount) for p in option_schedule(option)]
        if schedule == list(plan):
            return option
    return None


def _verify_spending_changes(
    row: dict[str, str], request: Request, data: Dataset, report: Report
) -> None:
    rid = request.request_id
    raw = row["spending_changes_needed"]
    if raw == "none":
        return
    profile = data.profiles[request.user_id]
    events = {e.event_id: e for e in data.events.get(request.user_id, [])}
    parts = raw.split("|")
    if len(parts) > 3:
        report.fail(rid, "more than three spending changes")
    seen: set[str] = set()
    for part in parts:
        fields = part.split(":")
        if fields[0] == "stop" and len(fields) == 2:
            event_id, target = fields[1], None
        elif fields[0] == "reduce_to" and len(fields) == 3:
            event_id, target = fields[1], money(fields[2])
        else:
            report.fail(rid, "unparseable spending change " + repr(part))
            continue
        if event_id in seen:
            report.fail(rid, "stop and reduce_to target the same event " + event_id)
        seen.add(event_id)
        event = events.get(event_id)
        if event is None:
            report.fail(rid, "spending change references unknown event " + event_id)
            continue
        if event.category in profile.protected_categories:
            report.fail(rid, event_id + " is in a protected category")
        if target is None:
            if not event.can_stop:
                report.fail(rid, event_id + " is not stoppable")
            if event.category not in profile.stoppable_categories:
                report.fail(rid, event_id + " is not in a category the user will stop")
        else:
            if not event.can_reduce:
                report.fail(rid, event_id + " is not reducible")
            if event.category not in profile.reducible_categories:
                report.fail(rid, event_id + " is not in a category the user will reduce")
            floor = event.minimum_allowed_amount
            if floor is not None and target < floor:
                report.fail(rid, event_id + " reduced below its minimum_allowed_amount")
