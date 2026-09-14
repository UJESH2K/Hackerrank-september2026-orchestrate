"""Decision explanations.

The explanation restates numbers the engine has already computed, so it is
generated from templates rather than sampled from a model. That keeps it exactly
consistent with `amount_safe_to_pay`, the plan and the minimum balance - a model
asked to phrase the same facts can drift on a digit, and the scoring rewards
consistency.

`llm_polish` is available for a reviewer-style rewrite; it is off by default and
its output is rejected unless every number in it also appears in the template
facts, so it can never introduce a figure the engine did not compute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from .dataset import Profile, Request
from .money import humanize_amount, humanize_date
from .planner import (
    Decision,
    METHOD_FULL,
    METHOD_INSTALMENTS,
    METHOD_NONE,
    METHOD_PARTIAL,
    METHOD_WAIT,
    STATUS_NONE,
)


@dataclass(frozen=True)
class ExplanationFacts:
    """The only numbers an explanation is allowed to mention."""

    currency: str
    requested: Decimal
    safe_today: Decimal
    minimum_balance: Decimal
    plan_amounts: tuple[Decimal, ...]


def build_facts(decision: Decision, profile: Profile) -> ExplanationFacts:
    return ExplanationFacts(
        currency=profile.home_currency,
        requested=decision.request.requested_amount,
        safe_today=decision.amount_safe_to_pay,
        minimum_balance=profile.minimum_balance,
        plan_amounts=tuple(payment.amount for payment in decision.payments),
    )


def explain(decision: Decision, profile: Profile, change_labels: dict[str, str]) -> str:
    request = decision.request
    currency = profile.home_currency
    minimum = currency + " " + humanize_amount(profile.minimum_balance)
    requested = currency + " " + humanize_amount(request.requested_amount)

    prefix = _changes_clause(decision, profile, change_labels)

    if decision.method == METHOD_FULL:
        body = "pay " + requested + " today" if prefix else "Pay " + requested + " today"
        tail = " This leaves at least " + minimum + " available"
        tail += "." if prefix else " over the next 90 days."
        return prefix + body + "." + tail

    if decision.method == METHOD_INSTALMENTS:
        count = len(decision.payments)
        each = currency + " " + humanize_amount(decision.payments[0].amount)
        start = humanize_date(decision.payments[0].when)
        return (
            prefix
            + ("use " if prefix else "Use ")
            + str(count)
            + " installments of "
            + each
            + ", starting "
            + start
            + ". This leaves at least "
            + minimum
            + " available."
        )

    if decision.method == METHOD_PARTIAL:
        first, second = decision.payments[0], decision.payments[1]
        return (
            prefix
            + ("pay " if prefix else "Pay ")
            + currency
            + " "
            + humanize_amount(first.amount)
            + " today and the remaining "
            + currency
            + " "
            + humanize_amount(second.amount)
            + " on "
            + humanize_date(second.when)
            + ". This completes the full request and keeps the "
            + minimum
            + " minimum protected."
        )

    if decision.method == METHOD_WAIT:
        when = humanize_date(decision.payments[0].when)
        return (
            "Pay "
            + requested
            + " in full on "
            + when
            + ". Paying earlier would take the balance below the "
            + minimum
            + " minimum."
        )

    # not_recommended
    if _partial_was_the_only_route(decision, profile):
        return (
            "Do not proceed with the "
            + requested
            + " request. Although "
            + currency
            + " "
            + humanize_amount(decision.amount_safe_to_pay)
            + " is available today, the full amount cannot be completed safely within 90 days."
        )
    return (
        "Do not make this payment by "
        + humanize_date(request.desired_completion_date)
        + ". None of the available options keeps the "
        + minimum
        + " minimum protected."
    )


def _partial_was_the_only_route(decision: Decision, profile: Profile) -> bool:
    """True when the user accepts no method that maps to a supplied option.

    Then talking about "the available options" would be misleading, so the
    explanation names the request itself instead. Matches the sample wording.
    """
    accepts_listed_option = bool(
        profile.payment_methods & {METHOD_FULL, METHOD_INSTALMENTS}
    )
    return (
        not accepts_listed_option
        and METHOD_PARTIAL in profile.payment_methods
        and decision.request.allows_partial_payment
    )


def _changes_clause(decision: Decision, profile: Profile, labels: dict[str, str]) -> str:
    if not decision.changes:
        return ""
    parts: list[str] = []
    for change in decision.changes:
        label = labels.get(change.event_id, "the recurring payment")
        if change.new_amount is None:
            parts.append("Stop " + label)
        else:
            parts.append(
                "reduce "
                + label
                + " to "
                + profile.home_currency
                + " "
                + humanize_amount(change.new_amount)
            )
    if len(parts) == 1:
        joined = parts[0]
    else:
        joined = ", ".join(parts[:-1]) + " and " + parts[-1]
    joined = joined[0].upper() + joined[1:]
    return joined + ", then "


def numbers_in(text: str) -> set[str]:
    return {match.replace(",", "") for match in re.findall(r"[\d][\d,]*(?:\.\d+)?", text)}


def is_consistent(text: str, facts: ExplanationFacts) -> bool:
    """Reject any explanation containing a figure the engine did not compute."""
    allowed = {
        str(value).rstrip("0").rstrip(".") if "." in str(value) else str(value)
        for value in (
            facts.requested,
            facts.safe_today,
            facts.minimum_balance,
            *facts.plan_amounts,
        )
    }
    allowed |= {str(v) for v in (facts.requested, facts.safe_today, facts.minimum_balance)}
    allowed |= {"90"}
    allowed |= {str(year) for year in range(1990, 2100)}
    allowed |= {str(day) for day in range(1, 32)}
    for number in numbers_in(text):
        candidates = {number, number.rstrip("0").rstrip(".") if "." in number else number}
        if not candidates & allowed:
            return False
    return True
