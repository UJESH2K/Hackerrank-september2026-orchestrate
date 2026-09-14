"""Confirmed salary vs. variable income - classified by Claude, not by variance.

`recurrence.detect()` groups every recurring credit series by (category,
direction, event_type) plus a day-of-month or fixed-interval pattern; it has
no way to tell a stable employer payroll from a gig-platform payout that
rotates through a different description every cycle ("Delivery platform
payout", "Weekly app earnings", "Task marketplace payout", ...) even though a
human reading the descriptions sees the difference immediately. An earlier
attempt to separate these with a coefficient-of-variation threshold on the
amounts was tested against dataset/sample_requests.csv and rejected: cutting
by variance alone helped the two or three gig-income examples but reduced the
aggregate score, because plenty of legitimate salaries also vary cycle to
cycle by rounding or small raises. This module asks Claude to read the actual
description history instead - the signal a numeric proxy cannot see - and only
ships if it measurably helps the same calibration set (see
evaluation/evaluate_samples.py before/after runs referenced in README.md).

Classification is done once per user over that user's FULL settled history
(not truncated to one request's cutoff), because "is this a confirmed salary"
is a fact about the income source, not about any one request's timing. The
result is reused by every request touching that user via the cache in
anthropic_client.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import config
from .anthropic_client import ModelUnavailable, complete_json
from .dataset import Dataset, Event
from .fx import MissingRate, RateTable
from .ledger import RECURRING_CREDIT
from .recurrence import Observation, Series, detect

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": ["confirmed", "variable"]},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["classification", "reasoning", "confidence"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You classify one recurring income stream for a personal finance
forecaster. You are given every historical payment in the series: its date,
amount, and the description printed on that specific transaction.

Decide between exactly two classes:

- "confirmed": a stable, contracted income the person can rely on recurring at
  a similar amount - typically an employer payroll, a fixed salary, or a
  pension. The description is usually the same each cycle (allowing for small
  wording variants of the same employer/payroll term), and the amount is
  stable or changes only via clearly identifiable raises.
- "variable": income that is real and has happened, but should not be assumed
  to repeat at a predictable amount - gig-platform payouts (delivery,
  rideshare, freelance task marketplaces), commission-heavy pay, or any series
  whose description names a different job, platform, or client each cycle
  even though the category is the same. A rotating description across
  otherwise-similar amounts is the strongest signal of gig/variable income -
  weigh it more than the amount's numeric spread.

Answer only from the transaction history given; do not assume anything about
amounts or dates not shown. Reply with confidence 0.0-1.0 for how sure you are."""


@dataclass(frozen=True)
class IncomeSeriesSummary:
    key: tuple
    category: str
    event_type: str
    occurrences: tuple[tuple[str, str, str], ...]  # (date, amount, description)


def _build_full_history_series(data: Dataset, user_id: str) -> list[Series]:
    """Every detectable credit series over a user's complete settled history."""
    profile = data.profiles[user_id]
    rates = RateTable(data.rates)
    groups: dict[tuple[str, str, str], list[Observation]] = {}
    for event in data.events.get(user_id, []):
        if not event.is_credit or (event.event_type, event.category) not in RECURRING_CREDIT:
            continue
        if event.status != "settled" or event.amount is None:
            continue
        try:
            converted = rates.convert(event.amount, event.currency, profile.home_currency, event.cash_date)
        except MissingRate:
            continue
        key = (event.category, event.direction, event.event_type)
        groups.setdefault(key, []).append(
            Observation(
                when=event.cash_date,
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
    return series


def _summarize(series: Series, user_id: str) -> IncomeSeriesSummary:
    return IncomeSeriesSummary(
        key=series.key(user_id),
        category=series.category,
        event_type=series.event_type,
        occurrences=tuple(
            (o.when.isoformat(), str(o.amount), o.description) for o in series.observations
        ),
    )


def _classify_one(summary: IncomeSeriesSummary, home_currency: str) -> tuple[bool, str, float] | None:
    history_lines = "\n".join(
        when + " | " + amount + " " + home_currency + " | " + (description or "(no description)")
        for when, amount, description in summary.occurrences
    )
    user_content = (
        "category: " + summary.category + "\n"
        "event_type: " + summary.event_type + "\n"
        "history (oldest first):\n" + history_lines
    )
    try:
        result = complete_json(
            system=SYSTEM_PROMPT,
            content=user_content,
            schema=CLASSIFICATION_SCHEMA,
            schema_name="income_stability",
            model=config.CLAUDE_TEXT_MODEL,
            max_tokens=1200,
        )
    except ModelUnavailable:
        return None
    classification = result.get("classification")
    if classification not in ("confirmed", "variable"):
        return None
    return (
        classification == "confirmed",
        str(result.get("reasoning", "")),
        float(result.get("confidence", 0.0) or 0.0),
    )


def classify_income_series(
    data: Dataset, user_ids: Sequence[str] | None = None
) -> dict[tuple, bool]:
    """Classify every recurring credit series across `user_ids` (default: all users).

    Returns a dict mapping a series' `Series.key(user_id)` to True ("confirmed",
    keep projecting it forward - today's default behavior) or False
    ("variable", stop projecting it forward). A series that fails to classify
    (no API key, a bad response, too short a history) is simply absent from
    the dict; `LedgerBuilder` treats an absent key as True, so this step can
    never cause the engine to invent a *smaller* forecast than before by
    silently dropping income it previously counted.
    """
    targets = user_ids if user_ids is not None else list(data.profiles)
    result: dict[tuple, bool] = {}
    for user_id in targets:
        profile = data.profiles.get(user_id)
        if profile is None:
            continue
        for series in _build_full_history_series(data, user_id):
            if len(series.observations) < 2:
                continue
            summary = _summarize(series, user_id)
            classified = _classify_one(summary, profile.home_currency)
            if classified is None:
                continue
            confirmed, _reasoning, _confidence = classified
            result[summary.key] = confirmed
    return result
