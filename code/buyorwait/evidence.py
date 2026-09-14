"""Structured evidence extracted from messages and images.

Messages and images are untrusted input. The perception layer is only allowed to
emit rows of the closed schema below; no free text from a message ever reaches
the forecaster, and no instruction inside a message can change a rule. Each row
is a claim about a number, a date or a status, and the ledger decides what to do
with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Iterable, Sequence

# The complete set of effects the engine understands. Anything the extractor
# cannot map to one of these is dropped rather than guessed at.
KINDS = (
    "income_amount_change",     # confirmed recurring pay changes to `amount` from `effective_date`
    "income_date_change",       # the confirmed pay date moves to `effective_date`
    "income_start",             # a first or new salary of `amount` begins on `effective_date`
    "income_end",               # no confirmed income after `effective_date`
    "recurring_expense_change", # a recurring bill in `category` changes by amount or percent
    "event_amount",             # the true amount of `related_event_id` (usually from an image)
    "event_cancelled",          # `related_event_id` will not happen
    "event_delayed",            # `related_event_id` now settles on `effective_date`
    "credit_not_confirmed",     # a payout, bonus, commission, prize or refund is not yet cash
    "unrealized_value",         # a valuation moved; no cash was generated
    "internal_transfer",        # a matching debit and credit between the user's own accounts
    "none",                     # informational only
)

# Effects that only restate rules the engine already applies. They are recorded
# for the explanation and the audit trail, but change no number.
INFORMATIONAL = {"credit_not_confirmed", "unrealized_value", "none"}


@dataclass(frozen=True)
class Amendment:
    kind: str
    source: str                      # message_id or image_id it came from
    user_id: str
    request_id: str = ""             # blank when the evidence is user-level
    sent_at: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
    amount: Decimal | None = None
    currency: str | None = None
    percent_change: Decimal | None = None
    effective_date: date | None = None
    category: str | None = None
    related_event_id: str = ""
    event_ids: tuple[str, ...] = ()
    scope: str = "ongoing"           # "ongoing" or "next_payment" for income changes
    note: str = ""
    confidence: float = 1.0

    def is_material(self) -> bool:
        return self.kind not in INFORMATIONAL


@dataclass(frozen=True)
class EvidenceSet:
    amendments: tuple[Amendment, ...] = ()

    @classmethod
    def empty(cls) -> "EvidenceSet":
        return cls(())

    @classmethod
    def of(cls, amendments: Iterable[Amendment]) -> "EvidenceSet":
        return cls(tuple(amendments))

    def for_request(self, user_id: str, request_id: str, request_date: date) -> list[Amendment]:
        """Evidence visible to this request: same user, addressed to it or to nobody,
        and sent on or before the request date."""
        selected = [
            a
            for a in self.amendments
            if a.user_id == user_id
            and a.request_id in ("", request_id)
            and a.sent_at.date() <= request_date
        ]
        selected.sort(key=lambda a: (a.sent_at, a.source))
        return _resolve_conflicts(selected)


def _resolve_conflicts(amendments: Sequence[Amendment]) -> list[Amendment]:
    """Conflict order from the problem statement.

    1. an explicit cancellation, settlement or amendment wins;
    2. otherwise the newer record from the same source wins;
    3. otherwise a settled figure beats a forecast;
    4. otherwise keep the financially safer reading.

    Concretely: for each (kind, target) we keep the latest row, except that a
    cancellation always outranks a later amendment of the same event.
    """
    cancelled = {a.related_event_id for a in amendments if a.kind == "event_cancelled"}
    # Successive payroll updates stack rather than replace: "salary resumes on X"
    # followed by "next salary is reduced" are two separate facts about two
    # different cycles, and the income policy applies them in order.
    stacking = {"income_amount_change", "income_start"}
    kept: list[Amendment] = []
    latest: dict[tuple[str, str], Amendment] = {}
    for amendment in amendments:
        target = amendment.related_event_id or amendment.category or ""
        if amendment.kind in {"event_amount", "event_delayed"} and amendment.related_event_id in cancelled:
            continue
        if amendment.kind in stacking:
            kept.append(amendment)
        else:
            latest[(amendment.kind, target)] = amendment
    kept.extend(latest.values())
    return sorted(kept, key=lambda a: (a.sent_at, a.source))


@dataclass
class ExtractionStats:
    messages_seen: int = 0
    messages_material: int = 0
    images_seen: int = 0
    images_resolved: int = 0
    failures: list[str] = field(default_factory=list)
