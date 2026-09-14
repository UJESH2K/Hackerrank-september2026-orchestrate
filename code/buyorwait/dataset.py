"""Typed loaders for dataset/. Every file is schema-checked before use."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from .config import DATASET_DIR
from .money import (
    money,
    parse_date,
    parse_optional_date,
    parse_optional_decimal,
    split_list,
)

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "requests.csv": {
        "request_id", "user_id", "request_date", "request_type", "requested_amount",
        "desired_completion_date", "allows_partial_payment", "request_text",
    },
    "financial_profiles.csv": {
        "user_id", "home_currency", "current_available_balance", "minimum_balance_to_keep",
        "financial_priorities", "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider", "max_installment_months",
    },
    "financial_events.csv": {
        "event_id", "user_id", "event_type", "description", "category", "direction",
        "amount", "currency", "event_date", "settlement_date", "status",
        "linked_event_id", "flexibility", "minimum_allowed_amount",
    },
    "request_payment_options.csv": {
        "payment_option_id", "request_id", "payment_method", "payment_amount",
        "number_of_payments", "first_payment_date", "payment_frequency_days",
        "financing_fee", "total_payable_amount",
    },
    "exchange_rates.csv": {"rate_date", "from_currency", "to_currency", "rate"},
    "messages.csv": {
        "message_id", "user_id", "request_id", "related_event_id", "sent_at",
        "source_type", "message_text",
    },
    "images.csv": {"image_id", "user_id", "request_id", "related_event_id"},
}

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

# Statuses that never move cash and must stay out of the forecast.
DEAD_STATUSES = frozenset({"cancelled", "failed", "reversed", "duplicate", "unrealized"})
FLEXIBLE_STOPPABLE = frozenset({"stoppable", "reducible_or_stoppable"})
FLEXIBLE_REDUCIBLE = frozenset({"reducible", "reducible_or_stoppable"})


@dataclass(frozen=True)
class Profile:
    user_id: str
    home_currency: str
    balance: Decimal
    minimum_balance: Decimal
    priorities: frozenset[str]
    protected_categories: frozenset[str]
    reducible_categories: frozenset[str]
    stoppable_categories: frozenset[str]
    payment_methods: frozenset[str]
    max_installment_months: int | None


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Decimal | None
    currency: str
    event_date: date
    settlement_date: date | None
    status: str
    linked_event_id: str
    flexibility: str
    minimum_allowed_amount: Decimal | None

    @property
    def cash_date(self) -> date:
        return self.settlement_date or self.event_date

    @property
    def is_credit(self) -> bool:
        return self.direction == "credit"

    @property
    def can_stop(self) -> bool:
        return self.flexibility in FLEXIBLE_STOPPABLE

    @property
    def can_reduce(self) -> bool:
        return self.flexibility in FLEXIBLE_REDUCIBLE


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str
    expected: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal

    @property
    def sort_key(self) -> tuple[int, str]:
        digits = "".join(ch for ch in self.payment_option_id if ch.isdigit())
        return (int(digits) if digits else 0, self.payment_option_id)


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: str
    related_event_id: str
    sent_at: datetime
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str

    def path(self, dataset_dir: Path = DATASET_DIR) -> Path:
        return dataset_dir / "media" / "images" / (self.image_id + ".png")


@dataclass(frozen=True)
class Dataset:
    requests: list[Request]
    profiles: dict[str, Profile]
    events: dict[str, list[Event]]
    options: dict[str, list[PaymentOption]]
    rates: dict[tuple[date, str, str], Decimal]
    messages: list[Message]
    images: list[ImageRef]

    def messages_for(self, user_id: str, request_id: str) -> list[Message]:
        """User-level messages plus messages addressed to this specific request."""
        return [
            m for m in self.messages
            if m.user_id == user_id and m.request_id in ("", request_id)
        ]

    def images_for(self, user_id: str) -> list[ImageRef]:
        return [i for i in self.images if i.user_id == user_id]


def _read(name: str, dataset_dir: Path, schema_key: str | None = None) -> list[dict[str, str]]:
    path = dataset_dir / name
    key = schema_key or name
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.get(key, set()) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(name + " is missing columns: " + repr(sorted(missing)))
        return [{k: (v if v is not None else "") for k, v in row.items()} for row in reader]


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError("Unrecognised boolean: " + repr(value))


def load_requests(name: str = "requests.csv", dataset_dir: Path = DATASET_DIR) -> list[Request]:
    """Load requests.csv or sample_requests.csv; the latter also carries solved columns."""
    rows = _read(name, dataset_dir, schema_key="requests.csv")
    seen: set[str] = set()
    requests: list[Request] = []
    for line, row in enumerate(rows, start=2):
        request_id = row["request_id"].strip()
        if not request_id:
            raise ValueError(name + " row " + str(line) + ": blank request_id")
        if request_id in seen:
            raise ValueError(name + " row " + str(line) + ": duplicate " + request_id)
        seen.add(request_id)
        expected = {key: row[key] for key in OUTPUT_COLUMNS[1:] if key in row}
        requests.append(
            Request(
                request_id=request_id,
                user_id=row["user_id"].strip(),
                request_date=parse_date(row["request_date"]),
                request_type=row["request_type"].strip(),
                requested_amount=money(row["requested_amount"]),
                desired_completion_date=parse_date(row["desired_completion_date"]),
                allows_partial_payment=_parse_bool(row["allows_partial_payment"]),
                request_text=row["request_text"],
                expected=expected,
            )
        )
    return requests


def load(request_file: str = "requests.csv", dataset_dir: Path = DATASET_DIR) -> Dataset:
    profiles: dict[str, Profile] = {}
    for row in _read("financial_profiles.csv", dataset_dir):
        months = row["max_installment_months"].strip()
        profiles[row["user_id"].strip()] = Profile(
            user_id=row["user_id"].strip(),
            home_currency=row["home_currency"].strip(),
            balance=money(row["current_available_balance"]),
            minimum_balance=money(row["minimum_balance_to_keep"]),
            priorities=split_list(row["financial_priorities"]),
            protected_categories=split_list(row["expense_categories_to_protect"]),
            reducible_categories=split_list(row["expense_categories_user_is_willing_to_reduce"]),
            stoppable_categories=split_list(row["expense_categories_user_is_willing_to_stop"]),
            payment_methods=split_list(row["payment_methods_user_will_consider"]),
            max_installment_months=int(months) if months else None,
        )

    events: dict[str, list[Event]] = {}
    for row in _read("financial_events.csv", dataset_dir):
        event = Event(
            event_id=row["event_id"].strip(),
            user_id=row["user_id"].strip(),
            event_type=row["event_type"].strip(),
            description=row["description"].strip(),
            category=row["category"].strip(),
            direction=row["direction"].strip(),
            amount=parse_optional_decimal(row["amount"]),
            currency=row["currency"].strip(),
            event_date=parse_date(row["event_date"]),
            settlement_date=parse_optional_date(row["settlement_date"]),
            status=row["status"].strip(),
            linked_event_id=row["linked_event_id"].strip(),
            flexibility=row["flexibility"].strip(),
            minimum_allowed_amount=parse_optional_decimal(row["minimum_allowed_amount"]),
        )
        events.setdefault(event.user_id, []).append(event)
    for bucket in events.values():
        bucket.sort(key=lambda e: (e.cash_date, e.event_id))

    options: dict[str, list[PaymentOption]] = {}
    for row in _read("request_payment_options.csv", dataset_dir):
        frequency = row["payment_frequency_days"].strip()
        option = PaymentOption(
            payment_option_id=row["payment_option_id"].strip(),
            request_id=row["request_id"].strip(),
            payment_method=row["payment_method"].strip(),
            payment_amount=money(row["payment_amount"]),
            number_of_payments=int(row["number_of_payments"]),
            first_payment_date=parse_date(row["first_payment_date"]),
            payment_frequency_days=int(frequency) if frequency else None,
            financing_fee=money(row["financing_fee"] or 0),
            total_payable_amount=money(row["total_payable_amount"]),
        )
        options.setdefault(option.request_id, []).append(option)
    for bucket in options.values():
        bucket.sort(key=lambda o: o.sort_key)

    rates = {
        (parse_date(row["rate_date"]), row["from_currency"].strip(), row["to_currency"].strip()):
            Decimal(row["rate"])
        for row in _read("exchange_rates.csv", dataset_dir)
    }

    messages = [
        Message(
            message_id=row["message_id"].strip(),
            user_id=row["user_id"].strip(),
            request_id=row["request_id"].strip(),
            related_event_id=row["related_event_id"].strip(),
            sent_at=datetime.fromisoformat(row["sent_at"].replace("Z", "+00:00")),
            source_type=row["source_type"].strip(),
            message_text=row["message_text"],
        )
        for row in _read("messages.csv", dataset_dir)
    ]
    messages.sort(key=lambda m: (m.sent_at, m.message_id))

    images = [
        ImageRef(
            image_id=row["image_id"].strip(),
            user_id=row["user_id"].strip(),
            request_id=row["request_id"].strip(),
            related_event_id=row["related_event_id"].strip(),
        )
        for row in _read("images.csv", dataset_dir)
    ]

    return Dataset(
        requests=load_requests(request_file, dataset_dir),
        profiles=profiles,
        events=events,
        options=options,
        rates=rates,
        messages=messages,
        images=images,
    )
