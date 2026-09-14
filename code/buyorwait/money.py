"""Exact decimal arithmetic and the output formatting rules."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP

CENT = Decimal("0.01")
ZERO = Decimal("0")


def money(value: Decimal | str | int | float) -> Decimal:
    """Round to two decimal places, the precision every amount in the dataset uses."""
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def floor_money(value: Decimal | str | int | float) -> Decimal:
    """Round towards zero. Used for safe amounts so rounding never overstates capacity."""
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_DOWN)


def parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def parse_optional_date(value: str | None) -> date | None:
    return parse_date(value) if value and value.strip() else None


def parse_optional_decimal(value: str | None) -> Decimal | None:
    if value is None or not value.strip():
        return None
    return Decimal(value.strip())


def split_list(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    return frozenset(part.strip() for part in value.split("|") if part.strip())


def format_amount(value: Decimal) -> str:
    """Trim trailing zeros the way sample_requests.csv does: 25256, 941.60 -> 941.6."""
    quantized = money(value)
    text = format(quantized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def format_plan_amount(value: Decimal) -> str:
    """Plan legs and reduce_to targets: whole numbers plain, everything else 2 dp.

    Matches dataset/sample_requests.csv exactly, e.g. `2024-09-04:28820` and
    `2026-01-03:620.40` appear in the same column.
    """
    quantized = money(value)
    if quantized == quantized.to_integral_value():
        return str(int(quantized))
    return format(quantized, "f")


def humanize_amount(value: Decimal) -> str:
    """Thousands-separated form used inside decision explanations."""
    quantized = money(value)
    whole = quantized.to_integral_value()
    if quantized == whole:
        return f"{int(whole):,}"
    return f"{quantized:,.2f}"


def humanize_date(value: date) -> str:
    return f"{value.day} {value.strftime('%B')} {value.year}"
