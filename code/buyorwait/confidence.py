"""Confidence layers for `amount_safe_to_pay`.

Four independent checks, each looking for a different failure mode. None of
them changes what gets written to `output.csv` - the output contract is
fixed by the spec and a number should never move just because a check felt
uneasy about it. What they *do* produce is a `ConfidenceReport`: a machine-
checkable explanation of *why* a given number might be fragile, so a human
(or a future automated gate) can decide what to do about it.

The four layers, in the order they run:

1. **Shadow simulation** - re-derives the same forecast with a completely
   separate, deliberately naive day-by-day loop instead of `Forecast`'s
   flow-indexed walk. If the two disagree, that is a real *software* bug in
   one of the two implementations, not estimator uncertainty - this layer
   exists to catch that class of error, not to catch disagreement with the
   hidden ground truth.
2. **Estimator sensitivity** - recomputes `amount_safe_to_pay` under every
   estimator combination `evaluate_samples.py` has ever swept (mean, mean3,
   median, last, max for expenses; last, median, mean, max for income) and
   reports the spread as a fraction of the user's minimum balance. A number
   that barely moves across estimator choices is a genuinely stable
   forecast; one that swings widely means the "right" answer is a matter of
   which policy you pick, not a fact the ledger determined.
3. **Confirmed-only floor** - recomputes the trough using *only* settled and
   already-scheduled/pending cash movements, with every projected
   (recurring, non-confirmed) flow stripped out. The gap between this floor
   and the full forecast measures how much of the reported number rests on
   projection rather than on record.
4. **Categorical stability** - reruns the *planner*, not just the forecast,
   under the same estimator grid as layer 2, and checks whether
   `affordability_status` is unanimous across all of them. A request whose
   amount barely moves (layer 2) can still flip status if it sits near the
   `affordable_now`/`affordable_with_plan` boundary; this catches that case
   directly instead of guessing at it from a balance heuristic.

Calibration note: this module was built and cross-checked against
`dataset/sample_requests.csv` specifically to see whether a low-confidence
flag predicts an actual mismatch (see
`code/evaluation/confidence_audit.py`). It measurably does - see that
script's README-quoted output - which is why it ships as a diagnostic layer
rather than a discarded experiment like the income classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from itertools import product

from .dataset import Dataset, Request
from .forecast import Forecast
from .ledger import EXPLICIT, LedgerBuilder, LedgerState
from .money import ZERO, money
from .planner import plan

# The estimator combinations actually tried against the calibration set
# (config.py picked mean3/median as the shipped default). Sweeping all of
# them again here is what makes "sensitivity" a measured number rather than
# a guess.
SENSITIVITY_EXPENSE_ESTIMATORS = ("mean", "mean3", "median", "last", "max")
SENSITIVITY_INCOME_ESTIMATORS = ("last", "median", "mean", "max")

HIGH_CONFIDENCE = "high"
MEDIUM_CONFIDENCE = "medium"
LOW_CONFIDENCE = "low"


@dataclass(frozen=True)
class ConfidenceReport:
    request_id: str
    reported_amount: Decimal
    shadow_amount: Decimal
    shadow_agrees: bool
    sensitivity_low: Decimal
    sensitivity_high: Decimal
    sensitivity_spread_pct: Decimal      # spread / minimum_balance, as a percent
    confirmed_only_floor: Decimal
    projection_reliance_pct: Decimal     # gap vs. confirmed-only floor, as % of minimum balance
    status_unanimous: bool
    status_variants: frozenset[str]
    tier: str
    reasons: tuple[str, ...]


def _shadow_min_balance(state: LedgerState) -> Decimal:
    """A deliberately different implementation of the same 90-day walk.

    `Forecast` only evaluates the balance at dates where a flow occurs
    (correct, since it is flat in between, but a bug there would not be
    caught by re-running the same code). This instead steps one calendar day
    at a time and accumulates every flow whose date has arrived, so a
    disagreement with `Forecast.min_balance()` points at a real defect in
    one of the two, not at an estimator choice - both use identical inputs.
    """
    by_date: dict[date, Decimal] = {}
    for flow in state.flows:
        by_date[flow.when] = by_date.get(flow.when, ZERO) + flow.amount
    balance = money(state.profile.balance)
    lowest = balance
    cursor = state.request_date
    while cursor <= state.horizon_end:
        balance += by_date.get(cursor, ZERO)
        lowest = min(lowest, balance)
        cursor += timedelta(days=1)
    return lowest


def _amount_safe_for_state(state: LedgerState, requested_amount: Decimal) -> Decimal:
    forecast = Forecast(state)
    return forecast.amount_safe_today(requested_amount)


def _rebuild_with_estimators(
    builder: LedgerBuilder, request: Request, expense: str, income: str
) -> LedgerState:
    """Rebuilds the ledger with a specific estimator pair, bypassing config.py.

    Reuses `LedgerBuilder` end to end (same conflict resolution, same
    recurrence detection) and only substitutes which estimator function
    turns a series' history into a projected amount - isolating exactly the
    one axis config.py currently calibrates.
    """
    import buyorwait.ledger as ledger_module

    original_expense = ledger_module.EXPENSE_ESTIMATOR
    original_income = ledger_module.INCOME_ESTIMATOR
    ledger_module.EXPENSE_ESTIMATOR = expense
    ledger_module.INCOME_ESTIMATOR = income
    try:
        return builder.build(request.user_id, request.request_date, request.request_id)
    finally:
        ledger_module.EXPENSE_ESTIMATOR = original_expense
        ledger_module.INCOME_ESTIMATOR = original_income


def _confirmed_only_state(state: LedgerState) -> LedgerState:
    """The same ledger with every projected (non-confirmed) flow stripped out."""
    confirmed_flows = tuple(flow for flow in state.flows if flow.source == EXPLICIT)
    return replace(state, flows=confirmed_flows)


def evaluate(
    data: Dataset,
    request: Request,
    reported_amount: Decimal,
    builder: LedgerBuilder | None = None,
) -> ConfidenceReport:
    """Runs all four layers for one request and returns a combined report."""
    profile = data.profiles[request.user_id]
    builder = builder or LedgerBuilder(data)
    state = builder.build(request.user_id, request.request_date, request.request_id)
    requested = money(request.requested_amount)
    reasons: list[str] = []

    # Layer 1: shadow simulation.
    forecast = Forecast(state)
    reported_trough = forecast.min_balance()
    shadow_trough = _shadow_min_balance(state)
    shadow_agrees = shadow_trough == reported_trough
    shadow_amount = max(ZERO, min(requested, money(shadow_trough - profile.minimum_balance)))
    if not shadow_agrees:
        reasons.append(
            "shadow simulation disagrees with the reported trough by "
            + str(money(shadow_trough - reported_trough))
            + " - investigate as a software bug, not estimator noise"
        )

    minimum_balance = money(profile.minimum_balance) or Decimal("1")
    options = data.options.get(request.request_id, [])

    # Layers 2 and 4 share one sweep: for each estimator pair, compute both
    # the forecast amount (layer 2) and the full plan's affordability_status
    # (layer 4), since the same rebuilt ledger drives both.
    amounts: list[Decimal] = []
    statuses: set[str] = set()
    for expense_estimator, income_estimator in product(
        SENSITIVITY_EXPENSE_ESTIMATORS, SENSITIVITY_INCOME_ESTIMATORS
    ):
        variant_state = _rebuild_with_estimators(builder, request, expense_estimator, income_estimator)
        amounts.append(_amount_safe_for_state(variant_state, requested))
        decision = plan(request, profile, variant_state, options)
        statuses.add(decision.status)
    sensitivity_low = min(amounts)
    sensitivity_high = max(amounts)
    sensitivity_spread_pct = money(
        (sensitivity_high - sensitivity_low) / minimum_balance * 100
    )
    if sensitivity_spread_pct > 15:
        reasons.append(
            "amount_safe_to_pay swings " + str(sensitivity_spread_pct)
            + "% of the minimum balance depending on the estimator chosen"
            + " (range " + str(sensitivity_low) + "-" + str(sensitivity_high) + ")"
        )

    status_unanimous = len(statuses) == 1
    if not status_unanimous:
        reasons.append(
            "affordability_status is not unanimous across estimator choices: "
            + ", ".join(sorted(statuses))
        )

    # Layer 3: confirmed-only floor, normalized against the minimum balance
    # (not the reported amount) so a small reported amount does not blow the
    # percentage up out of proportion.
    confirmed_state = _confirmed_only_state(state)
    confirmed_floor = _amount_safe_for_state(confirmed_state, requested)
    projection_reliance_pct = money(
        abs(reported_amount - confirmed_floor) / minimum_balance * 100
    )
    if projection_reliance_pct > 25:
        reasons.append(
            "amount_safe_to_pay differs from the confirmed-only floor by "
            + str(projection_reliance_pct)
            + "% of the minimum balance - most of the number rests on projected, "
            "not yet confirmed, cash flow"
        )

    if not shadow_agrees:
        tier = LOW_CONFIDENCE     # a software disagreement always overrides everything else
    elif not status_unanimous:
        tier = LOW_CONFIDENCE
    elif sensitivity_spread_pct > 15 or projection_reliance_pct > 25:
        tier = MEDIUM_CONFIDENCE
    else:
        tier = HIGH_CONFIDENCE

    return ConfidenceReport(
        request_id=request.request_id,
        reported_amount=reported_amount,
        shadow_amount=shadow_amount,
        shadow_agrees=shadow_agrees,
        sensitivity_low=sensitivity_low,
        sensitivity_high=sensitivity_high,
        sensitivity_spread_pct=sensitivity_spread_pct,
        confirmed_only_floor=confirmed_floor,
        projection_reliance_pct=projection_reliance_pct,
        status_unanimous=status_unanimous,
        status_variants=frozenset(statuses),
        tier=tier,
        reasons=tuple(reasons),
    )
