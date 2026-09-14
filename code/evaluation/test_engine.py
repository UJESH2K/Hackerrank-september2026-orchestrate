"""Unit tests for the deterministic core. No network access, no dataset files.

    python code/evaluation/test_engine.py
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buyorwait.dataset import Event, PaymentOption, Profile, Request
from buyorwait.evidence import Amendment, EvidenceSet
from buyorwait.forecast import Forecast, Payment, SpendingChange
from buyorwait.fx import MissingRate, RateTable
from buyorwait.ledger import LedgerBuilder
from buyorwait.dataset import Dataset
from buyorwait.money import format_amount, format_plan_amount, money
from buyorwait.planner import (
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
    plan,
)
from buyorwait.recurrence import Observation, detect, estimate_amount


def _profile(**overrides) -> Profile:
    base = dict(
        user_id="user_test",
        home_currency="USD",
        balance=money("1000"),
        minimum_balance=money("200"),
        priorities=frozenset(),
        protected_categories=frozenset({"rent"}),
        reducible_categories=frozenset({"dining"}),
        stoppable_categories=frozenset({"streaming"}),
        payment_methods=frozenset({"full_payment", "partial_payment", "installments"}),
        max_installment_months=12,
    )
    base.update(overrides)
    return Profile(**base)


def _event(**overrides) -> Event:
    base = dict(
        event_id="event_1",
        user_id="user_test",
        event_type="expense",
        description="Test expense",
        category="groceries",
        direction="debit",
        amount=Decimal("30"),
        currency="USD",
        event_date=date(2026, 1, 2),
        settlement_date=date(2026, 1, 2),
        status="settled",
        linked_event_id="",
        flexibility="fixed",
        minimum_allowed_amount=None,
    )
    base.update(overrides)
    return Event(**base)


def _dataset(profile: Profile, events: list[Event], options=None, rates=None) -> Dataset:
    return Dataset(
        requests=[],
        profiles={profile.user_id: profile},
        events={profile.user_id: events},
        options=options or {},
        rates=rates or {},
        messages=[],
        images=[],
    )


class MoneyFormattingTests(unittest.TestCase):
    def test_format_amount_trims_trailing_zeros(self) -> None:
        self.assertEqual(format_amount(Decimal("25256.00")), "25256")
        self.assertEqual(format_amount(Decimal("941.60")), "941.6")
        self.assertEqual(format_amount(Decimal("0")), "0")

    def test_format_plan_amount_keeps_two_decimals_unless_whole(self) -> None:
        self.assertEqual(format_plan_amount(Decimal("620.40")), "620.40")
        self.assertEqual(format_plan_amount(Decimal("28820")), "28820")
        self.assertEqual(format_plan_amount(Decimal("23.5")), "23.50")


class FxTests(unittest.TestCase):
    def test_direct_rate(self) -> None:
        table = RateTable({(date(2024, 1, 1), "USD", "EUR"): Decimal("0.9")})
        self.assertEqual(table.convert(Decimal("100"), "USD", "EUR", date(2024, 1, 1)), money("90"))

    def test_inverse_rate(self) -> None:
        table = RateTable({(date(2024, 1, 1), "EUR", "USD"): Decimal("1.1")})
        result = table.convert(Decimal("110"), "USD", "EUR", date(2024, 1, 1))
        self.assertEqual(result, money(Decimal("110") / Decimal("1.1")))

    def test_missing_rate_raises(self) -> None:
        table = RateTable({})
        with self.assertRaises(MissingRate):
            table.convert(Decimal("10"), "USD", "ZAR", date(2024, 1, 1))

    def test_same_currency_is_a_no_op(self) -> None:
        table = RateTable({})
        self.assertEqual(table.convert(Decimal("10"), "USD", "USD", date(2024, 1, 1)), money("10"))


class RecurrenceTests(unittest.TestCase):
    def test_monthly_series_detected_from_day_of_month(self) -> None:
        observations = [
            Observation(date(2025, 1, 15), Decimal("100"), "e1", "salary", "fixed", None),
            Observation(date(2025, 2, 15), Decimal("100"), "e2", "salary", "fixed", None),
            Observation(date(2025, 3, 15), Decimal("100"), "e3", "salary", "fixed", None),
        ]
        series = detect(observations, "credit", "salary", "income")
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].kind, "monthly")
        self.assertEqual(series[0].parameter, 15)

    def test_fixed_interval_series_detected(self) -> None:
        observations = [
            Observation(date(2025, 1, 1), Decimal("10"), "e1", "transport", "fixed", None),
            Observation(date(2025, 1, 8), Decimal("11"), "e2", "transport", "fixed", None),
            Observation(date(2025, 1, 15), Decimal("9"), "e3", "transport", "fixed", None),
            Observation(date(2025, 1, 22), Decimal("10"), "e4", "transport", "fixed", None),
        ]
        series = detect(observations, "debit", "transport", "expense")
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].kind, "fixed")
        self.assertEqual(series[0].parameter, 7)

    def test_single_observation_is_not_a_series(self) -> None:
        observations = [Observation(date(2025, 1, 1), Decimal("10"), "e1", "shopping", "fixed", None)]
        self.assertEqual(detect(observations, "debit", "shopping", "expense"), [])

    def test_estimators(self) -> None:
        obs = [
            Observation(date(2025, 1, 1), Decimal("10"), "e1", "x", "fixed", None),
            Observation(date(2025, 1, 8), Decimal("20"), "e2", "x", "fixed", None),
            Observation(date(2025, 1, 15), Decimal("30"), "e3", "x", "fixed", None),
        ]
        series = detect(obs, "debit", "x", "expense")[0]
        self.assertEqual(estimate_amount(series, "last"), money("30"))
        self.assertEqual(estimate_amount(series, "mean3"), money("20"))
        self.assertEqual(estimate_amount(series, "max"), money("30"))


class ForecastTests(unittest.TestCase):
    def _state(self, events: list[Event], profile: Profile | None = None):
        profile = profile or _profile()
        data = _dataset(profile, events)
        builder = LedgerBuilder(data, EvidenceSet.empty())
        return builder.build(profile.user_id, date(2026, 1, 1))

    def test_amount_safe_today_respects_minimum(self) -> None:
        events = [
            _event(event_id="rent", category="rent", amount=Decimal("30"),
                   event_date=date(2026, 1, 5), settlement_date=date(2026, 1, 5),
                   status="scheduled"),
            _event(event_id="salary", event_type="income", category="salary",
                   direction="credit", amount=Decimal("50"),
                   event_date=date(2026, 1, 10), settlement_date=date(2026, 1, 10),
                   status="scheduled"),
        ]
        state = self._state(events)
        forecast = Forecast(state)
        # opening 1000, minimum 200 -> trough before any payment is 1000-30=970
        self.assertEqual(forecast.amount_safe_today(Decimal("10000")), money("770"))

    def test_pending_debit_is_reserved_from_request_date(self) -> None:
        events = [
            _event(event_id="pending1", amount=Decimal("100"), status="pending",
                   event_date=date(2025, 12, 30), settlement_date=date(2026, 1, 10)),
        ]
        state = self._state(events)
        forecast = Forecast(state)
        self.assertEqual(forecast.min_balance(), money("900"))

    def test_pending_credit_is_not_counted(self) -> None:
        events = [
            _event(event_id="pendingcredit", amount=Decimal("500"), direction="credit",
                   status="pending", event_date=date(2026, 1, 5), settlement_date=date(2026, 1, 5)),
        ]
        state = self._state(events)
        forecast = Forecast(state)
        self.assertEqual(forecast.min_balance(), money("1000"))

    def test_cancelled_event_is_ignored(self) -> None:
        events = [
            _event(event_id="cancelled1", amount=Decimal("900"), status="cancelled",
                   event_date=date(2026, 1, 5), settlement_date=date(2026, 1, 5)),
        ]
        state = self._state(events)
        forecast = Forecast(state)
        self.assertEqual(forecast.min_balance(), money("1000"))

    def test_unrealized_investment_is_not_cash(self) -> None:
        events = [
            _event(event_id="unreal1", event_type="investment_valuation", category="investment",
                   direction="non_cash", amount=Decimal("5000"), status="unrealized",
                   event_date=date(2026, 1, 5), settlement_date=date(2026, 1, 5)),
        ]
        state = self._state(events)
        forecast = Forecast(state)
        self.assertEqual(forecast.min_balance(), money("1000"))

    def test_earliest_full_payment_date_is_conservative(self) -> None:
        events = [
            _event(event_id="rent", category="rent", amount=Decimal("700"),
                   event_date=date(2026, 1, 2), settlement_date=date(2026, 1, 2), status="scheduled"),
            _event(event_id="salary", event_type="income", category="salary", direction="credit",
                   amount=Decimal("800"), event_date=date(2026, 1, 15),
                   settlement_date=date(2026, 1, 15), status="scheduled"),
        ]
        state = self._state(events)
        forecast = Forecast(state)
        # balance after rent on day 1: 300; safe amount today for 900 request is 100.
        # full 900 only becomes safe once the salary lands on Jan 15: 300+800=1100 >= 900+200
        earliest = forecast.earliest_full_payment_date(Decimal("900"))
        self.assertEqual(earliest, date(2026, 1, 15))

    def test_spending_change_frees_up_headroom(self) -> None:
        events = [
            _event(event_id="stream1", event_type="subscription", category="streaming",
                   direction="debit", amount=Decimal("50"), flexibility="stoppable",
                   event_date=date(2026, 1, 1), settlement_date=date(2026, 1, 1), status="settled"),
            _event(event_id="stream2", event_type="subscription", category="streaming",
                   direction="debit", amount=Decimal("50"), flexibility="stoppable",
                   event_date=date(2025, 12, 1), settlement_date=date(2025, 12, 1), status="settled"),
            _event(event_id="stream3", event_type="subscription", category="streaming",
                   direction="debit", amount=Decimal("50"), flexibility="stoppable",
                   event_date=date(2026, 2, 1), settlement_date=date(2026, 2, 1), status="scheduled"),
        ]
        state = self._state(events)
        forecast = Forecast(state)
        without_change = forecast.min_balance()
        stop = SpendingChange("stream3", None)
        with_change = forecast.min_balance(changes=(stop,))
        self.assertGreater(with_change, without_change)


class PlannerTests(unittest.TestCase):
    def _request(self, **overrides) -> Request:
        base = dict(
            request_id="request_1",
            user_id="user_test",
            request_date=date(2026, 1, 1),
            request_type="purchase",
            requested_amount=money("500"),
            desired_completion_date=date(2026, 2, 1),
            allows_partial_payment=True,
            request_text="",
        )
        base.update(overrides)
        return Request(**base)

    def _plan_for(self, events, profile=None, request=None, options=None):
        profile = profile or _profile()
        request = request or self._request()
        data = _dataset(profile, events, options={request.request_id: options or []})
        builder = LedgerBuilder(data, EvidenceSet.empty())
        state = builder.build(profile.user_id, request.request_date, request.request_id)
        return plan(request, profile, state, options or [])

    def test_affordable_now_when_full_payment_is_safe(self) -> None:
        decision = self._plan_for([])
        self.assertEqual(decision.status, STATUS_NOW)
        self.assertEqual(decision.method, METHOD_FULL)
        self.assertEqual(decision.payments[0].when, date(2026, 1, 1))
        self.assertEqual(decision.amount_safe_to_pay, money("500"))

    def test_full_payment_never_breaches_minimum(self) -> None:
        request = self._request(requested_amount=money("850"))
        decision = self._plan_for([], request=request)
        # balance 1000, minimum 200 -> max safe is 800, so 850 must not be affordable_now
        self.assertNotEqual(decision.status, STATUS_NOW)

    def test_not_recommended_when_no_method_is_accepted(self) -> None:
        profile = _profile(payment_methods=frozenset(), max_installment_months=None)
        request = self._request(allows_partial_payment=False)
        decision = self._plan_for([], profile=profile, request=request)
        self.assertEqual(decision.method, METHOD_NONE)
        self.assertEqual(decision.payments, ())

    def test_installments_must_match_a_supplied_option(self) -> None:
        request = self._request(
            requested_amount=money("900"),
            allows_partial_payment=False,
            desired_completion_date=date(2026, 4, 1),
        )
        option = PaymentOption(
            payment_option_id="payment_option_01",
            request_id=request.request_id,
            payment_method="installments",
            payment_amount=money("300"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=money("0"),
            total_payable_amount=money("900"),
        )
        profile = _profile(payment_methods=frozenset({"installments"}), balance=money("1300"))
        decision = self._plan_for([], profile=profile, request=request, options=[option])
        self.assertEqual(decision.method, METHOD_INSTALMENTS)
        self.assertEqual([p.amount for p in decision.payments], [money("300")] * 3)
        self.assertEqual(decision.option.payment_option_id, "payment_option_01")

    def test_installments_blocked_when_max_installment_months_is_blank(self) -> None:
        request = self._request(requested_amount=money("900"), allows_partial_payment=False)
        option = PaymentOption(
            payment_option_id="payment_option_01",
            request_id=request.request_id,
            payment_method="installments",
            payment_amount=money("300"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=money("0"),
            total_payable_amount=money("900"),
        )
        profile = _profile(payment_methods=frozenset({"installments"}), max_installment_months=None)
        decision = self._plan_for([], profile=profile, request=request, options=[option])
        self.assertNotEqual(decision.method, METHOD_INSTALMENTS)

    def test_partial_payment_sums_to_requested_amount(self) -> None:
        request = self._request(requested_amount=money("850"))
        decision = self._plan_for([], request=request)
        if decision.method == METHOD_PARTIAL:
            total = sum((p.amount for p in decision.payments), money("0"))
            self.assertEqual(total, money("850"))
            self.assertEqual(decision.status, STATUS_PLAN)

    def test_tie_break_prefers_plan_with_no_spending_changes(self) -> None:
        # Full payment safe without changes should always beat a plan needing changes.
        decision = self._plan_for([])
        self.assertEqual(decision.changes, ())

    def test_wait_only_when_full_payment_is_accepted(self) -> None:
        profile = _profile(payment_methods=frozenset({"partial_payment"}))
        request = self._request(requested_amount=money("5000"), allows_partial_payment=False,
                                 desired_completion_date=date(2026, 1, 5))
        decision = self._plan_for([], profile=profile, request=request)
        self.assertNotEqual(decision.method, METHOD_WAIT)


class IncomeStartDeduplicationTests(unittest.TestCase):
    """A message confirming a "first salary" that already has real settled
    history must not create a second, parallel salary series - found by
    tracing request_15 by hand: 28 of 34 income_start amendments in the full
    dataset target a user who already has 2+ settled salary occurrences on
    that same day-of-month, and a naive implementation doubles their income
    every cycle.
    """

    def test_income_start_does_not_duplicate_existing_salary_history(self) -> None:
        from datetime import datetime, timezone

        events = [
            _event(event_id="salary1", event_type="income", category="salary",
                   direction="credit", amount=Decimal("1661"),
                   event_date=date(2025, 11, 15), settlement_date=date(2025, 11, 15),
                   status="settled"),
            _event(event_id="salary2", event_type="income", category="salary",
                   direction="credit", amount=Decimal("1661"),
                   event_date=date(2025, 12, 15), settlement_date=date(2025, 12, 15),
                   status="settled"),
        ]
        profile = _profile(home_currency="EUR")
        data = _dataset(profile, events)
        amendment = Amendment(
            kind="income_start",
            source="message_11",
            user_id=profile.user_id,
            sent_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
            amount=money("1661"),
            currency="EUR",
            effective_date=date(2026, 1, 15),
        )
        evidence = EvidenceSet.of([amendment])
        builder = LedgerBuilder(data, evidence)
        state = builder.build(profile.user_id, date(2026, 1, 6))
        january_salary = [
            f for f in state.flows if f.category == "salary" and f.when == date(2026, 1, 15)
        ]
        self.assertEqual(len(january_salary), 1)
        self.assertEqual(january_salary[0].amount, money("1661"))

    def test_income_start_still_fires_with_no_prior_history(self) -> None:
        from datetime import datetime, timezone

        profile = _profile(home_currency="EUR")
        data = _dataset(profile, [])
        amendment = Amendment(
            kind="income_start",
            source="message_1",
            user_id=profile.user_id,
            sent_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
            amount=money("2000"),
            currency="EUR",
            effective_date=date(2026, 1, 15),
        )
        evidence = EvidenceSet.of([amendment])
        builder = LedgerBuilder(data, evidence)
        state = builder.build(profile.user_id, date(2026, 1, 6))
        january_salary = [
            f for f in state.flows if f.category == "salary" and f.when == date(2026, 1, 15)
        ]
        self.assertEqual(len(january_salary), 1)
        self.assertEqual(january_salary[0].amount, money("2000"))


class EvidenceConflictTests(unittest.TestCase):
    def test_cancellation_beats_a_later_amendment(self) -> None:
        from datetime import datetime, timezone

        amendments = [
            Amendment(kind="event_amount", source="msg1", user_id="user_test",
                      sent_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                      amount=money("50"), related_event_id="event_1"),
            Amendment(kind="event_cancelled", source="msg2", user_id="user_test",
                      sent_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                      related_event_id="event_1"),
        ]
        evidence = EvidenceSet.of(amendments)
        resolved = evidence.for_request("user_test", "", date(2026, 1, 3))
        kinds = {a.kind for a in resolved}
        self.assertIn("event_cancelled", kinds)
        self.assertNotIn("event_amount", kinds)

    def test_only_evidence_up_to_request_date_is_visible(self) -> None:
        from datetime import datetime, timezone

        amendments = [
            Amendment(kind="event_amount", source="msg1", user_id="user_test",
                      sent_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
                      amount=money("50"), related_event_id="event_1"),
        ]
        evidence = EvidenceSet.of(amendments)
        resolved = evidence.for_request("user_test", "", date(2026, 1, 1))
        self.assertEqual(resolved, [])


class IncomeClassificationTests(unittest.TestCase):
    """The income-stability classifier (income_classifier.py) only ever
    *excludes* a series when a classification explicitly says "variable" -
    a missing or absent classification must fail open to today's behavior
    (project it), never silently drop income the engine previously counted.
    """

    def _series_and_events(self):
        observations = [
            Observation(date(2025, 11, 15), Decimal("1000"), "e1", "salary", "fixed", None),
            Observation(date(2025, 12, 15), Decimal("1000"), "e2", "salary", "fixed", None),
            Observation(date(2026, 1, 15), Decimal("1000"), "e3", "salary", "fixed", None),
        ]
        series = detect(observations, "credit", "salary", "income")
        self.assertEqual(len(series), 1)
        return series[0]

    def test_series_key_is_stable_and_user_scoped(self) -> None:
        series = self._series_and_events()
        key_a = series.key("user_01")
        key_b = series.key("user_01")
        key_other_user = series.key("user_02")
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_other_user)
        self.assertEqual(key_a[0], "user_01")

    def _events_for(self, user_id: str) -> list[Event]:
        return [
            _event(event_id="rent", category="rent", amount=Decimal("30"),
                   event_date=date(2026, 1, 5), settlement_date=date(2026, 1, 5),
                   status="scheduled"),
            _event(event_id="salary1", user_id=user_id, event_type="income",
                   category="salary", direction="credit", amount=Decimal("1000"),
                   event_date=date(2025, 11, 15), settlement_date=date(2025, 11, 15),
                   status="settled"),
            _event(event_id="salary2", user_id=user_id, event_type="income",
                   category="salary", direction="credit", amount=Decimal("1000"),
                   event_date=date(2025, 12, 15), settlement_date=date(2025, 12, 15),
                   status="settled"),
        ]

    def test_ledger_excludes_series_classified_variable(self) -> None:
        profile = _profile()
        events = self._events_for(profile.user_id)
        data = _dataset(profile, events)
        builder_default = LedgerBuilder(data, EvidenceSet.empty())
        state_default = builder_default.build(profile.user_id, date(2026, 1, 1))
        salary_flows_default = [f for f in state_default.flows if f.category == "salary"]
        self.assertTrue(salary_flows_default)  # projected forward by default

        series = self._series_and_events()  # same (category, direction, event_type, kind, parameter)
        key = series.key(profile.user_id)
        builder_excluded = LedgerBuilder(data, EvidenceSet.empty(), income_classification={key: False})
        state_excluded = builder_excluded.build(profile.user_id, date(2026, 1, 1))
        salary_flows_excluded = [f for f in state_excluded.flows if f.category == "salary"]
        self.assertEqual(salary_flows_excluded, [])  # no longer projected forward

    def test_missing_classification_fails_open(self) -> None:
        profile = _profile()
        events = self._events_for(profile.user_id)
        data = _dataset(profile, events)
        # An unrelated key in the dict must not affect this series - only an
        # explicit False on *its own* key excludes it.
        builder = LedgerBuilder(
            data, EvidenceSet.empty(), income_classification={("someone_else",): False}
        )
        state = builder.build(profile.user_id, date(2026, 1, 1))
        salary_flows = [f for f in state.flows if f.category == "salary"]
        self.assertTrue(salary_flows)


class AnthropicClientTests(unittest.TestCase):
    def test_cache_key_is_deterministic_and_content_sensitive(self) -> None:
        from buyorwait.anthropic_client import _cache_key

        key_a = _cache_key("claude-opus-5", "system", "content", "schema_a")
        key_b = _cache_key("claude-opus-5", "system", "content", "schema_a")
        key_c = _cache_key("claude-opus-5", "system", "different content", "schema_a")
        key_d = _cache_key("claude-opus-5", "system", "content", "schema_b")
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)
        self.assertNotEqual(key_a, key_d)


class VisionResolutionTests(unittest.TestCase):
    """resolve_vision_readings is a pure function (no network) so the
    dual-model disagreement policy is testable on its own: Claude wins on
    disagreement, a single available reading is used as-is, and no reading
    at all correctly yields no amendment rather than a fabricated one.
    """

    def _image(self) -> ImageRef:
        from buyorwait.dataset import ImageRef

        return ImageRef(image_id="image_99", user_id="user_test", request_id="request_1",
                         related_event_id="event_1")

    def test_agreement_uses_claude_reading(self) -> None:
        from buyorwait.extraction import VisionReading, resolve_vision_readings

        claude = VisionReading("claude", money("1000"), "USD", 0.95)
        groq = VisionReading("groq", money("1000.02"), "USD", 0.9)
        amendment, audit = resolve_vision_readings(self._image(), claude, groq)
        self.assertEqual(audit["agreement"], "agree")
        self.assertEqual(amendment.amount, money("1000"))
        self.assertEqual(audit["chosen_source"], "claude")

    def test_disagreement_prefers_claude(self) -> None:
        from buyorwait.extraction import VisionReading, resolve_vision_readings

        claude = VisionReading("claude", money("100000"), "INR", 0.95)
        groq = VisionReading("groq", money("1000000"), "INR", 0.95)  # the digit-grouping bug
        amendment, audit = resolve_vision_readings(self._image(), claude, groq)
        self.assertEqual(audit["agreement"], "DISAGREE")
        self.assertEqual(amendment.amount, money("100000"))
        self.assertEqual(audit["chosen_source"], "claude")

    def test_single_reading_is_used(self) -> None:
        from buyorwait.extraction import VisionReading, resolve_vision_readings

        groq = VisionReading("groq", money("500"), "EUR", 0.9)
        amendment, audit = resolve_vision_readings(self._image(), None, groq)
        self.assertEqual(audit["agreement"], "single reading")
        self.assertEqual(amendment.amount, money("500"))
        self.assertEqual(audit["chosen_source"], "groq")

    def test_no_reading_yields_no_amendment(self) -> None:
        from buyorwait.extraction import resolve_vision_readings

        amendment, audit = resolve_vision_readings(self._image(), None, None)
        self.assertIsNone(amendment)
        self.assertEqual(audit["agreement"], "no reading")


class ConfidenceLayerTests(unittest.TestCase):
    """The four guardrail layers in confidence.py. Calibrated against
    dataset/sample_requests.csv (see code/evaluation/confidence_audit.py):
    every one of the 4 exact amount matches has 0% confirmed-only-floor
    reliance, and every request whose affordability_status is not unanimous
    across estimator choices misses. These tests lock in the two properties
    that make that signal trustworthy: the shadow simulation must agree with
    Forecast on data it has no reason to disagree on, and a confirmed-only
    ledger (no projection at all) must report zero projection reliance.
    """

    def test_shadow_simulation_agrees_with_forecast(self) -> None:
        from buyorwait.confidence import _shadow_min_balance
        from buyorwait.forecast import Forecast

        profile = _profile()
        events = [
            _event(event_id="rent", category="rent", amount=Decimal("300"),
                   event_date=date(2026, 1, 5), settlement_date=date(2026, 1, 5),
                   status="scheduled"),
            _event(event_id="salary", event_type="income", category="salary",
                   direction="credit", amount=Decimal("500"),
                   event_date=date(2026, 1, 20), settlement_date=date(2026, 1, 20),
                   status="scheduled"),
        ]
        data = _dataset(profile, events)
        builder = LedgerBuilder(data, EvidenceSet.empty())
        state = builder.build(profile.user_id, date(2026, 1, 1))
        forecast = Forecast(state)
        self.assertEqual(_shadow_min_balance(state), forecast.min_balance())

    def test_confirmed_only_state_has_no_projected_flows(self) -> None:
        from buyorwait.confidence import _confirmed_only_state
        from buyorwait.ledger import EXPLICIT

        profile = _profile()
        events = [
            _event(event_id="rent1", category="rent", amount=Decimal("300"),
                   event_date=date(2025, 11, 2), settlement_date=date(2025, 11, 2),
                   status="settled"),
            _event(event_id="rent2", category="rent", amount=Decimal("300"),
                   event_date=date(2025, 12, 2), settlement_date=date(2025, 12, 2),
                   status="settled"),
            _event(event_id="pending1", category="groceries", amount=Decimal("50"),
                   event_date=date(2025, 12, 30), settlement_date=date(2026, 1, 10),
                   status="pending"),
        ]
        data = _dataset(profile, events)
        builder = LedgerBuilder(data, EvidenceSet.empty())
        state = builder.build(profile.user_id, date(2026, 1, 1))
        self.assertTrue(any(f.source != EXPLICIT for f in state.flows))  # a projected rent exists
        confirmed = _confirmed_only_state(state)
        self.assertTrue(all(f.source == EXPLICIT for f in confirmed.flows))
        self.assertTrue(any(f.event_id == "pending1" for f in confirmed.flows))

    def test_zero_reliance_when_nothing_is_projected(self) -> None:
        from buyorwait.confidence import evaluate

        profile = _profile()
        events = [
            _event(event_id="pending1", category="groceries", amount=Decimal("100"),
                   event_date=date(2025, 12, 30), settlement_date=date(2026, 1, 10),
                   status="pending"),
        ]
        data = Dataset(
            requests=[], profiles={profile.user_id: profile},
            events={profile.user_id: events}, options={}, rates={},
            messages=[], images=[],
        )
        request = Request(
            request_id="request_x", user_id=profile.user_id, request_date=date(2026, 1, 1),
            request_type="purchase", requested_amount=money("200"),
            desired_completion_date=date(2026, 2, 1), allows_partial_payment=False,
            request_text="",
        )
        report = evaluate(data, request, reported_amount=money("200"))
        self.assertEqual(report.projection_reliance_pct, Decimal("0.00"))


if __name__ == "__main__":
    unittest.main()
