"""End-to-end orchestration: dataset -> evidence -> ledger -> plan -> verified CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import OUTPUT_PATH
from .dataset import Dataset, OUTPUT_COLUMNS, Request, load
from .evidence import EvidenceSet
from .explain import explain
from .ledger import LedgerBuilder, LedgerState
from .money import format_amount, format_plan_amount
from .planner import Decision, plan
from .verify import Report, VerificationError, verify


@dataclass
class RunResult:
    rows: list[dict[str, str]]
    decisions: list[Decision]
    report: Report


def render(decision: Decision, state: LedgerState) -> dict[str, str]:
    labels = {item.event_id: item.description.lower() for item in state.flexible}
    plan_text = (
        "|".join(
            payment.when.isoformat() + ":" + format_plan_amount(payment.amount)
            for payment in decision.payments
        )
        or "none"
    )
    changes_text = "|".join(change.render() for change in decision.changes) or "none"
    return {
        "request_id": decision.request.request_id,
        "amount_safe_to_pay": format_amount(decision.amount_safe_to_pay),
        "affordability_status": decision.status,
        "recommended_payment_method": decision.method,
        "payment_plan": plan_text,
        "earliest_date_for_full_payment": (
            decision.earliest_full_payment.isoformat()
            if decision.earliest_full_payment
            else ""
        ),
        "spending_changes_needed": changes_text,
        "decision_explanation": explain(decision, state.profile, labels),
    }


def run(
    data: Dataset,
    evidence: EvidenceSet | None = None,
    requests: Sequence[Request] | None = None,
    income_classification: dict[tuple, bool] | None = None,
) -> RunResult:
    builder = LedgerBuilder(data, evidence, income_classification)
    targets = list(requests if requests is not None else data.requests)
    rows: list[dict[str, str]] = []
    decisions: list[Decision] = []
    for request in targets:
        profile = data.profiles[request.user_id]
        state = builder.build(request.user_id, request.request_date, request.request_id)
        decision = plan(request, profile, state, data.options.get(request.request_id, []))
        decisions.append(decision)
        rows.append(render(decision, state))
    report = verify(rows, targets, data)
    return RunResult(rows, decisions, report)


def write_csv(rows: Sequence[dict[str, str]], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_and_write(
    request_file: str = "requests.csv",
    output_path: Path = OUTPUT_PATH,
    evidence: EvidenceSet | None = None,
    income_classification: dict[tuple, bool] | None = None,
    strict: bool = True,
    data: Dataset | None = None,
) -> RunResult:
    data = data if data is not None else load(request_file)
    result = run(data, evidence, income_classification=income_classification)
    if strict and not result.report.ok:
        preview = "\n  ".join(result.report.problems[:20])
        raise VerificationError(
            str(len(result.report.problems)) + " invariant failures:\n  " + preview
        )
    write_csv(result.rows, output_path)
    return result
