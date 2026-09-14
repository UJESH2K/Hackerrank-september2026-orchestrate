"""Does a low-confidence flag actually predict a wrong amount_safe_to_pay?

    python code/evaluation/confidence_audit.py

Runs the four guardrail layers in buyorwait/confidence.py against every row
in dataset/sample_requests.csv and cross-tabulates each request's confidence
tier against whether its amount_safe_to_pay actually matched the reference.
This is the calibration check confidence.py's docstring refers to: the
layers are kept as shipped diagnostics only because this script shows they
correlate with real error, not because they sounded reasonable.
"""

from __future__ import annotations

import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buyorwait import config
from buyorwait.confidence import evaluate
from buyorwait.dataset import load
from buyorwait.evidence import EvidenceSet
from buyorwait.ledger import LedgerBuilder
from buyorwait.pipeline import run

TOLERANCE = Decimal("0.01")


def _decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        return None


def main() -> int:
    config.load_dotenv()
    data = load("sample_requests.csv")

    evidence = EvidenceSet.empty()
    try:
        from buyorwait.extraction import build_evidence

        evidence = build_evidence(data=data)
    except Exception:  # noqa: BLE001 - confidence audit still runs deterministic-only
        pass

    result = run(data, evidence)
    rows_by_id = {row["request_id"]: row for row in result.rows}
    expected = {r.request_id: r.expected for r in data.requests}
    builder = LedgerBuilder(data, evidence)

    print(
        f"{'request':10s} {'tier':7s} {'match?':7s} {'reported':>14s} "
        f"{'shadow_ok':9s} {'sens%':>7s} {'proj%':>7s} {'status_unanimous':16s}"
    )
    correct_by_tier: dict[str, list[bool]] = {"high": [], "medium": [], "low": []}
    for request in data.requests:
        row = rows_by_id[request.request_id]
        want = expected[request.request_id]
        reported = _decimal(row["amount_safe_to_pay"])
        want_amount = _decimal(want["amount_safe_to_pay"])
        if reported is None or want_amount is None:
            continue
        matched = abs(reported - want_amount) <= TOLERANCE
        report = evaluate(data, request, reported, builder=builder)
        correct_by_tier[report.tier].append(matched)
        print(
            f"{request.request_id:10s} {report.tier:7s} {str(matched):7s} "
            f"{reported!s:>14s} {str(report.shadow_agrees):9s} "
            f"{report.sensitivity_spread_pct!s:>6s}% {report.projection_reliance_pct!s:>6s}% "
            f"{str(report.status_unanimous):16s}"
        )
        for reason in report.reasons:
            print("    - " + reason)

    print()
    print("Match rate by confidence tier (this is the calibration check):")
    for tier in ("high", "medium", "low"):
        outcomes = correct_by_tier[tier]
        if not outcomes:
            print(f"  {tier:7s} n=0")
            continue
        rate = sum(outcomes) / len(outcomes) * 100
        print(f"  {tier:7s} n={len(outcomes):2d} exact-match rate={rate:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
