"""Score the engine against the 25 solved rows in dataset/sample_requests.csv.

Run it from the repo root:

    python code/evaluation/evaluate_samples.py            # with the evidence layer
    python code/evaluation/evaluate_samples.py --no-evidence
    python code/evaluation/evaluate_samples.py --detail   # per-row diffs

Exact-match rate is reported per field, plus a tolerance band on
`amount_safe_to_pay` because that column is a projection, not a lookup.
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buyorwait import config
from buyorwait.dataset import load
from buyorwait.evidence import EvidenceSet
from buyorwait.pipeline import run

EXACT_FIELDS = [
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]
TOLERANCE = Decimal("0.01")


def _decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-evidence", action="store_true")
    parser.add_argument(
        "--income-classifier", action="store_true",
        help=(
            "enable the experimental Claude income-stability classifier "
            "(off by default - see README.md Known Limitations)"
        ),
    )
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args(argv)

    config.load_dotenv()
    data = load("sample_requests.csv")

    evidence = EvidenceSet.empty()
    income_classification: dict[tuple, bool] = {}
    if not args.no_evidence:
        from buyorwait.extraction import build_evidence

        evidence = build_evidence(data=data)

        if args.income_classifier:
            from buyorwait.income_classifier import classify_income_series

            user_ids = sorted({r.user_id for r in data.requests})
            income_classification = classify_income_series(data, user_ids=user_ids)

    result = run(data, evidence, income_classification=income_classification)
    expected = {r.request_id: r.expected for r in data.requests}
    requested = {r.request_id: r.requested_amount for r in data.requests}

    hits = {field: 0 for field in EXACT_FIELDS}
    amount_exact = 0
    amount_close = 0
    relative_error = Decimal(0)
    perfect_rows = 0

    for row in result.rows:
        rid = row["request_id"]
        want = expected[rid]
        row_perfect = True
        differences: list[str] = []
        for field in EXACT_FIELDS:
            if row[field] == want[field]:
                hits[field] += 1
            else:
                row_perfect = False
                differences.append(
                    "  " + field + ": expected " + repr(want[field]) + " got " + repr(row[field])
                )
        got_amount = _decimal(row["amount_safe_to_pay"])
        want_amount = _decimal(want["amount_safe_to_pay"])
        if got_amount is not None and want_amount is not None:
            delta = abs(got_amount - want_amount)
            scale = max(requested[rid], Decimal(1))
            relative_error += delta / scale
            if delta <= TOLERANCE:
                amount_exact += 1
            else:
                row_perfect = False
                differences.append(
                    "  amount_safe_to_pay: expected " + str(want_amount) + " got " + str(got_amount)
                    + "  (" + str(round(delta / scale * 100, 2)) + "% of request)"
                )
            if delta / scale <= Decimal("0.02"):
                amount_close += 1
        perfect_rows += row_perfect
        if args.detail and differences:
            print(rid)
            print("\n".join(differences))

    total = len(result.rows)
    print("")
    print("sample rows:            " + str(total))
    print("fully correct rows:     " + str(perfect_rows) + "/" + str(total))
    for field in EXACT_FIELDS:
        print("  " + field.ljust(32) + str(hits[field]) + "/" + str(total))
    print("  amount_safe_to_pay exact        " + str(amount_exact) + "/" + str(total))
    print("  amount_safe_to_pay within 2%    " + str(amount_close) + "/" + str(total))
    print("  mean relative amount error      " + str(round(relative_error / total * 100, 3)) + "%")
    print("invariant checks:       " + ("all passed" if result.report.ok else "FAILED"))
    for problem in result.report.problems[:20]:
        print("  " + problem)
    return 0 if result.report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
