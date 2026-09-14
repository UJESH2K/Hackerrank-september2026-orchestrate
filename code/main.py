"""Buy or Wait? - command-line entry point.

    python code/main.py                        # full run, writes output.csv
    python code/main.py --no-evidence          # deterministic core only, no model calls
    python code/main.py --requests sample_requests.csv --output /tmp/samples.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from buyorwait import config
from buyorwait.dataset import load as load_dataset
from buyorwait.evidence import EvidenceSet
from buyorwait.pipeline import run_and_write


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buy or Wait? financial decision agent")
    parser.add_argument(
        "--requests", default="requests.csv",
        help="file inside dataset/ to score (default: requests.csv)",
    )
    parser.add_argument(
        "--output", default=None, help="where to write the predictions CSV",
    )
    parser.add_argument(
        "--no-evidence", action="store_true",
        help="skip the message and image perception layer entirely",
    )
    parser.add_argument(
        "--refresh-evidence", action="store_true",
        help="ignore the extraction cache and call the models again",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="write the CSV even if an invariant fails (diagnostics only)",
    )
    parser.add_argument(
        "--income-classifier", action="store_true",
        help=(
            "enable the experimental Claude income-stability classifier. "
            "Off by default: it measurably regresses accuracy on "
            "dataset/sample_requests.csv - see README.md Known Limitations."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.load_dotenv()

    # Loaded once: profiles/events/rates cover every user regardless of which
    # requests.csv slice is being scored, and both the evidence pass and the
    # income classifier need the full dataset, not just the scored rows.
    data = load_dataset(args.requests)

    evidence = EvidenceSet.empty()
    income_classification: dict[tuple, bool] = {}
    if not args.no_evidence:
        from buyorwait.extraction import build_evidence

        evidence = build_evidence(data=data, refresh=args.refresh_evidence)

        if args.income_classifier:
            from buyorwait.income_classifier import classify_income_series

            income_classification = classify_income_series(data)

    output_path = Path(args.output) if args.output else config.OUTPUT_PATH
    result = run_and_write(
        request_file=args.requests,
        output_path=output_path,
        evidence=evidence,
        income_classification=income_classification,
        strict=not args.no_verify,
        data=data,
    )
    print(
        "wrote " + str(len(result.rows)) + " rows to " + str(output_path)
        + " | invariant checks: " + ("all passed" if result.report.ok else "FAILED")
    )
    for problem in result.report.problems[:20]:
        print("  " + problem)
    return 0 if result.report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
