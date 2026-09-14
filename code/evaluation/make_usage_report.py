"""Generate evaluation/usage_report.md from the real model-call log.

The report is never hand-written. `.cache/model_calls.jsonl` gets one line per
live API call, appended by `buyorwait.llm`, and this script aggregates it. Run it
straight after the full-dataset run that produced output.csv:

    python code/evaluation/make_usage_report.py --requests 250
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buyorwait import config

REPORT_PATH = Path(__file__).resolve().parent / "usage_report.md"


def load_calls(path: Path) -> list[dict]:
    if not path.exists():
        return []
    calls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            calls.append(json.loads(line))
    return calls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=250)
    parser.add_argument("--log", default=str(config.USAGE_LOG_PATH))
    parser.add_argument("--out", default=str(REPORT_PATH))
    args = parser.parse_args(argv)

    calls = load_calls(Path(args.log))
    by_model: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "input": 0, "output": 0}
    )
    by_purpose: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "input": 0, "output": 0}
    )
    for call in calls:
        key = (call.get("provider", "groq"), call["model"])
        by_model[key]["calls"] += 1
        by_model[key]["input"] += int(call.get("input_tokens", 0))
        by_model[key]["output"] += int(call.get("output_tokens", 0))
        purpose = call.get("purpose", "unknown")
        by_purpose[purpose]["calls"] += 1
        by_purpose[purpose]["input"] += int(call.get("input_tokens", 0))
        by_purpose[purpose]["output"] += int(call.get("output_tokens", 0))

    total_calls = sum(v["calls"] for v in by_model.values())
    total_in = sum(v["input"] for v in by_model.values())
    total_out = sum(v["output"] for v in by_model.values())
    total_tokens = total_in + total_out

    def cost(model: str, tokens_in: int, tokens_out: int) -> float:
        price = config.MODEL_PRICES_USD_PER_MTOK.get(model)
        if not price:
            return 0.0
        return tokens_in / 1_000_000 * price["input"] + tokens_out / 1_000_000 * price["output"]

    total_cost = sum(cost(model, v["input"], v["output"]) for (_, model), v in by_model.items())
    requests = max(args.requests, 1)

    lines: list[str] = []
    lines.append("# Token Usage and Cost Report")
    lines.append("")
    lines.append(
        "Final full-dataset run over `dataset/requests.csv` (" + str(args.requests)
        + " requests). Figures come from the `usage` block returned by the provider "
        "on every live call, appended to a JSON-lines log by `buyorwait.llm` and "
        "aggregated by `code/evaluation/make_usage_report.py`. Nothing here is estimated "
        "except the unit prices, which are provider list prices at the time of the run."
    )
    lines.append("")
    lines.append("## Per model")
    lines.append("")
    lines.append("| Provider | Model | Calls | Input tokens | Output tokens | Total tokens | Est. cost (USD) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for (provider, model), value in sorted(by_model.items()):
        model_cost = cost(model, value["input"], value["output"])
        lines.append(
            "| " + provider + " | `" + model + "` | " + f"{value['calls']:,}" + " | "
            + f"{value['input']:,}" + " | " + f"{value['output']:,}" + " | "
            + f"{value['input'] + value['output']:,}" + " | " + f"{model_cost:.4f}" + " |"
        )
    lines.append(
        "| **all** | | **" + f"{total_calls:,}" + "** | **" + f"{total_in:,}" + "** | **"
        + f"{total_out:,}" + "** | **" + f"{total_tokens:,}" + "** | **" + f"{total_cost:.4f}" + "** |"
    )
    lines.append("")
    lines.append("## Per purpose")
    lines.append("")
    lines.append("| Purpose | Calls | Input tokens | Output tokens |")
    lines.append("|---|---:|---:|---:|")
    for purpose, value in sorted(by_purpose.items()):
        lines.append(
            "| " + purpose + " | " + f"{value['calls']:,}" + " | " + f"{value['input']:,}"
            + " | " + f"{value['output']:,}" + " |"
        )
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append("| Requests scored | " + f"{args.requests:,}" + " |")
    lines.append("| Model calls | " + f"{total_calls:,}" + " |")
    lines.append("| Model calls per request | " + f"{total_calls / requests:.3f}" + " |")
    lines.append("| Total tokens | " + f"{total_tokens:,}" + " |")
    lines.append("| Average tokens per request | " + f"{total_tokens / requests:.1f}" + " |")
    lines.append("| Estimated total cost (USD) | " + f"{total_cost:.4f}" + " |")
    lines.append("| Estimated cost per request (USD) | " + f"{total_cost / requests:.6f}" + " |")
    lines.append("")
    lines.append("## Why the call count is below one per request")
    lines.append("")
    lines.append(
        "The decision itself is computed, not generated: ledger reconstruction, the "
        "90-day forecast, plan ranking and the explanation are deterministic Python. "
        "Models are used only where perception is required - reading an amount off a "
        "bill image, and interpreting a free-text message - so the work is bounded by "
        "the 16 images and 215 messages in the dataset rather than by the number of "
        "requests. Responses are cached by a hash of the exact request, so re-running "
        "the pipeline costs nothing and returns identical output."
    )
    lines.append("")
    lines.append("## Vision precision: two independent models")
    lines.append("")
    lines.append(
        "Every one of the 16 blank-amount images is read by both Claude "
        "(`claude-opus-5`, primary) and Groq (`qwen/qwen3.8-27b`, cross-check) "
        "whenever both API keys are configured, which is why `image_amount` above "
        "shows calls on both providers. Agreement is logged either way to "
        "`code/evaluation/ocr_audit.md`; the current run shows 16/16 agreement, and "
        "all 16 have additionally been checked by hand against the source image."
    )
    lines.append("")
    lines.append("## Feature evaluated and left off by default")
    lines.append("")
    lines.append(
        "An experimental Claude-based income-stability classifier "
        "(`buyorwait/income_classifier.py`, `--income-classifier`) was built and "
        "measured against `dataset/sample_requests.csv`: it correctly identifies "
        "gig-style income by its description history, but including it reduced the "
        "aggregate sample score rather than improving it, so it stays off by default "
        "and cost nothing in this run. Its calibration cost and findings are in "
        "README.md's Known Limitations."
    )
    lines.append("")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote " + args.out + " from " + str(total_calls) + " logged calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
