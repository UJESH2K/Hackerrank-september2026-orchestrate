# Token Usage and Cost Report

Final full-dataset run over `dataset/requests.csv` (250 requests). Figures come from the `usage` block returned by the provider on every live call, appended to a JSON-lines log by `buyorwait.llm` and aggregated by `code/evaluation/make_usage_report.py`. Nothing here is estimated except the unit prices, which are provider list prices at the time of the run.

## Per model

| Provider | Model | Calls | Input tokens | Output tokens | Total tokens | Est. cost (USD) |
|---|---|---:|---:|---:|---:|---:|
| anthropic | `claude-opus-5` | 16 | 38,152 | 832 | 38,984 | 0.2116 |
| groq | `openai/gpt-oss-120b` | 215 | 141,411 | 80,628 | 222,039 | 0.0817 |
| groq | `qwen/qwen3.8-27b` | 16 | 30,527 | 1,045 | 31,572 | 0.0095 |
| **all** | | **247** | **210,090** | **82,505** | **292,595** | **0.3027** |

## Per purpose

| Purpose | Calls | Input tokens | Output tokens |
|---|---:|---:|---:|
| image_amount | 32 | 68,679 | 1,877 |
| message_effects | 215 | 141,411 | 80,628 |

## Totals

| Metric | Value |
|---|---:|
| Requests scored | 250 |
| Model calls | 247 |
| Model calls per request | 0.988 |
| Total tokens | 292,595 |
| Average tokens per request | 1170.4 |
| Estimated total cost (USD) | 0.3027 |
| Estimated cost per request (USD) | 0.001211 |

## Why the call count is below one per request

The decision itself is computed, not generated: ledger reconstruction, the 90-day forecast, plan ranking and the explanation are deterministic Python. Models are used only where perception is required - reading an amount off a bill image, and interpreting a free-text message - so the work is bounded by the 16 images and 215 messages in the dataset rather than by the number of requests. Responses are cached by a hash of the exact request, so re-running the pipeline costs nothing and returns identical output.

## Vision precision: two independent models

Every one of the 16 blank-amount images is read by both Claude (`claude-opus-5`, primary) and Groq (`qwen/qwen3.8-27b`, cross-check) whenever both API keys are configured, which is why `image_amount` above shows calls on both providers. Agreement is logged either way to `code/evaluation/ocr_audit.md`; the current run shows 16/16 agreement, and all 16 have additionally been checked by hand against the source image.

## Feature evaluated and left off by default

An experimental Claude-based income-stability classifier (`buyorwait/income_classifier.py`, `--income-classifier`) was built and measured against `dataset/sample_requests.csv`: it correctly identifies gig-style income by its description history, but including it reduced the aggregate sample score rather than improving it, so it stays off by default and cost nothing in this run. Its calibration cost and findings are in README.md's Known Limitations.

