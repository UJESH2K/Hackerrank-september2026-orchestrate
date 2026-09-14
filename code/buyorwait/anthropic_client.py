"""Claude client: on-disk caching and exact token accounting.

Mirrors `llm.py`'s shape (cache-by-request-hash, one JSON line per live call
appended to the shared usage log) but talks to Claude through the official
`anthropic` SDK rather than raw HTTP, per current API guidance. Claude is used
for the two places semantic judgment beats a numeric heuristic: reading a
receipt image, and telling a confirmed salary from a variable/gig income
stream by its description history (see extraction.py and
income_classifier.py). Every call requests a JSON-schema-constrained response,
so - unlike the Groq path - there is no fence-stripping fallback: the SDK
guarantees the first text block is valid JSON against the schema.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from . import config


class ModelUnavailable(RuntimeError):
    """No API key, or the endpoint refused every retry."""


def _client():
    try:
        import anthropic
    except ImportError as exc:
        raise ModelUnavailable(
            "the 'anthropic' package is not installed (pip install anthropic)"
        ) from exc
    api_key = config.anthropic_api_key()
    if not api_key:
        raise ModelUnavailable("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(
        api_key=api_key,
        max_retries=config.CLAUDE_MAX_RETRIES,
        timeout=config.CLAUDE_TIMEOUT_SECONDS,
    )


def _cache_path(key: str) -> Path:
    directory = config.CACHE_DIR / "responses"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (key + ".json")


def _cache_key(model: str, system: str, content: Any, schema_name: str) -> str:
    payload = json.dumps(
        {"provider": "anthropic", "model": model, "system": system, "content": content, "schema": schema_name},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def encode_image(path: Path) -> tuple[str, str]:
    """Returns (base64_data, media_type) for a local PNG."""
    return base64.standard_b64encode(path.read_bytes()).decode("ascii"), "image/png"


def complete_json(
    system: str,
    content: list[dict[str, Any]] | str,
    schema: dict[str, Any],
    schema_name: str,
    model: str | None = None,
    max_tokens: int = 1024,
    refresh: bool = False,
) -> dict[str, Any]:
    """One Claude call constrained to `schema`, returning the parsed JSON object.

    Caches on an exact hash of (model, system, content, schema_name), so a
    re-run makes no network call and returns byte-identical results.
    """
    model = model or config.CLAUDE_TEXT_MODEL
    key = _cache_key(model, system, content, schema_name)
    path = _cache_path(key)

    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached["data"]

    client = _client()
    messages = [{"role": "user", "content": content}]
    last_error: Exception | None = None
    for attempt in range(config.CLAUDE_MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                # These are narrow, single-turn extraction/classification calls,
                # not open-ended reasoning - "low" effort keeps Opus 5's
                # always-on thinking from eating the whole token budget before
                # it reaches the JSON body (thinking is on by default on this
                # model; a low max_tokens with default effort can truncate to
                # stop_reason "max_tokens" with no text block at all).
                output_config={
                    "format": {"type": "json_schema", "schema": schema},
                    "effort": "low",
                },
            )
            text_blocks = [block.text for block in response.content if block.type == "text"]
            if not text_blocks:
                raise ModelUnavailable(
                    "Claude returned no text block (stop_reason="
                    + str(response.stop_reason) + "); raise max_tokens"
                )
            data = json.loads(text_blocks[0])
            usage = response.usage
            record = {
                "provider": "anthropic",
                "model": model,
                "schema": schema_name,
                "data": data,
                "prompt_tokens": int(getattr(usage, "input_tokens", 0)),
                "completion_tokens": int(getattr(usage, "output_tokens", 0)),
            }
            path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            _append_call_log(record)
            return data
        except Exception as error:  # noqa: BLE001 - narrowed by SDK exception classes below
            last_error = error
            if _is_retryable(error):
                time.sleep(min(3 * (attempt + 1), 30))
                continue
            raise ModelUnavailable("Claude call failed: " + repr(error)) from error
    raise ModelUnavailable("Claude call failed after retries: " + repr(last_error))


def _is_retryable(error: Exception) -> bool:
    try:
        import anthropic
    except ImportError:
        return False
    return isinstance(
        error,
        (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError),
    )


def _append_call_log(record: dict[str, Any]) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "provider": "anthropic",
            "model": record["model"],
            "purpose": record["schema"],
            "input_tokens": record["prompt_tokens"],
            "output_tokens": record["completion_tokens"],
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    with config.USAGE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
