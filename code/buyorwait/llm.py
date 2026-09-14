"""Groq client with on-disk caching and exact token accounting.

Every model call goes through `complete`. It caches on a hash of the request, so
a re-run costs nothing and produces identical output, and it appends one JSON
line per real call to `.cache/model_calls.jsonl`, which is what
`evaluation/usage_report.md` is generated from. Nothing here reads a secret from
anywhere except the environment.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config


class ModelUnavailable(RuntimeError):
    """No API key, or the endpoint refused every retry."""


@dataclass
class Usage:
    provider: str = "groq"
    model: str = ""
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self) -> float:
        price = config.MODEL_PRICES_USD_PER_MTOK.get(self.model)
        if not price:
            return 0.0
        return (
            self.input_tokens / 1_000_000 * price["input"]
            + self.output_tokens / 1_000_000 * price["output"]
        )


@dataclass
class UsageLedger:
    per_model: dict[str, Usage] = field(default_factory=dict)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int, cached: bool) -> None:
        usage = self.per_model.setdefault(model, Usage(model=model))
        if cached:
            usage.cached_calls += 1
            return
        usage.calls += 1
        usage.input_tokens += prompt_tokens
        usage.output_tokens += completion_tokens

    def totals(self) -> Usage:
        combined = Usage(model="all")
        for usage in self.per_model.values():
            combined.calls += usage.calls
            combined.cached_calls += usage.cached_calls
            combined.input_tokens += usage.input_tokens
            combined.output_tokens += usage.output_tokens
        return combined

    def total_cost_usd(self) -> float:
        return sum(usage.cost_usd() for usage in self.per_model.values())


LEDGER = UsageLedger()


def _cache_path(key: str) -> Path:
    directory = config.CACHE_DIR / "responses"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (key + ".json")


def _cache_key(model: str, messages: list[dict[str, Any]], schema_name: str) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "schema": schema_name},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def encode_image(path: Path) -> str:
    """Data URL for a local PNG, so no file is ever uploaded to a third party by path."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def complete(
    messages: list[dict[str, Any]],
    model: str | None = None,
    schema_name: str = "generic",
    max_tokens: int = 700,
    refresh: bool = False,
) -> str:
    """Return the assistant text for one chat completion, using the cache when possible."""
    model = model or config.TEXT_MODEL
    key = _cache_key(model, messages, schema_name)
    path = _cache_path(key)

    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        LEDGER.record(model, cached.get("prompt_tokens", 0), cached.get("completion_tokens", 0), True)
        return cached["content"]

    api_key = config.api_key()
    if not api_key:
        raise ModelUnavailable("GROQ_API_KEY is not set")

    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": config.MODEL_TEMPERATURE,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "seed": 7,
        }
    ).encode("utf-8")

    last_error: Exception | None = None
    for attempt in range(config.MODEL_MAX_RETRIES):
        request = urllib.request.Request(
            config.GROQ_BASE_URL + "/chat/completions",
            data=body,
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "User-Agent": "buy-or-wait/1.0 (+python-urllib)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=config.MODEL_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage", {})
            record = {
                "model": model,
                "schema": schema_name,
                "content": content,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            }
            path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            _append_call_log(record)
            time.sleep(config.MODEL_CALL_DELAY_SECONDS)
            LEDGER.record(model, record["prompt_tokens"], record["completion_tokens"], False)
            return content
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code in (408, 429, 500, 502, 503, 529):
                retry_after = error.headers.get("retry-after") if error.headers else None
                delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
                time.sleep(delay if delay is not None else min(3 * (attempt + 1), 30))
                continue
            raise ModelUnavailable("groq returned " + str(error.code) + ": " + error.read().decode("utf-8", "ignore"))
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            time.sleep(min(2 ** attempt, 20))
    raise ModelUnavailable("groq call failed after retries: " + repr(last_error))


def _append_call_log(record: dict[str, Any]) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "provider": "groq",
            "model": record["model"],
            "purpose": record["schema"],
            "input_tokens": record["prompt_tokens"],
            "output_tokens": record["completion_tokens"],
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    with config.USAGE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def parse_json(text: str) -> dict[str, Any]:
    """Tolerant JSON parse: models occasionally wrap the object in a fence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        stripped = stripped.rsplit("```", 1)[0]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(stripped[start : end + 1])
