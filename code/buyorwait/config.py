"""Central configuration. Every tunable the engine uses lives here, nowhere else."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CODE_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = CODE_ROOT.parent

DATASET_DIR = Path(os.environ.get("BOW_DATASET_DIR", REPO_ROOT / "dataset"))
OUTPUT_PATH = Path(os.environ.get("BOW_OUTPUT_PATH", REPO_ROOT / "output.csv"))
CACHE_DIR = Path(os.environ.get("BOW_CACHE_DIR", CODE_ROOT / ".cache"))
USAGE_LOG_PATH = CACHE_DIR / "model_calls.jsonl"

# --- Financial rules fixed by the problem statement -------------------------
FORECAST_DAYS = 90
MAX_SPENDING_CHANGES = 3

# --- Forecast policy (calibrated against dataset/sample_requests.csv) -------
# Recurrence intervals the dataset generator actually uses, in days.
FIXED_INTERVALS = (5, 7, 10, 14, 21)
# A group is monthly when this share of its events sit on repeated days-of-month.
MONTHLY_DOM_COVERAGE = 0.8
MONTHLY_DOM_MAX_GROUPS = 4
# Share of gaps that must agree before a fixed-interval series is accepted.
FIXED_INTERVAL_AGREEMENT = 0.6
# Minimum observations before a series is treated as recurring at all.
MIN_OCCURRENCES = 2
# Amount estimators: expenses use a short trailing mean, income uses the most
# recently confirmed figure. Calibrated in evaluation/calibrate.py.
EXPENSE_ESTIMATOR = os.environ.get("BOW_EXPENSE_ESTIMATOR", "mean3")
INCOME_ESTIMATOR = os.environ.get("BOW_INCOME_ESTIMATOR", "median")

# --- Model layer ------------------------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
TEXT_MODEL = os.environ.get("BOW_TEXT_MODEL", "openai/gpt-oss-120b")
VISION_MODEL = os.environ.get("BOW_VISION_MODEL", "qwen/qwen3.8-27b")
MODEL_TEMPERATURE = 0.0
MODEL_MAX_RETRIES = 8
MODEL_TIMEOUT_SECONDS = 120
# Courtesy pause between live calls so a free-tier rate limit is not tripped.
MODEL_CALL_DELAY_SECONDS = float(os.environ.get("BOW_CALL_DELAY", "1.0"))
# Published Groq list prices, USD per million tokens, used only for reporting.
MODEL_PRICES_USD_PER_MTOK = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.75},
    "openai/gpt-oss-20b": {"input": 0.10, "output": 0.50},
    "qwen/qwen3.8-27b": {"input": 0.29, "output": 0.59},
    "qwen/qwen3.6-27b": {"input": 0.29, "output": 0.59},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
}

# Claude is the primary vision (OCR) reader and the income-stability
# classifier, used specifically where semantic judgment beats a numeric
# heuristic. Groq stays wired in as a second opinion on every image (see
# extraction.py) so a disagreement is visible rather than silent.
CLAUDE_VISION_MODEL = os.environ.get("BOW_CLAUDE_VISION_MODEL", "claude-opus-5")
CLAUDE_TEXT_MODEL = os.environ.get("BOW_CLAUDE_TEXT_MODEL", "claude-opus-5")
CLAUDE_MAX_RETRIES = 6
CLAUDE_TIMEOUT_SECONDS = 120
# Two vision readings on the same bill are treated as agreeing within this
# relative tolerance; a wider gap is logged to evaluation/ocr_audit.md.
OCR_AGREEMENT_TOLERANCE = 0.005


def api_key() -> str | None:
    """Secrets are read from the environment only, never from a tracked file."""
    return os.environ.get("GROQ_API_KEY") or None


def anthropic_api_key() -> str | None:
    """Same rule as api_key(): environment only, never a tracked file."""
    return os.environ.get("ANTHROPIC_API_KEY") or None


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader so a local run needs no extra dependency."""
    env_path = path or REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
