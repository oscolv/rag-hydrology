"""Input/output sanitization helpers.

These are the boundary-layer defenses for the RAG pipeline:

- `escape_braces`       protects prompt templates from `{var}` injection by
                        document content (LangChain uses str.format()).
- `clamp_text`          enforces hard length caps on user input and free-form
                        content before it reaches the LLM or retriever.
- `redact_secrets`      scrubs API-key-shaped tokens from strings destined for
                        logs.
- `safe_json_loads`     parses JSON defensively and returns a fallback on any
                        error (used for LLM-produced grader outputs).
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Prompt-template injection defense
# ---------------------------------------------------------------------------


def escape_braces(text: str) -> str:
    """Escape `{` and `}` so the string is safe to inject into a str.format() prompt.

    LangChain's ChatPromptTemplate interpolates variables with `.format()`. If
    a retrieved PDF chunk contains `{context}` literally, it will try to
    re-interpolate and either crash or leak unintended variables. Doubling the
    braces produces the original character after formatting.
    """
    return text.replace("{", "{{").replace("}", "}}")


# ---------------------------------------------------------------------------
# Input length caps
# ---------------------------------------------------------------------------

MAX_QUESTION_CHARS = 4000  # CLI user input
MAX_CHUNK_CHARS = 20_000  # upper bound for a single chunk before embedding
MAX_CONTEXT_CHARS = 120_000  # upper bound for concatenated retrieved context


def clamp_text(text: str, max_chars: int, *, ellipsis: str = "…") -> str:
    """Truncate text to max_chars, appending an ellipsis if it was cut."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(ellipsis)] + ellipsis


# ---------------------------------------------------------------------------
# Secret redaction for logs
# ---------------------------------------------------------------------------

# Matches OpenAI (`sk-...`), OpenRouter (`sk-or-v1-...`), Cohere (40+ hex/base64),
# and generic long token values after `api_key=`, `authorization:`, `token=`.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-(?:or-v1-|proj-|ant-api\d+-)?[A-Za-z0-9_-]{20,}"),
    re.compile(
        r"(api[_-]?key|authorization|bearer|token)\s*[:=]\s*['\"]?"
        r"([A-Za-z0-9_.\-]{20,})['\"]?",
        re.IGNORECASE,
    ),
)

_REDACTED = "***REDACTED***"


def redact_secrets(text: str) -> str:
    """Replace anything that looks like an API key or bearer token with a placeholder.

    Conservative: prefer false positives (a legitimate-looking 20+ char token
    near `api_key=` becomes `***REDACTED***`) over leaking credentials.
    """
    result = text
    # First pattern: bare sk- tokens
    result = _SECRET_PATTERNS[0].sub(_REDACTED, result)
    # Second pattern: key=value forms — preserve the key name, redact the value
    result = _SECRET_PATTERNS[1].sub(
        lambda m: f"{m.group(1)}={_REDACTED}", result,
    )
    return result


# ---------------------------------------------------------------------------
# Defensive JSON parsing for LLM-produced schema outputs
# ---------------------------------------------------------------------------


def safe_json_loads(text: str, fallback: T) -> Any | T:
    """Parse JSON from LLM output; return fallback on any failure.

    LLMs sometimes wrap JSON in ```json fences or prepend explanatory prose.
    This helper strips the most common decorators and falls back silently
    rather than raising into the generation path.
    """
    if not text:
        return fallback
    s = text.strip()
    # Strip markdown code fences
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s[3:]
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    # Best-effort trim: if there's leading prose, try to find the first { or [
    if s and s[0] not in "{[":
        for marker in ("{", "["):
            idx = s.find(marker)
            if idx >= 0:
                s = s[idx:]
                break
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return fallback
