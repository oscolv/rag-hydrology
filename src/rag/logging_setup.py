"""Structured logging with automatic secret redaction.

The CLI uses Rich for human-facing output, so this module is for machine-
readable diagnostic logs (to file or stderr) used in CI and production. A
logging filter pipes every message through `redact_secrets` before it is
emitted, which means even an accidental `logger.info(f"key={settings.llm_api_key}")`
cannot leak a credential.

Usage:

    from rag.logging_setup import get_logger
    log = get_logger(__name__)
    log.info("retrieval.start", extra={"query_len": len(q), "rid": rid})
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from rag.sanitize import redact_secrets

_CONFIGURED = False


class _RedactFilter(logging.Filter):
    """Pass every LogRecord message (and exception text) through redact_secrets."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_secrets(str(record.msg))
            if record.args:
                # Arguments get formatted into the message; redact each one.
                record.args = tuple(
                    redact_secrets(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
            if record.exc_text:
                record.exc_text = redact_secrets(record.exc_text)
        except Exception:  # noqa: S110 — never break logging
            pass
        return True


class _JsonFormatter(logging.Formatter):
    """One JSON object per line: suitable for `jq` and log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach any structured fields passed via `extra=` that aren't standard.
        skip = set(logging.LogRecord.__dict__.keys()) | {"message", "asctime"}
        for key, value in record.__dict__.items():
            if key in skip or key.startswith("_"):
                continue
            if key in payload:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = str(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | int | None = None) -> None:
    """Install the redact filter and JSON formatter on the root logger.

    Idempotent — calling it twice is a no-op. Level can also be set via the
    `RAG_LOG_LEVEL` env var (default: WARNING to keep the CLI quiet).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        level = os.environ.get("RAG_LOG_LEVEL", "WARNING")

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_RedactFilter())

    # Remove default handlers so we don't double-log.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring the root once on first call."""
    configure_logging()
    return logging.getLogger(name)


def new_request_id() -> str:
    """Short opaque id to correlate log lines for a single query."""
    return uuid.uuid4().hex[:12]
