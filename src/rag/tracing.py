"""Optional Langfuse-backed tracing for the RAG pipeline.

Activated by setting both ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY``
in the environment (and optionally ``LANGFUSE_HOST`` for self-hosted instances;
defaults to https://cloud.langfuse.com). When unset, every helper here
returns a no-op so the rest of the codebase can call ``trace_config(...)``
unconditionally without paying any cost.

Usage:

    from rag.tracing import trace_config
    cfg = trace_config("rag.query", {"request_id": rid, "collection": col})
    docs = retriever.invoke(question, config=cfg)
    answer = llm.invoke(messages, config=cfg)

A single ``CallbackHandler`` is reused across calls; Langfuse's batched
exporter handles flushing on its own thread.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from rag.logging_setup import get_logger

log = get_logger(__name__)

_DEFAULT_HOST = "https://cloud.langfuse.com"

_lock = threading.Lock()
_handler: Any = None
_handler_init_failed = False


def is_enabled() -> bool:
    """True when both Langfuse keys are present in the environment.

    Read fresh each call (don't cache) so toggling env vars at runtime
    (e.g. tests with monkeypatch) takes effect immediately.
    """
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def host() -> str:
    """The configured Langfuse host (used for the health endpoint UI hint)."""
    return os.environ.get("LANGFUSE_HOST", _DEFAULT_HOST)


def _build_handler() -> Any | None:
    """Lazily build a singleton CallbackHandler.

    Failures (missing dependency, network probe error during init) are logged
    once and the handler is set to ``None`` permanently for this process —
    we never want tracing to break a query.
    """
    global _handler, _handler_init_failed
    if _handler is not None or _handler_init_failed:
        return _handler
    if not is_enabled():
        return None
    with _lock:
        if _handler is not None or _handler_init_failed:
            return _handler
        try:
            from langfuse.langchain import CallbackHandler
            _handler = CallbackHandler()
            log.info("tracing.enabled", extra={"host": host()})
        except Exception as e:
            _handler_init_failed = True
            log.warning("tracing.init_failed", extra={"err": str(e)})
            _handler = None
        return _handler


def get_callbacks() -> list[Any]:
    """Return the Langfuse callback list, or [] when tracing is off."""
    handler = _build_handler()
    return [handler] if handler is not None else []


def trace_config(name: str, metadata: dict | None = None) -> dict:
    """Build a RunnableConfig dict for LangChain invokes.

    Returns an empty dict when tracing is disabled — passing ``config={}`` to
    LangChain is a safe no-op, so callers don't need a separate code path.

    The optional ``metadata`` dict is attached to the trace; ``request_id``,
    ``collection``, and ``model`` are typical values worth carrying through
    so traces are searchable from the feedback log.
    """
    cbs = get_callbacks()
    if not cbs:
        return {}
    cfg: dict[str, Any] = {"callbacks": cbs, "run_name": name}
    if metadata:
        cfg["metadata"] = metadata
    return cfg


def reset_for_tests() -> None:
    """Drop the cached handler. Test-only — production code should never call this."""
    global _handler, _handler_init_failed
    with _lock:
        _handler = None
        _handler_init_failed = False
