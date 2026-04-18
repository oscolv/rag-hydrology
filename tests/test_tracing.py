"""Tests for the tracing module — env-driven on/off, no-op when disabled.

These tests never hit the Langfuse network: they verify the module's contract
(``is_enabled``, ``trace_config``, ``get_callbacks``) so the rest of the
codebase can call ``trace_config(...)`` unconditionally.
"""

import pytest

from rag import tracing


@pytest.fixture(autouse=True)
def _isolate_handler():
    """Each test starts with a fresh handler cache and clean env."""
    tracing.reset_for_tests()
    yield
    tracing.reset_for_tests()


def test_is_enabled_false_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert tracing.is_enabled() is False


def test_is_enabled_requires_both_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert tracing.is_enabled() is False

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    assert tracing.is_enabled() is True


def test_get_callbacks_empty_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert tracing.get_callbacks() == []


def test_trace_config_returns_empty_dict_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert tracing.trace_config("rag.query") == {}
    # passing config={} to LangChain runnables is a safe no-op — that's the contract


def test_host_defaults_to_cloud(monkeypatch):
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    assert tracing.host() == "https://cloud.langfuse.com"


def test_host_honors_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    assert tracing.host() == "http://localhost:3000"


def test_trace_config_includes_callbacks_and_metadata_when_enabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")

    cfg = tracing.trace_config("rag.query", {"request_id": "abc", "collection": "c"})
    assert "callbacks" in cfg
    assert len(cfg["callbacks"]) == 1
    assert cfg["run_name"] == "rag.query"
    assert cfg["metadata"] == {"request_id": "abc", "collection": "c"}


def test_handler_is_singleton_within_process(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

    cbs1 = tracing.get_callbacks()
    cbs2 = tracing.get_callbacks()
    assert cbs1 and cbs2
    assert cbs1[0] is cbs2[0]


def test_init_failure_is_swallowed(monkeypatch):
    """If the Langfuse SDK errors during init, tracing must degrade silently."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

    # Force the import inside _build_handler() to fail
    import sys
    sentinel = sys.modules.pop("langfuse.langchain", None)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", None)
    try:
        cbs = tracing.get_callbacks()
        assert cbs == []
        # subsequent calls also stay empty (failure is sticky for the process)
        assert tracing.get_callbacks() == []
    finally:
        if sentinel is not None:
            sys.modules["langfuse.langchain"] = sentinel
        else:
            sys.modules.pop("langfuse.langchain", None)
