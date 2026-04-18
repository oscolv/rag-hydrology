"""Cached factories for LLM and embedding clients.

ChatOpenAI/OpenAIEmbeddings reuse internal HTTP sessions, so recreating them
per call adds startup overhead. These helpers cache one client per unique
(model, key, base_url) signature. Safe to share across threads — the underlying
httpx client is thread-safe.
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.config import Settings


@lru_cache(maxsize=8)
def _chat_llm(
    model: str,
    temperature: float,
    api_key: str,
    base_url: str | None,
    max_retries: int,
    request_timeout: int | None,
) -> ChatOpenAI:
    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "openai_api_key": api_key,
        "max_retries": max_retries,
    }
    if base_url:
        kwargs["openai_api_base"] = base_url
    if request_timeout is not None:
        kwargs["request_timeout"] = request_timeout
    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=4)
def _embeddings(model: str, api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model, openai_api_key=api_key)


def get_chat_llm(settings: Settings, *, temperature: float | None = None) -> ChatOpenAI:
    """Return a shared ChatOpenAI client for the generation LLM."""
    return _chat_llm(
        model=settings.llm.model,
        temperature=settings.llm.temperature if temperature is None else temperature,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        max_retries=2,
        request_timeout=None,
    )


def get_context_llm(settings: Settings) -> ChatOpenAI:
    """Return a shared ChatOpenAI client for contextual retrieval generation."""
    model = settings.chunking.context_model or settings.llm.model
    return _chat_llm(
        model=model,
        temperature=0.0,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        max_retries=5,
        request_timeout=60,
    )


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """Return a shared OpenAIEmbeddings client."""
    return _embeddings(
        model=settings.llm.embedding_model,
        api_key=settings.openai_api_key,
    )


def clear_cache() -> None:
    """Drop all cached clients. Useful for tests."""
    _chat_llm.cache_clear()
    _embeddings.cache_clear()
