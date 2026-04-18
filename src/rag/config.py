"""Centralized configuration using Pydantic Settings + YAML overrides."""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_COLLECTION = "default"
COLLECTIONS_DIR = "collections"
ACTIVE_FILE = ".active_collection"


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    semantic: bool = False  # Semantic chunking via embedding similarity
    similarity_threshold: float = 0.82  # Breakpoint threshold for semantic chunking
    contextual_retrieval: bool = False  # LLM-generated context per chunk (Anthropic technique)
    context_model: str | None = None  # Model for context generation (None = use llm.model)
    context_workers: int = 8  # Parallel threads for contextual retrieval LLM calls


class RetrievalConfig(BaseModel):
    dense_k: int = 20
    bm25_k: int = 20
    rerank_top_k: int = 5
    multi_query: bool = True
    self_rag: bool = False  # Self-RAG: document grading + hallucination check + retry
    self_rag_max_retries: int = 2  # Max retrieval retries when context is irrelevant


class LLMConfig(BaseModel):
    model: str = "gpt-4o"
    temperature: float = 0.1
    embedding_model: str = "text-embedding-3-small"
    base_url: str | None = None  # For OpenRouter: https://openrouter.ai/api/v1


class EvalConfig(BaseModel):
    test_set_size: int = 30
    eval_model: str = "gpt-4o-mini"  # Must be OpenAI model (RAGAS requires strict JSON parsing)


class DomainConfig(BaseModel):
    """Domain customization — makes the RAG usable for any PDF corpus."""

    name: str = "documents"  # Short label shown in CLI greetings
    collection_name: str = "rag_docs"  # ChromaDB collection name
    system_prompt: str = (
        "You are a research assistant. Answer questions using ONLY the provided "
        "context.\n\n"
        "Rules:\n"
        "- Cite sources inline using [N] where N matches the bracketed number "
        "at the start of each context block (e.g. 'GRACE measures gravity [1][3]')\n"
        "- Do NOT invent citation numbers — only cite blocks you actually used\n"
        "- If the context doesn't contain the answer, say so explicitly\n"
        "- For Spanish-language sources, translate relevant content when answering "
        "in English\n"
        "- Preserve technical accuracy, especially for equations, units, and "
        "measurements\n"
        "- Structure longer answers with clear paragraphs\n\n"
        "Context:\n{context}"
    )
    grader_description: str = "a research RAG system"  # Filled into the Self-RAG grader prompt
    example_queries: list[str] = [
        "What are the main findings of these documents?",
        "Summarize the key methodology used.",
    ]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    openai_api_key: str = ""
    cohere_api_key: str = ""
    openrouter_api_key: str = ""

    @property
    def llm_api_key(self) -> str:
        """Return the API key for the configured LLM provider."""
        if self.llm.base_url and "openrouter" in (self.llm.base_url or ""):
            return self.openrouter_api_key
        return self.openai_api_key

    @property
    def llm_base_url(self) -> str | None:
        """Return the base URL for the configured LLM provider."""
        return self.llm.base_url

    project_root: Path = Path(".")
    docs_dir: str = "docs"
    chroma_dir: str = "data/chroma"
    bm25_path: str = "data/bm25_index.pkl"
    data_dir: str = "data"

    # Active collection. Overridable via env RAG_COLLECTION or CLI --collection.
    # "default" maps to the legacy layout (data/chroma, data/bm25_index.pkl, docs/)
    # when those paths exist — keeps existing users' indices untouched.
    active_collection: str = DEFAULT_COLLECTION

    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    evaluation: EvalConfig = EvalConfig()
    domain: DomainConfig = DomainConfig()

    # ----- Collection-aware path resolution -----

    @property
    def data_path(self) -> Path:
        return self.project_root / self.data_dir

    @property
    def collections_root(self) -> Path:
        return self.data_path / COLLECTIONS_DIR

    @property
    def active_file_path(self) -> Path:
        return self.data_path / ACTIVE_FILE

    def _collection_dir(self, name: str) -> Path:
        return self.collections_root / name

    @property
    def _legacy_chroma(self) -> Path:
        return self.project_root / self.chroma_dir

    @property
    def _legacy_bm25(self) -> Path:
        return self.project_root / self.bm25_path

    @property
    def _legacy_docs(self) -> Path:
        return self.project_root / self.docs_dir

    def _use_legacy(self) -> bool:
        """True when active collection is 'default' and new layout doesn't exist.

        Lets existing users keep their pre-multi-collection index without
        manual migration. When they create a new collection, the new
        data/collections/<name>/ layout is used from then on.
        """
        if self.active_collection != DEFAULT_COLLECTION:
            return False
        new_path = self._collection_dir(DEFAULT_COLLECTION)
        return not new_path.exists() and (
            self._legacy_chroma.exists()
            or self._legacy_bm25.exists()
            or any(self._legacy_docs.glob("*.pdf"))
            if self._legacy_docs.exists() else False
        )

    @property
    def docs_path(self) -> Path:
        if self._use_legacy():
            return self._legacy_docs
        return self._collection_dir(self.active_collection) / "docs"

    @property
    def chroma_path(self) -> Path:
        if self._use_legacy():
            return self._legacy_chroma
        return self._collection_dir(self.active_collection) / "chroma"

    @property
    def bm25_full_path(self) -> Path:
        if self._use_legacy():
            return self._legacy_bm25
        return self._collection_dir(self.active_collection) / "bm25.pkl"

    @property
    def metrics_db_path(self) -> Path:
        return self.data_path / "metrics.sqlite3"


def _load_yaml_overrides(config_path: Path) -> dict:
    """Load YAML config file and return as flat dict for Settings overrides."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _resolve_active_collection(root: Path) -> str:
    """Resolve active collection: env var > marker file > 'default'."""
    env_val = os.environ.get("RAG_COLLECTION")
    if env_val:
        return env_val.strip()
    marker = root / "data" / ACTIVE_FILE
    if marker.exists():
        try:
            value = marker.read_text().strip()
            if value:
                return value
        except OSError:
            pass
    return DEFAULT_COLLECTION


@lru_cache
def get_settings(project_root: str = ".") -> Settings:
    """Load settings from .env + config.yaml with caching."""
    root = Path(project_root).resolve()
    yaml_data = _load_yaml_overrides(root / "config.yaml")

    kwargs: dict = {
        "project_root": root,
        "active_collection": _resolve_active_collection(root),
    }
    for key in ("chunking", "retrieval", "llm", "evaluation", "domain"):
        if key in yaml_data:
            kwargs[key] = yaml_data[key]

    return Settings(
        _env_file=str(root / ".env"),
        **kwargs,
    )
