"""Centralized configuration using Pydantic Settings + YAML overrides."""

from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200


class RetrievalConfig(BaseModel):
    dense_k: int = 20
    bm25_k: int = 20
    rerank_top_k: int = 5
    multi_query: bool = True


class LLMConfig(BaseModel):
    model: str = "gpt-4o"
    temperature: float = 0.1
    embedding_model: str = "text-embedding-3-small"
    base_url: str | None = None  # For OpenRouter: https://openrouter.ai/api/v1


class EvalConfig(BaseModel):
    test_set_size: int = 30
    eval_model: str = "gpt-4o-mini"


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

    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    evaluation: EvalConfig = EvalConfig()

    @property
    def docs_path(self) -> Path:
        return self.project_root / self.docs_dir

    @property
    def chroma_path(self) -> Path:
        return self.project_root / self.chroma_dir

    @property
    def bm25_full_path(self) -> Path:
        return self.project_root / self.bm25_path


def _load_yaml_overrides(config_path: Path) -> dict:
    """Load YAML config file and return as flat dict for Settings overrides."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_settings(project_root: str = ".") -> Settings:
    """Load settings from .env + config.yaml with caching."""
    root = Path(project_root).resolve()
    yaml_data = _load_yaml_overrides(root / "config.yaml")

    # Build nested config objects from YAML
    kwargs: dict = {"project_root": root}
    for key in ("chunking", "retrieval", "llm", "evaluation"):
        if key in yaml_data:
            kwargs[key] = yaml_data[key]

    return Settings(
        _env_file=str(root / ".env"),
        **kwargs,
    )
