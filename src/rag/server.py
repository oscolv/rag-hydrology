"""FastAPI server for the RAG system.

Provides:
- POST /api/query           — SSE streaming of the RAG answer
- GET  /api/collections     — list collections
- POST /api/collections     — create
- POST /api/collections/{name}/activate
- DELETE /api/collections/{name}
- GET  /api/stats           — query metrics
- GET  /api/health          — health check
- Static files under /      — web UI

The retrieval pipeline is cached per (collection_name, config_fingerprint) in
memory, so repeat queries for the same collection skip cold start (~3-5s on
first call, ~0ms after). Ingestion or config changes invalidate the cache.
"""

import hashlib
import json
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.collections import CollectionManager, valid_collection_name
from rag.config import Settings, get_settings
from rag.generation import build_rag_chain_streaming, extract_citation_numbers
from rag.logging_setup import get_logger
from rag.metrics import MetricsStore
from rag.retrieval import build_retriever

log = get_logger(__name__)

WEB_DIR = Path(__file__).parent / "web"


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------


@dataclass
class CachedPipeline:
    collection: str
    fingerprint: str
    retriever: object
    chain: object
    built_at: float = field(default_factory=time.time)


class PipelineCache:
    """Thread-safe cache of built retrievers/chains per collection.

    Keyed by (collection_name, config_fingerprint). If the user changes
    retrieval or LLM settings in config.yaml, the fingerprint changes and the
    next request builds a fresh pipeline.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._cache: dict[str, CachedPipeline] = {}

    @staticmethod
    def _fingerprint(settings: Settings) -> str:
        parts = [
            settings.llm.model,
            settings.llm.embedding_model,
            str(settings.llm.base_url or ""),
            str(settings.retrieval.dense_k),
            str(settings.retrieval.bm25_k),
            str(settings.retrieval.rerank_top_k),
            str(settings.retrieval.multi_query),
            str(settings.retrieval.self_rag),
            str(settings.retrieval.self_rag_max_retries),
            settings.domain.collection_name,
        ]
        return hashlib.md5(
            "|".join(parts).encode(), usedforsecurity=False,
        ).hexdigest()

    def get_or_build(self, settings: Settings) -> CachedPipeline:
        fp = self._fingerprint(settings)
        key = f"{settings.active_collection}:{fp}"
        with self._lock:
            hit = self._cache.get(key)
            if hit:
                return hit

            retriever = build_retriever(settings)
            chain = build_rag_chain_streaming(retriever, settings)
            entry = CachedPipeline(
                collection=settings.active_collection,
                fingerprint=fp,
                retriever=retriever,
                chain=chain,
            )
            self._cache[key] = entry
            log.info(
                "server.pipeline.built",
                extra={"collection": settings.active_collection, "fp": fp[:8]},
            )
            return entry

    def invalidate(self, collection: str | None = None) -> None:
        with self._lock:
            if collection is None:
                self._cache.clear()
            else:
                stale = [k for k in self._cache if k.startswith(f"{collection}:")]
                for k in stale:
                    del self._cache[k]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str
    collection: str | None = None  # override active collection for this call


class CollectionCreate(BaseModel):
    name: str
    display_name: str | None = None
    description: str = ""


def _settings_for_collection(
    project_root: str, collection: str | None,
) -> Settings:
    """Load settings and optionally override the active collection."""
    settings = get_settings(project_root)
    if collection and collection != settings.active_collection:
        # Build a fresh Settings instance with the override (don't mutate the
        # cached one — other requests may still use the default).
        settings = settings.model_copy(update={"active_collection": collection})
    return settings


def _doc_to_dict(doc) -> dict:
    return {
        "source": doc.metadata.get("source", "unknown"),
        "page": doc.metadata.get("page", "?"),
        "title": doc.metadata.get("title", ""),
        "section": doc.metadata.get("section", ""),
        "year": doc.metadata.get("year", ""),
        "content": doc.page_content,
    }


def _sse(event: str, data: dict) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode()


def _stream_to_sse(
    iterator: Iterator[dict],
    metrics: MetricsStore,
    collection: str,
    question: str,
) -> Iterator[bytes]:
    start = time.time()
    token_count = 0
    done_payload: dict | None = None
    error: str | None = None

    try:
        for event in iterator:
            kind = event.get("event", "message")
            if kind == "token":
                token_count += 1
                yield _sse("token", {"content": event.get("content", ""),
                                      "regenerated": event.get("regenerated", False)})
            elif kind == "sources":
                docs = event.get("documents", [])
                yield _sse("sources", {
                    "documents": [_doc_to_dict(d) for d in docs],
                })
            elif kind == "reflection":
                yield _sse("reflection", {"step": event.get("step", {})})
            elif kind == "retrieval_start":
                yield _sse("retrieval_start", {"request_id": event.get("request_id", "")})
            elif kind == "regenerating":
                yield _sse("regenerating", {})
            elif kind == "done":
                done_payload = event
                docs = event.get("source_documents", [])
                citations = extract_citation_numbers(event.get("answer", ""))
                yield _sse("done", {
                    "answer": event.get("answer", ""),
                    "documents": [_doc_to_dict(d) for d in docs],
                    "reflection": event.get("reflection", []),
                    "citations": citations,
                    "request_id": event.get("request_id", ""),
                })
            elif kind == "error":
                error = event.get("message", "unknown error")
                yield _sse("error", {"message": error})
    except Exception as e:  # last-resort safety net
        error = str(e)
        yield _sse("error", {"message": error})

    latency_ms = int((time.time() - start) * 1000)
    try:
        metrics.record_query(
            collection=collection,
            question=question,
            latency_ms=latency_ms,
            answer_len=len(done_payload.get("answer", "")) if done_payload else 0,
            token_count=token_count,
            doc_count=len(done_payload.get("source_documents", [])) if done_payload else 0,
            error=error,
        )
    except Exception as e:
        log.warning("server.metrics.record_failed", extra={"err": str(e)})


def create_app(project_root: str = ".") -> FastAPI:
    """Build the FastAPI app. project_root is where config.yaml / .env live."""
    cache = PipelineCache()

    # Resolve project root eagerly so settings loading uses an absolute path.
    root = str(Path(project_root).resolve())
    base_settings = get_settings(root)
    metrics = MetricsStore(base_settings.metrics_db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        metrics.ensure_schema()
        log.info("server.startup", extra={"project_root": root})
        yield
        log.info("server.shutdown")

    app = FastAPI(
        title="RAG API",
        version="1.0",
        lifespan=lifespan,
    )

    # ---------- Collections ----------

    @app.get("/api/collections")
    def list_collections():
        settings = get_settings(root)
        mgr = CollectionManager(settings)
        infos = mgr.list()
        return {
            "active": mgr.get_active(),
            "collections": [
                {
                    "name": c.name,
                    "display_name": c.display_name,
                    "description": c.description,
                    "created_at": c.created_at,
                    "is_active": c.is_active,
                    "is_legacy": c.is_legacy,
                    "pdf_count": c.pdf_count,
                    "has_index": c.has_index,
                }
                for c in infos
            ],
        }

    @app.post("/api/collections")
    def create_collection(body: CollectionCreate):
        if not valid_collection_name(body.name):
            raise HTTPException(400, "Invalid collection name")
        settings = get_settings(root)
        mgr = CollectionManager(settings)
        if mgr.exists(body.name):
            raise HTTPException(409, "Collection already exists")
        info = mgr.create(body.name, body.display_name, body.description)
        return {"name": info.name, "display_name": info.display_name}

    @app.post("/api/collections/{name}/activate")
    def activate_collection(name: str):
        settings = get_settings(root)
        mgr = CollectionManager(settings)
        if not mgr.exists(name):
            raise HTTPException(404, "Collection not found")
        mgr.set_active(name)
        # Invalidate any pipeline cached under other fingerprints for safety.
        cache.invalidate()
        get_settings.cache_clear()
        return {"active": name}

    @app.delete("/api/collections/{name}")
    def delete_collection(name: str):
        settings = get_settings(root)
        mgr = CollectionManager(settings)
        try:
            mgr.delete(name)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        cache.invalidate(name)
        return {"deleted": name}

    # ---------- Query (SSE) ----------

    @app.post("/api/query")
    def query(body: QueryRequest):
        settings = _settings_for_collection(root, body.collection)
        mgr = CollectionManager(settings)
        if not mgr.exists(settings.active_collection):
            raise HTTPException(404, f"Collection '{settings.active_collection}' not found")
        if not settings.chroma_path.exists() or not settings.bm25_full_path.exists():
            raise HTTPException(
                409,
                f"Collection '{settings.active_collection}' has no index. "
                "Run `rag ingest` for this collection first.",
            )

        pipeline = cache.get_or_build(settings)
        stream = pipeline.chain(body.question)
        return StreamingResponse(
            _stream_to_sse(stream, metrics, settings.active_collection, body.question),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---------- Stats ----------

    @app.get("/api/stats")
    def stats(limit: int = 50, collection: str | None = None):
        return metrics.summary(limit=limit, collection=collection)

    @app.post("/api/pipeline/invalidate")
    def invalidate(collection: str | None = None):
        cache.invalidate(collection)
        return {"invalidated": collection or "all"}

    # ---------- Health ----------

    @app.get("/api/health")
    def health():
        settings = get_settings(root)
        return {
            "ok": True,
            "active_collection": settings.active_collection,
            "model": settings.llm.model,
        }

    # ---------- Static web UI ----------

    if WEB_DIR.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")

        @app.get("/")
        def index():
            index_file = WEB_DIR / "index.html"
            if not index_file.exists():
                return JSONResponse({"error": "Web UI not built"}, status_code=500)
            return FileResponse(index_file)

    return app
