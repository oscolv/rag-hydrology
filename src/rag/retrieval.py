"""Hybrid retrieval pipeline: dense + BM25 with RRF fusion, multi-query, and reranking."""

import pickle  # nosec B403 — loaded file is gated by a magic header, see load_bm25_index
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_cohere import CohereRerank
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr
from rank_bm25 import BM25Okapi

from rag.config import Settings
from rag.factories import get_chat_llm, get_embeddings


class HybridRetriever(BaseRetriever):
    """Combines dense (ChromaDB) and sparse (BM25) retrieval with Reciprocal Rank Fusion."""

    vectorstore: Any = Field(exclude=True)
    dense_k: int = 20
    bm25_k: int = 20
    rrf_k: int = 60

    _bm25: BM25Okapi = PrivateAttr()
    _bm25_documents: list[Document] = PrivateAttr()

    def __init__(
        self,
        vectorstore: Chroma,
        bm25: BM25Okapi,
        bm25_documents: list[Document],
        dense_k: int = 20,
        bm25_k: int = 20,
        **kwargs: Any,
    ):
        super().__init__(
            vectorstore=vectorstore,
            dense_k=dense_k,
            bm25_k=bm25_k,
            **kwargs,
        )
        self._bm25 = bm25
        self._bm25_documents = bm25_documents

    def _dense_search(self, query: str) -> list[Document]:
        """Network-bound: ChromaDB similarity search."""
        return self.vectorstore.similarity_search(query, k=self.dense_k)

    def _sparse_search(self, query: str) -> list[Document]:
        """CPU-bound: BM25 scoring over the corpus."""
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        top_indices = bm25_scores.argsort()[-self.bm25_k :][::-1]
        return [self._bm25_documents[i] for i in top_indices]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        # Dense (network-bound) and BM25 (CPU-bound) are independent — run concurrently.
        # Two worker threads is enough; BM25 releases the GIL during numpy ops
        # inside rank_bm25, so it overlaps cleanly with the HTTP call.
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(self._dense_search, query)
            sparse_future = pool.submit(self._sparse_search, query)
            dense_results = dense_future.result()
            sparse_results = sparse_future.result()

        return self._rrf_merge(dense_results, sparse_results)

    def _rrf_merge(
        self,
        dense_results: list[Document],
        sparse_results: list[Document],
    ) -> list[Document]:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(dense_results):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(sparse_results):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0) + 1.0 / (self.rrf_k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys]

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """Create a unique key for a document based on content hash."""
        return f"{doc.metadata.get('source', '')}:{doc.metadata.get('page', '')}:{hash(doc.page_content[:200])}"


_BM25_MAGIC = b"RAG-BM25v1\n"


def load_bm25_index(bm25_path: Path) -> tuple[BM25Okapi, list[Document]]:
    """Load the serialized BM25 index.

    The on-disk format is a short magic header followed by a pickle payload.
    The header is a cheap integrity check: it won't stop an attacker who can
    write to the data directory, but it prevents accidental loading of
    arbitrary .pkl files into the pickle module.

    If you store this file in an untrusted location, re-generate it from your
    own PDFs rather than deserializing an inherited one.
    """
    with open(bm25_path, "rb") as f:
        header = f.read(len(_BM25_MAGIC))
        if header != _BM25_MAGIC:
            raise ValueError(
                f"BM25 index at {bm25_path} has an unrecognized format. "
                "Re-run `rag ingest --force` to rebuild it."
            )
        data = pickle.load(f)  # noqa: S301  # nosec B301 — magic header gates input
    return data["bm25"], data["documents"]


def build_retriever(settings: Settings) -> ContextualCompressionRetriever:
    """Build the full retrieval pipeline: hybrid search + multi-query + reranking."""
    embeddings = get_embeddings(settings)
    vectorstore = Chroma(
        persist_directory=str(settings.chroma_path),
        embedding_function=embeddings,
        collection_name=settings.domain.collection_name,
    )

    # Load BM25 index
    bm25, bm25_docs = load_bm25_index(settings.bm25_full_path)

    # Build hybrid retriever
    hybrid = HybridRetriever(
        vectorstore=vectorstore,
        bm25=bm25,
        bm25_documents=bm25_docs,
        dense_k=settings.retrieval.dense_k,
        bm25_k=settings.retrieval.bm25_k,
    )

    # Wrap with multi-query expansion
    if settings.retrieval.multi_query:
        llm = get_chat_llm(settings, temperature=0.0)
        base_retriever = MultiQueryRetriever.from_llm(
            retriever=hybrid,
            llm=llm,
        )
    else:
        base_retriever = hybrid

    # Add reranking
    reranker = CohereRerank(
        model="rerank-v3.5",
        top_n=settings.retrieval.rerank_top_k,
        cohere_api_key=settings.cohere_api_key,
    )

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )
