"""Hybrid retrieval pipeline: dense + BM25 with RRF fusion, multi-query, and reranking."""

import pickle
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from pydantic import Field, PrivateAttr
from rank_bm25 import BM25Okapi

from rag.config import Settings


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

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        # Dense retrieval
        dense_results = self.vectorstore.similarity_search(query, k=self.dense_k)

        # Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        top_indices = bm25_scores.argsort()[-self.bm25_k :][::-1]
        sparse_results = [self._bm25_documents[i] for i in top_indices]

        # Reciprocal Rank Fusion
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


def load_bm25_index(bm25_path: Path) -> tuple[BM25Okapi, list[Document]]:
    """Load the serialized BM25 index."""
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["documents"]


def build_retriever(settings: Settings) -> ContextualCompressionRetriever:
    """Build the full retrieval pipeline: hybrid search + multi-query + reranking."""
    # Load vector store
    embeddings = OpenAIEmbeddings(
        model=settings.llm.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
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
        llm_kwargs = {
            "model": settings.llm.model,
            "temperature": 0,
            "openai_api_key": settings.llm_api_key,
        }
        if settings.llm_base_url:
            llm_kwargs["openai_api_base"] = settings.llm_base_url
        llm = ChatOpenAI(**llm_kwargs)
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
