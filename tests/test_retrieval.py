"""Tests for the retrieval pipeline."""

from typing import Any
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.retrieval import HybridRetriever, ParentExpansionRetriever


def _make_doc(content: str, source: str = "test.pdf", page: int = 1) -> Document:
    return Document(page_content=content, metadata={"source": source, "page": page})


def test_rrf_merge():
    """Test Reciprocal Rank Fusion produces expected ordering."""
    from rank_bm25 import BM25Okapi

    dummy_bm25 = BM25Okapi([["dummy"]])
    dummy_docs = [_make_doc("dummy")]
    retriever = HybridRetriever(
        vectorstore=MagicMock(),
        bm25=dummy_bm25,
        bm25_documents=dummy_docs,
        dense_k=20,
        bm25_k=20,
    )

    # Doc A ranks #1 in dense, #3 in sparse
    # Doc B ranks #2 in dense, #1 in sparse
    # Doc C ranks #3 in dense, #2 in sparse
    doc_a = _make_doc("Doc A", "a.pdf", 1)
    doc_b = _make_doc("Doc B", "b.pdf", 1)
    doc_c = _make_doc("Doc C", "c.pdf", 1)

    dense = [doc_a, doc_b, doc_c]
    sparse = [doc_b, doc_c, doc_a]

    merged = retriever._rrf_merge(dense, sparse)

    # Doc B should rank highest (rank 2 + rank 1 gives best combined score)
    assert merged[0].page_content == "Doc B"
    assert len(merged) == 3


def test_doc_key_uniqueness():
    doc1 = _make_doc("Content A", "file1.pdf", 1)
    doc2 = _make_doc("Content B", "file1.pdf", 1)
    doc3 = _make_doc("Content A", "file1.pdf", 1)

    key1 = HybridRetriever._doc_key(doc1)
    key2 = HybridRetriever._doc_key(doc2)
    key3 = HybridRetriever._doc_key(doc3)

    assert key1 != key2  # Different content
    assert key1 == key3  # Same content and metadata


# ----------------------------------------------------------------------
# ParentExpansionRetriever
# ----------------------------------------------------------------------


class _StubRetriever(BaseRetriever):
    """Minimal retriever that returns a pre-set list of children."""

    docs: Any = None

    def _get_relevant_documents(self, query, *, run_manager=None):
        return list(self.docs or [])


def _child(content: str, parent_id: str) -> Document:
    return Document(
        page_content=content,
        metadata={"parent_id": parent_id, "is_child": True, "source": "x.pdf", "page": 1},
    )


def _parent(content: str, parent_id: str) -> Document:
    return Document(
        page_content=content,
        metadata={"parent_id": parent_id, "source": "x.pdf", "page": 1},
    )


def test_parent_expansion_substitutes_parents_and_dedupes():
    base = _StubRetriever(docs=[
        _child("c1a", "p1"),
        _child("c1b", "p1"),  # same parent → dedup
        _child("c2", "p2"),
    ])
    parents = {
        "p1": _parent("PARENT 1", "p1"),
        "p2": _parent("PARENT 2", "p2"),
    }
    retriever = ParentExpansionRetriever(base=base, parents=parents)
    out = retriever.invoke("ignored")

    assert [d.page_content for d in out] == ["PARENT 1", "PARENT 2"]


def test_parent_expansion_passes_through_children_with_unknown_parent_id():
    """Stale parents store: unknown IDs fall back to the child doc itself."""
    base = _StubRetriever(docs=[
        _child("orphan", "missing"),
        _child("c2", "p2"),
    ])
    retriever = ParentExpansionRetriever(
        base=base,
        parents={"p2": _parent("PARENT 2", "p2")},
    )
    out = retriever.invoke("ignored")

    assert [d.page_content for d in out] == ["orphan", "PARENT 2"]


def test_parent_expansion_passes_through_when_store_is_empty():
    children = [_child("c1", "p1"), _child("c2", "p2")]
    base = _StubRetriever(docs=children)
    retriever = ParentExpansionRetriever(base=base, parents={})
    out = retriever.invoke("ignored")

    assert out == children
