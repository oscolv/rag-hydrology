"""Tests for the generation pipeline."""

from langchain_core.documents import Document

from rag.generation import format_documents


def test_format_documents():
    docs = [
        Document(
            page_content="GRACE measures gravity.",
            metadata={"source": "grace.pdf", "page": 5},
        ),
        Document(
            page_content="Water storage varies seasonally.",
            metadata={"source": "tws.pdf", "page": 12},
        ),
    ]

    formatted = format_documents(docs)

    assert "[Source: grace.pdf, Page: 5]" in formatted
    assert "[Source: tws.pdf, Page: 12]" in formatted
    assert "GRACE measures gravity." in formatted
    assert "---" in formatted


def test_format_documents_missing_metadata():
    docs = [Document(page_content="Some content.", metadata={})]
    formatted = format_documents(docs)
    assert "[Source: unknown, Page: ?]" in formatted


def test_format_documents_empty():
    assert format_documents([]) == ""
