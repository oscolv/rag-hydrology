"""Tests for the ingestion pipeline."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag.config import ChunkingConfig, Settings
from rag.ingest import (
    _detect_language,
    _enforce_chunk_sizes,
    _extract_section_header,
    _extract_year,
    _make_parent_id,
    _split_sentences,
    build_chunks,
    load_parents_index,
    save_parents_index,
    to_children,
)


def test_extract_year_from_filename():
    assert _extract_year("gleeson2015.pdf", "") == "2015"
    assert _extract_year("landerer2012.pdf", "") == "2012"
    assert _extract_year("19740022614.pdf", "") == "1974"


def test_extract_year_from_content():
    assert _extract_year("unknown.pdf", "Published in 2023 by Elsevier") == "2023"


def test_extract_year_unknown():
    assert _extract_year("nodate.pdf", "no date here") == "unknown"


def test_detect_language_english():
    text = "The GRACE satellite mission measures gravity field variations"
    assert _detect_language(text) == "en"


def test_detect_language_spanish():
    text = "El atlas del agua en México presenta los datos de la disponibilidad del recurso hídrico en las regiones del país"
    assert _detect_language(text) == "es"


def test_extract_section_header():
    text = "# Introduction\nThis paper presents..."
    assert _extract_section_header(text) == "Introduction"


def test_extract_section_header_empty():
    text = "No headers here, just text."
    assert _extract_section_header(text) == ""


def test_build_chunks():
    pages = [
        {
            "text": "# Title\n\nThis is a test document about GRACE satellite.\n\n## Methods\n\nWe used data from 2002 to 2020.",
            "metadata": {"page": 0},
        }
    ]
    settings = MagicMock(spec=Settings)
    settings.chunking = ChunkingConfig(chunk_size=200, chunk_overlap=50)

    chunks = build_chunks(pages, "test.pdf", "abc123", settings)

    assert len(chunks) > 0
    assert all(isinstance(c, Document) for c in chunks)
    assert all(c.metadata["source"] == "test.pdf" for c in chunks)
    assert all("[From:" in c.page_content for c in chunks)


def test_split_sentences():
    text = "GRACE measures gravity. The mission launched in 2002. Data is available globally."
    sentences = _split_sentences(text)
    assert len(sentences) >= 2
    assert any("GRACE" in s for s in sentences)


def test_enforce_chunk_sizes_merge_small():
    chunks = ["Hi.", "Short.", "This is a longer chunk that should stay separate on its own."]
    result = _enforce_chunk_sizes(chunks, min_size=20, max_size=500)
    # Small chunks should be merged
    assert len(result) <= len(chunks)
    assert all(len(c) > 0 for c in result)


def test_enforce_chunk_sizes_split_large():
    large = "word " * 500  # ~2500 chars
    result = _enforce_chunk_sizes([large], min_size=50, max_size=200)
    assert len(result) > 1
    assert all(len(c) <= 250 for c in result)  # Allow some slack from splitter


def test_build_chunks_metadata():
    pages = [
        {
            "text": "# GRACE Mission Overview\n\nLaunched in 2002.",
            "metadata": {"page": 0},
        }
    ]
    settings = MagicMock(spec=Settings)
    settings.chunking = ChunkingConfig(chunk_size=500, chunk_overlap=100)

    chunks = build_chunks(pages, "grace2002.pdf", "hash123", settings)

    assert chunks[0].metadata["year"] == "2002"
    assert chunks[0].metadata["language"] == "en"
    assert chunks[0].metadata["file_hash"] == "hash123"
    assert chunks[0].metadata["page"] == 1  # 1-indexed


# ----------------------------------------------------------------------
# Parent-child (small-to-big) helpers
# ----------------------------------------------------------------------


def _make_doc(content: str, **meta) -> Document:
    base = {"file_hash": "abc", "page": 1, "source": "x.pdf"}
    base.update(meta)
    return Document(page_content=content, metadata=base)


def test_make_parent_id_deterministic():
    a = _make_doc("Some content here that is long enough to fingerprint.")
    b = _make_doc("Some content here that is long enough to fingerprint.")
    assert _make_parent_id(a) == _make_parent_id(b)


def test_make_parent_id_differs_for_different_content():
    a = _make_doc("Content A")
    b = _make_doc("Content B")
    assert _make_parent_id(a) != _make_parent_id(b)


def test_make_parent_id_differs_for_different_pages():
    a = _make_doc("Same body", page=1)
    b = _make_doc("Same body", page=2)
    assert _make_parent_id(a) != _make_parent_id(b)


def test_to_children_inherits_metadata_and_marks_children():
    settings = MagicMock(spec=Settings)
    settings.chunking = ChunkingConfig(child_chunk_size=80, child_chunk_overlap=10)

    long_text = " ".join(["GRACE measures gravity from orbit."] * 30)  # ~1000 chars
    parent = _make_doc(long_text, page=4, source="paper.pdf")
    parent.metadata["parent_id"] = _make_parent_id(parent)

    children = to_children([parent], settings)

    assert len(children) > 1
    for child in children:
        assert child.metadata["parent_id"] == parent.metadata["parent_id"]
        assert child.metadata["is_child"] is True
        assert child.metadata["source"] == "paper.pdf"
        assert child.metadata["page"] == 4


def test_to_children_skips_parents_without_id():
    settings = MagicMock(spec=Settings)
    settings.chunking = ChunkingConfig(child_chunk_size=80, child_chunk_overlap=10)

    parent_no_id = _make_doc("body without an id assigned")
    children = to_children([parent_no_id], settings)
    assert children == []


def test_parents_index_round_trip(tmp_path):
    parent = _make_doc("Body", page=2)
    parent.metadata["parent_id"] = _make_parent_id(parent)

    path = tmp_path / "parents.pkl"
    save_parents_index(path, [parent])

    loaded = load_parents_index(path)
    assert parent.metadata["parent_id"] in loaded
    assert loaded[parent.metadata["parent_id"]].page_content == "Body"


def test_load_parents_index_returns_empty_when_missing(tmp_path):
    assert load_parents_index(tmp_path / "absent.pkl") == {}


def test_load_parents_index_rejects_unknown_format(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not-the-magic-header")
    with pytest.raises(ValueError, match="unrecognized format"):
        load_parents_index(bad)
