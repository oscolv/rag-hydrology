"""Tests for the ingestion pipeline."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

from rag.config import Settings, ChunkingConfig
from rag.ingest import (
    _file_hash,
    _extract_year,
    _detect_language,
    _extract_section_header,
    build_chunks,
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
