"""Tests for CollectionManager and legacy-layout detection."""

from pathlib import Path

import pytest

from rag.collections import CollectionManager, valid_collection_name
from rag.config import DEFAULT_COLLECTION, Settings


def _make_settings(root: Path) -> Settings:
    return Settings(project_root=root)


# ---------------------------------------------------------------------------
# valid_collection_name
# ---------------------------------------------------------------------------


def test_valid_name_accepts_slugs():
    assert valid_collection_name("hydro")
    assert valid_collection_name("law_es")
    assert valid_collection_name("ProjectX-2025")


def test_valid_name_rejects_bad_chars():
    assert not valid_collection_name("")
    assert not valid_collection_name("has space")
    assert not valid_collection_name("dot.name")
    assert not valid_collection_name("slash/name")
    assert not valid_collection_name("a" * 65)


# ---------------------------------------------------------------------------
# Create / list / delete on a fresh root
# ---------------------------------------------------------------------------


def test_create_collection_writes_metadata(tmp_path):
    mgr = CollectionManager(_make_settings(tmp_path))
    info = mgr.create("papers", display_name="Research Papers", description="PDFs")

    assert info.name == "papers"
    assert info.display_name == "Research Papers"
    assert (tmp_path / "data" / "collections" / "papers" / "metadata.json").exists()
    assert (tmp_path / "data" / "collections" / "papers" / "docs").is_dir()
    assert (tmp_path / "data" / "collections" / "papers" / "chroma").is_dir()


def test_create_rejects_invalid_name(tmp_path):
    mgr = CollectionManager(_make_settings(tmp_path))
    with pytest.raises(ValueError, match="Invalid collection name"):
        mgr.create("bad name")


def test_create_rejects_duplicate(tmp_path):
    mgr = CollectionManager(_make_settings(tmp_path))
    mgr.create("papers")
    with pytest.raises(ValueError, match="already exists"):
        mgr.create("papers")


def test_set_active_and_get_active(tmp_path):
    mgr = CollectionManager(_make_settings(tmp_path))
    mgr.create("papers")
    mgr.set_active("papers")
    assert mgr.get_active() == "papers"
    assert (tmp_path / "data" / ".active_collection").read_text().strip() == "papers"


def test_set_active_rejects_unknown(tmp_path):
    mgr = CollectionManager(_make_settings(tmp_path))
    with pytest.raises(ValueError, match="does not exist"):
        mgr.set_active("ghost")


def test_delete_removes_files_and_falls_back(tmp_path):
    mgr = CollectionManager(_make_settings(tmp_path))
    mgr.create("papers")
    mgr.set_active("papers")

    mgr.delete("papers")

    assert not (tmp_path / "data" / "collections" / "papers").exists()
    assert mgr.get_active() == DEFAULT_COLLECTION


def test_cannot_delete_default(tmp_path):
    mgr = CollectionManager(_make_settings(tmp_path))
    with pytest.raises(ValueError, match="Cannot delete"):
        mgr.delete(DEFAULT_COLLECTION)


def test_list_empty_root(tmp_path):
    mgr = CollectionManager(_make_settings(tmp_path))
    assert mgr.list() == []


def test_list_after_create(tmp_path):
    mgr = CollectionManager(_make_settings(tmp_path))
    mgr.create("a")
    mgr.create("b")
    names = [c.name for c in mgr.list()]
    assert names == ["a", "b"]


# ---------------------------------------------------------------------------
# Legacy layout detection
# ---------------------------------------------------------------------------


def test_legacy_default_detected_from_docs_dir(tmp_path):
    """Pre-multi-collection install: only `docs/*.pdf` present."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "paper1.pdf").write_bytes(b"%PDF-1.4\n")

    settings = _make_settings(tmp_path)
    mgr = CollectionManager(settings)

    assert mgr.exists(DEFAULT_COLLECTION)
    infos = mgr.list()
    assert len(infos) == 1
    assert infos[0].name == DEFAULT_COLLECTION
    assert infos[0].is_legacy
    assert infos[0].pdf_count == 1


def test_legacy_default_detected_from_chroma_dir(tmp_path):
    """Legacy layout with built chroma index."""
    (tmp_path / "data" / "chroma").mkdir(parents=True)
    (tmp_path / "data" / "chroma" / "chroma.sqlite3").write_bytes(b"sqlite")

    settings = _make_settings(tmp_path)
    mgr = CollectionManager(settings)

    assert mgr.exists(DEFAULT_COLLECTION)
    infos = mgr.list()
    assert any(c.name == DEFAULT_COLLECTION and c.is_legacy for c in infos)


def test_no_legacy_no_default(tmp_path):
    """Clean install: `default` doesn't pre-exist until user ingests."""
    settings = _make_settings(tmp_path)
    mgr = CollectionManager(settings)
    assert not mgr.exists(DEFAULT_COLLECTION)


def test_legacy_docs_path_resolves_to_top_level(tmp_path):
    """When in legacy mode, Settings.docs_path must point at the top-level `docs/`."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "p.pdf").write_bytes(b"%PDF-1.4\n")

    settings = _make_settings(tmp_path)
    assert settings._use_legacy()
    assert settings.docs_path == tmp_path / "docs"


def test_non_default_collection_uses_new_layout(tmp_path):
    """Legacy default coexists with new named collections without interference."""
    # Seed legacy default.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "p.pdf").write_bytes(b"%PDF-1.4\n")

    mgr = CollectionManager(_make_settings(tmp_path))
    mgr.create("papers")
    mgr.set_active("papers")

    # When `papers` is active, paths resolve under the new layout — not legacy.
    active_settings = _make_settings(tmp_path).model_copy(
        update={"active_collection": "papers"}
    )
    assert not active_settings._use_legacy()
    assert active_settings.docs_path == tmp_path / "data" / "collections" / "papers" / "docs"

    # And the default legacy entry still surfaces in list().
    names = {c.name: c for c in mgr.list()}
    assert "default" in names and names["default"].is_legacy
    assert "papers" in names and not names["papers"].is_legacy
