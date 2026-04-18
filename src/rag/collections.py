"""Multi-collection management: create, list, switch, delete.

Each collection is an independent corpus with its own PDFs, ChromaDB,
and BM25 index. The active collection is persisted to `data/.active_collection`
and used by all CLI commands unless overridden via `--collection` or
`RAG_COLLECTION` env var.

Legacy layout (pre-multi-collection installs) is preserved: if the user has
`data/chroma` and `docs/` populated and no `data/collections/default/` yet,
the "default" collection transparently uses those legacy paths.
"""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rag.config import DEFAULT_COLLECTION, Settings

COLLECTION_NAME_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"


def valid_collection_name(name: str) -> bool:
    """Collection names must be filesystem-safe slugs."""
    if not name or len(name) > 64:
        return False
    return all(c in COLLECTION_NAME_CHARS for c in name)


@dataclass
class CollectionInfo:
    name: str
    display_name: str
    description: str = ""
    created_at: str = ""
    is_active: bool = False
    is_legacy: bool = False
    path: Path = field(default_factory=Path)
    pdf_count: int = 0
    has_index: bool = False


class CollectionManager:
    """Encapsulates collection directory layout and active-collection state."""

    def __init__(self, settings: Settings):
        self.settings = settings

    # ----- Active collection persistence -----

    def get_active(self) -> str:
        if self.settings.active_file_path.exists():
            value = self.settings.active_file_path.read_text().strip()
            if value:
                return value
        return DEFAULT_COLLECTION

    def set_active(self, name: str) -> None:
        if not self.exists(name):
            raise ValueError(f"Collection '{name}' does not exist")
        path = self.settings.active_file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)

    # ----- CRUD -----

    def exists(self, name: str) -> bool:
        if name == DEFAULT_COLLECTION and self._has_legacy_default():
            return True
        return self._collection_path(name).exists()

    def create(
        self,
        name: str,
        display_name: str | None = None,
        description: str = "",
    ) -> CollectionInfo:
        if not valid_collection_name(name):
            raise ValueError(
                f"Invalid collection name '{name}'. Use letters, digits, _ or -."
            )
        if self.exists(name):
            raise ValueError(f"Collection '{name}' already exists")

        base = self._collection_path(name)
        (base / "docs").mkdir(parents=True, exist_ok=True)
        (base / "chroma").mkdir(parents=True, exist_ok=True)

        meta = {
            "name": name,
            "display_name": display_name or name,
            "description": description,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (base / "metadata.json").write_text(json.dumps(meta, indent=2))
        return self._read_info(name)

    def delete(self, name: str) -> None:
        if name == DEFAULT_COLLECTION:
            raise ValueError(
                "Cannot delete the 'default' collection. "
                "Switch to another collection and remove its files manually if needed."
            )
        if not self.exists(name):
            raise ValueError(f"Collection '{name}' does not exist")

        path = self._collection_path(name)
        if path.exists():
            shutil.rmtree(path)

        if self.get_active() == name:
            # Fall back to default. It may not be materialized yet on a fresh
            # install, so clear the marker file directly — get_active() will
            # return DEFAULT_COLLECTION when the file is missing.
            marker = self.settings.active_file_path
            if marker.exists():
                marker.unlink()

    def list(self) -> list[CollectionInfo]:
        active = self.get_active()
        infos: list[CollectionInfo] = []

        # Legacy default (pre-multi-collection install)
        if self._has_legacy_default() and not self._collection_path(DEFAULT_COLLECTION).exists():
            infos.append(
                CollectionInfo(
                    name=DEFAULT_COLLECTION,
                    display_name="Default (legacy layout)",
                    is_active=active == DEFAULT_COLLECTION,
                    is_legacy=True,
                    path=self.settings._legacy_chroma.parent,
                    pdf_count=self._count_pdfs(self.settings._legacy_docs),
                    has_index=self.settings._legacy_chroma.exists()
                    and self.settings._legacy_bm25.exists(),
                )
            )

        root = self.settings.collections_root
        if root.exists():
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                infos.append(self._read_info(child.name, active=active))

        return infos

    def get(self, name: str) -> CollectionInfo:
        if not self.exists(name):
            raise ValueError(f"Collection '{name}' does not exist")
        active = self.get_active()
        if (
            name == DEFAULT_COLLECTION
            and self._has_legacy_default()
            and not self._collection_path(DEFAULT_COLLECTION).exists()
        ):
            return CollectionInfo(
                name=DEFAULT_COLLECTION,
                display_name="Default (legacy layout)",
                is_active=active == DEFAULT_COLLECTION,
                is_legacy=True,
                path=self.settings._legacy_chroma.parent,
                pdf_count=self._count_pdfs(self.settings._legacy_docs),
                has_index=self.settings._legacy_chroma.exists()
                and self.settings._legacy_bm25.exists(),
            )
        return self._read_info(name, active=active)

    # ----- Internals -----

    def _collection_path(self, name: str) -> Path:
        return self.settings.collections_root / name

    def _has_legacy_default(self) -> bool:
        s = self.settings
        return (
            s._legacy_chroma.exists()
            or s._legacy_bm25.exists()
            or (s._legacy_docs.exists() and any(s._legacy_docs.glob("*.pdf")))
        )

    @staticmethod
    def _count_pdfs(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for _ in path.glob("*.pdf"))

    def _read_info(self, name: str, active: str | None = None) -> CollectionInfo:
        base = self._collection_path(name)
        meta_file = base / "metadata.json"
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except (OSError, json.JSONDecodeError):
                meta = {}

        docs_dir = base / "docs"
        chroma_dir = base / "chroma"
        bm25_file = base / "bm25.pkl"

        if active is None:
            active = self.get_active()

        return CollectionInfo(
            name=name,
            display_name=meta.get("display_name", name),
            description=meta.get("description", ""),
            created_at=meta.get("created_at", ""),
            is_active=name == active,
            is_legacy=False,
            path=base,
            pdf_count=self._count_pdfs(docs_dir),
            has_index=chroma_dir.exists() and bm25_file.exists(),
        )
