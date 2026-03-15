from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from app.rag.types import RAGDocument


class FileSystemDocumentLoader:
    def __init__(self, root_dir: Path, docs_path: str) -> None:
        self._root_dir = root_dir
        self._docs_path = root_dir / docs_path

    @property
    def docs_path(self) -> Path:
        return self._docs_path

    def iter_source_files(self) -> Iterable[Path]:
        if not self._docs_path.exists():
            return []
        return sorted(path for path in self._docs_path.rglob("*") if path.suffix.lower() in {".md", ".txt"})

    def load(self) -> list[RAGDocument]:
        documents: list[RAGDocument] = []
        for path in self.iter_source_files():
            if path.is_relative_to(self._root_dir):
                source = str(path.relative_to(self._root_dir))
            else:
                source = str(path.relative_to(self._docs_path))
            documents.append(
                RAGDocument(
                    document_id=hashlib.sha1(source.encode("utf-8")).hexdigest(),
                    source=source,
                    content=path.read_text(encoding="utf-8"),
                    metadata={"source_path": source},
                )
            )
        return documents
