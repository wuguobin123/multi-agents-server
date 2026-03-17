from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from app.rag.parser import parse_document
from app.rag.types import RAGDocument


SUPPORTED_EXTENSIONS = {".md", ".txt", ".docx", ".xlsx", ".pdf"}


def build_document_record(
    *,
    path: Path,
    document_id: str,
    source: str,
    metadata: dict[str, object] | None = None,
) -> RAGDocument:
    content, parser_type = parse_document(path)
    payload = dict(metadata or {})
    payload.setdefault("source_path", source)
    payload.setdefault("parser_type", parser_type)
    return RAGDocument(
        document_id=document_id,
        source=source,
        content=content,
        metadata=payload,
    )


class FileSystemDocumentLoader:
    def __init__(self, root_dir: Path, docs_path: str) -> None:
        self._root_dir = root_dir
        self._docs_path = root_dir / docs_path if not Path(docs_path).is_absolute() else Path(docs_path)

    @property
    def docs_path(self) -> Path:
        return self._docs_path

    def iter_source_files(self) -> Iterable[Path]:
        if not self._docs_path.exists():
            return []
        return sorted(path for path in self._docs_path.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS)

    def load(self) -> list[RAGDocument]:
        documents: list[RAGDocument] = []
        for path in self.iter_source_files():
            if path.is_relative_to(self._root_dir):
                source = str(path.relative_to(self._root_dir))
            else:
                source = str(path.relative_to(self._docs_path))
            documents.append(
                build_document_record(
                    path=path,
                    document_id=hashlib.sha1(source.encode("utf-8")).hexdigest(),
                    source=source,
                    metadata={"source_path": source},
                )
            )
        return documents
