from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel


_FILENAME_SANITIZER = re.compile(r"[^A-Za-z0-9._-]+")


class StoredFile(BaseModel):
    source: str
    storage_path: str
    file_url: str | None = None
    file_hash: str
    file_size: int
    mime_type: str | None = None


class LocalFileStorage:
    def __init__(self, root_dir: Path, uploads_path: str, public_base_url: str | None = None) -> None:
        self._root_dir = root_dir
        self._uploads_dir = root_dir / uploads_path
        self._public_base_url = public_base_url.rstrip("/") if public_base_url else None

    def save_bytes(
        self,
        *,
        knowledge_base_id: str,
        document_id: str,
        filename: str,
        content: bytes,
        mime_type: str | None = None,
    ) -> StoredFile:
        safe_name = self._sanitize_filename(filename)
        target = self._uploads_dir / knowledge_base_id / document_id / safe_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        relative = target.relative_to(self._root_dir)
        file_hash = hashlib.sha256(content).hexdigest()
        file_url = f"{self._public_base_url}/{relative.as_posix()}" if self._public_base_url else relative.as_posix()
        return StoredFile(
            source=relative.as_posix(),
            storage_path=str(target),
            file_url=file_url,
            file_hash=file_hash,
            file_size=len(content),
            mime_type=mime_type,
        )

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        candidate = Path(filename or "document").name
        cleaned = _FILENAME_SANITIZER.sub("-", candidate).strip("-")
        return cleaned or "document"
