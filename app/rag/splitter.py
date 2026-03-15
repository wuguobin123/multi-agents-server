from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.types import ChunkRecord, RAGDocument


class DocumentSplitter:
    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: list[RAGDocument]) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for document in documents:
            split_docs = self._splitter.split_documents(
                [
                    Document(
                        page_content=document.content,
                        metadata={"source": document.source, "document_id": document.document_id},
                    )
                ]
            )
            for chunk_index, chunk in enumerate(split_docs):
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{document.document_id}:{chunk_index}",
                        document_id=document.document_id,
                        source=document.source,
                        text=chunk.page_content.strip(),
                        chunk_index=chunk_index,
                        metadata={"source": document.source},
                    )
                )
        return chunks
