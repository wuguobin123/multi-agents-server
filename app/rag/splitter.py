from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import ChunkingSettings
from app.rag.types import ChunkRecord, RAGDocument


def available_chunking_strategies() -> list[dict[str, Any]]:
    return [
        {
            "name": "recursive_character",
            "description": "Use recursive character splitting with configurable chunk size and overlap.",
            "supports": ["chunk_size", "chunk_overlap"],
        },
        {
            "name": "qa_pair",
            "description": "Split FAQ-style content into question-answer pairs, with recursive fallback for non-QA docs.",
            "supports": [
                "qa_question_prefixes",
                "qa_answer_prefixes",
                "qa_fallback_to_recursive",
                "chunk_size",
                "chunk_overlap",
            ],
        },
    ]


class RecursiveCharacterDocumentSplitter:
    def __init__(self, settings: ChunkingSettings) -> None:
        self._settings = settings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
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
                        metadata={
                            "source": document.source,
                            "chunk_strategy": "recursive_character",
                        },
                    )
                )
        return chunks


class QAPairDocumentSplitter:
    _heading_prefix = re.compile(r"^\s{0,3}(?:#+\s*)?")

    def __init__(self, settings: ChunkingSettings) -> None:
        self._settings = settings
        self._fallback = RecursiveCharacterDocumentSplitter(settings)

    def split(self, documents: list[RAGDocument]) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for document in documents:
            qa_chunks = self._split_document(document)
            if not qa_chunks and self._settings.qa_fallback_to_recursive:
                fallback_chunks = self._fallback.split([document])
                for chunk in fallback_chunks:
                    chunk.metadata["chunk_strategy"] = "qa_pair_fallback"
                chunks.extend(fallback_chunks)
                continue
            chunks.extend(qa_chunks)
        return chunks

    def _split_document(self, document: RAGDocument) -> list[ChunkRecord]:
        pairs: list[tuple[str, str]] = []
        current_question: list[str] = []
        current_answer: list[str] = []
        answer_started = False

        for raw_line in document.content.splitlines():
            line = raw_line.rstrip()
            normalized = self._normalize_line(line)
            question_value = self._match_prefix(normalized, self._settings.qa_question_prefixes)
            answer_value = self._match_prefix(normalized, self._settings.qa_answer_prefixes)

            if question_value is not None:
                self._flush_pair(pairs, current_question, current_answer)
                current_question = [question_value] if question_value else []
                current_answer = []
                answer_started = False
                continue

            if answer_value is not None and current_question:
                answer_started = True
                if answer_value:
                    current_answer.append(answer_value)
                continue

            if not current_question:
                continue
            if answer_started:
                if normalized:
                    current_answer.append(normalized)
                continue
            if normalized:
                current_question.append(normalized)

        self._flush_pair(pairs, current_question, current_answer)

        chunks: list[ChunkRecord] = []
        for chunk_index, (question, answer) in enumerate(pairs):
            content = "\n".join([f"Q: {question}", f"A: {answer}"]).strip()
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.document_id}:{chunk_index}",
                    document_id=document.document_id,
                    source=document.source,
                    text=content,
                    chunk_index=chunk_index,
                    metadata={
                        "source": document.source,
                        "chunk_strategy": "qa_pair",
                        "question": question,
                    },
                )
            )
        return chunks

    @classmethod
    def _normalize_line(cls, line: str) -> str:
        stripped = cls._heading_prefix.sub("", line).strip()
        return stripped

    @staticmethod
    def _match_prefix(line: str, prefixes: list[str]) -> str | None:
        lowered = line.casefold()
        for prefix in prefixes:
            normalized_prefix = prefix.strip()
            if lowered.startswith(normalized_prefix.casefold()):
                return line[len(normalized_prefix) :].strip()
        return None

    @staticmethod
    def _flush_pair(pairs: list[tuple[str, str]], question_parts: list[str], answer_parts: list[str]) -> None:
        if not question_parts or not answer_parts:
            return
        question = " ".join(part.strip() for part in question_parts if part.strip())
        answer = "\n".join(part.strip() for part in answer_parts if part.strip()).strip()
        if question and answer:
            pairs.append((question, answer))


class DocumentSplitter:
    def __init__(self, settings: ChunkingSettings) -> None:
        self._settings = settings
        self._splitter = self._build_splitter(settings)

    @staticmethod
    def _build_splitter(settings: ChunkingSettings) -> RecursiveCharacterDocumentSplitter | QAPairDocumentSplitter:
        if settings.strategy == "qa_pair":
            return QAPairDocumentSplitter(settings)
        return RecursiveCharacterDocumentSplitter(settings)

    def split(self, documents: list[RAGDocument]) -> list[ChunkRecord]:
        return self._splitter.split(documents)

    def profile(self) -> dict[str, Any]:
        return self._settings.profile()
