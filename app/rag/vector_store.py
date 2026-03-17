from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol

import httpx

from app.rag.types import SearchResult, VectorPoint


class VectorStore(Protocol):
    def reset(self) -> None: ...

    def upsert(self, points: list[VectorPoint]) -> None: ...

    def delete(self, point_ids: list[str]) -> None: ...

    def query(self, vector: list[float], *, limit: int) -> list[SearchResult]: ...

    def count(self) -> int: ...

    def vector_size(self) -> int | None: ...

    def readiness(self) -> dict[str, Any]: ...


class LocalVectorStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._points: dict[str, VectorPoint] = {}
        self._load()

    def reset(self) -> None:
        self._points = {}
        self._persist()

    def upsert(self, points: list[VectorPoint]) -> None:
        for point in points:
            self._points[point.point_id] = point
        self._persist()

    def delete(self, point_ids: list[str]) -> None:
        for point_id in point_ids:
            self._points.pop(point_id, None)
        self._persist()

    def query(self, vector: list[float], *, limit: int) -> list[SearchResult]:
        scored: list[tuple[float, VectorPoint]] = []
        for point in self._points.values():
            score = _cosine_similarity(vector, point.vector)
            if score > 0:
                scored.append((score, point))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(
                chunk_id=point.payload["chunk_id"],
                document_id=point.payload.get("document_id"),
                knowledge_base_id=point.payload.get("knowledge_base_id"),
                source=point.payload["source"],
                text=point.payload["text"],
                score=round(score, 4),
                metadata=point.payload.get("metadata", {}),
            )
            for score, point in scored[:limit]
        ]

    def count(self) -> int:
        return len(self._points)

    def vector_size(self) -> int | None:
        first_point = next(iter(self._points.values()), None)
        if first_point is None:
            return None
        return len(first_point.vector)

    def readiness(self) -> dict[str, Any]:
        return {
            "backend": "local",
            "status": "ok",
            "path": str(self._path),
            "point_count": len(self._points),
            "vector_size": self.vector_size(),
        }

    def _load(self) -> None:
        if not self._path.exists():
            return
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        self._points = {item["point_id"]: VectorPoint.model_validate(item) for item in payload}

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [point.model_dump() for point in self._points.values()]
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class QdrantHttpVectorStore:
    def __init__(
        self,
        *,
        base_url: str,
        collection_name: str,
        vector_size: int,
        api_key: str | None = None,
        timeout_seconds: int = 10,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._ensure_collection()

    def reset(self) -> None:
        self._request("DELETE", f"/collections/{self._collection_name}", allow_404=True)
        self._ensure_collection()

    def upsert(self, points: list[VectorPoint]) -> None:
        if not points:
            return
        payload = {
            "points": [
                {
                    "id": point.point_id,
                    "vector": point.vector,
                    "payload": point.payload,
                }
                for point in points
            ]
        }
        self._request("PUT", f"/collections/{self._collection_name}/points", json=payload)

    def delete(self, point_ids: list[str]) -> None:
        if not point_ids:
            return
        self._request(
            "POST",
            f"/collections/{self._collection_name}/points/delete",
            json={"points": point_ids},
        )

    def query(self, vector: list[float], *, limit: int) -> list[SearchResult]:
        payload = {
            "query": vector,
            "limit": limit,
            "with_payload": True,
        }
        response = self._request("POST", f"/collections/{self._collection_name}/points/query", json=payload)
        body = response.json()
        items = body.get("result", {}).get("points", [])
        return [
            SearchResult(
                chunk_id=item["payload"]["chunk_id"],
                document_id=item["payload"].get("document_id"),
                knowledge_base_id=item["payload"].get("knowledge_base_id"),
                source=item["payload"]["source"],
                text=item["payload"]["text"],
                score=round(float(item.get("score", 0.0)), 4),
                metadata=item["payload"].get("metadata", {}),
            )
            for item in items
        ]

    def count(self) -> int:
        response = self._request(
            "POST",
            f"/collections/{self._collection_name}/points/count",
            json={"exact": True},
        )
        body = response.json()
        return int(body.get("result", {}).get("count", 0))

    def vector_size(self) -> int | None:
        collection = self._get_collection()
        if collection is None:
            return None
        config = collection.get("config", {}).get("params", {}).get("vectors", {})
        size = config.get("size")
        return int(size) if size is not None else None

    def readiness(self) -> dict[str, Any]:
        collection = self._get_collection()
        status = "ok" if collection is not None else "empty"
        return {
            "backend": "qdrant",
            "status": status,
            "url": self._base_url,
            "collection": self._collection_name,
            "vector_size": self.vector_size(),
        }

    def _ensure_collection(self) -> None:
        payload = {
            "vectors": {
                "size": self._vector_size,
                "distance": "Cosine",
            }
        }
        self._request(
            "PUT",
            f"/collections/{self._collection_name}",
            json=payload,
            allow_statuses={409},
        )

    def _get_collection(self) -> dict[str, Any] | None:
        response = self._request("GET", f"/collections/{self._collection_name}", allow_404=True)
        if response.status_code == 404:
            return None
        body = response.json()
        return body.get("result")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        allow_404: bool = False,
        allow_statuses: set[int] | None = None,
    ) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["api-key"] = self._api_key
        response = httpx.request(
            method,
            f"{self._base_url}{path}",
            json=json,
            headers=headers,
            timeout=self._timeout,
            trust_env=False,
        )
        if allow_404 and response.status_code == 404:
            return response
        if allow_statuses and response.status_code in allow_statuses:
            return response
        response.raise_for_status()
        return response


def _cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    if not lhs or not rhs:
        return 0.0
    numerator = sum(a * b for a, b in zip(lhs, rhs, strict=False))
    lhs_norm = math.sqrt(sum(value * value for value in lhs))
    rhs_norm = math.sqrt(sum(value * value for value in rhs))
    if lhs_norm == 0 or rhs_norm == 0:
        return 0.0
    return numerator / (lhs_norm * rhs_norm)
