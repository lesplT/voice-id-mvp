from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from filelock import FileLock
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .config import get_settings


VECTOR_SIZE = 128


class VoiceRepository:
    def __init__(self) -> None:
        self.settings = get_settings()

    @contextmanager
    def _client(self) -> Iterator[QdrantClient]:
        url = self.settings.qdrant_url.strip()
        if url and url.lower() not in {"local", "embedded", "none"}:
            client = QdrantClient(url=url, timeout=15, check_compatibility=False)
            try:
                yield client
            finally:
                client.close()
            return

        path = Path(self.settings.qdrant_local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(path) + ".access.lock", timeout=30)
        with lock:
            client = QdrantClient(path=str(path))
            try:
                yield client
            finally:
                client.close()

    def ensure_collection(self) -> None:
        with self._client() as client:
            expected_size = 192 if self.settings.speaker_engine == "ecapa" else VECTOR_SIZE
            if not client.collection_exists(self.settings.qdrant_collection):
                client.create_collection(
                    collection_name=self.settings.qdrant_collection,
                    vectors_config=models.VectorParams(
                        size=expected_size,
                        distance=models.Distance.COSINE
                    ),
                )
            elif client.get_collection(self.settings.qdrant_collection).config.params.vectors.size != expected_size:
                raise ValueError("Speaker model and Qdrant vector dimensions differ. Select a separate QDRANT_COLLECTION; existing data was preserved.")

    def add_enrollment(
        self,
        speaker_id: str,
        embeddings: list[list[float]],
        centroid: list[float],
        source_name: str,
    ) -> str:
        self.ensure_collection()
        enrollment_id = uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        points = []
        for index, vector in enumerate(embeddings):
            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={
                        "speaker_id": speaker_id,
                        "kind": "reference",
                        "enrollment_id": enrollment_id,
                        "embedding_index": index,
                        "source_name": source_name,
                        "created_at": now,
                    },
                )
            )
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=centroid,
                payload={
                    "speaker_id": speaker_id,
                    "kind": "centroid",
                    "enrollment_id": enrollment_id,
                    "source_name": source_name,
                    "created_at": now,
                },
            )
        )
        with self._client() as client:
            client.upsert(
                collection_name=self.settings.qdrant_collection,
                points=points,
                wait=True,
            )
        return enrollment_id

    def search(self, vector: list[float], limit: int) -> list[dict]:
        self.ensure_collection()
        with self._client() as client:
            response = client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector,
                query_filter=models.Filter(must=[
                    models.FieldCondition(key="kind", match=models.MatchValue(value="reference"))
                ]),
                limit=limit,
                with_payload=True,
            )
        return [
            {
                "speaker_id": point.payload.get("speaker_id"),
                "kind": point.payload.get("kind"),
                "enrollment_id": point.payload.get("enrollment_id"),
                "score": float(point.score),
            }
            for point in response.points
            if point.payload and point.payload.get("speaker_id")
        ]

    def list_speakers(self) -> list[dict]:
        self.ensure_collection()
        offset = None
        stats: dict[str, dict] = {}
        with self._client() as client:
            while True:
                points, offset = client.scroll(
                    collection_name=self.settings.qdrant_collection,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    speaker = payload.get("speaker_id")
                    if not speaker:
                        continue
                    item = stats.setdefault(
                        speaker,
                        {"speaker_id": speaker, "references": 0, "centroids": 0, "enrollments": set()},
                    )
                    item["references" if payload.get("kind") == "reference" else "centroids"] += 1
                    if payload.get("enrollment_id"):
                        item["enrollments"].add(payload["enrollment_id"])
                if offset is None:
                    break
        return [
            {**item, "enrollments": len(item["enrollments"])}
            for item in sorted(stats.values(), key=lambda x: x["speaker_id"])
        ]

    def reference_vectors(self) -> dict[str, list[list[float]]]:
        self.ensure_collection()
        result: dict[str, list[list[float]]] = {}
        offset = None
        with self._client() as client:
            while True:
                points, offset = client.scroll(collection_name=self.settings.qdrant_collection,
                    limit=256, offset=offset, with_payload=True, with_vectors=True)
                for point in points:
                    payload = point.payload or {}
                    if payload.get("kind") == "reference" and payload.get("speaker_id"):
                        result.setdefault(payload["speaker_id"], []).append(point.vector)
                if offset is None:
                    break
        return result

    def delete_speaker(self, speaker_id: str) -> None:
        self.ensure_collection()
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="speaker_id", match=models.MatchValue(value=speaker_id)
                    )
                ]
            )
        )
        with self._client() as client:
            client.delete(
                collection_name=self.settings.qdrant_collection,
                points_selector=selector,
                wait=True,
            )
