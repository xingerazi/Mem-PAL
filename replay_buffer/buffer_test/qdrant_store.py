import logging
from typing import List, Dict, Optional, Any, Union

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)
import uuid
logger = logging.getLogger(__name__)


VectorInput = Union[List[float], Dict[str, List[float]]]


class QdrantStore:
    """
    A minimal, opinionated Qdrant wrapper for InteractionUnit storage.

    Design assumptions:
    - Docker-based Qdrant only
    - Single collection
    - point_id is provided externally (iu_id)
    - Payload filtering is first-class
    - Supports BOTH:
        - single-vector collection
        - named-vector collection (e.g. user_query / topic)
    """

    def __init__(
        self,
        collection_name: str,
        vector_dim: int,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        distance: Distance = Distance.COSINE,

        # ⭐ NEW: 命名向量配置（可选）
        vector_names: Optional[List[str]] = None,
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.vector_names = vector_names  # None = 单向量模式

        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
        )

        self._ensure_collection(distance)
        self._ensure_payload_indexes()

    # =========================
    # Collection & Index
    # =========================

    def _ensure_collection(self, distance: Distance):
        collections = self.client.get_collections().collections
        for col in collections:
            if col.name == self.collection_name:
                logger.info(f"Using existing collection: {self.collection_name}")
                return

        logger.info(f"Creating collection: {self.collection_name}")

        # ⭐ NEW: 根据是否传 vector_names 决定 collection 结构
        if self.vector_names:
            vectors_config = {
                name: VectorParams(
                    size=self.vector_dim,
                    distance=distance,
                )
                for name in self.vector_names
            }
        else:
            vectors_config = VectorParams(
                size=self.vector_dim,
                distance=distance,
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
        )

    def _ensure_payload_indexes(self):
        index_fields = {
            "person_id": "keyword",
            "topic": "keyword",
            "user_feedback": "keyword",
            "timestamp": "keyword",
        }

        for field, schema in index_fields.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )
                logger.info(f"Created payload index: {field}")
            except Exception:
                pass

    # =========================
    # Insert / Update
    # =========================

    def upsert(
        self,
        iu_id: str,
        vector: VectorInput,
        payload: Dict[str, Any],
    ):
        """
        Insert or update a single InteractionUnit.

        vector:
        - List[float]                    -> single-vector collection
        - Dict[str, List[float]]         -> named-vector collection
        """

        # ⭐ NEW: 轻量校验
        if self.vector_names:
            if not isinstance(vector, dict):
                raise ValueError("Named-vector collection requires dict[str, vector]")
        else:
            if not isinstance(vector, list):
                raise ValueError("Single-vector collection requires List[float]")
        point_id = uuid.uuid5(uuid.NAMESPACE_DNS, iu_id)
        point = PointStruct(
            id=str(point_id),
            vector=vector,
            payload=payload,
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def upsert_batch(
        self,
        points: List[Dict[str, Any]],
    ):
        qdrant_points = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload", {}),
            )
            for p in points
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=qdrant_points,
        )

    # =========================
    # Search
    # =========================

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,

        # ⭐ NEW: 命名向量时指定用哪个向量空间
        using: Optional[str] = None,
    ):
        """
        Vector similarity search with optional payload filters.

        using:
        - None           -> 单向量 collection
        - "user_query"   -> 命名向量 collection
        """

        query_filter = self._build_filter(filters)

        # ⭐ NEW: 根据是否 using 选择调用方式
        if using:
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=using,
                query_filter=query_filter,
                limit=limit,
            )
            return hits.points
        else:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
            )
            return hits

    # =========================
    # Filter Builder
    # =========================

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        if not filters:
            return None

        conditions = []

        for key, value in filters.items():
            if isinstance(value, dict) and ("gte" in value or "lte" in value):
                conditions.append(
                    FieldCondition(
                        key=key,
                        range=Range(
                            gte=value.get("gte"),
                            lte=value.get("lte"),
                        ),
                    )
                )
            else:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )

        return Filter(must=conditions)

    # =========================
    # Delete / Get
    # =========================

    def delete(self, iu_id: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[iu_id],
        )

    def retrieve(self, iu_id: str):
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[iu_id],
            with_payload=True,
            with_vectors=False,
        )
        return result[0] if result else None

    # =========================
    # Debug / Maintenance
    # =========================

    def count(self) -> int:
        return self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count
