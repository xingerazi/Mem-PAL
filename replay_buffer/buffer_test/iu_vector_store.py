import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from openai import OpenAI

from qdrant_store import QdrantStore  # 你已有的
logger = logging.getLogger(__name__)


class IUVectorStore:
    """
    High-level vector store for InteractionUnit.

    Responsibilities:
    - build embeddings (user_query / topic)
    - call QdrantStore
    - hide vector/db details from upper layers
    """

    def __init__(
        self,
        collection_name: str,
        vector_dim: int,
        openai_client: OpenAI,
        embedding_model: str,

        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ):
        self.embedding_model = embedding_model
        self.client = openai_client

        self.vectorstore = QdrantStore(
            collection_name=collection_name,
            vector_dim=vector_dim,
            host=qdrant_host,
            port=qdrant_port,
            vector_names=["user_query", "topic"],
        )

    # =========================
    # Embedding
    # =========================

    def _embed(self, text: str) -> List[float]:
        text = text.replace("\n", " ").strip()
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return resp.data[0].embedding

    # =========================
    # Public API
    # =========================

    def add_unit(self, unit: Dict[str, Any]):
        iu_id = unit["iu_id"]

        # 只 embed 真实存在、且你明确想用来检索的东西
        uq_text = unit["user_query"].strip()
        topic_text = unit["topic"].strip()

        if not uq_text or not topic_text:
            logger.warning(f"Skip IU {iu_id}: empty user_query or topic")
            return

        uq_vec = self._embed(uq_text)
        topic_vec = self._embed(topic_text)

        payload = {
            "person_id": unit["person_id"],
            "topic": unit["topic"],
            "user_feedback": unit["user_feedback"]["type"],
            "timestamp": unit["timestamp"],
            "user_query": unit["user_query"],
            "insight": unit.get("insight"),
        }

        self.vectorstore.upsert(
            iu_id=iu_id,
            vector={
                "user_query": uq_vec,
                "topic": topic_vec,
            },
            payload=payload,
        )

    def search_by_query(
        self,
        query_text: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ):
        """
        Search IU by user_query semantic space.
        """
        vec = self._embed(query_text)

        results = self.vectorstore.search(
            query_vector=vec,
            using="user_query",
            limit=limit,
            filters=filters,
        )

        return results  