# qdrant_search_service.py

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

logger = logging.getLogger(__name__)


class QdrantSearchService:
    """
    基于 Qdrant 的搜索封装（不依赖 IUVectorStore）
    - 负责 embedding + vector search
    - 支持并发查询
    """

    def __init__(
        self,
        *,
        qdrant_host: str,
        qdrant_port: int,
        collection_name: str,
        embedding_model: str,
        openai_client: OpenAI,
        max_workers: int = 8,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.openai_client = openai_client

        self.qdrant = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
        )

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers

        logger.info(
            f"[QdrantSearchService] init "
            f"(collection={collection_name}, workers={max_workers})"
        )

    # -------------------------
    # Embedding
    # -------------------------
    def _embed(self, text: str) -> List[float]:
        resp = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return resp.data[0].embedding

    # -------------------------
    # 单条搜索（内部）
    # -------------------------
    def _search_one(
        self,
        query_text: str,
        *,
        limit: int,
        filters: Optional[Dict] = None,
    ):
        vector = self._embed(query_text)

        qdrant_filter = None
        if filters:
            # 直接把 dict 交给 Qdrant（payload filter）
            qdrant_filter = Filter(**filters)

        hits = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=("user_query", vector),  # ✅ 正确
            limit=limit,
            query_filter=qdrant_filter,
        )

        return hits

    # -------------------------
    # 单条搜索（同步）
    # -------------------------
    def search(
        self,
        query_text: str,
        *,
        limit: int = 5,
        filters: Optional[Dict] = None,
    ):
        return self._search_one(
            query_text,
            limit=limit,
            filters=filters,
        )

    # -------------------------
    # 多条并发搜索
    # -------------------------
    def batch_search(
        self,
        query_list: List[str],
        *,
        limit: int = 5,
        filters: Optional[Dict] = None,
    ):
        """
        并发搜索多个 query

        Returns:
            [
              {
                "query": "...",
                "hits": [...]
              },
              ...
            ]
        """
        futures = {}
        results = []

        for q in query_list:
            fut = self.executor.submit(
                self._search_one,
                q,
                limit=limit,
                filters=filters,
            )
            futures[fut] = q

        for fut in as_completed(futures):
            query = futures[fut]
            try:
                hits = fut.result()
                results.append(
                    {
                        "query": query,
                        "hits": hits,
                    }
                )
            except Exception as e:
                logger.exception(f"[Qdrant search failed] query={query}")
                results.append(
                    {
                        "query": query,
                        "error": str(e),
                        "hits": [],
                    }
                )

        return results
