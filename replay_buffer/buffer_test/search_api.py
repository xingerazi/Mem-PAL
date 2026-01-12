# search_api.py

import logging
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Config
# =========================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "mem_pal_test_v4"

EMBEDDING_MODEL = "text-embedding-3-small"

VECTOR_QUERY = "user_query"
VECTOR_TOPIC = "topic"
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
# =========================
# Init clients
# =========================

openai_client = OpenAI()
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

app = FastAPI(title="Qdrant Search API (query / topic)")


# =========================
# Schema
# =========================

class SearchRequest(BaseModel):
    text: str
    limit: int = 3
    filters: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]


# =========================
# Internal util
# =========================

def embed_text(text: str) -> List[float]:
    return openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    ).data[0].embedding


def qdrant_search(
    *,
    vector_name: str,
    vector: List[float],
    limit: int,
    filters: Optional[Dict[str, Any]],
):
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=(vector_name, vector),
        limit=limit,
        query_filter=filters,
    )

    results = []
    for p in hits:
        results.append(
            {
                "id": p.id,
                "score": p.score,
                "payload": p.payload,
            }
        )
    return results


# =========================
# APIs
# =========================

@app.post("/search/query", response_model=SearchResponse)
async def search_by_query(req: SearchRequest):
    """
    使用 user_query 向量搜索（用户自然语言问题）
    """
    logger.info(f"[SEARCH:QUERY] {req.text}")

    vector = embed_text(req.text)
    results = qdrant_search(
        vector_name=VECTOR_QUERY,
        vector=vector,
        limit=req.limit,
        filters=req.filters,
    )
    return {"results": results}


@app.post("/search/topic", response_model=SearchResponse)
async def search_by_topic(req: SearchRequest):
    """
    使用 topic 向量搜索（主题 / 抽象语义）
    """
    logger.info(f"[SEARCH:TOPIC] {req.text}")

    vector = embed_text(req.text)
    results = qdrant_search(
        vector_name=VECTOR_TOPIC,
        vector=vector,
        limit=req.limit,
        filters=req.filters,
    )
    return {"results": results}
