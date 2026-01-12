import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from iu_vector_store import IUVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Config（和 main.py 保持一致）
# =========================

COLLECTION_NAME = "mem_pal_test_v4"
VECTOR_DIM = 1536
EMBEDDING_MODEL = "text-embedding-3-small"

# 如果你想只搜某个用户
PERSON_ID = "pal_0000"   # 不想限制就设为 None


def main():
    load_dotenv()

    client = OpenAI()

    store = IUVectorStore(
        collection_name=COLLECTION_NAME,
        vector_dim=VECTOR_DIM,
        openai_client=client,
        embedding_model=EMBEDDING_MODEL,
    )

    # =========================
    # Query
    # =========================
    query_text = "我最近胃有点难受，怎么办？"
    logger.info(f"Search query: {query_text}")

    filters = None
    if PERSON_ID:
        filters = {"person_id": PERSON_ID}

    hits = store.search_by_query(
        query_text=query_text,
        limit=5,
        filters=filters,
    )

    # =========================
    # Print results（用 payload 里的 iu_id）
    # =========================
    logger.info("Search results:")
    for i, p in enumerate(hits, 1):
        payload = p.payload or {}
        logger.info(
            f"{i}. person_id={payload.get('person_id')}, "
            f"user_query={payload.get('user_query')}, "
            f"topic={payload.get('topic')}, "
            f"timestamp={payload.get('timestamp')}"
        )


if __name__ == "__main__":
    main()


