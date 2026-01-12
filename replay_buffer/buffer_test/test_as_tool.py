# test_qdrant_search.py

import logging
from dotenv import load_dotenv
from openai import OpenAI

from qdrant_search_service import QdrantSearchService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Config
# =========================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "mem_pal_test_v4"
EMBEDDING_MODEL = "text-embedding-3-small"

PERSON_ID = "pal_0000"   # None 表示不限制


def main():
    load_dotenv()
    client = OpenAI()

    search_service = QdrantSearchService(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        openai_client=client,
        max_workers=4,   # 并发度
    )

    query_list = [
        "我最近胃有点难受，怎么办？",
        "长期出差会影响身体吗？",
        "已婚有孩子的人压力大吗？",
        "经常熬夜会带来哪些健康问题？",
    ]

    filters = None
    if PERSON_ID:
        # 假设你的 payload 里有 person_id
        filters = {
            "must": [
                {
                    "key": "person_id",
                    "match": {"value": PERSON_ID},
                }
            ]
        }

    logger.info("Start concurrent search...")
    results = search_service.batch_search(
        query_list=query_list,
        limit=5,
        filters=filters,
    )

    logger.info("Search results:")
    for idx, item in enumerate(results, 1):
        logger.info(f"\n[{idx}] Query: {item['query']}")
        for i, p in enumerate(item["hits"], 1):
            payload = p.payload or {}
            logger.info(
                f"  {i}. person_id={payload.get('person_id')}, "
                f"user_query={payload.get('user_query')}, "
                f"topic={payload.get('topic')}, "
                f"timestamp={payload.get('timestamp')}"
            )


if __name__ == "__main__":
    main()

# from qdrant_client import QdrantClient

# client = QdrantClient(host="localhost", port=6333)
# info = client.get_collection("mem_pal_test_v4")
# print(info.config.params.vectors)