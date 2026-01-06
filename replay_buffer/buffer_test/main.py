import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from iu_vector_store import IUVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


UNITS_PATH = Path("replay_buffer/buffer_test/0000_units.jsonl")

COLLECTION_NAME = "mem_pal_test_v2"
VECTOR_DIM = 1536
EMBEDDING_MODEL = "text-embedding-3-small"


def iter_units(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error at line {line_no}") from e


def main():
    load_dotenv()

    client = OpenAI()

    store = IUVectorStore(
        collection_name=COLLECTION_NAME,
        vector_dim=VECTOR_DIM,
        openai_client=client,
        embedding_model=EMBEDDING_MODEL,
    )

    count = 0

    for unit in iter_units(UNITS_PATH):
        store.add_unit(unit)
        count += 1

        if count % 5 == 0:
            logger.info(f"Inserted {count} units")

    logger.info(f"Finished. Total inserted: {count}")

    # 简单验证一次 search
    test_query = "我最近在减肥，晚饭吃什么比较好？"
    hits = store.search_by_query(test_query, limit=5)

    logger.info("Sample search results:")
    for i, p in enumerate(hits, 1):
        logger.info(
            f"{i}. iu_id={p.id}, topic={p.payload.get('topic')}"
        )


if __name__ == "__main__":
    main()
