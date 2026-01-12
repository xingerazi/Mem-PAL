import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from iu_vector_store import IUVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== 路径：改成目录 =====
UNITS_DIR = Path(
    r"E:\gitmyrepo\mem_pal_self\Mem-PAL\replay_buffer\interaction_units"
)

COLLECTION_NAME = "mem_pal_test_v4"
VECTOR_DIM = 1536
EMBEDDING_MODEL = "text-embedding-3-small"


def iter_units(path: Path):
    """逐行读取一个 jsonl 文件"""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"JSON decode error in {path.name} at line {line_no}"
                ) from e


def main():
    load_dotenv()

    client = OpenAI()

    if not UNITS_DIR.exists():
        raise FileNotFoundError(f"Units dir not found: {UNITS_DIR.resolve()}")

    # ===== 找到 0000–0099 =====
    unit_files = sorted(UNITS_DIR.glob("00*_units.jsonl"))

    if not unit_files:
        raise RuntimeError("No unit files found (00*_units.jsonl)")

    logger.info(f"Found {len(unit_files)} unit files")

    store = IUVectorStore(
        collection_name=COLLECTION_NAME,
        vector_dim=VECTOR_DIM,
        openai_client=client,
        embedding_model=EMBEDDING_MODEL,
    )

    total_count = 0

    for file_path in unit_files:
        logger.info(f"Processing file: {file_path.name}")

        for unit in iter_units(file_path):
            store.add_unit(unit)
            total_count += 1

            if total_count % 5 == 0:
                logger.info(f"Inserted {total_count} units")

    logger.info(f"Finished. Total inserted: {total_count}")

    # ===== 简单验证一次 search =====
    test_query = "我最近胃有点难受，怎么办？"
    hits = store.search_by_query(test_query, limit=5)

    logger.info("Sample search results:")
    for i, p in enumerate(hits, 1):
        logger.info(
            f"{i}. iu_id={p.id}, topic={p.payload.get('topic')}"
        )


if __name__ == "__main__":
    main()
