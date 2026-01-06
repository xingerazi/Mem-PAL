from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import random

# =========================
# 1. 连接本地 Qdrant
# =========================
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "test_user_query"
VECTOR_NAME = "user_query"
DIM = 4  # 测试用，故意设小一点

# =========================
# 2. 创建 collection（安全写法）
# =========================
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            VECTOR_NAME: VectorParams(
                size=DIM,
                distance=Distance.COSINE
            )
        }
    )
    print("Collection created.")
else:
    print("Collection already exists.")

# =========================
# 3. 构造一些“假向量”
# =========================
def random_vec():
    return [random.random() for _ in range(DIM)]

points = [
    {
        "id": 1,
        "vector": {VECTOR_NAME: [1.0, 0.0, 0.0, 0.0]},
        "payload": {"text": "用户不喜欢香菜"}
    },
    {
        "id": 2,
        "vector": {VECTOR_NAME: [0.9, 0.1, 0.0, 0.0]},
        "payload": {"text": "用户讨厌吃香菜"}
    },
    {
        "id": 3,
        "vector": {VECTOR_NAME: [0.0, 1.0, 0.0, 0.0]},
        "payload": {"text": "用户喜欢吃辣"}
    },
]

# =========================
# 4. 插入数据
# =========================
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("Points inserted.")

# =========================
# 5. 查询相似向量
# =========================
query_vector = [1.0, 0.05, 0.0, 0.0]

results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=(VECTOR_NAME, query_vector),
    limit=3
)

# =========================
# 6. 打印结果
# =========================
print("\nSearch results:")
for r in results:
    print(
        f"id={r.id}, "
        f"score={round(r.score, 4)}, "
        f"text={r.payload.get('text')}"
    )
