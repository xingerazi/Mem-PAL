import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from prompt import LONG_TERM_INSIGHT_PROMPT


# ======================
# 0. 读取 .env（不校验）
# ======================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ======================
# 1. 初始化 Client
# ======================
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)


# ======================
# 2. 读取原始数据
# ======================
with open(
    r"E:\gitmyrepo\mem_pal_self\Mem-PAL\output_users\0000.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

# ======================
# 3. 单 sample logs → LLM
# ======================
def extract_insight_for_logs(logs):
    user_input = json.dumps(logs, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": LONG_TERM_INSIGHT_PROMPT},
        {"role": "user", "content": f"用户行为日志如下：\n{user_input}"}
    ]

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2
    )

    content = resp.choices[0].message.content

    try:
        return json.loads(content)
    except Exception:
        return []


# ======================
# 4. 遍历 history / query（不合并）
# ======================
results = []

for user_id, user_data in data.items():
    for split in ("history", "query"):
        for sample in user_data.get(split, []):
            logs = sample.get("logs", [])
            if not logs:
                continue

            insights = extract_insight_for_logs(logs)

            results.append({
                "user_id": user_id,
                "split": split,
                "sample_id": sample.get("sample_id"),
                "insights": insights
            })

            print(f"✅ {user_id} | {split} | {sample.get('sample_id')}")


# ======================
# 5. 保存结果
# ======================
with open("long_term_insights.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("🎉 Done. Saved to long_term_insights.json")
