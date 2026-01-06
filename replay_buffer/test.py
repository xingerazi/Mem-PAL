import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# ======================
# 0. Env
# ======================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ======================
# 1. Path
# ======================
INPUT_JSON_PATH = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\replay_buffer\dia\0000.json"
OUTPUT_JSON_PATH = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\replay_buffer\processed_dia\0000.json"

# ======================
# 2. Prompt
# ======================
PROMPT = """
你是一个对话三元组抽取器。

输入是一个 dialogue JSON（turn_1, turn_2 ...）。
请你按对话顺序抽取若干条 triplet，每条包含：

- user_query：用户提出的一个明确问题或需求（尽量使用原句）
- model_answer：assistant 给出的核心可执行建议，用一句话概括
- user_feedback：用户对该建议的反馈或态度；若无则为 null，比如"我会尝试","这样不行！我在减肥"

要求：
1. 不要 topic / summary / 分类
2. 不要 turn_id / action / timestamp
3. 不要分析说明
4. 只输出 JSON，格式如下：

{
  "triplets": [
    {
      "user_query": "...",
      "model_answer": "...",
      "user_feedback": ".."
    }
  ]
}
"""

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

def extract_triplets(dialogue: dict, sample_id: str) -> dict:
    print(f"\n[LLM] Extracting triplets for sample_id = {sample_id}")
    print(f"[LLM] Dialogue turns = {len(dialogue)}")

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": PROMPT.strip()},
            {"role": "user", "content": json.dumps(dialogue, ensure_ascii=False)}
        ],
        temperature=0
    )

    content = resp.choices[0].message.content
    print("[LLM] Raw response:")
    print(content)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("[ERROR] JSON parse failed:", e)
        return {"triplets": []}

# ======================
# 4. Main
# ======================
def main():
    print("[INFO] Loading input json:", INPUT_JSON_PATH)
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = {}

    for user_id, user_block in data.items():
        print(f"\n[USER] Processing user_id = {user_id}")
        new_data[user_id] = {}

        for section in ["history", "query"]:
            if section not in user_block:
                continue

            print(f"[SECTION] {section} | num_items = {len(user_block[section])}")
            new_items = []

            for idx, item in enumerate(user_block[section]):
                sample_id = item.get("sample_id")
                print(f"\n  ▶ Processing {section}[{idx}] | sample_id = {sample_id}")

                dialogue = item.get("dialogue")
                if not dialogue:
                    print("  ⚠ No dialogue found, skip.")
                    continue

                result = extract_triplets(dialogue, sample_id)
                triplets = result.get("triplets", [])

                print(f"  ✓ Extracted {len(triplets)} triplets")

                new_items.append({
                    "sample_id": sample_id,
                    "dialogue_timestamp": item.get("dialogue_timestamp"),
                    "triplets": triplets
                })

            new_data[user_id][section] = new_items

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Done. Output saved to:", OUTPUT_JSON_PATH)


if __name__ == "__main__":
    main()