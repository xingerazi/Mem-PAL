import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from prompt import PROFILE_EDIT_PROMPT


# ======================
# 0. env & client
# ======================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ======================
# 1. 路径配置
# ======================
BASE_DIR = r"E:\gitmyrepo\mem_pal_self\Mem-PAL"

INPUT_DIR = os.path.join(BASE_DIR, "output_users")
PROFILE_OUT_DIR = os.path.join(BASE_DIR, "change", "extract", "output")
OPS_OUT_DIR = os.path.join(BASE_DIR, "change", "extract", "output_log")

os.makedirs(PROFILE_OUT_DIR, exist_ok=True)
os.makedirs(OPS_OUT_DIR, exist_ok=True)


# ======================
# 2. profile 读 / 写
# ======================
def load_profile(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "work": {},
        "health": {},
        "family": {},
        "leisure": {}
    }


def save_profile(path, profile):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


# ======================
# 3. ID 生成（你掌控）
# ======================
PREFIX = {
    "work": "wk",
    "health": "hl",
    "family": "fm",
    "leisure": "ls"
}

def next_id(profile, category):
    pref = PREFIX[category]
    used = profile.get(category, {})
    max_n = 0
    for k in used.keys():
        if k.startswith(pref):
            try:
                max_n = max(max_n, int(k[len(pref):]))
            except:
                pass
    return f"{pref}{max_n + 1}"


# ======================
# 4. LLM：logs → operations
# ======================
def get_profile_operations(old_profile, logs):
    prompt = PROFILE_EDIT_PROMPT.format(
        old_profile=json.dumps(old_profile, ensure_ascii=False, indent=2),
        logs=json.dumps(logs, ensure_ascii=False, indent=2)
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2
    )

    content = resp.choices[0].message.content

    print("\n====== LLM RAW RESPONSE ======")
    print(content)
    print("====== END LLM RESPONSE ======\n")

    match = re.search(r'\{[\s\S]*\}', content)
    if not match:
        print("⚠️ No JSON found")
        return []

    try:
        obj = json.loads(match.group(0))
    except Exception as e:
        print("⚠️ JSON parse failed:", e)
        return []

    return obj.get("operations", [])


# ======================
# 5. 执行 operations
# ======================
def apply_operations(profile, operations, max_per_cat=5):
    for op in operations:
        cat = op.get("category")
        if cat not in PREFIX:
            continue

        if op.get("op") == "add":
            nid = next_id(profile, cat)
            profile[cat][nid] = op.get("content")

        elif op.get("op") == "update":
            _id = op.get("id")
            if _id in profile[cat]:
                profile[cat][_id] = op.get("content")

    # 每类最多 max_per_cat 条
    for cat, pref in PREFIX.items():
        keys = sorted(
            profile[cat].keys(),
            key=lambda k: int(k[len(pref):]) if k[len(pref):].isdigit() else 10**9
        )
        for k in keys[max_per_cat:]:
            profile[cat].pop(k, None)

    return profile


# ======================
# 6. 主函数（你要的）
# ======================
def main(start_id: int, end_id: int):
    for uid in range(start_id, end_id + 1):
        user_id = f"{uid:04d}"
        user_path = os.path.join(INPUT_DIR, f"{user_id}.json")

        if not os.path.exists(user_path):
            print(f"⚠️ Skip missing {user_id}.json")
            continue

        print(f"\n🚀 Processing user {user_id}")

        with open(user_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile_path = os.path.join(PROFILE_OUT_DIR, f"{user_id}_profile.json")
        ops_path = os.path.join(OPS_OUT_DIR, f"{user_id}_ops.json")

        for _, user_data in data.items():
            for split in ("history", "query"):
                for sample in user_data.get(split, []):
                    logs = sample.get("logs", [])
                    if not logs:
                        continue

                    profile = load_profile(profile_path)

                    operations = get_profile_operations(profile, logs)
                    profile = apply_operations(profile, operations)
                    save_profile(profile_path, profile)

                    # 追加 ops
                    record = {
                        "user_id": user_id,
                        "split": split,
                        "sample_id": sample.get("sample_id"),
                        "operations": operations
                    }

                    if os.path.exists(ops_path):
                        with open(ops_path, "r", encoding="utf-8") as f:
                            ops = json.load(f)
                    else:
                        ops = []

                    ops.append(record)

                    with open(ops_path, "w", encoding="utf-8") as f:
                        json.dump(ops, f, ensure_ascii=False, indent=2)

        print(f"✅ Finished user {user_id}")


# ======================
# 7. CLI 入口
# ======================
if __name__ == "__main__":
    main(start_id=0, end_id=99)
