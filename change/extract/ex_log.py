import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from prompt import PROFILE_EDIT_PROMPT
import re

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
# 2. 路径（按你现在的工程）
# ======================
USER_JSON_PATH = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\output_users\0000.json"

PROFILE_PATH = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\change\extract\0000_profile.json"

OPS_LOG_PATH = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\change\extract\0000_profile_ops_log.json"


# ======================
# 3. 读用户原始数据
# ======================
with open(USER_JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


# ======================
# 4. profile 读 / 写
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
# 5. ID 生成（你掌控）
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
# 6. LLM：返回 operations
# ======================

def get_profile_operations(old_profile, logs):
    prompt = PROFILE_EDIT_PROMPT.format(
        old_profile=json.dumps(old_profile, ensure_ascii=False, indent=2),
        logs=json.dumps(logs, ensure_ascii=False, indent=2)
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2
    )

    content = resp.choices[0].message.content

    print("\n====== LLM RAW RESPONSE ======")
    print(content)
    print("====== END LLM RESPONSE ======\n")

    # ✅ 关键修改就在这里：从输出里“捞”JSON
    match = re.search(r'\{[\s\S]*\}', content)
    if not match:
        print("⚠️ No JSON found in LLM output")
        return []

    json_str = match.group(0)

    try:
        obj = json.loads(json_str)
    except Exception as e:
        print("⚠️ JSON parse failed:", e)
        print("⚠️ Extracted JSON was:")
        print(json_str)
        return []

    return obj.get("operations", [])
# ======================
# 7. 执行 operations（add / update）
# ======================
def apply_operations(profile, operations, max_per_cat=5):
    for op in operations:
        cat = op.get("category")
        if cat not in PREFIX:
            continue

        profile.setdefault(cat, {})

        if op.get("op") == "add":
            content = op.get("content")
            if not content:
                continue
            nid = next_id(profile, cat)
            profile[cat][nid] = content

        elif op.get("op") == "update":
            _id = op.get("id")
            content = op.get("content")
            if not _id or not content:
                continue
            if _id in profile[cat]:
                profile[cat][_id] = content

    # 每类最多保留 max_per_cat 条
    for cat, pref in PREFIX.items():
        items = profile.get(cat, {})
        keys = sorted(
            items.keys(),
            key=lambda k: int(k[len(pref):]) if k[len(pref):].isdigit() else 10**9
        )
        for k in keys[max_per_cat:]:
            items.pop(k, None)

    return profile


# ======================
# 8. 主流程：每个 sample 独立 load / apply / save
# ======================

for user_id, user_data in data.items():
    for split in ("history", "query"):
        for sample in user_data.get(split, []):
            logs = sample.get("logs", [])
            if not logs:
                continue

            sample_id = sample.get("sample_id")

            # 🔁 每次都从磁盘读 profile
            profile = load_profile(PROFILE_PATH)

            operations = get_profile_operations(profile, logs)
            profile = apply_operations(profile, operations)
            
            # 💾 每次都立刻写回
            save_profile(PROFILE_PATH, profile)

            print("------ PROFILE AFTER APPLY ------")
            print(json.dumps(profile, ensure_ascii=False, indent=2))
            print("------ END PROFILE ------\n")

            op_record = {
                "user_id": user_id,
                "split": split,
                "sample_id": sample_id,
                "operations": operations
            }

            # 📌 每一步都立即写 ops（追加式）
            if os.path.exists(OPS_LOG_PATH):
                with open(OPS_LOG_PATH, "r", encoding="utf-8") as f:
                    existing_ops = json.load(f)
            else:
                existing_ops = []

            existing_ops.append(op_record)

            with open(OPS_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(existing_ops, f, ensure_ascii=False, indent=2)

            print(f"✅ updated profile | {user_id} | {split} | {sample_id}")


# ======================
# 9. 写 ops 日志
# ======================

print("🎉 Done.")
print(f"Profile saved to: {PROFILE_PATH}")
print(f"Ops log saved to: {OPS_LOG_PATH}")




