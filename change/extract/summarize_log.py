import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from prompt import PROFILE_CONSOLIDATE_PROMPT


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
# 1. 路径
# ======================
PROFILE_PATH = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\change\extract\0000_profile.json"
OUTPUT_PATH = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\change\extract\0000_profile_consolidated.json"


# ======================
# 2. 读取原 profile
# ======================
with open(PROFILE_PATH, "r", encoding="utf-8") as f:
    profile = json.load(f)


# ======================
# 3. 调用 LLM 做 consolidation
# ======================
prompt = PROFILE_CONSOLIDATE_PROMPT.format(
    profile=json.dumps(profile, ensure_ascii=False, indent=2)
)

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "system", "content": prompt}],
    temperature=0.2
)

raw = resp.choices[0].message.content

print("====== RAW CONSOLIDATED PROFILE ======")
print(raw)
print("====== END ======")

# ======================
# 4. 稳健解析 JSON
# ======================
match = re.search(r'\{[\s\S]*\}', raw)
if not match:
    raise ValueError("No JSON object found in model output")

new_profile = json.loads(match.group(0))


# ======================
# 5. 写回结果
# ======================
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(new_profile, f, ensure_ascii=False, indent=2)

print("🎉 Consolidated profile saved to:")
print(OUTPUT_PATH)