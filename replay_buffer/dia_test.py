import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from prompt import InteractionUnit_Extractor_Prompt
# ======================
# 0. Env
# ======================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ======================
# 1. Path
# ======================
INPUT_JSON_PATH = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\replay_buffer\dia\0000.json"



# ======================
# 3. Load 第一个 dialogue
# ======================
with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 只取第一个 user、query、dialogue
user_id = list(data.keys())[0]
first_dialogue = data[user_id]["query"][0]["dialogue"]

print(f"[INFO] Testing user_id={user_id}, sample_id={data[user_id]['query'][0]['sample_id']}")
print("[INFO] Dialogue preview (keys):", list(first_dialogue.keys()))

# ======================
# 4. Call LLM
# ======================
prompt = InteractionUnit_Extractor_Prompt.format(
    dialogue=json.dumps(first_dialogue, ensure_ascii=False, indent=2)
)

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": prompt}
    ],
    temperature=0
)

print("\n================ RAW MODEL OUTPUT ================\n")
print(resp.choices[0].message.content)
print("\n==================================================\n")