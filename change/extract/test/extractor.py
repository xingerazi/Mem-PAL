import json
from pathlib import Path

INPUT_PATH = "E:\gitmyrepo\mem_pal_self\Mem-PAL\data_synthesis_v2\data_en\input_en.json"
OUTPUT_DIR = "output_users_en"

# 1. 创建输出目录
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 2. 读取原始 JSON
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, dict):
    raise ValueError("Top-level JSON must be a dict")

# 3. 批量抽取
for user_id, user_data in data.items():
    output_data = {
        user_id: user_data
    }

    output_path = Path(OUTPUT_DIR) / f"{user_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ Extracted {len(data)} users into '{OUTPUT_DIR}/'")