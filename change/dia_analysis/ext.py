# 删除 history 和 query 中的 logs 字段

import json
from pathlib import Path

INPUT_DIR = Path(r"E:\gitmyrepo\mem_pal_self\Mem-PAL\output_users")   
OUTPUT_DIR = Path(r"E:\gitmyrepo\mem_pal_self\Mem-PAL\change\dia_analysis\dia") 


def remove_logs(obj):
    if isinstance(obj, dict):
        obj.pop("logs", None)
        for v in obj.values():
            remove_logs(v)
    elif isinstance(obj, list):
        for item in obj:
            remove_logs(item)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for i in range(100):
    input_path = INPUT_DIR / f"{i:04d}.json"
    output_path = OUTPUT_DIR / f"{i:04d}.json"

    if not input_path.exists():
        continue

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    remove_logs(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


        