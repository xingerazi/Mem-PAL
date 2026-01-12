import json

# 1. 读取原始 JSON
with open(r"E:\gitmyrepo\mem_pal_self\Mem-PAL\data_synthesis_v2\data_en\background\background_en.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 抽取 id -> name
name_cn = {
    user_id: user_info["name"]
    for user_id, user_info in data.items()
    if "name" in user_info
}

name_en = {
    user_id: user_info["name"]
    for user_id, user_info in data.items()
    if "name" in user_info
}

# 3. 写入新文件
with open("name_en.json", "w", encoding="utf-8") as f:
    json.dump(name_en, f, ensure_ascii=False, indent=2)

print("已生成 name_en.json")
