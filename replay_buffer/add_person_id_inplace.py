import json
from pathlib import Path

# =========================
# Config
# =========================

FILE_PATH = Path(
    r"E:\gitmyrepo\mem_pal_self\Mem-PAL\replay_buffer\interaction_units\0000_units.jsonl"
)

PERSON_ID = "pal_0000"


def main():
    assert FILE_PATH.exists(), f"File not found: {FILE_PATH}"

    # 先全部读入内存（0000 一般不大，安全）
    lines = FILE_PATH.read_text(encoding="utf-8").splitlines()

    new_lines = []
    count = 0

    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            unit = json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON decode error at line {line_no}") from e

        # ✅ 核心操作：直接覆盖 / 添加 person_id
        unit["person_id"] = PERSON_ID

        new_lines.append(json.dumps(unit, ensure_ascii=False))
        count += 1

    # 覆盖写回原文件
    FILE_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    print(f"✅ Done. Updated {count} units in-place.")
    print(f"📄 File: {FILE_PATH}")


if __name__ == "__main__":
    main()
