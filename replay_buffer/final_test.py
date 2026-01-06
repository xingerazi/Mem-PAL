import json
import os

INPUT_JSON_PATH = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\replay_buffer\dia\0000.json"
OUTPUT_JSON_PATH = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\replay_buffer\dia_segmented\0000.json"


def split_dialogue_into_segments(dialogue: dict):
    """
    Split a dialogue into segments based on user.action == '话题询问'.
    Each segment is a dict of turns with only user/assistant content.
    """
    segments = []
    current_segment = None
    seg_turn_idx = 1

    for turn_id in sorted(dialogue.keys(), key=lambda x: int(x.split("_")[1])):
        turn = dialogue[turn_id]
        user = turn.get("user", {})
        assistant = turn.get("assistant", {})

        user_action = user.get("action", "")
        user_content = user.get("content", "")
        assistant_content = assistant.get("content", "")

        # New segment starts
        if user_action == "话题询问":
            if current_segment:
                segments.append(current_segment)

            current_segment = {}
            seg_turn_idx = 1

        # If no segment started yet (safety, normally shouldn't happen)
        if current_segment is None:
            current_segment = {}
            seg_turn_idx = 1

        current_segment[f"turn_{seg_turn_idx}"] = {
            "user": user_content,
            "assistant": assistant_content
        }
        seg_turn_idx += 1

    if current_segment:
        segments.append(current_segment)

    return segments


def main():
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = {}

    for user_id, user_block in data.items():
        print(f"[INFO] Processing user_id = {user_id}")
        all_segments = []

        for section in ["history", "query"]:
            if section not in user_block:
                continue

            for item in user_block[section]:
                dialogue = item.get("dialogue")
                if not dialogue:
                    continue

                segments = split_dialogue_into_segments(dialogue)
                all_segments.extend(segments)

        output[user_id] = all_segments

        print(f"  -> extracted {len(all_segments)} segments")

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("[DONE] Saved to:", OUTPUT_JSON_PATH)


if __name__ == "__main__":
    main()