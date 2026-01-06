import json
import os

INPUT_DIR = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\replay_buffer\dia"
OUTPUT_DIR = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\replay_buffer\dia_segmented_time"


def split_dialogue_into_segments(dialogue: dict):
    """
    Split dialogue into segments by user.action == '话题询问'.
    Return:
      - segments: list of segment dicts
      - topic_query_count: number of '话题询问'
    """
    segments = []
    current_segment = None
    seg_turn_idx = 1
    topic_query_count = 0

    for turn_id in sorted(dialogue.keys(), key=lambda x: int(x.split("_")[1])):
        turn = dialogue[turn_id]
        user = turn.get("user", {})
        assistant = turn.get("assistant", {})

        user_action = user.get("action", "")
        user_content = user.get("content", "")
        assistant_content = assistant.get("content", "")

        # New segment trigger
        if user_action == "话题询问":
            topic_query_count += 1
            if current_segment:
                segments.append(current_segment)
            current_segment = {}
            seg_turn_idx = 1

        # Safety: if dialogue doesn't start with 话题询问
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

    return segments, topic_query_count


def process_file(input_path, output_path, file_prefix):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = {}
    total_topic_queries = 0
    total_samples = 0

    for user_id, user_block in data.items():
        samples = []
        user_topic_queries = 0

        for section in ["history", "query"]:
            if section not in user_block:
                continue

            for item_idx, item in enumerate(user_block[section]):
                dialogue = item.get("dialogue")
                if not dialogue:
                    continue

                segments, tq_count = split_dialogue_into_segments(dialogue)

                sample = {
                    "sample_id": f"{file_prefix}_sample{item_idx}",
                    "dialogue_timestamp": item.get("dialogue_timestamp"),
                    "dialogue": segments      # ✅ dialogue 是数组（切好的）
                }

                samples.append(sample)
                user_topic_queries += tq_count
                total_samples += 1

        output[user_id] = samples
        total_topic_queries += user_topic_queries

        print(
            f"    [USER {user_id}] "
            f"话题询问={user_topic_queries} | samples={len(samples)}"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return total_topic_queries, total_samples


def main():
    print("====== Batch Dialogue Segmentation ======")

    for i in range(100):
        fname = f"{i:04d}.json"
        file_prefix = f"{i:04d}"

        input_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, fname)

        if not os.path.exists(input_path):
            print(f"[SKIP] {fname} not found")
            continue

        print(f"\n[FILE] {fname}")
        tq, sample_cnt = process_file(input_path, output_path, file_prefix)

        print(
            f"[SUMMARY] {fname} | "
            f"话题询问总数={tq} | sample总数={sample_cnt}"
        )

    print("\n====== DONE ======")


if __name__ == "__main__":
    main()