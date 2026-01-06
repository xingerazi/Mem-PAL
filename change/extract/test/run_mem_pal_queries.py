import json
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI
from prompt import Requirement_Generation_Prompt

import os
from dotenv import load_dotenv

load_dotenv()  # 自动读取当前目录下的 .env
# ======================
# OpenAI client
# ======================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")

# ======================
# Functions
# ======================

def build_requirement_prompt(dialogue: dict, user_query: str) -> str:
    return Requirement_Generation_Prompt.format(
        dialogue=json.dumps(dialogue, ensure_ascii=False, indent=2),
        user_query=user_query
    )


def call_llm(prompt: str) -> str:
    if LLM_BACKEND == "openai":
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()

    else:
        # TODO: vllm / local / other backend
        pass
# ======================
# Core logic (Requirement only)
# ======================

def process_topic(
    user_id: str,
    sample_id: str,
    topic_id: str,
    dialogue: dict,
    topic_data: dict
) -> None:
    """
    只做 requirement generation，直接 print
    """

    user_query = topic_data["user_query"]

    prompt = build_requirement_prompt(
        dialogue=dialogue,
        user_query=user_query
    )

    output = call_llm(prompt)

    print("=" * 80)
    print(f"user_id   : {user_id}")
    print(f"sample_id : {sample_id}")
    print(f"topic_id  : {topic_id}")
    print(f"user_query: {user_query}")
    print("\n[Generated Requirement]")
    print(output)
    print("=" * 80)


def process_query_sample(
    user_id: str,
    query_sample: Dict[str, Any]
) -> None:

    sample_id = query_sample["sample_id"]
    dialogue = query_sample["dialogue"]
    topics = query_sample.get("topics", {})

    for topic_id, topic_data in topics.items():
        process_topic(
            user_id=user_id,
            sample_id=sample_id,
            topic_id=topic_id,
            dialogue=dialogue,
            topic_data=topic_data
        )


def run_on_user_file(user_json_path: Path) -> None:
    """
    输入：单个 user 的 json（如 0000.json）
    直接打印所有 topic 的 requirement
    """

    with user_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    user_id, user_data = next(iter(data.items()))

    for query_sample in user_data.get("query", []):
        process_query_sample(
            user_id=user_id,
            query_sample=query_sample
        )


# ======================
# Entry
# ======================

if __name__ == "__main__":
    run_on_user_file(Path(r"E:\gitmyrepo\mem_pal_self\Mem-PAL\output_users\0000.json"))
