import os
import json
import re
import uuid
from dataclasses import asdict
from typing import Dict, Any, Iterable, List

from dotenv import load_dotenv
from openai import OpenAI

from ..prompt import InteractionUnit_Build_Prompt
from .interaction_unit import InteractionUnit, UserFeedback, UserFeedbackType

def init_llm_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未设置")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(client: OpenAI, prompt: str, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"无法提取 JSON：\n{text}")
    return json.loads(m.group(0))


def validate_llm_output(d: Dict[str, Any]) -> Dict[str, Any]:
    required = {"topic", "user_query", "user_feedback", "insight"}
    if not required.issubset(d):
        raise ValueError(f"LLM 输出字段不完整: {d}")

    if d["user_feedback"] not in {"confirm", "reject", "revise"}:
        raise ValueError(f"user_feedback 非法: {d['user_feedback']}")

    return d


def make_iu_id(file_id: str, sample_idx: int, segment_idx: int) -> str:
    """
    iu_p0000s0d1
    """
    return f"iu_p{file_id}s{sample_idx}d{segment_idx}"


def build_interaction_unit(
    iu_id: str,
    llm_out: Dict[str, Any],
    segment: Dict[str, Any],
    timestamp: str,
    file_id: str,          # ✅ 新增
) -> InteractionUnit:
    return InteractionUnit(
        iu_id=iu_id,
        person_id=f"pal_{file_id}",   # ✅ 关键一行
        topic=llm_out["topic"],
        user_query=llm_out["user_query"],
        trajectory=segment,
        user_feedback=UserFeedback(
            type=UserFeedbackType(llm_out["user_feedback"])
        ),
        insight=llm_out["insight"],
        timestamp=timestamp
    )



def iter_interaction_units(
    data: Dict[str, Any],
    file_id: str,
    client: OpenAI,
    model: str
) -> Iterable[InteractionUnit]:
    """
    从一个 dia_segmented 文件中，逐条 yield InteractionUnit
    """
    for user_id, samples in data.items():
        for sample_idx, sample in enumerate(samples):
            timestamp = sample.get("dialogue_timestamp")
            segments = sample.get("dialogue", [])

            for seg_idx, segment in enumerate(segments):
                iu_id = make_iu_id(file_id, sample_idx, seg_idx)

                prompt = InteractionUnit_Build_Prompt.format(
                    dialogue=json.dumps(segment, ensure_ascii=False)
                )

                raw = call_llm(client, prompt, model)
                llm_out = extract_json(raw)
                llm_out = validate_llm_output(llm_out)

                yield build_interaction_unit(
                    iu_id=iu_id,
                    llm_out=llm_out,
                    segment=segment,
                    timestamp=timestamp,
                    file_id=file_id,      # ✅ 传进来
                )


def write_units_jsonl(units: Iterable[InteractionUnit], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for iu in units:
            f.write(json.dumps(asdict(iu), ensure_ascii=False) + "\n")
            f.flush()




def main():
    BASE_INPUT_DIR = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\replay_buffer\dia_segmented_time"
    BASE_OUTPUT_DIR = r"E:\gitmyrepo\mem_pal_self\Mem-PAL\replay_buffer\interaction_units"

    client = init_llm_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    for idx in range(67, 100):
        file_id = f"{idx:04d}"

        input_path = os.path.join(BASE_INPUT_DIR, f"{file_id}.json")
        output_path = os.path.join(BASE_OUTPUT_DIR, f"{file_id}_units.jsonl")

        if not os.path.exists(input_path):
            print(f"⚠️ Skip {file_id}: input file not found")
            continue

        print(f"▶ Processing {file_id}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        units = iter_interaction_units(
            data=data,
            file_id=file_id,
            client=client,
            model=model
        )

        write_units_jsonl(units, output_path)

        print(f"✅ Finished {file_id} → {output_path}")


if __name__ == "__main__":
    main()