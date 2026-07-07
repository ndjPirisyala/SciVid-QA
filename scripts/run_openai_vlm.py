#!/usr/bin/env python3

import argparse
import base64
import json
import os
import re
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """You are a careful visual evaluator for scientific video contact sheets.

You will see a contact sheet made from sampled frames of one generated video.
Answer the given question using ONLY visible evidence in the contact sheet.

Important rules:
- Do not assume the original prompt.
- Do not infer what should be there unless it is visible.
- If the relevant object or process is not clear, use visibility = "not_visible" or "ambiguous".
- Return ONLY valid JSON.
"""


def encode_image_as_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        raise ValueError(f"Unsupported image type: {image_path}")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response:\n{text}")

    return json.loads(match.group(0))


def build_user_prompt(job: dict) -> str:
    qa = job["qa_item"]

    answer_type = qa["answer_type"]
    choices = qa.get("choices")

    schema = {
        "answer": "short answer to the question",
        "visibility": "visible | not_visible | ambiguous",
        "confidence": "low | medium | high",
        "evidence_frames": ["frame labels visible in the contact sheet, e.g. frame_00000"],
        "short_reason": "one brief sentence based only on visible evidence"
    }

    lines = [
        "Answer this scientific QA item from the contact sheet only.",
        "",
        f"Question: {qa['question']}",
        f"Answer type: {answer_type}",
    ]

    if choices:
        lines.append(f"Allowed choices: {choices}")

    lines += [
        "",
        "Return ONLY valid JSON with this schema:",
        json.dumps(schema, indent=2),
        "",
        "For yes/no questions, answer must be exactly yes, no, or unclear.",
        "For count questions, answer with a number word or digit if visible; otherwise use unclear.",
        "For multiple-choice questions, answer using one of the allowed choices if possible; otherwise use unclear."
    ]

    return "\n".join(lines)


def run_one_job(client: OpenAI, model: str, job: dict) -> dict:
    image_path = Path(job["contact_sheet_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"Missing contact sheet: {image_path}")

    image_data_url = encode_image_as_data_url(image_path)
    user_prompt = build_user_prompt(job)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": "high"
                    },
                ],
            },
        ],
        max_output_tokens=600,
    )

    raw_text = response.output_text
    parsed = extract_json(raw_text)

    return {
    "job_id": job["job_id"],
    "condition_id": job["condition_id"],
    "prompt_id": job["prompt_id"],
    "qa_item_id": job["qa_item"]["id"],
    "category": job["qa_item"]["category"],
    "question": job["qa_item"]["question"],
    "gold_answer": job["qa_item"]["gold_answer"],
    "answer_type": job["qa_item"]["answer_type"],
    "weight": job["qa_item"]["weight"],
    "contact_sheet_path": job["contact_sheet_path"],
    "model": model,
    "vlm_raw_text": raw_text,
    "vlm_answer": parsed,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    args = parser.parse_args()

    client = OpenAI()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0

    with open(args.jobs, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue

            job = json.loads(line)
            print(f"Running {job['job_id']}")

            result = run_one_job(client, args.model, job)
            fout.write(json.dumps(result) + "\n")
            fout.flush()

            n += 1
            if args.limit is not None and n >= args.limit:
                break

    print(f"Wrote {n} VLM answers to {out_path}")


if __name__ == "__main__":
    main()