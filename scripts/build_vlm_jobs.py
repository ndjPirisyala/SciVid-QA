#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", required=True)
    parser.add_argument("--contact_sheets", required=True)
    parser.add_argument("--out", default="outputs/vlm_jobs/dev_10_vlm_jobs.jsonl")
    args = parser.parse_args()

    qa = load_json(args.qa)
    contact_sheets = load_json(args.contact_sheets)

    qa_by_prompt = {
        item["prompt_id"]: item["items"]
        for item in qa["qa_items"]
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_jobs = 0

    with open(out_path, "w") as f:
        for record in contact_sheets["records"]:
            condition_id = record["condition_id"]
            prompt_id = record["prompt_id"]
            contact_sheet_path = record["contact_sheet_path"]

            if prompt_id not in qa_by_prompt:
                raise ValueError(f"No QA items found for prompt_id: {prompt_id}")

            if not Path(contact_sheet_path).exists():
                raise FileNotFoundError(f"Missing contact sheet: {contact_sheet_path}")

            for qa_item in qa_by_prompt[prompt_id]:
                job = {
                    "job_id": f"{condition_id}__{prompt_id}__{qa_item['id']}",
                    "condition_id": condition_id,
                    "prompt_id": prompt_id,
                    "contact_sheet_path": contact_sheet_path,
                    "qa_item": qa_item
                }
                f.write(json.dumps(job) + "\n")
                num_jobs += 1

    print(f"Wrote {num_jobs} VLM jobs to {out_path}")


if __name__ == "__main__":
    main()