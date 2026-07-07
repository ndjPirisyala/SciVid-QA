#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict


COUNT_MAP = {
    "zero": "0",
    "none": "0",
    "one": "1",
    "a": "1",
    "an": "1",
    "single": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def normalize_text(x):
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = x.replace("-", "_")
    x = re.sub(r"[^a-z0-9_ ]+", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_count(x):
    x = normalize_text(x)
    if x in COUNT_MAP:
        return COUNT_MAP[x]

    match = re.search(r"\d+", x)
    if match:
        return match.group(0)

    return x


def score_answer(answer_type, gold, pred):
    gold_n = normalize_text(gold)
    pred_n = normalize_text(pred)

    if pred_n in {"unclear", "unknown", "not_visible", "ambiguous", ""}:
        return 0.0, "unclear"

    if answer_type == "yes_no":
        if pred_n.startswith("yes"):
            pred_n = "yes"
        elif pred_n.startswith("no"):
            pred_n = "no"
        return (1.0 if pred_n == gold_n else 0.0), pred_n

    if answer_type == "count":
        gold_c = normalize_count(gold)
        pred_c = normalize_count(pred)
        return (1.0 if pred_c == gold_c else 0.0), pred_c

    if answer_type == "multiple_choice":
        return (1.0 if pred_n == gold_n else 0.0), pred_n

    if answer_type == "short_phrase":
        return (1.0 if gold_n in pred_n or pred_n in gold_n else 0.0), pred_n

    raise ValueError(f"Unknown answer_type: {answer_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out_csv", default="outputs/scores/item_scores.csv")
    parser.add_argument("--out_summary", default="outputs/scores/summary.json")
    args = parser.parse_args()

    answers_path = Path(args.answers)
    out_csv = Path(args.out_csv)
    out_summary = Path(args.out_summary)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    with open(answers_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            rec = json.loads(line)
            vlm = rec["vlm_answer"]

            visibility = normalize_text(vlm.get("visibility", ""))
            observable = visibility == "visible"

            pred_answer = vlm.get("answer", "")
            raw_score, normalized_pred = score_answer(
                rec["answer_type"],
                rec["gold_answer"],
                pred_answer,
            )

            weight = float(rec.get("weight", 1.0))

            # If the VLM says the relevant evidence is not visible or ambiguous,
            # do not award scientific correctness points.
            if not observable and weight > 0:
                raw_score = 0.0

            weighted_score = raw_score * weight

            rows.append({
                "job_id": rec["job_id"],
                "condition_id": rec["condition_id"],
                "prompt_id": rec["prompt_id"],
                "qa_item_id": rec["qa_item_id"],
                "category": rec.get("category", "unknown"),
                "answer_type": rec["answer_type"],
                "question": rec["question"],
                "gold_answer": rec["gold_answer"],
                "vlm_answer": pred_answer,
                "normalized_vlm_answer": normalized_pred,
                "visibility": visibility,
                "observable": observable,
                "confidence": vlm.get("confidence", ""),
                "weight": weight,
                "raw_score": raw_score,
                "weighted_score": weighted_score,
                "short_reason": vlm.get("short_reason", ""),
            })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_condition = defaultdict(lambda: {"score": 0.0, "weight": 0.0, "observable": 0, "total": 0})
    by_condition_category = defaultdict(lambda: {"score": 0.0, "weight": 0.0, "observable": 0, "total": 0})

    for r in rows:
        condition = r["condition_id"]
        category = r["category"]
        weight = r["weight"]

        by_condition[condition]["score"] += r["weighted_score"]
        by_condition[condition]["weight"] += weight
        by_condition[condition]["observable"] += int(r["observable"])
        by_condition[condition]["total"] += 1

        key = f"{condition}::{category}"
        by_condition_category[key]["score"] += r["weighted_score"]
        by_condition_category[key]["weight"] += weight
        by_condition_category[key]["observable"] += int(r["observable"])
        by_condition_category[key]["total"] += 1

    summary = {
        "answers_file": str(answers_path),
        "num_items": len(rows),
        "by_condition": {},
        "by_condition_category": {},
    }

    for condition, s in by_condition.items():
        summary["by_condition"][condition] = {
            "scientific_qa_score": 100.0 * s["score"] / s["weight"] if s["weight"] > 0 else None,
            "observability_rate": 100.0 * s["observable"] / s["total"] if s["total"] > 0 else None,
            "num_items": s["total"],
        }

    for key, s in by_condition_category.items():
        summary["by_condition_category"][key] = {
            "scientific_qa_score": 100.0 * s["score"] / s["weight"] if s["weight"] > 0 else None,
            "observability_rate": 100.0 * s["observable"] / s["total"] if s["total"] > 0 else None,
            "num_items": s["total"],
        }

    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote item scores to {out_csv}")
    print(f"Wrote summary to {out_summary}")


if __name__ == "__main__":
    main()