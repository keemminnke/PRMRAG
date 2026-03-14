#!/usr/bin/env python3
"""Prepare data for PRM scaling experiment.

Convert FlashRAG test data to generate_trajectories.py format,
sample N questions per dataset with fixed seed.

Usage:
    python scripts/prepare_prm_data.py \
        --datasets popqa hotpotqa 2wikimultihopqa musique \
        --limit 500 \
        --seed 42 \
        --output-dir data/prm_scaling
"""

import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["popqa", "hotpotqa", "2wikimultihopqa", "musique"])
    parser.add_argument("--data-dir", type=str, default="data/flashrag")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/prm_scaling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ds in args.datasets:
        input_path = os.path.join(args.data_dir, ds, "test.jsonl")
        output_path = os.path.join(args.output_dir, f"{ds}_{args.limit}.jsonl")

        # Load FlashRAG format
        questions = []
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))

        # Seed-based sampling
        random.seed(args.seed)
        if len(questions) > args.limit:
            questions = random.sample(questions, args.limit)

        # Convert format: FlashRAG -> generate_trajectories.py
        converted = []
        for q in questions:
            # FlashRAG uses 'golden_answers' (list), 'id', 'question'
            # generate_trajectories.py expects '_id', 'question', 'answer'
            gold_answers = q.get("golden_answers", [])
            answer = gold_answers[0] if gold_answers else q.get("answer", "")

            converted.append({
                "_id": q.get("id", q.get("_id", "")),
                "question": q["question"],
                "answer": answer,
                "golden_answers": gold_answers,  # Keep for multi-answer eval
            })

        with open(output_path, "w") as f:
            for item in converted:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"[{ds}] {len(questions)} -> {output_path}")


if __name__ == "__main__":
    main()
