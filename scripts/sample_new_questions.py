#!/usr/bin/env python3
"""Sample new questions that don't overlap with existing ones.

Usage:
    python scripts/sample_new_questions.py \
        --source data/raw/questions/hotpotqa_train.jsonl \
        --existing outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl \
        --output data/raw/questions/hotpotqa_train_round2.jsonl \
        --num 1000 \
        --seed 42
"""

import argparse
import json
import random


def extract_question_ids_from_trajectories(path: str) -> set:
    """Extract unique question IDs from trajectory file.

    trajectory_id format: {question_id}_sample_{N}
    """
    ids = set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            tid = d["trajectory_id"]
            # Remove _sample_N suffix to get question ID
            qid = tid.rsplit("_sample_", 1)[0]
            ids.add(qid)
    return ids


def extract_question_ids_from_questions(path: str) -> set:
    """Extract question IDs from a questions JSONL file."""
    ids = set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            ids.add(d["_id"])
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="Source question file (e.g., hotpotqa_train.jsonl)")
    parser.add_argument("--existing", nargs="+", required=True,
                        help="Existing trajectory/question files to exclude. "
                             "Auto-detects format (trajectory vs question JSONL)")
    parser.add_argument("--output", required=True,
                        help="Output path for new questions")
    parser.add_argument("--num", type=int, default=1000,
                        help="Number of questions to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Collect existing question IDs
    existing_ids = set()
    for path in args.existing:
        print(f"Loading existing IDs from: {path}")
        with open(path) as f:
            first_line = json.loads(f.readline())

        if "trajectory_id" in first_line:
            ids = extract_question_ids_from_trajectories(path)
        elif "_id" in first_line:
            ids = extract_question_ids_from_questions(path)
        else:
            raise ValueError(f"Unknown format in {path}: keys={list(first_line.keys())}")

        print(f"  Found {len(ids)} unique question IDs")
        existing_ids |= ids

    print(f"\nTotal existing question IDs: {len(existing_ids)}")

    # Load source questions
    candidates = []
    total = 0
    with open(args.source) as f:
        for line in f:
            total += 1
            d = json.loads(line)
            if d["_id"] not in existing_ids:
                candidates.append(d)

    print(f"Source: {total} total, {len(candidates)} available (after excluding {total - len(candidates)} existing)")

    if len(candidates) < args.num:
        print(f"WARNING: Only {len(candidates)} available, requested {args.num}")
        args.num = len(candidates)

    # Sample
    random.seed(args.seed)
    sampled = random.sample(candidates, args.num)

    # Write output
    with open(args.output, "w") as f:
        for d in sampled:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nSampled {len(sampled)} questions → {args.output}")

    # Quick stats
    types = {}
    levels = {}
    for d in sampled:
        t = d.get("type", "unknown")
        l = d.get("level", "unknown")
        types[t] = types.get(t, 0) + 1
        levels[l] = levels.get(l, 0) + 1

    print(f"Types: {types}")
    print(f"Levels: {levels}")


if __name__ == "__main__":
    main()
