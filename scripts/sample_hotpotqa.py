#!/usr/bin/env python3
"""Sample questions from HotpotQA by difficulty level.

Usage:
    python scripts/sample_hotpotqa.py \
        --output_path data/sampled/hotpotqa_sampled_500.jsonl \
        --easy 100 --medium 300 --hard 100 \
        --seed 42
"""

import json
import random
import argparse
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Sample HotpotQA questions by difficulty")
    parser.add_argument('--input_path', type=str,
                        default='data/raw/questions/hotpotqa_train.jsonl',
                        help='Path to HotpotQA train JSONL')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output sampled JSONL')
    parser.add_argument('--easy', type=int, default=100,
                        help='Number of easy questions')
    parser.add_argument('--medium', type=int, default=300,
                        help='Number of medium questions')
    parser.add_argument('--hard', type=int, default=100,
                        help='Number of hard questions')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)

    # Load and group by level
    questions_by_level = defaultdict(list)

    print(f"Loading questions from {args.input_path}...")
    with open(args.input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            level = data.get('level', 'unknown')
            questions_by_level[level].append(data)

    print(f"Loaded questions by level:")
    for level, questions in questions_by_level.items():
        print(f"  {level}: {len(questions)}")

    # Sample
    target_counts = {
        'easy': args.easy,
        'medium': args.medium,
        'hard': args.hard,
    }

    sampled = []
    for level, count in target_counts.items():
        available = questions_by_level.get(level, [])
        if len(available) < count:
            print(f"Warning: Only {len(available)} {level} questions available, requested {count}")
            count = len(available)

        selected = random.sample(available, count)
        sampled.extend(selected)
        print(f"Sampled {count} {level} questions")

    # Shuffle the final list
    random.shuffle(sampled)

    # Save
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, 'w') as f:
        for q in sampled:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(sampled)} questions to {args.output_path}")
    print(f"  Easy: {args.easy}")
    print(f"  Medium: {args.medium}")
    print(f"  Hard: {args.hard}")


if __name__ == '__main__':
    main()
