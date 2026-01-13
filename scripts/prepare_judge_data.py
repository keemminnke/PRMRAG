#!/usr/bin/env python3
"""Prepare clean data for GPT-OSS-120B judge labeling.

This script extracts only the essential fields needed for judge evaluation:
- question
- gold_answer
- steps (only step_num, content, and rpe_label for comparison)

Removes all noisy metadata (mc_before, mc_after, rpe scores, etc.)
"""

import json
import argparse
from pathlib import Path


def prepare_judge_sample(trajectory: dict) -> dict:
    """Extract structured fields for judge evaluation.

    Args:
        trajectory: Full trajectory with all metadata

    Returns:
        Clean sample with only reasoning fields (no metadata, no RPE labels)
    """
    # Extract essential trajectory info
    clean_sample = {
        "question_id": trajectory["question_id"],
        "question": trajectory["question"],
        "gold_answer": trajectory["gold_answer"],
        "predicted_answer": trajectory.get("predicted_answer"),
        "is_correct": trajectory.get("is_correct"),
        "num_steps": len(trajectory["steps"]),
    }

    # Extract only structured reasoning fields (no content, no metadata)
    clean_steps = []
    for step in trajectory["steps"]:
        clean_step = {
            "step_num": step["step_num"],
            "thought": step.get("thought"),
            "action": step.get("action"),
            "action_input": step.get("action_input"),
            "observation": step.get("observation"),
            "sub_answer": step.get("sub_answer"),
        }
        clean_steps.append(clean_step)

    clean_sample["steps"] = clean_steps

    return clean_sample


def main():
    parser = argparse.ArgumentParser(description='Prepare data for judge labeling')
    parser.add_argument('input_file', type=Path, help='Input JSONL file (full data)')
    parser.add_argument('output_file', type=Path, help='Output JSONL file (clean for judge)')
    parser.add_argument('--limit', type=int, help='Limit number of samples')

    args = parser.parse_args()

    # Check input exists
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    # Create output directory
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Preparing judge data from {args.input_file}")
    print(f"Output: {args.output_file}")
    print()

    # Process trajectories
    count = 0
    total_steps = 0

    with open(args.input_file, 'r') as infile, open(args.output_file, 'w') as outfile:
        for line in infile:
            if args.limit and count >= args.limit:
                break

            trajectory = json.loads(line)
            clean_sample = prepare_judge_sample(trajectory)

            # Write to output
            outfile.write(json.dumps(clean_sample) + '\n')

            count += 1
            total_steps += len(clean_sample["steps"])

            if count % 100 == 0:
                print(f"Processed {count} trajectories...")

    print()
    print(f"✓ Prepared {count} trajectories")
    print(f"✓ Total {total_steps} steps to judge")
    print(f"✓ Output: {args.output_file}")
    print()

    # Show sample
    print("Sample output:")
    with open(args.output_file, 'r') as f:
        sample = json.loads(f.readline())
        print(json.dumps(sample, indent=2)[:500] + "...")

    # Size comparison
    import os
    input_size = os.path.getsize(args.input_file) / 1024 / 1024
    output_size = os.path.getsize(args.output_file) / 1024 / 1024
    reduction = (1 - output_size / input_size) * 100

    print()
    print(f"Size: {input_size:.1f}MB → {output_size:.1f}MB ({reduction:.1f}% reduction)")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
