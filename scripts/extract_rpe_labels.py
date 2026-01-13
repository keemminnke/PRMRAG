#!/usr/bin/env python3
"""Extract RPE labels from trajectory data for consensus filtering.

This script extracts only question_id, step_num, and RPE label
to compare with Judge labels later.
"""

import json
import argparse
from pathlib import Path


def extract_rpe_labels(trajectory: dict) -> dict:
    """Extract RPE labels for each step.

    Args:
        trajectory: Full trajectory with RPE labels

    Returns:
        Minimal structure with only IDs and RPE labels
    """
    rpe_sample = {
        "question_id": trajectory["question_id"],
        "steps": []
    }

    for step in trajectory["steps"]:
        rpe_step = {
            "step_num": step["step_num"],
            "rpe_label": step.get("label"),  # "good" or "bad"
        }
        rpe_sample["steps"].append(rpe_step)

    return rpe_sample


def main():
    parser = argparse.ArgumentParser(description='Extract RPE labels for consensus filtering')
    parser.add_argument('input_file', type=Path, help='Input JSONL file (full data)')
    parser.add_argument('output_file', type=Path, help='Output JSONL file (RPE labels only)')
    parser.add_argument('--limit', type=int, help='Limit number of samples')

    args = parser.parse_args()

    # Check input exists
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    # Create output directory
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting RPE labels from {args.input_file}")
    print(f"Output: {args.output_file}")
    print()

    # Process trajectories
    count = 0
    total_steps = 0
    label_counts = {"good": 0, "bad": 0, "none": 0}

    with open(args.input_file, 'r') as infile, open(args.output_file, 'w') as outfile:
        for line in infile:
            if args.limit and count >= args.limit:
                break

            trajectory = json.loads(line)
            rpe_sample = extract_rpe_labels(trajectory)

            # Count labels
            for step in rpe_sample["steps"]:
                label = step["rpe_label"]
                if label == "good":
                    label_counts["good"] += 1
                elif label == "bad":
                    label_counts["bad"] += 1
                else:
                    label_counts["none"] += 1

            # Write to output
            outfile.write(json.dumps(rpe_sample) + '\n')

            count += 1
            total_steps += len(rpe_sample["steps"])

            if count % 100 == 0:
                print(f"Processed {count} trajectories...")

    print()
    print(f"✓ Extracted {count} trajectories")
    print(f"✓ Total {total_steps} steps")
    print()
    print("RPE Label Distribution:")
    print(f"  Good: {label_counts['good']} ({label_counts['good']/total_steps*100:.1f}%)")
    print(f"  Bad:  {label_counts['bad']} ({label_counts['bad']/total_steps*100:.1f}%)")
    if label_counts['none'] > 0:
        print(f"  None: {label_counts['none']} ({label_counts['none']/total_steps*100:.1f}%)")
    print()
    print(f"✓ Output: {args.output_file}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
