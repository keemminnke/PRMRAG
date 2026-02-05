#!/usr/bin/env python3
"""Clean judge labels data for critic training.

Fixes:
1. Remove steps with "Failed to parse response" reasoning
2. Convert last reason step to answer if predicted_answer exists
3. Optionally exclude reason steps from training

Usage:
    python scripts/clean_judge_data.py \
        --input outputs/judge_labels_500q_16s.jsonl \
        --output outputs/judge_labels_clean.jsonl \
        --exclude-reason
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def clean_trajectory(data: dict, exclude_reason: bool = False) -> dict:
    """Clean a single trajectory.

    Returns:
        Cleaned trajectory dict, or None if trajectory should be excluded
    """
    steps = data['steps']
    cleaned_steps = []

    for step in steps:
        reasoning = step.get('judge_reasoning', '')
        step_type = step.get('step_type', '')

        # 1. Skip steps with "Failed to parse response"
        if 'Failed to parse' in reasoning:
            continue

        # 2. Optionally skip reason steps
        if exclude_reason and step_type == 'reason':
            continue

        cleaned_steps.append(step)

    # 3. Fix missing answer step
    # If no answer step but predicted_answer exists, convert last step to answer
    has_answer = any(s.get('step_type') == 'answer' for s in cleaned_steps)
    predicted_answer = data.get('predicted_answer', '')

    if not has_answer and predicted_answer and cleaned_steps:
        last_step = cleaned_steps[-1]
        # Convert to answer step
        last_step['step_type'] = 'answer'
        last_step['answer'] = predicted_answer
        # Remove search if present (it's now an answer step)
        if 'search' in last_step:
            del last_step['search']

    # Skip trajectory if no valid steps remain
    if not cleaned_steps:
        return None

    # Update step_ids
    for i, step in enumerate(cleaned_steps):
        step['step_id'] = i + 1

    cleaned_data = {
        'trajectory_id': data['trajectory_id'],
        'question': data['question'],
        'gold_answer': data['gold_answer'],
        'predicted_answer': data.get('predicted_answer', ''),
        'is_correct': data.get('is_correct', False),
        'steps': cleaned_steps,
    }

    return cleaned_data


def main():
    parser = argparse.ArgumentParser(description="Clean judge labels data")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--exclude-reason", action="store_true",
                        help="Exclude reason steps (only keep search+answer)")
    args = parser.parse_args()

    print("=" * 70)
    print("CLEANING JUDGE LABELS DATA")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Exclude reason steps: {args.exclude_reason}")
    print()

    # Statistics
    total_trajs = 0
    kept_trajs = 0
    removed_trajs = 0

    total_steps_before = 0
    total_steps_after = 0

    removed_failed_parse = 0
    removed_reason = 0
    converted_to_answer = 0

    step_types_before = Counter()
    step_types_after = Counter()

    with open(args.input, 'r') as f_in, open(args.output, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            total_trajs += 1

            # Count before
            for step in data['steps']:
                total_steps_before += 1
                step_types_before[step.get('step_type', 'unknown')] += 1

            # Check for conversions
            has_answer_before = any(s.get('step_type') == 'answer' for s in data['steps'])

            # Clean
            cleaned = clean_trajectory(data, exclude_reason=args.exclude_reason)

            if cleaned is None:
                removed_trajs += 1
                continue

            kept_trajs += 1

            # Count after
            has_answer_after = any(s.get('step_type') == 'answer' for s in cleaned['steps'])
            if not has_answer_before and has_answer_after:
                converted_to_answer += 1

            for step in cleaned['steps']:
                total_steps_after += 1
                step_types_after[step.get('step_type', 'unknown')] += 1

            f_out.write(json.dumps(cleaned, ensure_ascii=False) + '\n')

    # Calculate removed
    removed_failed_parse = sum(1 for _ in open(args.input)
                               for s in json.loads(_)['steps']
                               if 'Failed to parse' in s.get('judge_reasoning', ''))

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Trajectories: {total_trajs} → {kept_trajs} (removed {removed_trajs})")
    print(f"Steps: {total_steps_before} → {total_steps_after} (removed {total_steps_before - total_steps_after})")
    print(f"Converted to answer: {converted_to_answer}")
    print()
    print("Step types BEFORE:")
    for st, cnt in step_types_before.most_common():
        print(f"  {st}: {cnt}")
    print()
    print("Step types AFTER:")
    for st, cnt in step_types_after.most_common():
        print(f"  {st}: {cnt}")
    print()
    print(f"✓ Saved to: {args.output}")


if __name__ == "__main__":
    main()
