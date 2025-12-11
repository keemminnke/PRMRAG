#!/usr/bin/env python3
"""Re-parse old trajectory data to add structured ReAct fields.

This script:
1. Reads old trajectory data with content but no structured fields
2. Applies the same parsing logic as the improved adaptive_generator
3. Saves re-parsed data with thought, action, observation, sub_answer fields
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.generation.adaptive_generator import parse_rag_content, extract_intermediate_answer


def reparse_step(step: Dict) -> Dict:
    """Re-parse a single step to add structured fields.

    Args:
        step: Original step dict with content field

    Returns:
        Updated step dict with parsed fields
    """
    content = step.get('content', '')
    step_type = 'RAG' if step.get('num_passages', 0) > 0 else 'CoT'

    if step_type == 'RAG':
        # RAG step: use parse_rag_content
        parsed = parse_rag_content(content)

        step['thought'] = parsed['thought']
        step['action'] = parsed['action'] if parsed['action'] else 'Search'
        step['action_input'] = parsed['action_input']
        step['observation'] = parsed['observation']
        step['sub_answer'] = parsed['sub_answer']

    else:
        # CoT step: use semantic mapping
        # Parse content first
        parsed = parse_rag_content(content)

        # Extract intermediate answer
        intermediate_ans = extract_intermediate_answer(content)

        # Semantic mapping for CoT
        step['thought'] = content if parsed['thought'] is None else parsed['thought']
        step['action'] = 'Reason'  # CoT always uses Reason
        step['action_input'] = None  # No search query
        step['observation'] = intermediate_ans  # Intermediate conclusion
        step['sub_answer'] = None  # CoT doesn't use sub_answer

    return step


def reparse_trajectory(trajectory: Dict) -> Dict:
    """Re-parse all steps in a trajectory.

    Args:
        trajectory: Original trajectory dict

    Returns:
        Updated trajectory dict with parsed fields in all steps
    """
    # Re-parse each step
    for step in trajectory.get('steps', []):
        reparse_step(step)

    return trajectory


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Re-parse old trajectory data')
    parser.add_argument('input_file', type=Path, help='Input JSONL file')
    parser.add_argument('output_file', type=Path, help='Output JSONL file')
    parser.add_argument('--limit', type=int, help='Limit number of trajectories to process')

    args = parser.parse_args()

    # Check input file exists
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    # Create output directory if needed
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Re-parsing trajectories from {args.input_file}")
    print(f"Output will be saved to {args.output_file}")
    print()

    # Process trajectories
    count = 0
    reparsed_count = 0

    with open(args.input_file, 'r') as infile, open(args.output_file, 'w') as outfile:
        for line in infile:
            if args.limit and count >= args.limit:
                break

            count += 1

            # Parse trajectory
            trajectory = json.loads(line)

            # Re-parse
            reparsed = reparse_trajectory(trajectory)
            reparsed_count += 1

            # Write to output
            outfile.write(json.dumps(reparsed) + '\n')

            # Progress
            if count % 100 == 0:
                print(f"Processed {count} trajectories...")

    print()
    print(f"✓ Re-parsed {reparsed_count} trajectories")
    print(f"✓ Output saved to {args.output_file}")

    # Show sample
    print()
    print("Sample re-parsed step:")
    with open(args.output_file, 'r') as f:
        sample = json.loads(f.readline())
        step1 = sample['steps'][0]
        print(f"  Step 1:")
        print(f"    thought: {step1['thought'][:100] if step1['thought'] else None}...")
        print(f"    action: {step1['action']}")
        print(f"    observation: {step1['observation'][:100] if step1['observation'] else None}...")

    return 0


if __name__ == '__main__':
    sys.exit(main())
