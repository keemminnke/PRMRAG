#!/usr/bin/env python3
"""View individual trajectories in detail for manual inspection.

This script lets you browse through trajectories to manually verify
whether RAG actually improved reasoning quality.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def print_trajectory_detail(result: Dict[str, Any], show_passages: bool = False):
    """Print a single trajectory in detail."""

    print("\n" + "=" * 70)
    print(f"TRAJECTORY: {result['question_id']}")
    print("=" * 70)

    # Question and answer
    print(f"\n📝 Question:")
    print(f"   {result['question']}")

    print(f"\n✅ Gold Answer:")
    print(f"   {result['gold_answer']}")

    status = "✅ CORRECT" if result['is_correct'] else "❌ INCORRECT"
    print(f"\n🤖 Predicted Answer: {status}")
    print(f"   {result['predicted_answer']}")

    # Summary
    print(f"\n📊 Summary:")
    print(f"   Total steps: {result['num_steps']}")
    print(f"   - CoT steps: {result['num_cot_steps']}")
    print(f"   - RAG steps: {result['num_rag_steps']}")

    # Step-by-step breakdown
    print(f"\n🔍 Step-by-Step Breakdown:")
    print("   " + "-" * 66)

    for step in result['steps']:
        step_type = "🔵 CoT" if step['type'] == 'cot' else "🟢 RAG"
        label_marker = "✓" if step['label'] == 'good' else "✗"

        print(f"\n   Step {step['step_num']} {step_type} [{label_marker} {step['label']}]")
        print(f"   MC: {step['mc_before']:.3f} → {step['mc_after']:.3f} (RPE: {step['rpe']:.3f})")

        # Show content
        content_lines = step['content'].split('\n')
        for line in content_lines:
            if line.strip():
                print(f"   │ {line}")

        # Show passages if RAG step
        if step['is_rag'] and show_passages:
            print(f"   │")
            print(f"   │ Retrieved passages: {step['num_passages']}")
            for i, title in enumerate(step.get('passage_titles', []), 1):
                print(f"   │   {i}. {title}")

        print(f"   " + "─" * 66)

    # RAG Impact Summary
    if result['has_rag']:
        print(f"\n💡 RAG Impact Summary:")
        for intervention in result['rag_interventions']:
            label_marker = "✓" if intervention['label'] == 'good' else "✗"
            mc_change = intervention['mc_improvement']
            mc_marker = "↑" if mc_change > 0 else "↓" if mc_change < 0 else "="

            print(f"   Step {intervention['step_num']}: MC {mc_marker} {mc_change:+.3f} [{label_marker} {intervention['label']}]")
            if show_passages:
                print(f"      Passages: {', '.join(intervention['passage_titles'][:3])}")


def main():
    parser = argparse.ArgumentParser(
        description="View trajectories in detail for manual inspection"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to results JSONL file from batch_test_adaptive.py",
    )
    parser.add_argument(
        "--filter-rag",
        action="store_true",
        help="Only show trajectories that used RAG",
    )
    parser.add_argument(
        "--filter-correct",
        action="store_true",
        help="Only show correct trajectories",
    )
    parser.add_argument(
        "--filter-incorrect",
        action="store_true",
        help="Only show incorrect trajectories",
    )
    parser.add_argument(
        "--show-passages",
        action="store_true",
        help="Show retrieved passage titles",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        help="Show specific question by ID",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of trajectories to show (default: 10)",
    )
    args = parser.parse_args()

    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)

    # Load results
    print(f"Loading results from: {results_file}")
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Loaded {len(results)} results")

    # Filter by question ID
    if args.question_id:
        results = [r for r in results if r['question_id'] == args.question_id]
        if not results:
            print(f"Error: Question ID not found: {args.question_id}")
            sys.exit(1)

    # Apply filters
    if args.filter_rag:
        results = [r for r in results if r['has_rag']]
        print(f"Filtered to {len(results)} results with RAG")

    if args.filter_correct:
        results = [r for r in results if r['is_correct']]
        print(f"Filtered to {len(results)} correct results")

    if args.filter_incorrect:
        results = [r for r in results if not r['is_correct']]
        print(f"Filtered to {len(results)} incorrect results")

    # Limit
    if len(results) > args.limit:
        print(f"Showing first {args.limit} of {len(results)} results")
        results = results[:args.limit]

    # Display trajectories
    for i, result in enumerate(results, 1):
        print(f"\n\n{'#' * 70}")
        print(f"# TRAJECTORY {i} of {len(results)}")
        print(f"{'#' * 70}")
        print_trajectory_detail(result, show_passages=args.show_passages)

    print("\n" + "=" * 70)
    print("END OF TRAJECTORIES")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
